import torch
from transformers import AdamW

def train_siamese_network(model, dataloaders, num_epochs, device):
    """
    Train the given SNN model.

    :param model: SNN model
    :param dataloaders: a dict that contains train data loader and validation data loader
    :param num_epochs: number of epochs
    :param device: 'cpu' or 'cuda'

    :return:  train_loss_history - list of train losses by epochs
              val_loss_history -  list of validation losses by epochs

    """
    train_loss_history = []
    val_loss_history = []
    matching_similarity = []
    non_matching_similarity = []

    val_matching_similarity = []
    val_non_matching_similarity = []

    criterion = torch.nn.BCELoss(reduction='mean') #ContrastiveLoss(margin=1)  #losses.ContrastiveLoss(pos_margin=0, neg_margin=1) # torch.nn.BCEWithLogitsLoss(reduction='mean') # the labels are same class (1) vs. different class (0)
    learning_rate = 0.005 # 0.005 # 0.1
    optimizer  =  AdamW(model.parameters(),lr =learning_rate)#torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # AdamW(model.parameters(),lr =learning_rate)  1e-5

    # lr = lr * factor
    # mode='min': look for the min validation loss to track
    # patience: number of epochs - 1 where loss plateaus before decreasing LR
    # patience = 0, after 1 bad epoch, reduce LR
    # factor: decaying factor

    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True, min_lr=0.0001)  ########################################################
    #cyclic_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1, cycle_momentum=False) ########################################################

    for epoch in range(num_epochs):  # loop over the train dataset multiple times

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for i, batch in enumerate(dataloaders[phase]):

                  seq1, seq2, mask1, mask2, label = batch

                  if device == 'cuda':
                    seq1, seq2, mask1, mask2, label = seq1.to(device), seq2.to(device), mask1.to(device), mask2.to(device), label.to(device)

                  # zero the parameter gradients
                  optimizer.zero_grad()

                  # track history  only in train
                  with torch.set_grad_enabled(phase == 'train'):

                      # forward
                      output = model.forward(seq1, seq2, mask1, mask2)
                      loss = criterion(output, label.view(output.size())) # criterion(output.squeeze(0), label.view(1))    label.view((trainLoader.batch_size,1))

                      # backward + optimize only if in training phase
                      if phase == 'train': #  with torch.no_grad() if phae == 'val'?
                          loss.backward()
                          torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
                          optimizer.step()
                          #cyclic_scheduler.step() ########################################################

                          # save similarity scores for training data
                          output = output.cpu().detach().numpy()
                          label = label.cpu().numpy()
                          non_matching_similarity.append((sum(output[label == 0]) / sum(label == 0)).item())
                          matching_similarity.append((sum(output[label == 1]) / sum(label == 1)).item())

                      if phase == 'val':
                         val_non_matching_similarity.append((sum(output[label == 0]) / sum(label == 0)).item())
                         val_matching_similarity.append((sum(output[label == 1]) / sum(label == 1)).item())

                  running_loss += loss.item() * seq1.size(0)  #we multiply by the batch size (note that the batch size in the last batch may not be the batch size we did since the batch size dont necceraly divide the train size)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if phase == 'train':
               train_loss_history.append(epoch_loss)
            else:
               val_loss_history.append(epoch_loss)
               #scheduler.step(epoch_loss) ########################################################

            print('Epoch {} | {} loss: {:.3f}'.format(epoch, phase, epoch_loss))


    return train_loss_history, val_loss_history, [non_matching_similarity, matching_similarity, val_non_matching_similarity, val_matching_similarity]

