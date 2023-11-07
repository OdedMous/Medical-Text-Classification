import torch
import torch.nn as nn
import torch.nn.functional as F

class BERT_Arch(nn.Module):

    def __init__(self, bert):

      super(BERT_Arch, self).__init__()

      self.bert = bert
      self.conv1 = nn.Conv1d(in_channels=768, out_channels=128, kernel_size=3, stride=1) # kernal_size=3 == three-grams
      self.avg_pooling = nn.AvgPool1d(kernel_size=2)
      self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1)
      self.flatten = nn.Flatten()
      self.fc = nn.Linear(64,128)
      self.dropout = nn.Dropout(0.2)

    def forward(self, seq, mask):


      hs, cls_hs = self.bert(seq, attention_mask=mask, return_dict=False)

      x = hs.permute(0, 2, 1).contiguous()          # Permute `hs` to match input shape requirement of `nn.Conv1d`
                                                    # The contiguous() ensures the memory of the tensor is stored contiguously
                                                    # which helps avoid potential issues during processing.
                                                    # Output shape: (b, 768, 70) = (b, embed_dim, max_len_seq).

      x = F.relu(self.conv1(x))                     # Output shape: (b, 128, *)  * depends on kernel size and padding
      x = self.avg_pooling(x)                       # Output shape: (b, 128, *)
      x = F.relu(self.conv2(x))                     # Output shape: (b, 128, *)
      x = F.max_pool1d(x, kernel_size=x.shape[2])   # Output shape: (b, 128, 1) # trick: we use kernel of size x.shape[2] to reduce from * to 1
      x = self.flatten(x)                           # Output shape: (b, 128)
      x = self.fc(x)                                # Output shape: (b, 128)
      x = self.dropout(x)

      return x


class SiameseNeuralNetwork(nn.Module):

    def __init__(self, bert_arch):
        super().__init__()

        self.bert_arch = bert_arch
        self.distance_layer = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())  # if we would use BCEWithLogitsLoss as loss function, we should delte the sigmoid since we dont need it after the linear layer a sigmoid layer


    def forward(self, seq1, seq2, mask1, mask2):
        feature_vec1 = self.bert_arch(seq1, mask1) # feature_vec1 shape:  [batch_size, embedding_size]
        feature_vec2 = self.bert_arch(seq2, mask2)
        difference = torch.abs(feature_vec1 - feature_vec2)
        out = self.distance_layer(difference)
        return out

class ContrastiveLoss(nn.Module):
    """
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() *  torch.nn.functional.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()
