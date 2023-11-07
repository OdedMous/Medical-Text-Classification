
import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
import consts

def construct_train_matrix(SNN_model, trainLoader_simple):
    """
    Embed the training data using the trained SNN model.
    """

    # Get intermidate layer output.
    # Source: https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301/2
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    # SNN_model.bert_arch.fc1.register_forward_hook(get_activation('fc1'))
    SNN_model.bert_arch.fc.register_forward_hook(get_activation('fc'))

    train_matrix = []
    print("num batches:", len(trainLoader_simple))
    with torch.no_grad():
        for i, batch in enumerate(trainLoader_simple):
            print(i, "/", len(trainLoader_simple), " batches")
            seq1, mask1, label1 = batch
            if consts.device == 'cuda':
                seq1, mask1, label1 = seq1.to(consts.device), mask1.to(consts.device), label1.to(consts.device)

            SNN_model.eval()
            output = SNN_model(seq1, seq1, mask1, mask1)
            train_matrix.append(activation[
                                    'fc'].cpu().numpy())  # activation['fc1'] return a tensor in cuda with size (batch_size, embedding_dim), so we move it to cpu, than to numpy array.

    return np.vstack(
        train_matrix)  # we combine all the batches, so now we return matrix of size (num_samples_train, embedding_dim)


def extract_prototypes(k, trainLoader_simple, train_labels, train_matrix):
    """
    Compute kc (= k/num_classes_train) prototypes for each class in the trainset.
    (if k % num_classes_train != 0 then take the highest k0 <= k which is divisable by num_classes_train)

    :param k:
    :param train_matrix: size (num_samples_train, embedding_dim)
    :return:
    """
    train_labels = list(train_labels)
    train_dataset = trainLoader_simple.dataset  # contains triples of (seq, mask, label)

    # construct a hash table, each key is a class of diagnosis
    # and the value is a list of the indexs of the sentences which belong to this class
    hash_table = {}  # format: {"diagnosis" : [i1, i2, ...]}
    for i in range(len(train_labels)):
        lbl = train_labels[i]
        if lbl in hash_table:
            hash_table[lbl].append(i)
        else:
            hash_table[lbl] = [i]

    # Create prototypess
    prototypes_list = {diagnosis: [] for diagnosis in hash_table.keys()}
    num_classes_train = len(hash_table)
    assert k >= num_classes_train, "k should be greater than the numbrer of uniqe labels in the train set'"
    kc = int(k / num_classes_train)
    print("kc:", kc)

    for diagnosis in hash_table.keys():

        print("diagnosis:", diagnosis)

        if len(hash_table[
                   diagnosis]) <= 1:  # if there is only a single sentence in some diagnosis sentences list - take it as the prototype of this class
            prototypes_list += list(train_matrix[hash_table[diagnosis]])

        else:
            # fit on all sentences which belongs to the same class (diagnosis)
            kmeans = KMeans(n_clusters=kc, init='k-means++').fit(train_matrix[hash_table[diagnosis]])
            # extract for each centroid the closest real sample, and add it as a prototype
            for centroid in kmeans.cluster_centers_:
                # print(train_matrix[hash_table[diagnosis]].shape)
                best_match_index = None
                best_match_dist = float('inf')
                for sentence_index in hash_table[diagnosis]:
                    # print(sentence_index)
                    embedded_sent = train_matrix[sentence_index]
                    dist = distance.euclidean(centroid, embedded_sent)
                    # print("dist:", best_match_dist)
                    if dist < best_match_dist:
                        best_match_dist = dist
                        best_match_index = sentence_index
                # print(best_match_index)
                prototypes_list[diagnosis].append(train_dataset[best_match_index])

    return prototypes_list

def project_to_dissimilarity_space(dataLoader, SNN_model, prototypes_list):
    """

    Parameters
    ----------
    dataLoader :
    SNN_model :


    Returns
    -------
    projected_data : numpy array of shape (num_samples_data, projection_dim)
    """

    projected_data = []
    with torch.no_grad():

        for batch in dataLoader:
            print("****new batch***")

            projected_sentence = []
            seq1, mask1, label1 = batch
            if consts.device == 'cuda':
                seq1, mask1, label1 = seq1.to(consts.device), mask1.to(consts.device), label1.to(consts.device)

            for diagnosis in prototypes_list:
                # print("diagnosis:", diagnosis)
                for centroid in prototypes_list[diagnosis]:  # centroid contains a triple of (seq, mask, label)
                    seq2, mask2 = centroid[0].repeat(seq1.shape[0], 1), centroid[1].repeat(mask1.shape[0],
                                                                                           1)  # we stack to seq2 and mask2 replications of them such that they will fit the batch size of seq1 and mask2
                    if consts.device == 'cuda':
                        seq2, mask2 = seq2.to(consts.device), mask2.to(consts.device)

                    SNN_model.eval()
                    distance = SNN_model(seq1, seq2, mask1, mask2)
                    projected_sentence.append(distance.squeeze().cpu().numpy())

            projected_data.append(np.array(projected_sentence).T)
            # break # TODO delete

    projected_data = np.vstack(projected_data)
    return projected_data