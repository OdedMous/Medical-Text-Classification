from torch.utils.data import Dataset, DataLoader
import torch
import random

class ContrastiveDataset(Dataset):

    def __init__(self, train_seq, train_mask, train_y, positive_prob=0.5):

        super().__init__()
        self.train_seq = train_seq
        self.train_mask = train_mask
        self.train_y = train_y
        self.positive_prob = positive_prob  # probability to sample two texts with the same category

        self.hash_table = {}  # format: {"category" : [i1, i2, ...]}

        # construct a hash table, each key is a category
        # and the value is a list of the indexs of the texts which belong to this category
        for i in range(len(self.train_seq)):
            label = self.train_y[i].item()
            if label in self.hash_table:
                self.hash_table[label].append(i)
            else:
                self.hash_table[label] = [i]

    def __getitem__(self, index):
        """
        Sample two texts from the same category with probability self.positive_prob
        :param index: index (int)
        :return:  seq_0 - a sequence of IDs (each ID represent a word in the vocabulary)
                  seq_1 - a sequence which differnt from seq0 (different text)
                  mask_0 - attention mask for seq1
                  mask_1 - attention mask for seq1
                  same_class - 1 if seq0 and seq1 are both from the same category, 0 otherwise
        """
        same_class = random.uniform(0, 1)
        same_class = same_class > self.positive_prob

        seq_0 = self.train_seq[index]
        mask_0 = self.train_mask[index]
        label_0 = self.train_y[index].item()
        class_samples = self.hash_table[label_0]

        if len(class_samples) < 2:  # handle the case where there are only a single text in some category (in this case we can't draw another text from this category...)
            same_class = False

        if same_class:
            while True:
                rnd_idx = random.randint(0, len(class_samples) - 1)
                index_1 = class_samples[rnd_idx]
                if index_1 != index:
                    seq_1 = self.train_seq[index_1]
                    mask_1 = self.train_mask[index_1]
                    label_1 = self.train_y[index_1].item()
                    break
        else:
            while True:
                index_1 = random.randint(0, self.__len__() - 1)
                if index_1 != index:
                    seq_1 = self.train_seq[index_1]
                    mask_1 = self.train_mask[index_1]
                    label_1 = self.train_y[index_1].item()
                    if label_1 != label_0:
                        break

        return seq_0, seq_1, mask_0, mask_1, torch.tensor(same_class, dtype=torch.float)

    def __len__(self):
        return len(self.train_seq)


class SimpleDataset(Dataset):

    def __init__(self, seq, mask, y):
        super().__init__()
        self.seq = seq
        self.mask = mask
        self.y = y

    def __getitem__(self, index):
        """
        Sample texts by the order of the training set.
        :param index: index (int)
        :return: seq - a sequence of IDs (each ID represent a word in the vocabulary)
                 mask - attention mask for seq
                 y - the category of this text

        """
        return self.seq[index], self.mask[index], torch.tensor(self.y[index].item())

    def __len__(self):
        return len(self.seq)