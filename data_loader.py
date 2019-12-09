import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import re

class CustomDataset(Dataset):
    def __init__(self, data_root):
        # we should tokenize the input, but we will ignore that for now
        # build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
        self.data, self.labels, transform_word2num, self.vocab = self.read_data(path=data_root)
        unk = ['__PADDING__'] * 2
        self.data = unk + self.data + unk

        self.mapping = list(map(transform_word2num.get, self.data))
        self.samples = [[self.mapping[i - 2], self.mapping[i - 1], self.mapping[i],
                         self.mapping[i + 1], self.mapping[i + 2]]
                        for i in range(2, len(self.mapping) - 2)]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def read_data(path):
        """Read data from csv file, split into two groups (word, ner)
        Args:
        -----
          path: str, address of file.

        Returns:
        --------
          dict: dictionary of words as keys and part of speech as value

        """
        l_words = []
        l_labels = []

        with open(path) as file:
            for i, line in enumerate(file):
                try:
                    pairs = line.split('\t')
                    word, ner = pairs[0], re.sub(r'\W', '', pairs[1])
                    l_words.append(word), l_labels.append(ner)
                except IndexError:
                    pass

        arr_labels = np.array(l_labels)
        condlist = [arr_labels == 'LOC', arr_labels == 'PER',
                    arr_labels == 'O', arr_labels == 'ORG', arr_labels == 'MISC']
        choicelist = [0, 1, 2, 3, 4]
        labels = np.select(condlist, choicelist)

        vocab = set(l_words)
        word_to_num = {word: i for i, word in enumerate(vocab, 2)}
        word_to_num['__PADDING__'] = 0
        return (l_words, labels, word_to_num, vocab)

    def __getitem__(self, idx):
        # maybe add double __START__ in the begining of the corpus and __END__ in the end,
        # convert all numbers to __NUMBERS__, maybe in the test we doesn't have vocab words must
        # to add to vocab __UNKNOW__
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        sample = torch.tensor(self.samples[idx])

        return sample, label


if __name__ == '__main__':
    dataset = CustomDataset(
        data_root=r"/home/vova/PycharmProjects/Deep_Exe2/Ass2DL/data/ner/train")
    dataloader = DataLoader(dataset, batch_size=50, shuffle=True, num_workers=2)
    for i, batch in enumerate(dataloader):
        print(i, batch)


