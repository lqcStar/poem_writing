import numpy as np
import torch
from torch.utils.data import Dataset


class PoetryDataset(Dataset):
    def __init__(self, path):
        datas = np.load(path, allow_pickle=True)
        data = datas['data']
        self.ix2word = datas['ix2word'].item()
        self.word2ix = datas['word2ix'].item()
        self.data = torch.from_numpy(data).long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][:-1], self.data[index][1:]


if __name__ == '__main__':
    dataset = PoetryDataset('tang.npz')
    id2word, word2id = dataset.ix2word, dataset.word2ix
    print(len(id2word))
    print(len(dataset))
    # print(id2word[7435])
    # for i in dataset.data[0]:
    #     print(id2word[i.item()], end=' ')
    # print(len(dataset.data))

