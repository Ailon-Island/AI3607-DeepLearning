from jittor.dataset.dataset import Dataset
import jittor as jt
import os
from PIL import Image
import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


class CIFAR10(Dataset):
    def __init__(self, opt, train=True, shuffle=False, augment=None, transform=None, target_transform=None):
        super().__init__()
        self.data_dir = os.path.join(opt.data_root, 'cifar-10-batches-py')
        self.train = train
        self.batch_size = opt.batch_size
        self.shuffle = shuffle
        if self.train:
            self.imbalanced = opt.imbalanced_data
            if self.imbalanced:
                self.rebalanced = opt.rebalanced_data

        self.total_len = 0
        self.data, self.targets, self.num_data = [], [], []
        self.get_data()
        if self.total_len == 0:
            self.total_len = len(self.data)

        if self.train:
            self.augment = augment
        self.transform = transform
        self.target_transform = target_transform

        # set_attrs must be called to set batch size total len and shuffle like __len__ function in pytorch
        self.set_attrs(batch_size=self.batch_size, total_len=self.total_len,
                       shuffle=self.shuffle)  # bs , total_len, shuffle

    def get_data(self):
        if self.train:
            for i in range(1, 6):
                data_dict = unpickle(os.path.join(self.data_dir, 'data_batch_{}'.format(i)))
                self.data += [data_dict[b'data']]
                self.targets += data_dict[b'labels']
            self.data = np.concatenate(self.data, axis=0)
        else:  # test
            data_dict = unpickle(os.path.join(self.data_dir, 'test_batch'))
            self.data = data_dict[b'data']
            self.targets = data_dict[b'labels']
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose(0, 2, 3, 1)
        self.targets = np.array(self.targets)

        if self.train and self.imbalanced:
            if self.rebalanced:
                self.total_len = len(self.data)
            data, targets = self.data, self.targets
            self.data, self.targets = [], []
            for label in range(5):
                idx = np.where(targets == label)[0]
                num = len(idx)//10
                np.random.shuffle(idx)
                idx = idx[:num]
                self.data += [data[idx]]
                self.targets += [targets[idx]]
                self.num_data += [num]
            mask = targets >= 5
            self.data += [data[mask]]
            self.targets += [targets[mask]]
            self.data = np.concatenate(self.data, axis=0)
            self.targets = np.concatenate(self.targets, axis=0)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        if self.train and self.imbalanced and self.rebalanced:
            if idx < self.total_len // 2:
                idx = idx // 10
            else:  # normal half
                idx -= self.total_len * 9 // 20
        data, target = self.data[idx], self.targets[idx]
        data = Image.fromarray(data)

        if self.train and self.augment:
            data = self.augment(data)
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            target = self.target_transform(target)

        return data, target
