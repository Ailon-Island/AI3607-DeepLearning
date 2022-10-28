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


class CIFAR10Perm(Dataset):
    def __init__(self, opt, train=True, shuffle=False, augment=None, transform=None, target_transform=None):
        super(CIFAR10Perm, self).__init__()
        self.data_dir = os.path.join(opt.data_root, 'cifar-10-batches-py')
        self.train = train
        self.batch_size = opt.batch_size
        self.shuffle = shuffle

        self.total_len = 0
        self.data = []
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
            self.data = np.concatenate(self.data, axis=0)
        else:  # test
            data_dict = unpickle(os.path.join(self.data_dir, 'test_batch'))
            self.data = data_dict[b'data']
        self.data = self.data.reshape(-1, 3, 32, 32)
        self.data = self.data.transpose(0, 2, 3, 1)

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data = self.data[idx]
        data = Image.fromarray(data)

        if self.train and self.augment:
            data = self.augment(data)
        if self.transform:
            data = self.transform(data)

        H_2, W_2 = data.shape[1] // 2, data.shape[2] // 2
        perm = [data[:, :H_2, :W_2], data[:, :H_2, W_2:], data[:, H_2:, :W_2], data[:, H_2:, W_2:]]
        perm = np.stack(perm, axis=0)
        target = jt.randperm(4)
        perm = perm[target]

        return perm, target


