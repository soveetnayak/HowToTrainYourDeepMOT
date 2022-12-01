import torch
from torch.utils import data
import numpy as np
import os

class Dataset(data.Dataset):

    def __init__(self, data_dir,train=True):
        self.data_dir = data_dir
        self.train = train
        self.data = self.load_data()
        self.labels = self.load_labels()

    def load_data(self):
        data = []
        if self.train:
            des = os.path.join(self.data_dir, 'train/')
        else:
            des = os.path.join(self.data_dir, 'test/')

        dirs = os.listdir(des)
        for dir in dirs:
            files_path = os.path.join(des, dir)
            files = os.listdir(files_path)
            for file in files:
                if '_m.npy' in file:
                    data.append(os.path.join(files_path, file))
        return data

    def load_labels(self):
        labels = []
        if self.train:
            des = os.path.join(self.data_dir, 'train/')
        else:
            des = os.path.join(self.data_dir, 'test/')

        for d in self.data:
            sp = d.split('_')
            label = '_'.join(sp[0:-1]) + '_t.npy'
            labels.append(label)
        return labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = np.load(self.data[index]).astype(np.float32)
        label = np.load(self.labels[index]).astype(np.int32)

        data = torch.from_numpy(data)
        label = torch.from_numpy(label)

        if torch.rand(1).item() < 0.5:
            idx = torch.randperm(data.shape[1])
            data = data[0:, idx, 0:]
            label = label[0:, idx, 0:]
            print('vertical flip')

        if torch.rand(1).item() < 0.5:
            idx = torch.randperm(data.shape[2])
            data = data[0:, 0:, idx]
            label = label[0:, 0:, idx]
            print('horizontal flip')


        return data, label


# data_dir = './Data/'
# train_dataset = Dataset(data_dir, train=True)
# test_dataset = Dataset(data_dir, train=False)

# train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
# test_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# print(len(train_dataset))
# print(len(test_dataset))

# for i, (data, label) in enumerate(train_loader):
#     print(data.shape)
#     print(data)
#     print(label.shape)
#     print(label)
#     break

