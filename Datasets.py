import os

import numpy as np
import torch
from PIL import Image


class DatasetObject:
    def __init__(self, dataset, n_client, seed, result_path='', data_dir='./data'):
        self.dataset = dataset
        self.n_client = n_client
        self.seed = seed
        self.data_dir = data_dir
        self.name = "%s_%d_%d" % (self.dataset, self.n_client, self.seed)
        self.result_path = result_path
        self.set_data()

    def set_data(self):
        if self.dataset in ['DIGIT']:
            self.channels = 3
            self.width = 32
            self.height = 32
            self.n_cls = 10

        if self.dataset == 'DIGIT':
            dir_name = 'digit_dataset'
            curruption_name_list = ['MNIST', 'MNIST_M', 'SVHN', 'SynthDigits', 'USPS']

            mean = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape((1, 1, 1, 3))
            std = np.array([0.5, 0.5, 0.5], dtype=np.float32).reshape((1, 1, 1, 3))

            if self.n_client != len(curruption_name_list):
                raise ValueError()

            clnt_x = []
            clnt_y = []
            for _ in range(self.n_client):
                clnt_x.append(list())
                clnt_y.append(list())

            tst_x = []
            tst_y = []

            for client_idx, curruption_name in enumerate(curruption_name_list):
                npy_path = os.path.join(self.data_dir, dir_name, curruption_name, 'test.pkl')
                data, label = np.load(npy_path, allow_pickle=True)

                _data = []
                for img in data:
                    img = Image.fromarray(img).convert('RGB').resize((32, 32))
                    img = np.array(img)
                    _data.append(img)
                data = np.asarray(_data)

                data = data.astype(np.float32) / 255
                data = (data - mean) / std
                data = data.transpose((0, 3, 1, 2))
                tst_x.extend(data)
                tst_y.extend(label)

            for client_idx, curruption_name in enumerate(curruption_name_list):
                data, label = [], []
                npy_path = os.path.join(self.data_dir, dir_name, curruption_name, 'partitions', f'train_part0.pkl')
                data_part, label_part = np.load(npy_path, allow_pickle=True)
                data.extend(data_part)
                label.extend(label_part)

                data = data[:(len(data) // 2)]
                label = label[:(len(label) // 2)]

                _data = []
                for img in data:
                    img = Image.fromarray(img).convert('RGB').resize((32, 32))
                    img = np.array(img)
                    _data.append(img)
                data = np.asarray(_data)

                data = data.astype(np.float32) / 255
                data = (data - mean) / std
                data = data.transpose((0, 3, 1, 2))
                clnt_x[client_idx].extend(data)
                clnt_y[client_idx].extend(label)

            for client_idx in range(self.n_client):
                clnt_x[client_idx] = np.asarray(clnt_x[client_idx])
                clnt_y[client_idx] = np.asarray(clnt_y[client_idx])

        else:
            raise ValueError()

        self.clnt_x = clnt_x
        self.clnt_y = clnt_y
        self.tst_x = tst_x
        self.tst_y = tst_y

        print('Class frequencies:')
        count = 0
        for clnt in range(self.n_client):
            print("Client %3d: " % clnt +
                  ', '.join(["%.3f" % np.mean(self.clnt_y[clnt] == cls) for cls in range(self.n_cls)]) +
                  ', Amount:%d' % len(self.clnt_y[clnt]))
            count += len(self.clnt_y[clnt])

        print('Total Amount:%d' % count)
        print('--------')

        print("      Test: " +
              ', '.join(["%.3f" % np.mean(self.tst_y == cls) for cls in range(self.n_cls)]) +
              ', Amount:%d' % len(self.tst_y))


class Dataset(torch.utils.data.Dataset):

    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        data_y = np.array(data_y)  # convert to numpy
        if self.name == 'DIGIT':
            self.X_data = torch.tensor(np.array(data_x)).float()
            self.y_data = data_y
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        if self.name == 'DIGIT':
            X = self.X_data[idx, :]
            y = self.y_data[idx]
            return X, y
