import torch
import numpy as np
import os


def mnist():
    # exchange with the corrupted mnist dataset
    train_filenames = [f'train_{i}.npz' for i in range(5)]
    data_path = '../../../data/corruptmnist'
    loaded_train_files = [np.load(os.path.join(data_path, i)) for i in train_filenames]
    loaded_train_images = [torch.tensor(i['images']) for i in loaded_train_files]
    loaded_train_labels = [torch.tensor(i['labels']) for i in loaded_train_files]
    train_images = torch.concat(loaded_train_images)
    train_labels = torch.concat(loaded_train_labels)

    test_file = np.load(os.path.join(data_path, 'test.npz'))
    test_images = test_file['images']
    test_labels = test_file['labels']
    
    train = {'images': train_images, 'labels': train_labels}
    test = {'images': test_images, 'labels': test_labels}
    return train, test


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        super().__init__()
        self.data_dict = data_dict

    def __getitem__(self, i):
        return self.data_dict['images'][i], self.dadta_dict['lables'][i]

    def __len__(self):
        return self.data_dict['images'].shape[0]


if __name__ == '__main__':
    train, test = mnist()
