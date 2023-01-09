import numpy as np
import torch
import numpy as np
import os

def npz_to_tensors(path: str) -> dict:
    npz_object = np.load(path)
    images = npz_object['images']
    labels = npz_object['labels']

    # convert numpy arrays to tensors
    images = torch.tensor(images, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    return {'images': images, 'labels': labels}



def get_image_and_label_tensors() -> dict[str: dict]:
    data_path = '/home/victor/Programming/dtu_mlops/data/corruptmnist'
    n_train_files = 5

    train_filenames = [f'train_{i}.npz' for i in range(n_train_files)]
    train_dicts = [npz_to_tensors(os.path.join(data_path, i)) for i in train_filenames]
    train_images = torch.concat([i['images'] for i in train_dicts])
    train_labels = torch.concat([i['labels'] for i in train_dicts])

    test_file = npz_to_tensors(os.path.join(data_path, 'test.npz'))
    test_images = test_file['images']
    test_labels = test_file['labels']
    
    train = {'images': train_images, 'labels': train_labels}
    test = {'images': test_images, 'labels': test_labels}

    return {'train': train, 'test': test}

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        super().__init__()
        self.data_dict = data_dict

    def __getitem__(self, i):
        return self.data_dict['images'][i].type(torch.float32), self.data_dict['labels'][i]

    def __len__(self):
        return self.data_dict['images'].shape[0]
