from data import npz_to_tensors, get_image_and_label_tensors, MNISTDataset
import torch

def test_npz_to_tensors():
    tensors_dict = npz_to_tensors('/home/victor/Programming/dtu_mlops/data/corruptmnist/train_0.npz')
    assert type(tensors_dict) == dict
    assert type(tensors_dict['images'] == torch.Tensor)
    assert type(tensors_dict['labels']  == torch.Tensor)
    assert tensors_dict['images'].shape[0] == tensors_dict['labels'].shape[0]

def test_get_image_and_labels_tensors():
    data_dict = get_image_and_label_tensors()
    assert type(data_dict) == dict
    train_dict = data_dict['train']
    test_dict = data_dict['test']
    assert type(train_dict['images'] == torch.Tensor)
    assert type(train_dict['labels']  == torch.Tensor)
    assert type(test_dict['images'] == torch.Tensor)
    assert type(test_dict['labels']  == torch.Tensor)
    assert train_dict['images'].shape[0] == train_dict['labels'].shape[0]
    assert test_dict['images'].shape[0] == test_dict['labels'].shape[0]

def test_dataset():
    images = torch.randn((2, 64, 64))
    labels = torch.randn(2)
    data_dict = {'images': images, 'labels': labels}
    ds = MNISTDataset(data_dict)

    sample = ds[0]
    assert sample[0].shape == (64, 64)
    assert sample[1].ndim == 0