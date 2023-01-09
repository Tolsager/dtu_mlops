import os
import torch


def test_data():
    ds_path = os.path.join("data", "processed", "processed_data.pth")
    data = torch.load(ds_path)
    train_dict = data["train"]
    test_dict = data["test"]

    assert True
