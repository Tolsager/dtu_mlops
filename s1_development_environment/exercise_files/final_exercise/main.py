import argparse
import sys

import torch
import click

from data import get_image_and_label_tensors, MNISTDataset
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
@click.option("--epochs", default=10, help='number of epochs to train for')
@click.option("--batch_size", default=64, help='amount of samples in each batch')
@click.option("--device", default='cuda', help='the device to train the model on')
def train(lr, epochs, batch_size, device):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    train_dict = get_image_and_label_tensors()['train']
    ds_train = MNISTDataset(train_dict)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        n_correct = 0
        n_samples = 0
        for images, labels in dl_train:
            images = images.to(device)

            out = model(images)
            labels = labels.type(torch.LongTensor)
            labels = labels.to(device)
            loss = criterion(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metrics
            n_samples += labels.shape[0]
            _, predictions = out.topk(1, dim=1)
            predictions = predictions.view(-1)
            n_correct += (predictions == labels).sum()
        
        print(f"Accuracy: {n_correct / n_samples}")

    checkpoint = {'device': device, 'state_dict': model.state_dict()}
    torch.save(checkpoint, 'checkpoint.pth')



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = MyAwesomeModel()

    checkpoint = torch.load(model_checkpoint)
    device = checkpoint['device']
    model.to(device)
    model.load_state_dict(checkpoint['state_dict'])

    test_dict = get_image_and_label_tensors()['test']
    ds_test = MNISTDataset(test_dict)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=64)

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in dl_test:
            images = images.to(device)

            out = model(images)
            labels = labels.to(device)

            # metrics
            n_samples += labels.shape[0]
            _, predictions = out.topk(1, dim=1)
            predictions = predictions.view(-1)
            n_correct += (predictions == labels).sum()
        
        print(f"Accuracy: {n_correct / n_samples}")

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

  