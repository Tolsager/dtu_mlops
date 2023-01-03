from torch import nn
import torch.nn.functional as F


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = 784
        fc1_neurons = 300
        fc2_neurons = 100
        fc3_neurons = 10
        self.fc1 = nn.Linear(in_dim, fc1_neurons)
        self.fc2 = nn.Linear(fc1_neurons, fc2_neurons)
        self.fc3 = nn.Linear(fc2_neurons, fc3_neurons)
        self.bn1 = nn.BatchNorm1d(fc1_neurons)
        self.bn2 = nn.BatchNorm1d(fc2_neurons)
    
    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = self.bn1(x)

        x = F.relu(self.fc2(x))
        x = self.bn2(x)

        x = self.fc3(x)
        return x



