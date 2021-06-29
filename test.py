import torch
import torch.nn as nn

class toy(nn.Module):
    def __init__(self):
        super(toy, self).__init__()
        self.layer1 = nn.Linear(256, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x

    torch.Tensor()