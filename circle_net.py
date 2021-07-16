import torch.nn as nn
import torch.nn.functional as F


class CircleNet(nn.Module):
    def __init__(self):
        super(CircleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 10),
            nn.LeakyReLU(0.2),
            nn.Linear(10,5),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(0.2),
            nn.Linear(5, 2)
        )

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=1)