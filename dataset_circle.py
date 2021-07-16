from torch.utils.data import Dataset
import random
import torch

class DatasetCircle(Dataset):
    def __init__(self, radius):
        self.radius = radius
        self.size = 10000
        self.x = torch.linspace(-radius, radius, self.size)
        self.y = self._x_to_y(self.x)

        fake_x = torch.linspace(-radius, radius, self.size)
        fake_y = torch.rand_like(fake_x)*self.radius
        for i in range(len(fake_y)):
            if(random.randint(0,1)==1):
                fake_y[i] = -fake_y[i]

        self.x = torch.cat((self.x, fake_x),0)
        self.y = torch.cat((self.y, fake_y),0)

    def _x_to_y(self, x):
        y = torch.sqrt(self.radius**2 - x**2)
        for i in range(len(y)):
            if(random.randint(0,1)==1):
                y[i] = -y[i]
        return y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        truth = 0
        if idx > self.y.shape[0]/2:
            truth = 1
        return torch.tensor([self.x[idx], self.y[idx]]), truth
