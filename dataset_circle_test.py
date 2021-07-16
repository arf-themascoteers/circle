from dataset_circle import DatasetCircle
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = DatasetCircle(radius=10)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

for x, y, truth in dataloader:
    for i in range(10):
        color = "red"
        if truth[i].item():
            color = "green"
        plt.scatter(x[i].item(), y[i].item(), color=color)

plt.show()