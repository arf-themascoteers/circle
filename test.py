from circle_net import CircleNet
from torch.utils.data import DataLoader
import torch
from dataset_circle import DatasetCircle
import matplotlib.pyplot as plt

def test():
    BATCH_SIZE = 1000
    dataset = DatasetCircle(10, is_train=False)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = CircleNet()
    model.load_state_dict(torch.load("models/ckd.h5"))
    model.eval()
    correct = 0
    total = 0
    print(f"Test started ...")
    with torch.no_grad():
        for data, y_true in dataloader:
            y_pred = model(data)
            pred = torch.argmax(y_pred, dim=1, keepdim=True)
            correct += pred.eq(y_true.data.view_as(pred)).sum()
            total += 1
            for i in range(len(data)):
                color = "red"
                if pred[i].item() == 1:
                    color = "green"
                plt.scatter(data[i][0].item(), data[i][1].item(), color=color)

    print(f"{correct} correct among {len(dataset)}")
    plt.show()

if __name__ == "__main__":
    test()