from circle_net import CircleNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from dataset_circle import DatasetCircle

def geo_loss(data, radius, y_pred):
    x = data[:,0]
    y = data[:,1]
    dis = torch.abs(radius - torch.sqrt(x**2 + y**2))
    dis = dis * y_pred[:,0]
    dis = torch.mean(torch.sqrt(dis))
    return dis

def train():
    NUM_EPOCHS = 20
    BATCH_SIZE = 10000

    dataset = DatasetCircle(10, is_train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = CircleNet()
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    LAMBDA = 0.1
    for epoch in range(0, NUM_EPOCHS):
        for data, y_true in dataloader:
            optimizer.zero_grad()
            y_pred = model(data)
            gl = geo_loss(data, 10, y_pred )
            loss = criterion(y_pred, y_true) + LAMBDA *  gl
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    print("Training done. Machine saved to models/ckd.h5")
    torch.save(model.state_dict(), 'models/ckd.h5')
    return model


if __name__ == "__main__":
    train()




