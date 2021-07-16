from circle_net import CircleNet
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from dataset_circle import DatasetCircle

def train():
    NUM_EPOCHS = 200
    BATCH_SIZE = 10000

    dataset = DatasetCircle(10)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = CircleNet()
    model.train()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for epoch in range(0, NUM_EPOCHS):
        for data, y_true in dataloader:
            optimizer.zero_grad()
            y_pred = model(data)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    print("Training done. Machine saved to models/ckd.h5")
    torch.save(model.state_dict(), 'models/ckd.h5')
    return model


if __name__ == "__main__":
    train()




