from dataset import *
import torch.utils.data

batch_size = 10
epochs = 100
def train():
    dataset = WikiArt("data", 64)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, False)
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):









if __name__ == '__main__':
    print('PyCharm')

