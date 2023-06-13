import copy

import torch
from torch.utils.data import DataLoader

from data_util import PoetryDataset
from model import RNN


def train(model, loss_func, optimizer, train_data, total_epoch, device):
    loss = 0
    for epoch in range(1, total_epoch + 1):
        for step, (train_x, train_y) in enumerate(train_data):
            out, _ = model(train_x.to(device))
            train_y = train_y.view(-1)
            loss = loss_func(out, train_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch-{:02} loss: {:.6f}'.format(epoch, loss.item()))
    return model


if __name__ == '__main__':
    # args
    total_epoch = 50
    bath_size = 32
    learning_rate = 3e-4

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda') if use_cuda else torch.device('cpu')

    # data
    dataset = PoetryDataset('tang.npz')
    id2word, word2id = dataset.ix2word, dataset.word2ix
    loader = DataLoader(
        dataset=dataset,
        batch_size=bath_size,
        shuffle=True,
    )

    vocab_size = len(word2id)

    # model
    model = RNN(vocab_size,
                embedding_dim=256,
                hidden_dim=512,
                num_layers=3,
                bidirectional=False,
                device=device).to(device)

    # train
    model = model.train()
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = train(model, loss_func, optimizer, loader, total_epoch, device)

    # save
    torch.save(model.state_dict(), 'params.pth')
