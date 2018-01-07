import torch
from torch.autograd import Variable
from torch import nn
from torch import optim
from model import rnn
from utils import config
from utils import data_helper
from itertools import product
from tqdm import tqdm


def train(inputs, labels, model, optimizer, criterion):
    hidden = model.initHidden()
    size = inputs.size()[0]
    model.zero_grad()
    loss = 0
    for i in tqdm(range(size)):
        output, hidden = model(inputs[i].unsqueeze(0), hidden)
        loss += criterion(output, labels[i])

    loss.backward()
    optimizer.step()

    return loss.data[0]/size


def process(data):
    inputs = []
    labels = []
    for d in data:
        t_inputs = d['value'].as_matrix()
        t_labels = d['label'].as_matrix()
        inputs.append(Variable(torch.from_numpy(t_inputs).float()))
        labels.append(Variable(torch.from_numpy(t_labels).long()))
    return inputs, labels

def main():
    cfg = config.config()
    model = rnn.RNN(cfg.n_input, cfg.n_hidden, cfg.n_categories)
    train_data = data_helper.get_data(cfg.train_file)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr)
    criterion = nn.NLLLoss()
    for epoch in range(cfg.max_epoch):
        inputs, labels = process(train_data)
        for idx in range(len(inputs)):
            print('[%d/%d] %d/%d'%(epoch+1, cfg.max_epoch, idx+1, len(inputs)))
            loss = train(inputs[idx], labels[idx], model, optimizer, criterion)
            print('[%d/%d] %d/%d loss: %.3f'%(
                  epoch+1, cfg.max_epoch, idx+1, len(inputs), loss))


if __name__ == '__main__':
    main()