import torch
from model import rnn
from utils import config
from utils import data_helper


def main():
    cfg = config.config()
    model = rnn.RNN(cfg.n_input, cfg.n_hidden, cfg.n_categories)
    train_data = data_helper.get_data(cfg.train_file)


if __name__ == '__main__':
    main()