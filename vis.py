import numpy as np
import matplotlib.pyplot as plt
from utils import config
from utils import data_helper
import pandas as pd


def vis_data(data):
    data.plot(x='time', y=['value', 'label'])
    plt.title(data.name, '.')


def vis_datas(data, idx):
    for i in idx:
        vis_data(data[i])
    plt.show()


def main():
    cfg = config.config()
    train_data = data_helper.get_data(cfg.train_file)
    # train_data = data_helper.get_data(cfg.val_file)
    # apply the index for visualization
    vis_datas(train_data, list(range(27)))


if __name__=='__main__':
    main()