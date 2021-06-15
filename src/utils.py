import torch
import numpy as np
import matplotlib.pyplot as plt


# import pandas as pd
# import matplotlib
# from matplotlib import font_manager


def load_npz_kmnist(folder, files):
    """
    Load npz files from disk and make tensors of them
    :param folder: str.
    :param files: list of str.
    :return: list of tensors
    """
    return [torch.Tensor(np.load(folder + p)['arr_0']) for p in files]


def plot_kmnist(x_array, y_array, labelpath, idx):
    """

    :param x_array: 3d np.array (imageindex, height, width)
    :param y_array: 1d np.array (label for respective imageindex)
    :param labelpath: path to csv, where the images labels are converted to
    japanese (non-ascii) characters
    :param idx: int. image index, which is to be plotted
    :return: None. plotting an image
    """

    # TODO: add japanese (non-ascii) language support to matplotlib to display
    # matplotlib.rc('font', family='TakaoPGothic')
    # label_mapping = pd.read_csv(labelpath)
    # label = label_mapping['char'].iloc[y_array[idx]]
    # plt.title('label class:{}, label:{}'.format(label, y_array[idx]))

    plt.title('label class:{}'.format(y_array[idx]))
    plt.imshow(x_array[idx], cmap='gray')
