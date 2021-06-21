import torch
import numpy as np
import matplotlib.pyplot as plt
import subprocess


# import pandas as pd
# import matplotlib
# from matplotlib import font_manager


def load_npz_kmnist(folder, files):
    """
    Load npz files from disk and make tensors of them
    :param folder: str. specifying the folderpath where the images reside
    :param files: list of str. specifying the npz in folder to be loaded.
    :return: list of tensors according to the order of the npz files
    """
    return [torch.Tensor(np.load(folder + p)['arr_0']) for p in files]


def plot_kmnist(x_array, y_array, idx, labelpath=None):
    """
    Plot a single example
    :param x_array: 3d np.array (imageindex, height, width)
    :param y_array: 1d np.array (label for respective imageindex)
    :param labelpath: path to csv, where the images labels are converted to
    japanese (non-ascii) characters
    :param idx: int. image index, which is to be plotted
    (first dimension of the array).
    :return: None. plotting an image.
    """

    # TODO: add japanese (non-ascii) language support to matplotlib to display
    # matplotlib.rc('font', family='TakaoPGothic')
    # label_mapping = pd.read_csv(labelpath)
    # label = label_mapping['char'].iloc[y_array[idx]]
    # plt.title('label class:{}, label:{}'.format(label, y_array[idx]))

    plt.title('label class:{}'.format(y_array[idx]))
    plt.imshow(x_array[idx], cmap='gray')


def get_git_revision_short_hash():
    """
    Get the current (short) commit-hash.
    Code taken from https://stackoverflow.com/a/21901260.
    """
    return subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD']).strip().decode()