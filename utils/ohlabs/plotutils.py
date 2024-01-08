import os.path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


funcs_name = ["alkane", "methyl", "alkene", "alkyne", "alcohols", "amines", "nitriles", "aromatics",
              "alkyl halides", "esters", "ketones", "aldehydes", "carboxylic acids",
              "ether", "acyl halides", "amides", "nitro"]


def plot_attention_map(spectra, result, att_map):
    if att_map is not None:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = plt.twinx().twiny()
        ax2.set_xlim(0, len(spectra))
        ax1.set_yticks(np.arange(len(funcs_name)), labels=funcs_name)
        # ax1.set_xlim(0, 64)

        tem = np.zeros_like(att_map)
        for i, r in enumerate(funcs_name):
            if r in result:
                tem[i, :] = att_map[i, :]

        ax1.imshow(tem[:, :], cmap='viridis', interpolation='nearest', aspect="auto")
        ax2.plot(spectra)
        plt.show()

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])

def plot_data(data_dis, save_dir="./", save_name="fg.png"):
    fig = plt.figure(figsize=(20, 6))
    x_pos = np.arange(len(funcs_name))
    plt.bar(funcs_name, data_dis[0], align='center', alpha=0.5)
    addlabels(x_pos, data_dis[0])
    plt.xticks(x_pos, funcs_name)
    plt.ylabel('Total samples')
    plt.title('Functional Groups Data Distribution')
    # plt.show()
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)

def plot_conf(conf, label=["1", "0"], title=None, save_dir="./", save_name='fg.png'):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=label)
    disp.plot()
    if title is not None:
        plt.title(title)
    # plt.show()
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)

def func_confusion(target, result, th=0.5):
    num_cls = len(funcs_name)
    conf_matrix = np.zeros((num_cls, 2, 2))
    target = target.astype(np.uint8)
    result = (result >= th).astype(np.uint8)
    for idx, name in enumerate(funcs_name):
        cf = confusion_matrix(target[:, idx], result[:, idx], labels=[1, 0])
        conf_matrix[idx, :] = cf
    return conf_matrix

def subs_confusion(target, result, th=0.5):
    conf_matrix = np.zeros((1, 2))
    target = target.astype(np.uint8)
    result = (result >= th).astype(np.uint8)
    batch_size = target.shape[0]
    for idx in range(batch_size):
        tg = target[idx]
        pd = result[idx]
        res = np.array_equal(tg, pd)
        if res:
            conf_matrix[0, 0] += 1
        else:
            conf_matrix[0, 1] += 1
    return conf_matrix


if __name__ == '__main__':
    tg = np.array([[1, 1, 1, 1], [0, 1, 1, 1]])
    rs = np.array([[0.9, 0.8, 0.75, 0.9], [0.1, 0.8, 0.75, 0.9]])
    subs_confusion(tg, rs, th=0.5)