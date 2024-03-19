import os.path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd


funcs_name = ["alkane", "methyl", "alkene", "alkyne", "alcohols", "amines", "nitriles", "aromatics",
              "alkyl halides", "esters", "ketones", "aldehydes", "carboxylic acids",
              "ether", "acyl halides", "amides", "nitro"]


def plot_cross_attention_map(spectra, result, att_map):
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


def plot_self_attention_map(spectra, att_map, offset=400):
    if att_map is not None:
        fig, ax1 = plt.subplots(figsize=(12, 12))
        ax2 = plt.twinx().twiny()
        ax2.set_xlim(offset, offset+len(spectra))
        ax1.set_xlabel('Patch index', fontsize=18)
        ax1.set_ylabel('Patch index', fontsize=18)
        ax1.imshow(att_map[1:, 1:], cmap='inferno', interpolation='nearest', aspect="auto")
        # tem_x = np.zeros_like(spectra)
        x = np.linspace(offset, offset + len(spectra), len(spectra))
        # for i, s in enumerate(spectra):
        #     tem_x[i] = i + 400
        ax2.set_xlabel('Wavelength', fontsize=18)
        ax2.set_ylabel('Intensity (a.u.)', fontsize=18)
        ax2.plot(x, spectra)
        plt.show()

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i,y[i],y[i])

def plot_data(data_dis, save_dir="./", save_name="fg.png"):
    fig = plt.figure(figsize=(20, 10))
    x_pos = np.arange(len(funcs_name))
    plt.bar(funcs_name, data_dis[0], align='center', alpha=0.5)
    addlabels(x_pos, data_dis[0])
    plt.xticks(x_pos, funcs_name)
    plt.ylabel('Total samples')
    plt.title('Functional Groups Data Distribution')
    # plt.show()
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)

def plot_roc_pr_curve(precision, recall, fpr, save_dir="./", save_name="funcs_roc_pr_curve.png"):
    fig = plt.figure(figsize=(14, 8))
    plt.style.use('ggplot')
    # Add value
    precision.insert(0, 0.0001)
    recall.insert(0, 0.9999)
    fpr.insert(0, 0.9999)

    precision.append(0.9999)
    recall.append(0.0001)
    fpr.append(0.0001)

    # ROC curve
    plt.subplot(121)
    plt.plot(fpr, recall, linewidth=2)
    plt.title('ROC Curve', fontsize=18, fontweight="bold", y=1.05)
    plt.fill_between(fpr, recall, facecolor='blue', alpha=0.1)
    plt.text(0.55, 0.4, 'AUC', fontsize=30)
    # styling figure
    plt.xlabel('False Positive Rate', fontsize=16, labelpad=13)
    plt.ylabel('True Positive Rate', fontsize=16, labelpad=13)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)

    # PR Curve
    plt.subplot(122)
    plt.plot(recall, precision)
    plt.title('PR Curve', fontsize=18, fontweight="bold", y=1.05)
    plt.ylabel('Precision', fontsize=16, labelpad=13)
    plt.xlabel('Recall', fontsize=16, labelpad=13)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    # plt.show()
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.close()

def plot_conf(conf, labelX=["1", "0"], labelY=["1", "0"], title=None, save_dir="./", save_name='fg.png', size=None):
    if size is None:
        fig = plt.figure(figsize=(15, 12))
    else:
        fig = plt.figure(figsize=size)
    plt.style.use('classic')
    # disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=label)
    # disp.plot(values_format='')
    ax = plt.subplot()
    sns.heatmap(conf, annot=True, fmt='g', ax=ax, cmap='Blues')  # annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    # ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(labelX)
    ax.yaxis.set_ticklabels(labelY)

    if title is not None:
        ax.set_title('Confusion Matrix')
        # plt.title(title)
    # plt.show()
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.close()

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

def subs_len_confusion(target, result, th=0.5):
    conf_matrix = np.zeros((8, 2))
    target = target.astype(np.uint8)
    result = (result >= th).astype(np.uint8)
    batch_size = target.shape[0]
    for idx in range(batch_size):
        tg = target[idx]
        pd = result[idx]
        res = np.array_equal(tg, pd)
        len = np.sum(tg)
        if res:
            conf_matrix[len-1, 0] += 1
            conf_matrix[7, 0] += 1
        else:
            conf_matrix[len-1, 1] += 1
            conf_matrix[7, 1] += 1
    return conf_matrix

def plot_loss_from_csv(fcg_loss, ircnn_loss, lr_dir):
    fcg_loss = pd.read_csv(fcg_loss)
    fcg_loss_value = fcg_loss['Value'].tolist()
    ircnn_loss = pd.read_csv(ircnn_loss)
    ircnn_loss_value = ircnn_loss['Value'].tolist()

    lr = pd.read_csv(lr_dir)
    lr_value = lr['Value'].tolist()

    fig, ax = plt.subplots(2, 1, figsize=(12, 10))

    ax[0].plot(fcg_loss_value, '-b', label='Fcg-Former')
    ax[0].plot(ircnn_loss_value, '-r', label='IRCNN')
    # ax[0].axis('equal')

    y_fgc, x_fcg = min(fcg_loss_value), fcg_loss_value.index(min(fcg_loss_value))
    ax[0].annotate('best checkpoint', xy=(x_fcg, y_fgc), xytext=(x_fcg-50, y_fgc+0.5),
                arrowprops={'arrowstyle': '->', 'ls': 'dashed', 'color': 'red'}, va='center')

    y_ircnn, x_ircnn = min(ircnn_loss_value), ircnn_loss_value.index(min(ircnn_loss_value))
    ax[0].annotate('best checkpoint', xy=(x_ircnn, y_ircnn), xytext=(x_ircnn-50, y_ircnn+0.5),
                arrowprops={'arrowstyle': '->', 'ls': 'dashed', 'color': 'black'}, va='center')

    y_ircnn_stop, x_ircnn_stop = ircnn_loss_value[-1], len(ircnn_loss)
    ax[0].annotate('early stop point', xy=(x_ircnn_stop, y_ircnn_stop), xytext=(x_ircnn_stop+50, y_ircnn_stop+0.5),
                arrowprops={'arrowstyle': '->', 'ls': 'dashed', 'color': 'black'}, va='center')

    ax[0].set_title('Validation Loss', fontsize=18)
    ax[0].set_xlabel('Epochs', fontsize=16)
    ax[0].set_ylabel('Loss', fontsize=16)
    ax[0].set_xlim([0, 600])
    ax[0].set_ylim([0, 1.2])


    ax[1].plot(lr_value, label='Learning rate')
    ax[1].set_title('Learning rate scheduler', fontsize=18)
    ax[1].set_xlabel('Training steps', fontsize=16)
    ax[1].set_ylabel('Learning rate', fontsize=16)
    ax[1].set_xlim([0, 1000])
    ax[1].set_ylim([0, 0.00025])

    leg = ax[0].legend()
    leg2 = ax[1].legend()
    fig.tight_layout(pad=3.0)
    plt.show()


if __name__ == '__main__':
    # tg = np.array([[1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0]])
    # rs = np.array([[0.9, 0.8, 0.75, 0.9], [0.1, 0.8, 0.75, 0.9], [0.9, 0.9, 0.9, 0.1]])
    # cf = subs_len_confusion(tg, rs, th=0.5)
    fcg_p = 'D:/lycaoduong/workspace/paper/fcg-former/loss/fcg/run-Loss_val-tag-Loss.csv'
    lr_p = 'D:/lycaoduong/workspace/paper/fcg-former/loss/fcg/run-.-tag-learning_rate.csv'
    ircnn_p = 'D:/lycaoduong/workspace/paper/fcg-former/loss/ircnn/run-Loss_val-tag-Loss.csv'
    plot_loss_from_csv(fcg_p, ircnn_p, lr_p)