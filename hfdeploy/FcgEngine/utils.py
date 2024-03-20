import matplotlib.pyplot as plt
import numpy as np


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
        return fig

def plot_spectra(spectra):
    fig = plt.figure()
    # fg.title("Spectra Signal")
    plt.plot(spectra)
    return fig

def plot_result(name, score):
    fig, ax = plt.subplots(figsize=(10, 10))
    y_pos = np.arange(len(name))
    ax.barh(y_pos, score, align='center')
    ax.set_yticks(y_pos, labels=name)
    for value, pos in zip(score, y_pos):
        ax.text(value-0.2, pos, value, ha='right')
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Confidence')
    ax.set_title('Prediction')
    return fig
