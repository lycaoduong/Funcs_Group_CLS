import numpy as np
import torch
import cv2


class Normalizer(object):
    def __init__(self, with_std=False, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])
        self.with_std = with_std

    def __call__(self, sample):
        signal, tokenizer, fn = sample['signal'], sample['tokenizer'], sample['fn']
        max = np.max(signal, axis=1)
        min = np.min(signal, axis=1)
        if self.with_std:
            signal = (((signal.astype(np.float32) - min) / (max - min)) - self.mean) / self.std
        else:
            signal = ((signal.astype(np.float32) - min) / (max - min))
        sample = {'signal': signal, 'tokenizer': tokenizer, 'fn': fn}
        return sample


class Resizer(object):
    def __init__(self, signal_size=1024):
        self.signal_size = signal_size

    def __call__(self, sample):
        signal, tokenizer, fn = sample['signal'], sample['tokenizer'], sample['fn']

        signal = cv2.resize(signal, (self.signal_size, 1), interpolation=cv2.INTER_CUBIC)

        sample = {'signal': torch.from_numpy(signal).to(torch.float32), 'tokenizer': torch.from_numpy(tokenizer).to(torch.long), 'fn': fn}
        return sample


def collater(data):
    signal = [s['signal'] for s in data]
    tokenizers = [s['tokenizer'] for s in data]
    fns = [s['fn'] for s in data]
    signal = torch.from_numpy(np.stack(signal, axis=0))
    tokenizers = torch.from_numpy(np.stack(tokenizers, axis=0))
    sample = {'signal': signal, 'tokenizer': tokenizers, 'fn': fns}
    return sample
