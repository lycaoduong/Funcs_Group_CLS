import numpy as np
import cv2
from torch.utils.data import Dataset
import os


class FCGDescriptionDataset(Dataset):
    def __init__(self, root_dir, list_data, voca_dic, pos_dic, max_sequence, transform=None):
        self.root = root_dir
        self.list = list_data
        self.voca_dic = voca_dic
        self.pos_dic = pos_dic
        self.max_sequence = max_sequence
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        signal, tokenizer, fn = self.load_data(idx)
        data_loader = {'signal': signal, 'tokenizer': tokenizer, 'fn': fn}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def load_data(self, idx):
        signal_dir = os.path.join(self.root, self.list[idx])
        annot_dir = os.path.join(self.root, self.list[idx][:-4] + '.txt')
        fn = self.list[idx]
        signal = np.load(signal_dir)
        signal = np.expand_dims(signal, axis=0)
        description = [line.rstrip('\n') for line in open(annot_dir)][0]
        tokenizers = [1] # BOS token
        components = description.split(" ")
        for idx, component in enumerate(components):
            if int(component) == 1:
                fcname = self.pos_dic[idx]
                token = self.voca_dic[fcname]
                tokenizers.append(token)
        for i in range(self.max_sequence - len(tokenizers)):
            tokenizers.append(0) # Add padding zero
        tokenizers[-1] = 2 # EOS
        tokenizers = np.array(tokenizers)
        return signal, tokenizers, fn


class FCGClassificationDataset(Dataset):
    def __init__(self, root_dir, list_data, voca_dic, pos_dic, max_sequence, transform=None):
        self.root = root_dir
        self.list = list_data
        self.voca_dic = voca_dic
        self.pos_dic = pos_dic
        self.max_sequence = max_sequence
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        signal, tokenizer, fn = self.load_data(idx)
        data_loader = {'signal': signal, 'tokenizer': tokenizer, 'fn': fn}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def load_data(self, idx):
        signal_dir = os.path.join(self.root, self.list[idx])
        annot_dir = os.path.join(self.root, self.list[idx][:-4] + '.txt')
        fn = self.list[idx]
        signal = np.load(signal_dir)
        signal = np.expand_dims(signal, axis=0)
        description = [line.rstrip('\n') for line in open(annot_dir)][0]
        tokenizers = []
        components = description.split(" ")
        for idx, component in enumerate(components):
            tokenizers.append(component)
        tokenizers = np.array(tokenizers).astype(np.float32)
        return signal, tokenizers, fn


class FCGClipDataset(Dataset):
    def __init__(self, root_dir, list_data, voca_dic, pos_dic, max_sequence, transform=None):
        self.root = root_dir
        self.list = list_data
        self.voca_dic = voca_dic
        self.pos_dic = pos_dic
        self.max_sequence = max_sequence
        self.transform = transform

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        signal, tokenizer, fn = self.load_data(idx)
        data_loader = {'signal': signal, 'tokenizer': tokenizer, 'fn': fn}
        if self.transform is not None:
            data_loader = self.transform(data_loader)
        return data_loader

    def load_data(self, idx):
        signal_dir = os.path.join(self.root, self.list[idx])
        annot_dir = os.path.join(self.root, self.list[idx][:-4] + '.txt')
        fn = self.list[idx]
        signal = np.load(signal_dir)
        signal = np.expand_dims(signal, axis=0)
        description = [line.rstrip('\n') for line in open(annot_dir)][0]
        tokenizers = []
        components = description.split(" ")
        for idx, component in enumerate(components):
            tokenizers.append(component)
        tokenizers = np.array(tokenizers).astype(np.float32)
        return signal, tokenizers, fn
