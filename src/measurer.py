import datetime
import os
import json
from utils.ohlabs.trainutils import YamlRead
from utils.ohlabs import transform as tr
from torchvision import transforms
from utils.ohlabs.dataloader import FCGClassificationDataset
from torch.utils.data import DataLoader
from networks.OblabsFcg.model import FCGClassification, FCGClassFormer
from networks.ircharacercnn.ircnn import IrCNN
from torch import nn
import torch
from tqdm.autonotebook import tqdm
import traceback
import numpy as np
from utils.ohlabs.plotutils import func_confusion, subs_confusion, plot_data, plot_conf, plot_roc_pr_curve, subs_len_confusion
from prettytable import PrettyTable
import cv2
import time


class Measurer(object):
    def __init__(self, eval_opt):
        self.dataset = eval_opt.dataset
        self.model_name = eval_opt.model

        # Read dataset
        dataset_configs = YamlRead(f'configs/dataset/{self.dataset}.yaml')
        self.eval_dir = dataset_configs.test_dir
        self.mean = dataset_configs.mean
        self.std = dataset_configs.std
        self.num_cls = dataset_configs.num_cls

        #Read Model configs
        model_configs = YamlRead(f'configs/model/{self.model_name}.yaml')
        self.signal_size = model_configs.signal_size

        self.eval_list = []
        for file in os.listdir(self.eval_dir):
            if file.endswith(".npy"):
                self.eval_list.append(file)

        self.device = eval_opt.device

        # Model
        if self.model_name == "Fcg-S" or self.model_name == "Fcg-B" or self.model_name == "Fcg-L" \
                or self.model_name == "Fcg-H" or self.model_name == "Fcg-Bbk":
            model = FCGClassification(embed_dim=model_configs.embed_dim, signal_size=model_configs.signal_size,
                                      patch_size=model_configs.patch_size, num_layers=model_configs.num_layers,
                                      expansion_factor=model_configs.expansion_factor, n_heads=model_configs.n_heads,
                                      num_cls=dataset_configs.num_cls)
        elif self.model_name == "IRCNN":
            model = IrCNN(signal_size=model_configs.signal_size)
        else:
            model = FCGClassFormer(embed_dim=model_configs.embed_dim, signal_size=model_configs.signal_size,
                                   patch_size=model_configs.patch_size, num_layers=model_configs.num_layers,
                                   expansion_factor=model_configs.expansion_factor, n_heads=model_configs.n_heads,
                                   num_cls=dataset_configs.num_cls)

        if eval_opt.ckpt is not None:
            weight = torch.load(eval_opt.ckpt, map_location=self.device)
            model.load_state_dict(weight, strict=True)

        self.model = model.to(self.device)
        self.model.eval()

    def to_pt_tensor(self, array):
        array = np.expand_dims(array, axis=0)
        max = np.max(array, axis=1)
        min = np.min(array, axis=1)
        array = ((array.astype(np.float32) - min) / (max - min))
        array = cv2.resize(array, (self.signal_size, 1), interpolation=cv2.INTER_CUBIC)
        array = torch.from_numpy(array)
        tensor = torch.unsqueeze(array, dim=0).to(torch.float32).to(self.device)
        return tensor

    def count_parameters(self):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        trainable_params = 0
        non_train_params = 0
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                non_train_params += parameter.numel()
                # continue
            else:
                trainable_params += parameter.numel()
            params = parameter.numel()
            table.add_row([name, params])

        total_params = trainable_params + non_train_params
        print(table)
        print(f"Total Params: {total_params}")
        print(f"Total Trainable Params: {trainable_params}")
        print(f"Total non_Trainable Params: {non_train_params}")
        estimate = total_params * 24 / 1048576
        print(f"GPU RAM estimate: {estimate} MB")
        return total_params

    def inference(self):
        init_in = torch.rand(1, 1, self.signal_size).to(self.device)
        with torch.no_grad():
            o = self.model(init_in)
        times = []
        for file in self.eval_list:
                array = np.load(os.path.join(self.eval_dir, file))
                tensor = self.to_pt_tensor(array)
                with torch.no_grad():
                    starttime = time.time()
                    o = self.model(tensor)
                    times.append(time.time() - starttime)
        times = np.array(times)
        mean = np.mean(times)
        std = np.std(times)
        print("Inference time: {} +/- {}".format(mean, std))

    def start(self):
        self.inference()
        self.count_parameters()
