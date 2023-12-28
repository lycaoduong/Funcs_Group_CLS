import torch
import numpy as np
from utils.ohlabs.trainutils import YamlRead
from networks.OblabsFcg.model import FCGDescription, FCGClassification
import cv2
import pandas as pd
from torch import nn

data_cfg = f'configs/dataset/ohlabsFcg.yaml'


class Processor(object):
    def __init__(self, dataset_configs, model_configs, device='cuda'):
        self.device = device
        self.dataset_configs = dataset_configs
        self.model_configs = model_configs
        df = pd.read_csv(dataset_configs.token_dir, encoding='utf-8').to_dict()
        tokenize = df['Voca']
        self.tokenize = dict((v, k) for k, v in tokenize.items())
        self.vocadic = dict((k, v) for k, v in tokenize.items())
        self.cls_dic = dataset_configs.pos_dic

    def pre_process(self, signal):
        signal = np.expand_dims(signal, axis=0)
        max = np.max(signal, axis=1)
        min = np.min(signal, axis=1)
        signal = ((signal.astype(np.float32) - min) / (max - min))
        signal = cv2.resize(signal, (self.model_configs.signal_size, 1), interpolation=cv2.INTER_CUBIC)
        signal = torch.from_numpy(signal).to(torch.float32).to(self.device)
        signal = torch.unsqueeze(signal, dim=0)
        return signal

    def tokenizer(self, voca):
        token = self.tokenize[voca]
        return token

    def getVoca(self, token):
        voca = self.vocadic[token]
        return voca

    def getFcgname(self, token):
        name = self.cls_dic[token]
        return name


class PredictorDes(object):
    def __init__(self, model_name='Fcg-B', ckpt=None, device='cuda'):
        model_configs = YamlRead(f'configs/model/{model_name}.yaml')
        dataset_configs = YamlRead(data_cfg)
        self.model = FCGDescription(embed_dim=model_configs.embed_dim, signal_size=model_configs.signal_size,
                               patch_size=model_configs.patch_size, target_vocab_size=model_configs.voca_size,
                               seq_length=dataset_configs.max_sequence, num_layers=model_configs.num_layers,
                               expansion_factor=model_configs.expansion_factor, n_heads=model_configs.n_heads,
                               num_cls=dataset_configs.num_cls)
        self.device = device
        if ckpt is not None:
            weight = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(weight, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.processor = Processor(dataset_configs=dataset_configs, model_configs=model_configs, device=device)
        self.seq_len = dataset_configs.max_sequence

    def __call__(self, signal):
        signal = self.processor.pre_process(signal)
        bos_token = self.processor.tokenizer('bos')
        with torch.no_grad():
            outputs = self.model.prediction(signal)
        # fcg_des = ''
        # for o in outputs:
        #     token = self.processor.getVoca(o)
        #     fcg_des += token + ' '
        # return fcg_des


class PredictorCls(object):
    def __init__(self, model_name='Fcg-B', ckpt=None, device='cuda'):
        model_configs = YamlRead(f'configs/model/{model_name}.yaml')
        dataset_configs = YamlRead(data_cfg)
        self.model = FCGClassification(embed_dim=model_configs.embed_dim, signal_size=model_configs.signal_size,
                               patch_size=model_configs.patch_size, target_vocab_size=model_configs.voca_size,
                               seq_length=dataset_configs.max_sequence, num_layers=model_configs.num_layers,
                               expansion_factor=model_configs.expansion_factor, n_heads=model_configs.n_heads,
                               num_cls=dataset_configs.num_cls)
        self.device = device
        if ckpt is not None:
            weight = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(weight, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.processor = Processor(dataset_configs=dataset_configs, model_configs=model_configs, device=device)

    def __call__(self, signal, th=0.3):
        signal = self.processor.pre_process(signal)
        with torch.no_grad():
            outputs = self.model(signal)
        outputs = torch.sigmoid(outputs)
        outputs = torch.squeeze(outputs)
        outputs = outputs.cpu().numpy()
        clses = list(np.where(outputs >= th))[0]
        results = []
        for cls in clses:
            cls_name = self.processor.getFcgname(int(cls))
            results.append(cls_name)
        return results

