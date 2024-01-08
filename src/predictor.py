import torch
import numpy as np
from utils.ohlabs.trainutils import YamlRead
from networks.OblabsFcg.model import FCGDescription, FCGClassificationXPORT, FCGClassFormerXPORT
import cv2
import pandas as pd
from torch import nn
import onnxruntime

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
        self.model_name = model_name
        model_configs = YamlRead(f'configs/model/{model_name}.yaml')
        dataset_configs = YamlRead(data_cfg)
        if model_name == 'Fcg-B':
            self.model = FCGClassificationXPORT(embed_dim=model_configs.embed_dim, signal_size=model_configs.signal_size,
                                                patch_size=model_configs.patch_size,
                                                target_vocab_size=model_configs.voca_size,
                                                num_layers=model_configs.num_layers,
                                                expansion_factor=model_configs.expansion_factor,
                                                n_heads=model_configs.n_heads, num_cls=dataset_configs.num_cls)
        else:
            self.model = FCGClassFormerXPORT(embed_dim=model_configs.embed_dim, signal_size=model_configs.signal_size,
                                                patch_size=model_configs.patch_size,
                                                target_vocab_size=model_configs.voca_size,
                                                num_layers=model_configs.num_layers,
                                                expansion_factor=model_configs.expansion_factor,
                                                n_heads=model_configs.n_heads, num_cls=dataset_configs.num_cls)
        self.device = device
        if ckpt is not None:
            weight = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(weight, strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.processor = Processor(dataset_configs=dataset_configs, model_configs=model_configs, device=device)
        self.signal_size = model_configs.signal_size

    def to_onnx(self, save_dir):
        input_names = ["signal"]
        if self.model_name == 'Fcg-B':
            output_names = ["result"]
        else:
            output_names = ["result", "attmap"]
        x = torch.randn((1, 1, self.signal_size), requires_grad=False).to(self.device)
        with torch.no_grad():
            o = self.model(x)
        dynamic_axes = None
        torch.onnx.export(self.model,
                          x,
                          save_dir,
                          export_params=True,
                          opset_version=14,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

    def __call__(self, signal, th=0.3):
        signal = self.processor.pre_process(signal)
        with torch.no_grad():
            if self.model_name == 'FcgFormer-B':
                outputs, cross_atts = self.model(signal)
                cross_atts = cross_atts.cpu().numpy()
                cross_atts = np.mean(cross_atts, axis=1)[0]
            else:
                outputs = self.model(signal)
        outputs = torch.squeeze(outputs)
        outputs = outputs.cpu().numpy()
        clses = list(np.where(outputs >= th))[0]
        results = []
        for cls in clses:
            cls_name = self.processor.getFcgname(int(cls))
            results.append(cls_name)
        if self.model_name == 'FcgFormer-B':
            return {"result": results, "att_map": cross_atts}
        return {"result": results, "att_map": None}


class PredictorClsONNX(object):
    def __init__(self, onnx_dir, signal_size=1024, device='cpu'):
        if device == 'cuda':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 0.5 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })
            ]
            session = onnxruntime.InferenceSession(onnx_dir, None, providers=providers)
        else:
            session = onnxruntime.InferenceSession(onnx_dir, providers=['CPUExecutionProvider'])
        session.get_modelmeta()
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.model = session
        self.device = device
        self.signal_size = signal_size
        dataset_configs = YamlRead(data_cfg)
        self.cls_dic = dataset_configs.pos_dic

    def __call__(self, signal, th=0.3):
        signal = np.expand_dims(signal, axis=0)
        max = np.max(signal, axis=1)
        min = np.min(signal, axis=1)
        signal = ((signal.astype(np.float32) - min) / (max - min))
        signal = cv2.resize(signal, (self.signal_size, 1), interpolation=cv2.INTER_CUBIC)
        input_blob = np.expand_dims(signal, axis=0).astype(np.float32)
        outputs = self.model.run([self.output_name], {self.input_name: input_blob})[0][0]
        clses = list(np.where(outputs >= th))[0]
        results = []
        for cls in clses:
            cls_name = self.cls_dic[cls]
            results.append(cls_name)
        return results


class PredictorClsFormerONNX(object):
    def __init__(self, onnx_dir, signal_size=1024, device='cpu'):
        if device == 'cuda':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 0.5 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })
            ]
            session = onnxruntime.InferenceSession(onnx_dir, None, providers=providers)
        else:
            session = onnxruntime.InferenceSession(onnx_dir, providers=['CPUExecutionProvider'])
        session.get_modelmeta()
        self.input_name = session.get_inputs()[0].name
        self.result_name = session.get_outputs()[0].name
        self.att_name = session.get_outputs()[1].name
        self.model = session
        self.device = device
        self.signal_size = signal_size
        dataset_configs = YamlRead(data_cfg)
        self.cls_dic = dataset_configs.pos_dic

    def __call__(self, signal, th=0.3):
        signal = np.expand_dims(signal, axis=0)
        max = np.max(signal, axis=1)
        min = np.min(signal, axis=1)
        signal = ((signal.astype(np.float32) - min) / (max - min))
        signal = cv2.resize(signal, (self.signal_size, 1), interpolation=cv2.INTER_CUBIC)
        input_blob = np.expand_dims(signal, axis=0).astype(np.float32)
        outputs, att_map = self.model.run([self.result_name, self.att_name], {self.input_name: input_blob})
        outputs = outputs[0]
        clses = list(np.where(outputs >= th))[0]
        results = []
        for cls in clses:
            cls_name = self.cls_dic[cls]
            results.append(cls_name)
        return results, np.mean(att_map, axis=1)[0]
