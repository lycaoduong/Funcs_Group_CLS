import os
import json
import onnxruntime
import numpy as np
import cv2
from transformers import AutoModelForImageClassification, AutoConfig
import torch


class PredictorClsOnnx(object):
    def __init__(self, model_dir, device='cpu', gpu_allocate=1):
        model_bin = os.path.join(model_dir, 'model.bin')
        with open(os.path.join(model_dir, 'configs.json')) as f:
            configs = json.load(f)
        self.ids = list(configs["ids"])
        self.signal_size = configs["signal_len"]
        if device == 'cuda':
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': gpu_allocate * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                    'do_copy_in_default_stream': True,
                })
            ]
            session = onnxruntime.InferenceSession(model_bin, None, providers=providers)
        else:
            session = onnxruntime.InferenceSession(model_bin, providers=['CPUExecutionProvider'])
        session.get_modelmeta()
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.model = session
        self.device = device

    def __call__(self, signal):
        signal = np.expand_dims(signal, axis=0)
        max = np.max(signal, axis=1)
        min = np.min(signal, axis=1)
        signal = ((signal.astype(np.float32) - min) / (max - min))
        signal = cv2.resize(signal, (self.signal_size, 1), interpolation=cv2.INTER_CUBIC)
        input_blob = np.expand_dims(signal, axis=0).astype(np.float32)
        outputs = self.model.run([self.output_name], {self.input_name: input_blob})[0][0]
        return outputs

    def decode(self, result, th=0.5):
        predict_cls = list(np.where(result >= th))[0]
        description = "Signal contains"
        for ids in predict_cls:
            description += ' '
            cls_name = self.ids[ids]
            prob = result[ids]
            description += '{} ({:.4f});'.format(cls_name.capitalize(), prob)
        return description


class PredictorCls(object):
    def __init__(self, model_path='lycaoduong/FcgFormer', device='cpu'):
        self.model = AutoModelForImageClassification.from_pretrained(model_path, trust_remote_code=True)
        self.model.to(device)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.ids = list(config.cls_name.keys())
        self.device = device

    def __call__(self, spectra):
        tensor = self.model.to_pt_tensor(spectra).to(self.device)
        with torch.no_grad():
            o = self.model(tensor)['logits']
        outputs = torch.sigmoid(o).cpu().numpy()
        return outputs[0]

    def decode(self, result, th=0.5):
        predict_cls = list(np.where(result >= th))[0]
        description = "Signal contains"
        for ids in predict_cls:
            description += ' '
            cls_name = self.ids[ids]
            prob = result[ids]
            description += '{} ({:.4f});'.format(cls_name.capitalize(), prob)
        return description
