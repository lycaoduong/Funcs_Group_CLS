import yaml
import numpy as np
import torch


class YamlRead:
    def __init__(self, params_path):
        self.params = yaml.safe_load(open(params_path, encoding='utf-8').read())

    def update(self, dictionary):
        self.params = dictionary

    def __getattr__(self, item):
        return self.params.get(item, None)


class toOpt(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])
