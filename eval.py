import argparse

import numpy as np

from src.evalator import Evaluation
from utils.ohlabs.plotutils import plot_conf, plot_data


def get_args():
    parser = argparse.ArgumentParser('Functional Groups Classification Pytorch')
    parser.add_argument('-p', '--project', type=str, default='FCG_Classification', help='Project Name')
    parser.add_argument('-m', '--model', type=str, default='IRCNN', help='Choosing Model')
    parser.add_argument('-w', '--ckpt', type=str, default='./ckpts/last.pt', help='Loading pretrained weighted')
    parser.add_argument('-d', '--dataset', type=str, default='ohlabsFcg', help='Loading dataset configs file')
    parser.add_argument('-dv', '--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Init train batch size')
    parser.add_argument('-nw', '--num_worker', type=int, default=8, help='Number of worker for Dataloader')
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help='Threshold')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    print("Starting evaluation Model: {}".format(opt.model))
    evaltor = Evaluation(opt)
    evaltor.eval_data_analysis()
    evaltor.start()

