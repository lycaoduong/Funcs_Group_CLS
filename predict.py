import numpy as np
from src.predictor import PredictorCls
import argparse


def get_args():
    parser = argparse.ArgumentParser('Functional Groups Classification Pytorch')
    parser.add_argument('-p', '--project', type=str, default='FCG Classification', help='Project Name')
    parser.add_argument('-m', '--model', type=str, default='Fcg-B', help='Choosing Model')
    parser.add_argument('-w', '--ckpt', type=str, default='./best_val_loss.pt', help='Loading pretrained weighted')
    parser.add_argument('-dv', '--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-op', '--input', type=str, default='./100016.npy', help='Choosing input')
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help='Choosing threshold')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    model = PredictorCls(model_name=opt.model, ckpt=opt.ckpt, device=opt.device)
    spectral = np.load(opt.input)
    result = model(spectral, th=0.5)
    print(result)
