import numpy as np
from src.predictor import PredictorCls, PredictorClsONNX, PredictorClsFormerONNX
import argparse
from utils.ohlabs.plotutils import plot_attention_map


def get_args():
    parser = argparse.ArgumentParser('Functional Groups Classification Pytorch')
    parser.add_argument('-p', '--project', type=str, default='FCG Classification', help='Project Name')
    parser.add_argument('-m', '--model', type=str, default='FcgFormer-B', help='Choosing Model')
    parser.add_argument('-w', '--ckpt', type=str, default='./ckpts/dec/last.pt', help='Loading pretrained weighted')
    parser.add_argument('-dv', '--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-op', '--input', type=str, default='./ckpts/75832.npy', help='Choosing input')
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help='Choosing threshold')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    # model = PredictorCls(model_name=opt.model, ckpt=opt.ckpt, device=opt.device)
    # spectra = np.load(opt.input)
    # o = model(spectra, th=0.5)
    # result, att_map = o["result"], o["att_map"]
    # print(result)
    # model.to_onnx('./cls_former.onnx')

    # plot_attention_map(spectra, result, att_map)

    model = PredictorClsFormerONNX(onnx_dir='cls_former.onnx')
    spectral = np.load(opt.input)
    result, att_map = model(spectral)
    print(result)
    plot_attention_map(spectral, result, att_map)
