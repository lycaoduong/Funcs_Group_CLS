import numpy as np
from src.predictor import PredictorCls, PredictorClsONNX, PredictorClsFormerONNX
import argparse
from utils.ohlabs.plotutils import plot_self_attention_map
import json


def get_args():
    parser = argparse.ArgumentParser('Functional Groups Classification Pytorch')
    parser.add_argument('-p', '--project', type=str, default='FCG Classification', help='Project Name')
    parser.add_argument('-m', '--model', type=str, default='Fcg-B', help='Choosing Model')
    parser.add_argument('-w', '--ckpt', type=str, default='./best_val_loss.pt', help='Loading pretrained weighted')
    parser.add_argument('-dv', '--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-op', '--input', type=str, default='./ckpts/17451182.npy', help='Choosing input')
    parser.add_argument('-th', '--threshold', type=float, default=0.5, help='Choosing threshold')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    engine = PredictorCls(model_name=opt.model, ckpt=opt.ckpt, device=opt.device)
    spectra = np.load(opt.input)
    o = engine(spectra, th=0.5)
    att = engine.model.get_self_attention(layer_value=1)
    att = np.sum(att[0], axis=0)
    plot_self_attention_map(spectra, att)
    # print(1)
    # result, att_map = o["result"], o["att_map"]
    # print(result)
    # model.to_onnx('./Fcg-H.onnx')

    # cls_dic = {"ids": {"alkane": 0, "methyl": 1, "alkene": 2, "alkyne": 3, "alcohols": 4, "amines": 5, "nitriles": 6,
    #                 "aromatics": 7, "alkyl halides": 8, "esters": 9, "ketones": 10, "aldehydes": 11,
    #                 "carboxylic acids": 12, "ether": 13, "acyl halides": 14, "amides": 15, "nitro": 16},
    #            "signal_len": 1024
    #            }
    #
    # with open('ckpts/configs.json', 'w') as fp:
    #     json.dump(cls_dic, fp)

    # plot_attention_map(spectra, result, att_map)

    # model = PredictorClsFormerONNX(onnx_dir='cls_former.onnx')
    # spectral = np.load(opt.input)
    # result, att_map = engine.model.(spectral)
    # print(result)
    # plot_attention_map(spectral, result, att_map)
