import argparse
from src.measurer import Measurer


def get_args():
    parser = argparse.ArgumentParser('Functional Groups Classification Pytorch')
    parser.add_argument('-p', '--project', type=str, default='FCG_Classification', help='Project Name')
    parser.add_argument('-m', '--model', type=str, default='Fcg-H', help='Choosing Model')
    parser.add_argument('-w', '--ckpt', type=str, default='./best_val_loss.pt', help='Loading pretrained weighted')
    parser.add_argument('-d', '--dataset', type=str, default='ohlabsFcg', help='Loading dataset configs file')
    parser.add_argument('-dv', '--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    print("Starting measure Model: {}".format(opt.model))
    measurer = Measurer(opt)
    measurer.start()
