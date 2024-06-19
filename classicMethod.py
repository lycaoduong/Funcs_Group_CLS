from networks.classicalNet.classicalNet import ClassNet
import argparse


def get_args():
    parser = argparse.ArgumentParser('Functional Groups Classical Method')
    parser.add_argument('-d', '--dataset', type=str, default='ohlabsFcg', help='Dataset Name')
    parser.add_argument('-m', '--method', type=str, default='DecisionTree', help='Choosing Method: DecisionTree, RandomForest, KNeighbors')
    parser.add_argument('-s', '--signal_size', type=int, default=1024, help='Signal size')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    save_name = './ckpts/{}.pickle'.format(opt.method)
    engine = ClassNet(opt)
    # engine.start(save_dir=save_name)
    # engine.eval()
    engine.loadModel(save_name)
    engine.eval()
