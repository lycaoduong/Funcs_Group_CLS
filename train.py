import argparse
from src.train_cls import Trainer
import nni
from utils.ohlabs.trainutils import toOpt


def get_args():
    parser = argparse.ArgumentParser('Functional Groups Classification Pytorch')
    parser.add_argument('-p', '--project', type=str, default='FCG_Classification', help='Project Name')
    parser.add_argument('-m', '--model', type=str, default='IRCNN', help='Choosing Model')
    parser.add_argument('-w', '--ckpt', type=str, default=None, help='Loading pretrained weighted')
    parser.add_argument('-d', '--dataset', type=str, default='ohlabsFcg', help='Loading dataset configs file')
    parser.add_argument('-au', '--aug', type=str, default='aug', help='Loading Augmentation configs file')
    parser.add_argument('-lr', '--lr', type=float, default=2e-4, help='Init Learning Rate')
    parser.add_argument('-ep', '--epochs', type=int, default=600, help='Init number of train epochs') #1240
    parser.add_argument('-dv', '--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-bs', '--batch_size', type=int, default=4, help='Init train batch size')
    parser.add_argument('-nw', '--num_worker', type=int, default=8, help='Number of worker for Dataloader')
    parser.add_argument('-op', '--optimizer', type=str, default='adamw', help='Choosing optimizer')
    parser.add_argument('-ls', '--lr_scheduler', type=str, default='cosine', help='Choosing learning rate scheduler')
    parser.add_argument('-lw', '--lossW', type=bool, default=True, help='Loss with weight')
    # args = parser.parse_args()
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    # models = ["IRCNN", "Fcg-S", "Fcg-B", "Fcg-L", "Fcg-H"]
    # for model in models:
    #     print("Starting train Model: {}".format(model))

    opt = get_args()
    nni_params = nni.get_next_parameter()
    if len(nni_params):
        opt.update(nni_params)
        run_nni = {"nni": True}
        opt.update(run_nni)

    opt = toOpt(opt)
    trainer = Trainer(opt)
    trainer.data_analysis()
    # trainer.start()
