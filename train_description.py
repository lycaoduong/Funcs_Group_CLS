import argparse
from src.train_des import Trainer
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def get_args():
    parser = argparse.ArgumentParser('Functional Groups Description Pytorch')
    parser.add_argument('-p', '--project', type=str, default='FCG_Description', help='Project Name')
    parser.add_argument('-m', '--model', type=str, default='Fcg-B', help='Choosing Model')
    parser.add_argument('-w', '--ckpt', type=str, default=None, help='Loading pretrained weighted')
    parser.add_argument('-d', '--dataset', type=str, default='ohlabsFcg', help='Loading dataset configs file')
    parser.add_argument('-lr', '--lr', type=float, default=2e-4, help='Init Learning Rate')
    parser.add_argument('-ep', '--epochs', type=int, default=550, help='Init number of train epochs')
    parser.add_argument('-dv', '--device', type=str, default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('-bs', '--batch_size', type=int, default=1, help='Init train batch size')
    parser.add_argument('-nw', '--num_worker', type=int, default=8, help='Number of worker for Dataloader')
    parser.add_argument('-op', '--optimizer', type=str, default='adamw', help='Choosing optimizer')
    parser.add_argument('-ls', '--lr_scheduler', type=str, default='cosine', help='Choosing learning rate scheduler')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    opt = get_args()
    trainer = Trainer(opt)
    trainer.start()
