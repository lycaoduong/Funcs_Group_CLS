import datetime
import os
import json
from utils.ohlabs.trainutils import YamlRead
from utils.ohlabs import transform as tr
from torchvision import transforms
from utils.ohlabs.dataloader import FCGDescriptionDataset
from torch.utils.data import DataLoader
from networks.OblabsFcg.model import FCGDescription
from torch import nn
import torch
from tqdm.autonotebook import tqdm
import traceback
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd


class ModelWithLoss(nn.Module):
    def __init__(self, model, reduction='mean', sq_len=13, voca_size=27):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        self.reduction = reduction
        self.sq_len = sq_len
        self.voca_size = voca_size

    def forward(self, image, target, **kwargs):
        o = self.model(image, target[:, :-1])
        shifted_target = target[:, 1:]
        bs = target.shape[0]
        losses = 0.0
        for b in range(bs):
            loss = self.criterion(o[b], shifted_target[b])
            losses += loss
        if self.reduction == "mean":
            losses /= bs
        # outputs_flat = o.view(bs * (self.sq_len-1), self.voca_size)
        # targets_flat = shifted_target.flatten()
        # loss = self.criterion(o, shifted_target)
        return losses


class Trainer(object):
    def __init__(self, train_opt):
        self.project = train_opt.project
        self.model_name = train_opt.model
        self.dataset = train_opt.dataset

        print('Project Name: ', self.project)
        print('Model and Dataset: ', self.model_name, self.dataset)
        date_time = datetime.datetime.now()
        date_time = date_time.strftime("%Y.%m.%d_%H.%M.%S")
        print('Date Access: ', date_time)

        if hasattr(train_opt, 'nni'):
            exp_name, _, trial_name = os.environ['NNI_OUTPUT_DIR'].split('\\')[-3:]
            self.nni_writer = True
        else:
            exp_name, trial_name = 'single_run', date_time
            self.nni_writer = False
        self.save_dir = '../runs/train/{}/{}_{}/{}/{}/'.format(self.project, self.model_name, self.dataset, exp_name, trial_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.logs = self.save_dir + 'logs/'
        os.makedirs(self.logs, exist_ok=True)
        self.writer = SummaryWriter(self.logs)

        # Save train parameters
        with open('{}/train_params.txt'.format(self.save_dir), 'w') as f:
            json.dump(train_opt.__dict__, f, indent=2)

        # Read dataset
        dataset_configs = YamlRead(f'configs/dataset/{self.dataset}.yaml')
        self.train_dir = dataset_configs.train_dir
        self.val_dir = dataset_configs.val_dir
        self.token_dir = dataset_configs.token_dir
        self.mean = dataset_configs.mean
        self.std = dataset_configs.std
        self.pos_dic = dataset_configs.pos_dic
        self.max_sequence = dataset_configs.max_sequence
        self.num_cls = dataset_configs.num_cls

        #Read Model configs
        model_configs = YamlRead(f'configs/model/{self.model_name}.yaml')
        self.signal_size = model_configs.signal_size

        if isinstance(self.val_dir, float):
            list_dataset = []
            for file in os.listdir(self.train_dir):
                if file.endswith(".npy"):
                    label = os.path.join(self.train_dir, file[:-4] + '.txt')
                    if os.path.isfile(label):
                        list_dataset.append(file)
            train_list, val_list = train_test_split(list_dataset, test_size=self.val_dir, random_state=42)
        else:
            train_list = [line.rstrip('\n') for line in open(self.train_dir)]
            val_list = [line.rstrip('\n') for line in open(self.val_dir)]

        # Data Loader
        self.device = train_opt.device
        self.batch_size = train_opt.batch_size
        df = pd.read_csv(self.token_dir, encoding='utf-8').to_dict()
        token_voca = df['Voca']
        voca_token = dict((v, k) for k, v in token_voca.items())
        self.voca_size = len(voca_token)

        train_transforms = [
            tr.Normalizer(with_std=False),
            tr.Resizer(signal_size=self.signal_size)
        ]

        training_set = FCGDescriptionDataset(root_dir=self.train_dir, list_data=train_list, voca_dic=voca_token,
                                             pos_dic=self.pos_dic, max_sequence=self.max_sequence,
                                             transform=transforms.Compose(train_transforms))

        train_params = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'drop_last': True,
            'num_workers': train_opt.num_worker
        }

        self.training_generator = DataLoader(training_set, collate_fn=tr.collater, **train_params)

        validation_transforms = [
            tr.Normalizer(with_std=False),
            tr.Resizer(signal_size=self.signal_size)
        ]

        val_set = FCGDescriptionDataset(root_dir=self.train_dir, list_data=val_list, voca_dic=voca_token,
                                        pos_dic=self.pos_dic, max_sequence=self.max_sequence,
                                        transform=transforms.Compose(validation_transforms))

        val_params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'drop_last': False,
            'num_workers': train_opt.num_worker
        }
        self.val_generator = DataLoader(val_set, collate_fn=tr.collater, **val_params)

        # Model

        model = FCGDescription(embed_dim=model_configs.embed_dim, signal_size=model_configs.signal_size,
                               patch_size=model_configs.patch_size, target_vocab_size=self.voca_size,
                               seq_length=dataset_configs.max_sequence, num_layers=model_configs.num_layers,
                               expansion_factor=model_configs.expansion_factor, n_heads=model_configs.n_heads,
                               num_cls=dataset_configs.num_cls)

        # if torch.cuda.is_available():
        #     model = nn.DataParallel(model)
        if train_opt.ckpt is not None:
            weight = torch.load(train_opt.ckpt, map_location=self.device)
            model.load_state_dict(weight, strict=True)

        self.model = ModelWithLoss(model=model, reduction='mean', sq_len=dataset_configs.max_sequence
                                   , voca_size=self.voca_size)
        self.model = self.model.to(self.device)

        # Optimizer and Learning rate scheduler

        self.opti = train_opt.optimizer
        self.l_rate = train_opt.lr
        self.lr_scheduler = train_opt.lr_scheduler

        if self.opti == 'adamw':
            self.optimizer = torch.optim.Adam(params=model.parameters(),
                                              lr=self.l_rate)
        else:
            self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                             lr=self.l_rate,
                                             momentum=0.9)

        if self.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=self.optimizer,
                                                                                  T_0=10,
                                                                                  T_mult=2)
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

        self.num_iter_per_epoch = len(self.training_generator)
        self.step = 0
        self.best_loss = 1e5
        self.epochs = train_opt.epochs

    def train(self, epoch):
        self.model.train()
        last_epoch = self.step // self.num_iter_per_epoch
        progress_bar = tqdm(self.training_generator)
        epoch_loss = []

        for iter, data in enumerate(progress_bar):
            if iter < self.step - last_epoch * self.num_iter_per_epoch:
                progress_bar.update()
                continue
            try:
                signals, tokenizer = data['signal'], data['tokenizer']
                signals = signals.to(self.device)
                tokenizer = tokenizer.to(self.device)
                self.optimizer.zero_grad()

                loss = self.model(signals, tokenizer)

                epoch_loss.append(float(loss))

                loss.backward()

                self.optimizer.step()

                current_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('learning_rate', current_lr, self.step)

                self.writer.add_scalars('Step_Loss', {'loss': loss}, self.step)

                epoch_loss.append(loss.item())

                descriptor = '[Train] Step: {}. Epoch: {}/{}. Iteration: {}/{}. Loss: {:.6f}.'.format(
                        self.step, epoch+1, self.epochs, iter + 1, self.num_iter_per_epoch, loss.item())
                progress_bar.set_description(descriptor)
                self.step += 1

                if self.lr_scheduler == 'cosine':
                    self.scheduler.step(epoch + iter / self.num_iter_per_epoch)

            except Exception as e:
                print('[Error]', traceback.format_exc())
                print(e)
                continue

        if self.lr_scheduler == 'reduce':
            self.scheduler.step(np.mean(epoch_loss))
        # if self.lr_scheduler == 'cosine':
        #     self.scheduler.step()

        mean_loss = np.mean(epoch_loss)

        train_descrip = '[Train] Epoch: {}. Mean Loss: {:.6f}.'.format(epoch+1, mean_loss)
        print(train_descrip)
        self.writer.add_scalars('Loss', {'train': mean_loss}, epoch)

    def validation(self, epoch):
        self.model.eval()
        progress_bar = tqdm(self.val_generator)
        epoch_loss = []

        for iter, data in enumerate(progress_bar):
            with torch.no_grad():
                try:
                    signals, tokenizer = data['signal'], data['tokenizer']
                    signals = signals.to(self.device)
                    tokenizer = tokenizer.to(self.device)

                    loss = self.model(signals, tokenizer)

                    epoch_loss.append(loss.item())

                    descriptor = '[Valid] Step: {}. Epoch: {}/{}. Iteration: {}/{}. Loss: {:.6f}.'.format(
                        epoch * len(progress_bar) + iter, epoch, self.epochs, iter + 1, len(progress_bar), loss.item())
                    progress_bar.set_description(descriptor)

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue

        mean_loss = np.mean(epoch_loss)
        val_descrip = '[Validation] Epoch: {}. Mean Loss: {:.6f}.'.format(epoch + 1, mean_loss)
        print(val_descrip)

        self.writer.add_scalars('Loss', {'val': mean_loss}, epoch)

        self.save_checkpoint(self.model, self.save_dir, 'last.pt')

        if self.best_loss > mean_loss:
            self.best_loss = mean_loss
            self.save_checkpoint(self.model, self.save_dir, 'best_val_loss.pt')

    def start(self):
        for epoch in range(self.epochs):
            self.train(epoch)
            self.validation(epoch)

    def save_checkpoint(self, model, saved_path, name):
        torch.save(model.model.state_dict(), saved_path + name)
