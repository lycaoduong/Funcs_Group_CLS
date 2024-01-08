import datetime
import os
import json
from utils.ohlabs.trainutils import YamlRead
from utils.ohlabs import transform as tr
from torchvision import transforms
from utils.ohlabs.dataloader import FCGClassificationDataset
from torch.utils.data import DataLoader
from networks.OblabsFcg.model import FCGClassification, FCGClassFormer
from networks.ircharacercnn.ircnn import IrCNN
from torch import nn
import torch
from tqdm.autonotebook import tqdm
import traceback
import numpy as np
from utils.ohlabs.plotutils import func_confusion, subs_confusion, plot_data, plot_conf


class ModelWithLoss(nn.Module):
    def __init__(self, model, reduction='mean'):
        super().__init__()
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, signal, target, **kwargs):
        o = self.model(signal)
        o = torch.squeeze(o)
        losses = self.criterion(o, target)
        return losses, torch.sigmoid(o)


class Evaluation(object):
    def __init__(self, eval_opt):
        self.project = eval_opt.project
        self.model_name = eval_opt.model
        self.dataset = eval_opt.dataset

        print('Project Name: ', self.project)
        print('Model and Dataset: ', self.model_name, self.dataset)
        date_time = datetime.datetime.now()
        date_time = date_time.strftime("%Y.%m.%d_%H.%M.%S")
        print('Date Access: ', date_time)


        exp_name, trial_name = 'single_run', date_time
        self.save_dir = '../runs/eval/{}/{}_{}/{}/{}/'.format(self.project, self.model_name, self.dataset, exp_name,
                                                              trial_name)
        os.makedirs(self.save_dir, exist_ok=True)


        # Save train parameters
        with open('{}/eval_params.txt'.format(self.save_dir), 'w') as f:
            json.dump(eval_opt.__dict__, f, indent=2)

        # Read dataset
        dataset_configs = YamlRead(f'configs/dataset/{self.dataset}.yaml')
        self.eval_dir = dataset_configs.eval_dir
        self.mean = dataset_configs.mean
        self.std = dataset_configs.std
        self.num_cls = dataset_configs.num_cls

        #Read Model configs
        model_configs = YamlRead(f'configs/model/{self.model_name}.yaml')
        self.signal_size = model_configs.signal_size

        eval_list = []
        for file in os.listdir(self.eval_dir):
            if file.endswith(".npy"):
                label = os.path.join(self.eval_dir, file[:-4] + '.txt')
                if os.path.isfile(label):
                    eval_list.append(file)

        # Data Loader

        self.device = eval_opt.device
        self.batch_size = eval_opt.batch_size


        eval_transforms = [
            tr.Normalizer(with_std=False),
            tr.Resizer(signal_size=self.signal_size)
        ]

        eval_set = FCGClassificationDataset(root_dir=self.eval_dir, list_data=eval_list, voca_dic=None, pos_dic=None,
                                            max_sequence=None, transform=transforms.Compose(eval_transforms))

        eval_params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'drop_last': False,
            'num_workers': eval_opt.num_worker
        }
        self.eval_generator = DataLoader(eval_set, collate_fn=tr.collater, **eval_params)

        # Model
        if self.model_name == "Fcg-B" or self.model_name == "Fcg-L":
            model = FCGClassification(embed_dim=model_configs.embed_dim, signal_size=model_configs.signal_size,
                                      patch_size=model_configs.patch_size, num_layers=model_configs.num_layers,
                                      expansion_factor=model_configs.expansion_factor, n_heads=model_configs.n_heads,
                                      num_cls=dataset_configs.num_cls)
        elif self.model_name == "IRCNN":
            model = IrCNN(signal_size=model_configs.signal_size)
        else:
            model = FCGClassFormer(embed_dim=model_configs.embed_dim, signal_size=model_configs.signal_size,
                                   patch_size=model_configs.patch_size, num_layers=model_configs.num_layers,
                                   expansion_factor=model_configs.expansion_factor, n_heads=model_configs.n_heads,
                                   num_cls=dataset_configs.num_cls)

        # if torch.cuda.is_available():
        #     model = nn.DataParallel(model)
        if eval_opt.ckpt is not None:
            weight = torch.load(eval_opt.ckpt, map_location=self.device)
            model.load_state_dict(weight, strict=True)

        self.model = ModelWithLoss(model=model, reduction='mean')
        self.model = self.model.to(self.device)

        self.num_iter_per_epoch = len(self.eval_generator)
        self.step = 0

        self.funcs_confusion = np.zeros((self.num_cls, 2, 2))
        self.substance_confusion = np.zeros((1, 2))
        self.cls_dic = dataset_configs.pos_dic
        self.data_dis = np.zeros((1, self.num_cls))
        self.cls_weight = np.zeros((1, self.num_cls))
        self.loss_weight = np.zeros((1, self.num_cls))
        self.th = eval_opt.threshold
        
    def eval_data_analysis(self):
        progress_bar = tqdm(self.eval_generator)
        for iter, data in enumerate(progress_bar):
            signals, tokenizer = data['signal'], data['tokenizer']
            tokenizer = tokenizer.cpu().numpy()
            self.data_dis += np.sum(tokenizer, axis=0)
        sum = np.sum(self.data_dis, axis=1)
        self.cls_weight = self.data_dis / sum
        self.loss_weight = (1.0 / self.cls_weight)
        plot_data(data_dis=self.data_dis, save_dir=self.save_dir, save_name="data_distribution.png")

    def plot_confusion_matrix(self):
        # Plot total functional groups cf
        plot_conf(np.sum(self.funcs_confusion, axis=0), label=["Positive", "Negative"],
                  title="Total functional groups confusion matrix",
                  save_dir=self.save_dir, save_name="fngs_cf.png")
        # Plot subs cf
        tem = np.zeros((2, 2))
        tem[0] = self.substance_confusion
        plot_conf(tem, label=["True", "False"], title="Total substances confusion matrix", save_dir=self.save_dir,
                  save_name="subs_cf.png")

        # Plot each functional group cf
        for i in self.cls_dic.keys():
            fng_name = self.cls_dic[i]
            conf = self.funcs_confusion[i]
            plot_conf(conf, label=["Positive", "Negative"], title="{} confusion matrix".format(fng_name),
                      save_dir=self.save_dir,
                      save_name="{}_cf.png".format(fng_name))

        print("Finish plot Confusion matrix, check save path: {}".format(self.save_dir))
        
    def eval(self):
        self.model.eval()
        last_epoch = self.step // self.num_iter_per_epoch
        progress_bar = tqdm(self.eval_generator)
        losses = []

        for iter, data in enumerate(progress_bar):
            if iter < self.step - last_epoch * self.num_iter_per_epoch:
                progress_bar.update()
                continue
            try:
                signals, tokenizer = data['signal'], data['tokenizer']
                signals = signals.to(self.device)
                tokenizer = tokenizer.to(self.device)
                tokenizer = tokenizer.to(torch.float32)

                with torch.no_grad():
                    loss, predict = self.model(signals, tokenizer)


                fun_conf = func_confusion(target=tokenizer.cpu().numpy(), result=predict.cpu().numpy(), th=self.th)
                self.funcs_confusion += fun_conf

                subs_conf = subs_confusion(target=tokenizer.cpu().numpy(), result=predict.cpu().numpy(), th=self.th)
                self.substance_confusion += subs_conf

                losses.append(float(loss.item()))

                descriptor = '[Eval] Step: {}. Iteration: {}/{}. Loss: {:.6f}.'.format(
                        self.step, iter + 1, self.num_iter_per_epoch, loss.item())
                progress_bar.set_description(descriptor)
                self.step += 1

            except Exception as e:
                print('[Error]', traceback.format_exc())
                print(e)
                continue

        mean_loss = np.mean(losses)
        eval_descrip = '[Eval]. Mean Loss: {:.6f}.'.format(mean_loss)
        print(eval_descrip)
        self.plot_confusion_matrix()

    def start(self):
        # self.eval_data_analysis()
        self.eval()
