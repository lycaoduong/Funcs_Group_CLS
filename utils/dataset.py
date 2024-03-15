import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class DatasetPreprocess(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        # self.ir_dir = os.path.join(root_dir, "ir_1.csv")
        # self.target_dir = os.path.join(root_dir, "target.csv")
        # self.data_post = os.path.join(self.root_dir, "ir_processed.csv")

    def fill_nan_value(self, infile='ir_1.csv', save_dir=None):
        ir_dir = os.path.join(self.root_dir, infile)
        df = pd.read_csv(ir_dir)
        df = df.interpolate(limit_area='inside').fillna(0)
        df_new = df.rename(columns={'Unnamed: 0': 'lambda'})
        if save_dir is not None:
            df_new.to_csv(save_dir, index=False)
        else:
            save_dir = os.path.join(self.root_dir, "ir_processed.csv")
            df_new.to_csv(save_dir, index=False)
            print("Finish - Check output folder at {}".format(self.root_dir))

    def clean_dataset(self, ir_file="ir_processed.csv", target_file="target.csv"):
        target_dir = os.path.join(self.root_dir, target_file)
        ir_dir = os.path.join(self.root_dir, ir_file)
        ir_df = pd.read_csv(ir_dir)
        col_names = ir_df.columns.tolist()
        lambda_name = ir_df['lambda'].tolist()
        target = pd.read_csv(target_dir)
        tg_col_name = target.columns.tolist()
        target_names = target['cas'].tolist()
        new_target = []
        new_df = [lambda_name]
        df_col_name =["lambda"]
        for i, tg in enumerate(target_names):
            if str(tg) in col_names:
                new_target.append(target.loc[i].tolist())
                new_df.append(ir_df[str(tg)])
                df_col_name.append(str(tg))

        new_target = np.array(new_target)
        sub_data = new_target[:, 1:]
        sum = np.sum(sub_data, axis=1)
        max_fcs = np.max(sum)
        print("Max FCN / sample: {}".format(max_fcs))
        new_target = pd.DataFrame(new_target, columns=tg_col_name)
        save_dir = os.path.join(self.root_dir, "target_filtered.csv")
        new_target.to_csv(save_dir, index=False)

        new_df = np.array(new_df)
        print("Total data: {}".format(new_df.shape[0]-1))
        new_df = np.transpose(new_df)
        new_df = pd.DataFrame(new_df, columns=df_col_name)
        save_dir = os.path.join(self.root_dir, "ir_filtered.csv")
        new_df.to_csv(save_dir, index=False)
        print("Finish - Check output folder at {}".format(self.root_dir))

    def tral_val_dataset(self, ir, target, val_size=0.2):
        train_dir = os.path.join(self.root_dir, 'train')
        os.makedirs(train_dir, exist_ok=True)
        val_dir = os.path.join(self.root_dir, 'test')
        os.makedirs(val_dir, exist_ok=True)
        labels = pd.read_csv(os.path.join(self.root_dir, target))
        datasets = pd.read_csv(os.path.join(self.root_dir, ir))
        total_label = labels['cas'].tolist()
        train, val = train_test_split(total_label, test_size=val_size, random_state=42)
        print("Extracting train dataset\n")
        for file in tqdm(train):
            value = datasets[str(file)].to_numpy()
            index = total_label.index(file)
            target = labels.iloc[index].tolist()[1:]
            np.save(os.path.join(train_dir, '{}.npy'.format(file)), value)
            with open(os.path.join(train_dir, '{}.txt'.format(file)), 'w') as f:
                for i, item in enumerate(target):
                    if i != len(target)-1:
                        f.write(f"{item} ")
                    else:
                        f.write(f"{item}")

        print("Extracting val dataset\n")
        for file in tqdm(val):
            value = datasets[str(file)].to_numpy()
            index = total_label.index(file)
            target = labels.iloc[index].tolist()[1:]
            np.save(os.path.join(val_dir, '{}.npy'.format(file)), value)
            with open(os.path.join(val_dir, '{}.txt'.format(file)), 'w') as f:
                for i, item in enumerate(target):
                    if i != len(target)-1:
                        f.write(f"{item} ")
                    else:
                        f.write(f"{item}")

    def split_by_num(self, ir, target, split_size=[0.75, 0.15, 0.1]):
        val_test_size = split_size[1] + split_size[2]
        test_size = split_size[2] / (split_size[1] + split_size[2])
        train_dir = os.path.join(self.root_dir, 'by_len', 'train')
        os.makedirs(train_dir, exist_ok=True)
        val_dir = os.path.join(self.root_dir, 'by_len', 'val')
        os.makedirs(val_dir, exist_ok=True)
        test_dir = os.path.join(self.root_dir, 'by_len', 'test')
        os.makedirs(test_dir, exist_ok=True)
        labels = pd.read_csv(os.path.join(self.root_dir, target))
        datasets = pd.read_csv(os.path.join(self.root_dir, ir))
        total_label = labels['cas'].tolist()
        label_array = labels.to_numpy()[:, 1:]
        sum = np.sum(label_array, axis=1)
        max_len = np.max(sum)
        # min_len = np.min(sum)
        for i in range(1, max_len+1):
            filter_id = np.where(sum == i)[0]
            train_id, val_test_id = train_test_split(filter_id, test_size=val_test_size, random_state=42)
            val_id, test_id = train_test_split(val_test_id, test_size=test_size, random_state=42)

            filter_train_df = labels.iloc[train_id]
            filter_val_df = labels.iloc[val_id]
            filter_test_df = labels.iloc[test_id]

            file_name = filter_train_df['cas'].tolist()
            print("Extracting train dataset\n")
            for file in file_name:
                value = datasets[str(file)].to_numpy()
                index = total_label.index(file)
                target = labels.iloc[index].tolist()[1:]
                np.save(os.path.join(train_dir, '{}.npy'.format(file)), value)
                with open(os.path.join(train_dir, '{}.txt'.format(file)), 'w') as f:
                    for i, item in enumerate(target):
                        if i != len(target) - 1:
                            f.write(f"{item} ")
                        else:
                            f.write(f"{item}")

            file_name = filter_val_df['cas'].tolist()
            print("Extracting val dataset\n")
            for file in tqdm(file_name):
                value = datasets[str(file)].to_numpy()
                index = total_label.index(file)
                target = labels.iloc[index].tolist()[1:]
                np.save(os.path.join(val_dir, '{}.npy'.format(file)), value)
                with open(os.path.join(val_dir, '{}.txt'.format(file)), 'w') as f:
                    for i, item in enumerate(target):
                        if i != len(target) - 1:
                            f.write(f"{item} ")
                        else:
                            f.write(f"{item}")

            file_name = filter_test_df['cas'].tolist()
            print("Extracting test dataset\n")
            for file in tqdm(file_name):
                value = datasets[str(file)].to_numpy()
                index = total_label.index(file)
                target = labels.iloc[index].tolist()[1:]
                np.save(os.path.join(test_dir, '{}.npy'.format(file)), value)
                with open(os.path.join(test_dir, '{}.txt'.format(file)), 'w') as f:
                    for i, item in enumerate(target):
                        if i != len(target) - 1:
                            f.write(f"{item} ")
                        else:
                            f.write(f"{item}")
