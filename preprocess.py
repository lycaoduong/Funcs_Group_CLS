import os
import shutil

from utils.dataset import DatasetPreprocess

if __name__ == '__main__':
    data_dir = r"D:\lycaoduong\workspace\datasets\ohlabs\minh\dataset\private\15compounds"
    label_dir = r"D:\lycaoduong\workspace\datasets\ohlabs\minh\dataset\private\npyRename"
    data_processor = DatasetPreprocess(root_dir=data_dir)
    # process.fill_nan_value(save_dir=None)
    # ir = "ir_processed.csv"
    # tg = "target.csv"
    # data_processor.clean_dataset(ir_file=ir, target_file=tg)
    # ir = "ir_cleaned.csv"
    # tg = "target_cleaned.csv"
    # data_processor.tral_val_dataset(ir=ir, target=tg, val=0.1)
    # data_processor.split_by_num(ir=ir, target=tg)
    # all_csv = os.listdir(data_dir)
    all_file = ['100425.npy', '108907.npy', '1634044.npy', '67663.npy', '67721.npy', '78933.npy', '79469.npy',
                '80626.npy', '88062.npy', '95534.npy', '98953.npy', '57147.npy', '57578.npy', '77474.npy', '78591.npy',
                '79118.npy', '87683.npy']

    for file in all_file:
        if file.endswith('.npy'):
            # file_p = os.path.join(data_dir, file)
            # data_processor.csv2numpy(file_p)
            bname = file.split('.')[0]
            src = os.path.join(label_dir, '{}.txt'.format(bname))
            trg = os.path.join(data_dir, '{}.txt'.format(bname))
            shutil.copy(src, trg)
            src = os.path.join(label_dir, '{}.npy'.format(bname))
            trg = os.path.join(data_dir, '{}.npy'.format(bname))
            shutil.copy(src, trg)
