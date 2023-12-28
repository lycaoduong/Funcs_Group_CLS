from utils.dataset import DatasetPreprocess

if __name__ == '__main__':
    data_dir = "D:/lycaoduong/workspace/datasets/ohlabs/minh/origin"
    data_processor = DatasetPreprocess(root_dir=data_dir)
    # process.fill_nan_value(save_dir=None)
    ir = "ir_processed.csv"
    tg = "target.csv"
    data_processor.clean_dataset(ir_file=ir, target_file=tg)
    # ir = "ir_filtered.csv"
    # tg = "target_filtered.csv"
    # data_processor.tral_val_dataset(save_dir=data_dir, ir=ir, target=tg, val=0.2)
