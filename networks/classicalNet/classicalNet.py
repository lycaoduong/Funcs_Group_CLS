from utils.ohlabs.trainutils import YamlRead
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
from tqdm import tqdm
import cv2
from utils.ohlabs.plotutils import func_confusion, subs_confusion, plot_data, plot_conf, plot_roc_pr_curve, subs_len_confusion
import pickle


class ClassNet(object):
    def __init__(self, opt):
        self.dataset = opt.dataset
        # Read dataset
        dataset_configs = YamlRead(f'configs/dataset/{self.dataset}.yaml')
        self.train_dir = dataset_configs.train_dir
        self.val_dir = dataset_configs.val_dir
        self.test_dir = dataset_configs.test_dir
        self.token_dir = dataset_configs.token_dir
        self.mean = dataset_configs.mean
        self.std = dataset_configs.std
        self.pos_dic = dataset_configs.pos_dic
        self.max_sequence = dataset_configs.max_sequence
        self.num_cls = dataset_configs.num_cls
        self.signal_size = opt.signal_size

        self.cls_dic = dataset_configs.pos_dic

        self.save_dir = '../runs/eval_paper/{}/'.format(opt.method)
        os.makedirs(self.save_dir, exist_ok=True)

        #Method
        if "DecisionTree" in opt.method:
            self.clf = MultiOutputClassifier(DecisionTreeClassifier(random_state=0))
        elif "RandomForest" in opt.method:
            self.clf = MultiOutputClassifier(RandomForestClassifier(random_state=0))
        else:
            self.clf = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=self.num_cls))

        train_X = []
        train_y = []

        print("Loading train data...")
        for file in tqdm(os.listdir(self.train_dir)):
            if file.endswith(".npy"):
                spectra = os.path.join(self.train_dir, file)
                label = os.path.join(self.train_dir, file[:-4] + '.txt')
                if os.path.isfile(label):
                    signal = np.load(spectra)
                    max = np.max(signal)
                    min = np.min(signal)
                    signal = ((signal.astype(np.float32) - min) / (max - min))
                    signal = np.expand_dims(signal, axis=0)
                    signal = cv2.resize(signal, (self.signal_size, 1), interpolation=cv2.INTER_CUBIC)

                    train_X.append(signal[0])

                    description = [line.rstrip('\n') for line in open(label)][0]
                    elements = []
                    components = description.split(" ")
                    for idx, component in enumerate(components):
                        elements.append(int(component))
                    elements = np.array(elements)
                    # gt = np.nonzero(elements)[0]
                    train_y.append(elements)
        self.train_X = np.array(train_X)
        self.train_y = train_y

        test_X = []
        test_y = []

        print("Loading test data...")
        for file in tqdm(os.listdir(self.test_dir)):
            if file.endswith(".npy"):
                spectra = os.path.join(self.test_dir, file)
                label = os.path.join(self.test_dir, file[:-4] + '.txt')
                if os.path.isfile(label):
                    signal = np.load(spectra)
                    max = np.max(signal)
                    min = np.min(signal)
                    signal = ((signal.astype(np.float32) - min) / (max - min))
                    signal = np.expand_dims(signal, axis=0)
                    signal = cv2.resize(signal, (self.signal_size, 1), interpolation=cv2.INTER_CUBIC)

                    test_X.append(signal[0])

                    description = [line.rstrip('\n') for line in open(label)][0]
                    elements = []
                    components = description.split(" ")
                    for idx, component in enumerate(components):
                        elements.append(int(component))
                    elements = np.array(elements)
                    # gt = np.nonzero(elements)[0]
                    test_y.append(elements)
        self.test_X = np.array(test_X)
        self.test_y = test_y
        self.predict = None
        self.funcs_confusion = np.zeros((self.num_cls, 2, 2))
        self.substance_confusion = np.zeros((8, 2))


    def start(self, save_dir=None):
        self.clf.fit(self.train_X, self.train_y)
        if save_dir is not None:
            self.saveModel(save_dir)

    def saveModel(self, filename):
        pickle.dump(self.clf, open(filename, "wb"))


    def loadModel(self, filename):
        self.clf = pickle.load(open(filename, "rb"))
        print(1)


    def eval(self):
        self.predict = self.clf.predict(self.test_X)
        target = np.array(self.test_y)
        self.funcs_confusion = func_confusion(target=target, result=self.predict, th=0.5)
        self.substance_confusion, _ = subs_len_confusion(target=target, result=self.predict, th=0.5)
        self.plot_confusion_matrix()

    def plot_confusion_matrix(self):
        # Plot total functional groups cf
        plot_conf(np.sum(self.funcs_confusion, axis=0), labelX=["Positive", "Negative"],
                  labelY=["Positive", "Negative"],
                  title="Total Functional groups Confusion Matrix",
                  save_dir=self.save_dir, save_name="fngs_cf.png", rotationY=90)
        # Plot subs cf
        # tem = np.zeros((2, 2))
        # tem[0] = self.substance_confusion
        plot_conf(self.substance_confusion, labelX=["True", "False"],
                  labelY=["1-Group", "2-Group", "3-Group", "4-Group", "5-Group", "6-Group", "7-Group", "Total"],
                  title="Molecule Confusion Matrix", save_dir=self.save_dir,
                  size=(19, 12),
                  save_name="subs_cf.png", rotationY=0)

        # Plot each functional group cf
        for i in self.cls_dic.keys():
            fng_name = self.cls_dic[i]
            conf = self.funcs_confusion[i]
            plot_conf(conf, labelX=["Positive", "Negative"],
                      labelY=["Positive", "Negative"],
                      title="{} Confusion Matrix".format(fng_name.capitalize()),
                      save_dir=self.save_dir,
                      save_name="{}_cf.png".format(fng_name), rotationY=90)

        print("Finish plot Confusion matrix, check save path: {}".format(self.save_dir))
