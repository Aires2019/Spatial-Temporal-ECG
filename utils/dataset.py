import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from data_preprocessing.preprocess_ECG import load_dataset, compute_label_aggregations, select_data, \
    preprocess_signals, data_slice, hf_dataset, load_raw_data_hf


## First run the dataset file to generate the processed data

class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    """

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        super(ECGDataset, self).__init__()
        self.data = signals
        self.label = labels
        self.num_classes = self.label.shape[1]

        self.cls_num_list = np.sum(self.label, axis=0)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.label[index]

        x = x.transpose()

        x = torch.tensor(x.copy(), dtype=torch.float)

        y = torch.tensor(y, dtype=torch.float)
        y = y.squeeze()
        return x, y

    def __len__(self):
        return len(self.data)


class DownLoadECGData:
    '''
        All experiments data
    '''

    def __init__(self, experiment_name, task, datafolder, sampling_frequency=100, min_samples=0,
                 train_fold=8, val_fold=9, test_fold=10):
        self.min_samples = min_samples
        self.task = task
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold
        self.experiment_name = experiment_name
        self.datafolder = datafolder
        self.sampling_frequency = sampling_frequency

    def preprocess_data(self):
        # Load PTB-XL data
        data, raw_labels = load_dataset(self.datafolder, self.sampling_frequency)
        # Preprocess label data
        labels = compute_label_aggregations(raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        data, labels, Y, _ = select_data(data, labels, self.task, self.min_samples)

        if self.datafolder == 'your_data_path/data/CPCS/':
            # 进行数据切片 长度1000
            data = data_slice(data)

        # 10th fold for testing (9th for now)
        X_test = data[labels.strat_fold == self.test_fold]
        y_test = Y[labels.strat_fold == self.test_fold]
        # 9th fold for validation (8th for now)
        X_val = data[labels.strat_fold == self.val_fold]
        y_val = Y[labels.strat_fold == self.val_fold]
        # rest for training
        X_train = data[(labels.strat_fold != self.test_fold) & (labels.strat_fold != self.val_fold)]
        y_train = Y[(labels.strat_fold != self.test_fold) & (labels.strat_fold != self.val_fold)]

        # Preprocess signal data
        X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test)

        return X_train, y_train, X_val, y_val, X_test, y_test, str(self.train_fold)


def load_datasets(datafolder=None, experiment=None):
    '''
    Load the final dataset
    '''
    experiment = experiment

    if datafolder == 'your_data_path/data/PTB_XL/raw/':
        save_path = 'your_data_path/data/PTB_XL/process_data_MI/'
        experiments = {
            'exp0': ('exp0', 'all'),
            'exp1': ('exp1', 'diagnostic'),
            'exp1.2': ('exp1.2', 'diagnostic_MI'),
            'exp1.1': ('exp1.1', 'subdiagnostic'),
            'exp1.1.1': ('exp1.1.1', 'superdiagnostic'),
            'exp2': ('exp2', 'form'),
            'exp3': ('exp3', 'rhythm')
        }
        name, task = experiments[experiment]
        ded = DownLoadECGData(name, task, datafolder)
        X_train, y_train, X_val, y_val, X_test, y_test, split = ded.preprocess_data()

        np.save(save_path + f'X_train_{task}_{split}.npy', X_train)
        np.save(save_path + f'y_train_{task}_{split}.npy', y_train)
        np.save(save_path + f'X_val_{task}_{split}.npy', X_val)
        np.save(save_path + f'y_val_{task}_{split}.npy', y_val)
        np.save(save_path + f'X_test_{task}_{split}.npy', X_test)
        np.save(save_path + f'y_test_{task}_{split}.npy', y_test)

    elif datafolder == 'your_data_path/data/CPCS/':
        save_path = 'your_data_path/data/CPCS/process_data/'
        ded = DownLoadECGData('exp_CPSC', 'all', datafolder)
        X_train, y_train, X_val, y_val, X_test, y_test, split = ded.preprocess_data()
        np.save(save_path + f'X_train_{split}.npy', X_train)
        np.save(save_path + f'y_train_{split}.npy', y_train)
        np.save(save_path + f'X_val_{split}.npy', X_val)
        np.save(save_path + f'y_val_{split}.npy', y_val)
        np.save(save_path + f'X_test_{split}.npy', X_test)
        np.save(save_path + f'y_test_{split}.npy', y_test)
    else:
        data, label = load_raw_data_hf(datafolder)
        # X_train, y_train, X_val, y_val, X_test, y_test = hf_dataset(datafolder)
        save_path = 'your_data_path/data/HFHC/process_data/'
        # np.save(save_path + 'X_train.npy', X_train)
        # np.save(save_path + 'y_train.npy', y_train)
        # np.save(save_path + 'X_val.npy', X_val)
        # np.save(save_path + 'y_val.npy', y_val)
        # np.save(save_path + 'X_test.npy', X_test)
        # np.save(save_path + 'y_test.npy', y_test)
        np.save(save_path + 'data.npy', data)
        np.save(save_path + 'label.npy', label)


if __name__ == "__main__":
    # PTB-XL
    # task = 'exp1.2'
    # load_datasets('your_data_path/data/PTB_XL/raw/', task)
    # # CPCS
    # load_datasets('your_data_path/data/CPCS/')
    # HFHC
    load_datasets('your_data_path/data/HFHC/raw/')
