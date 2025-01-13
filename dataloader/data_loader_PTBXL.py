import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, data, label, args):
        super(Load_Dataset, self).__init__()

        X_train = data
        y_train = label

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).float()
        else:
            self.x_data = X_train.float()
            self.y_data = y_train.float()

        self.len = X_train.shape[0]
        shape = self.x_data.size()
        self.x_data = self.x_data.reshape(shape[0], shape[1], args.time_denpen_len, args.patch_size)
        self.x_data = torch.transpose(self.x_data, 1, 2)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_ptbxl(data_path, k, args):
    # if args.dataset == 'PTB_XL':
    #     train_data_multi = np.load(data_path + f"X_train_superdiagnostic_{k}.npy")
    #     train_label_multi = np.load(data_path + f"y_train_superdiagnostic_{k}.npy")
    #     test_data_multi = np.load(data_path + f"X_test_superdiagnostic_{k}.npy")
    #     test_label_multi = np.load(data_path + f"y_test_superdiagnostic_{k}.npy")
    #     val_data_multi = np.load(data_path + f"X_val_superdiagnostic_{k}.npy")
    #     val_label_multi = np.load(data_path + f"y_val_superdiagnostic_{k}.npy")
    if args.dataset == 'PTB_XL':
        train_data_multi = np.load(data_path + f"X_train_diagnostic_MI_{k}.npy")
        train_label_multi = np.load(data_path + f"y_train_diagnostic_MI_{k}.npy")
        test_data_multi = np.load(data_path + f"X_test_diagnostic_MI_{k}.npy")
        test_label_multi = np.load(data_path + f"y_test_diagnostic_MI_{k}.npy")
        val_data_multi = np.load(data_path + f"X_val_diagnostic_MI_{k}.npy")
        val_label_multi = np.load(data_path + f"y_val_diagnostic_MI_{k}.npy")
    elif args.dataset == 'ICBEB':
        train_data_multi = np.load(data_path + "X_train_[8, 9, 10].npy")
        train_label_multi = np.load(data_path + "y_train_[8, 9, 10].npy")
        test_data_multi = np.load(data_path + "X_test_[8, 9, 10].npy")
        test_label_multi = np.load(data_path + "y_test_[8, 9, 10].npy")
        val_data_multi = np.load(data_path + "X_val_[8, 9, 10].npy")
        val_label_multi = np.load(data_path + "y_val_[8, 9, 10].npy")
    elif args.dataset == 'HFHC':
        # train_data_multi = np.load(data_path + f"X_train.npy")
        # train_label_multi = np.load(data_path + f"y_train.npy")
        test_data_multi = np.load(data_path + f"X_test.npy")
        test_label_multi = np.load(data_path + f"y_test.npy")
        val_data_multi = np.load(data_path + f"X_val.npy")
        val_label_multi = np.load(data_path + f"y_val.npy")
        # train 代表全部
        train_data_multi = np.load(data_path + f"data.npy")
        train_label_multi = np.load(data_path + f"label.npy")

    train_dataset = torch.tensor(train_data_multi)
    train_label = torch.tensor(train_label_multi)
    valid_dataset = torch.tensor(val_data_multi)
    valid_label = torch.tensor(val_label_multi)
    test_dataset = torch.tensor(test_data_multi)
    test_label = torch.tensor(test_label_multi)

    train_dataset = Load_Dataset(train_dataset, train_label, args)
    valid_dataset = Load_Dataset(valid_dataset, valid_label, args)
    test_dataset = Load_Dataset(test_dataset, test_label, args)

    if args.dataset == 'HFHC':
        return train_dataset

    else:

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, drop_last=args.drop_last,
                                                   num_workers=0)
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
                                                   shuffle=False, drop_last=args.drop_last,
                                                   num_workers=0)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                                  shuffle=False, drop_last=False,
                                                  num_workers=0)

        return train_loader, valid_loader, test_loader
