import os.path

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from tqdm import tqdm
import random
from dataloader.data_loader_PTBXL import data_generator_ptbxl
from dataloader.data_loader import data_generator_non
from dataloader.data_loader_2D import data_generator_2D
from utils.Metric import top_k_accuracy, top_k_recall, top_k_precision, top_k_f1_score, Metrics, AUC, \
    metric_summary, \
    FocalLoss
import torch as tr
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time

from model.FC_Model import STAR
import warnings

warnings.filterwarnings("ignore")
local_date = time.strftime('%m.%d', time.localtime(time.time()))
load_data_path = 'your_data_path/data/'


class Train():
    def __init__(self, args, k, seed):
        self.args = args
        self.k = k
        self.seed = seed
        if args.model_name == 'STAR':
            # 加载数据生成器和模型
            self.data_all = data_generator_ptbxl(load_data_path + f'{args.dataset}/process_data/',
                                                 k, args=args)
            self.net = STAR(args.space_out_dim, args.time_out_dim, args.conv_kernel, args.hidden_dim,
                            args.time_denpen_len, args.num_sensor, args.num_windows, args.decay,
                            args.pool_ratio, args.n_class)
        self.net = self.net.cuda(args.gpu) if tr.cuda.is_available() else self.net
        self.loss_function = nn.BCEWithLogitsLoss()
        self.my_loss = FocalLoss()
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr)
        self.scheduler = tr.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[5, 10, 15], gamma=0.5, last_epoch=-1)
        # 10-fold Cross Validation setup
        self.kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    def Train_batch(self):
        self.net.train()
        loss_ = 0
        count = 0
        for data, label in tqdm(self.train, desc='Train'):
            data = data.cuda(args.gpu) if tr.cuda.is_available() else data
            label = label.cuda(args.gpu) if tr.cuda.is_available() else label
            self.optim.zero_grad()
            prediction = self.net(data)
            loss = self.loss_function(prediction, label)
            loss.backward()
            self.optim.step()
            loss_ = loss_ + loss.item()
            count += 1
        return loss_ / count

    def Train_model(self):
        epoch = self.args.epoch
        cross_auc = 0
        train_time = 0
        results_acc = []
        results_auc = []
        results_f1 = []

        # Loop over the K-fold splits
        for fold, (train_idx, test_idx) in enumerate(self.kfold.split(self.data_all)):
            # if fold == 1:
            #     break
            print(f"Training fold {fold + 1}")
            # Create train and test data for the fold
            train_data = tr.utils.data.Subset(self.data_all, train_idx)
            test_data = tr.utils.data.Subset(self.data_all, test_idx)

            self.train = tr.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size,
                                                  shuffle=True, drop_last=args.drop_last,
                                                  num_workers=0)
            self.test = tr.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size,
                                                 shuffle=True, drop_last=args.drop_last,
                                                 num_workers=0)

            for i in range(epoch):
                time0 = time.time()
                loss = self.Train_batch()
                print(f"TRAIN Epoch {i} loss is: {loss}")
                duration = time.time() - time0
                train_time += duration
                msg_train = f'TRAIN, Epoch {i}, loss {loss}'
                self.log_msg(msg_train, './experiment/{}/{}/folds_10/{}_{}_train_{}.txt'.format(
                    args.dataset, args.model_name, args.model_name, local_date, args.k))

                # Validation
                if i % self.args.show_interval == 0:
                    loss_val, accuracy_val, auc_val = self.Cross_validation()
                    self.scheduler.step(loss_val)
                    print(f"VALID Epoch {i} loss_val is: {loss_val}, acc_val is: {accuracy_val}, auc_val is: {auc_val}")
                    msg_valid = f'VALID, Epoch {i}, loss {loss_val}, acc {accuracy_val}, auc {auc_val}'
                    self.log_msg(msg_valid,
                                 './experiment/{}/{}/folds_10/{}_{}_valid_{}.txt'.format(
                                     args.dataset, args.model_name, args.model_name, local_date, args.k))

                    if auc_val > cross_auc:
                        cross_auc = auc_val
                        paths = args.save_path + 'ECG.model_' + str(fold) + '_' + str(self.seed)
                        tr.save(self.net.state_dict(), paths)

            # Testing
            print(f'final train time = {train_time / int(epoch):.4f}')
            self.net.load_state_dict(tr.load(paths))
            mean_accuracy_test, roc_auc_test, f1_socre = self.Prediction()
            results_acc.append(mean_accuracy_test)
            results_auc.append(roc_auc_test)
            results_f1.append(f1_socre)

        flag = 1
        if flag == 1:  # 为1时保存
            results1 = np.array(results_acc)
            results2 = np.array(results_auc)
            results3 = np.array(results_f1)
            print(np.mean(results1, 0))
            print(np.mean(results2, 0))
            log_file = f'./experiment/HFHC/{args.model_name}/folds_10/total/result2.txt'
            with open(log_file, 'a') as f:
                f.write(args.model_name)
                f.write('\n')
                f.write('10 Fold Accuracy:\n')
                f.write('result: ' + str(results1))
                f.write('\n')
                f.write('mean: ' + str(np.mean(results1, 0)))
                f.write('\n')
                f.write('std: ' + str(np.std(results1, 0)))

                f.write('\n')
                f.write('10 Fold AUC:\n')
                f.write('result_new: ' + str(results2))
                f.write('\n')
                f.write('mean_new: ' + str(np.mean(results2, 0)))
                f.write('\n')
                f.write('std_new: ' + str(np.std(results2, 0)))

                f.write('\n')
                f.write('10 Fold F1:\n')
                f.write('result_new: ' + str(results3))
                f.write('\n')
                f.write('mean_new: ' + str(np.mean(results3, 0)))
                f.write('\n')
                f.write('std_new: ' + str(np.std(results3, 0)))

        return mean_accuracy_test, roc_auc_test, f1_socre

    def cuda_(self, x):
        x = tr.Tensor(np.array(x))
        if tr.cuda.is_available():
            return x.cuda(args.gpu)
        else:
            return x

    def Cross_validation(self):
        self.net.eval()
        prediction_ = []
        real_ = []
        loss_ = []
        with tr.no_grad():
            for data, label in tqdm(self.test, desc="Valid"):
                data = data.cuda(args.gpu) if tr.cuda.is_available() else data
                label = label.cuda(args.gpu) if tr.cuda.is_available() else label
                prediction = self.net(data)
                # 计算loss
                loss = self.loss_function(prediction, label)
                loss_.append(loss.detach().cpu().numpy())
                real_.append(label.detach().cpu())  # cpu标签
                prediction_.append(prediction.detach().cpu())
            prediction_ = tr.cat(prediction_, 0)
            real_ = tr.cat(real_, 0)
            valid_columns = []
            for i in range(real_.shape[1]):
                unique_vals = np.unique(real_[:, i].cpu().numpy())
                if len(unique_vals) > 1:
                    valid_columns.append(i)

            real_ = real_[:, valid_columns]
            prediction_ = prediction_[:, valid_columns]

            roc_score = roc_auc_score(real_, prediction_, average="macro")
            _, mean_acc = Metrics(real_, prediction_)

        return np.mean(loss_), mean_acc, roc_score

    def Prediction(self):
        self.net.eval()
        prediction_ = []
        real_ = []
        for data, label in tqdm(self.test, desc="Test"):
            data = data.cuda(args.gpu) if tr.cuda.is_available() else data
            real_.append(label)
            prediction = self.net(data)
            prediction_.append(prediction.detach().cpu())
        prediction_ = tr.cat(prediction_, 0)
        real_ = tr.cat(real_, 0)
        valid_columns = []
        for i in range(real_.shape[1]):
            unique_vals = np.unique(real_[:, i].cpu().numpy())
            if len(unique_vals) > 1:
                valid_columns.append(i)
        real_ = real_[:, valid_columns]
        prediction_ = prediction_[:, valid_columns]

        roc_score = roc_auc_score(real_, prediction_)
        acc, mean_acc = Metrics(real_, prediction_)  # acc 每类疾病的
        class_auc = AUC(real_, prediction_)
        summary = metric_summary(real_.numpy(), prediction_.numpy())
        f1 = summary[0]

        # ecg_data.challenge_metrics(y_test, one_hot(np.argmax(pred_all, axis=1), num_classes))
        print(f"class wise accuracy: {acc}")
        print(f"accuracy: {mean_acc}")
        print(f"roc_score : {roc_score}")
        print(f"f1 : {f1}")
        print(f"class wise AUC : {class_auc}")
        print(f"class wise precision, recall, f1 score : {summary}")

        # save log
        msg_test = (f'class wise accuracy: {acc}\n'
                    f'accuracy {mean_acc}, roc_auc {roc_score}, class wise AUC : {class_auc} \n'
                    f'class wise precision, recall, f1 score : {summary} \n\n')
        self.log_msg(msg_test,
                     './experiment/{}/{}/folds_10/{}_{}_test.txt'.format(args.dataset, args.model_name,
                                                                         args.model_name,
                                                                         local_date))

        return mean_acc, roc_score, f1

    def log_msg(self, message, log_file):
        with open(log_file, 'a') as f:
            print(message, file=f)


if __name__ == '__main__':
    from args import args

    args = args()


    def args_config_HFHC(args):
        args.epoch = 30
        args.k = 1
        args.window_sample = 1000

        args.decay = 0.8
        args.pool_ratio = 0.2
        args.lr = 1e-4
        args.batch_size = 32

        args.conv_kernel = [1, 3, 5]
        args.patch_size = 500
        args.time_denpen_len = int(args.window_sample / args.patch_size)
        args.conv_out = 14
        args.num_windows = 2
        args.conv_time_CNN = 6

        args.lstmout_dim = 64
        args.hidden_dim = 32
        args.lstmhidden_dim = 32

        args.num_sensor = 12
        args.dataset = 'HFHC'
        args.n_class = 34  # HFHC
        args.model_name = 'STAR'
        return args


    args = args_config_HFHC(args)
    print("=======================Start Train================================")
    print("Model:{}   Dataset:{}   Epoch:{}   GPU:{} ".format(args.model_name, args.dataset, args.epoch, args.gpu))
    args.save_path = f'your_path_savv/save/{args.dataset}/{args.model_name}/'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    trainer = Train(args, 1, seed=42)  # Set your seed here
    accuracy, auc, f1 = trainer.Train_model()
