import os.path

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import random
from dataloader.data_loader_PTBXL import data_generator_ptbxl
from utils.Metric import Metrics, AUC, metric_summary, FocalLoss
import torch as tr
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from model.FC_Model import STAR
import warnings

warnings.filterwarnings("ignore")
local_date = time.strftime('%m.%d', time.localtime(time.time()))
load_data_path = 'your_path/data/'


class Train:
    def __init__(self, args, k, seed):
        self.args = args
        self.k = k
        self.seed = seed
        if args.model_name == 'STAR':
            # load dataloader and model
            self.train, self.valid, self.test = data_generator_ptbxl(load_data_path + f'{args.dataset}/process_data/',
                                                                     k, args=args)
            self.net = STAR(args.space_out_dim, args.time_out_dim, args.conv_kernel, args.hidden_dim,
                            args.time_denpen_len, args.num_sensor, args.num_windows, args.decay,
                            args.pool_ratio, args.n_class)

        self.net = self.net.cuda(args.gpu) if tr.cuda.is_available() else self.net
        # if tr.cuda.device_count() > 1:
        #     self.net = tr.nn.DataParallel(self.net, device_ids=[2, 3])
        self.loss_function = nn.BCEWithLogitsLoss()
        self.my_loss = FocalLoss()
        self.optim = optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=1e-4)
        self.scheduler = tr.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[5, 10, 15], gamma=0.5, last_epoch=-1)

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
        cross_accu = 0
        train_time = 0
        test_best_acc = 0
        test_best_auc = 0

        for i in range(epoch):
            time0 = time.time()
            loss = self.Train_batch()
            self.scheduler.step()
            print("TRAIN Epoch {} loss is:{}".format(i, loss))
            # save
            msg_train = f'TRAIN, Epoch {i}, loss {loss}'
            self.log_msg(msg_train,
                         './experiment/{}/{}/MI/{}_{}_train_{}.txt'.format(args.dataset, args.model_name,
                                                                           args.model_name,
                                                                           local_date, args.k))
            duration = time.time() - time0
            train_time += duration
            if i % self.args.show_interval == 0:
                loss_val, accuracy_val, auc_val = self.Cross_validation()
                print("VALID Epoch {} loss_val is:{}, acc_val is:{}, auc_val is:{}".format(i, loss_val, accuracy_val,
                                                                                           auc_val))
                # save
                msg_valid = f'VALID, Epoch {i}, loss {loss_val}, acc {accuracy_val}, auc {auc_val}'
                self.log_msg(msg_valid,
                             './experiment/{}/{}/MI/{}_{}_valid_{}.txt'.format(args.dataset, args.model_name,
                                                                               args.model_name, local_date,
                                                                               args.k))

                if accuracy_val > cross_accu:
                    cross_accu = accuracy_val
                    paths = args.save_path + 'ECG.model_test_' + str(self.k) + '_' + str(self.seed)
                    tr.save(self.net.state_dict(), paths)

        print('final train time = %.4f' % (train_time / int(epoch)))
        self.net.load_state_dict(tr.load(paths))
        mean_accuracy_test, roc_auc_test = self.Prediction()
        return mean_accuracy_test, roc_auc_test

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
            for data, label in tqdm(self.valid, desc="Valid"):
                data = data.cuda(args.gpu) if tr.cuda.is_available() else data
                label = label.cuda(args.gpu) if tr.cuda.is_available() else label
                prediction = self.net(data)
                loss = self.loss_function(prediction, label)
                loss_.append(loss.detach().cpu().numpy())
                real_.append(label.detach().cpu())
                prediction_.append(prediction.detach().cpu())
            prediction_ = tr.cat(prediction_, 0)
            real_ = tr.cat(real_, 0)

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

        roc_score = roc_auc_score(real_, prediction_)
        acc, mean_acc = Metrics(real_, prediction_)  # acc 每类疾病的
        class_auc = AUC(real_, prediction_)
        summary = metric_summary(real_.numpy(), prediction_.numpy())

        print(f"class wise accuracy: {acc}")
        print(f"accuracy: {mean_acc}")
        print(f"roc_score : {roc_score}")
        print(f"class wise AUC : {class_auc}")
        print(f"class wise precision, recall, f1 score : {summary}")
        # save log
        msg_test = (
            f'class wise accuracy: {acc}\n'
            f'accuracy {mean_acc}, roc_auc {roc_score}, class wise AUC : {class_auc} \n'
            f'class wise precision, recall, f1 score : {summary} \n\n')
        self.log_msg(msg_test,
                     './experiment/{}/{}/MI/{}_{}_test.txt'.format(args.dataset, args.model_name, args.model_name,
                                                                   local_date))
        return mean_acc, roc_score

    def log_msg(self, message, log_file):
        with open(log_file, 'a') as f:
            print(message, file=f)


if __name__ == '__main__':
    from args import args

    args = args()

    def args_config_PTBXL(args):
        args.epoch = 30
        args.k = 1
        args.window_sample = 1000

        args.decay = 0.8
        args.pool_ratio = 0.2
        args.lr = 1e-3
        args.batch_size = 32

        args.conv_kernel = [1, 3, 5, 7, 9]
        args.patch_size = 500
        args.time_denpen_len = int(args.window_sample / args.patch_size)
        args.conv_out = 64  # STAR_original
        args.num_windows = 2  # STAR_original
        args.conv_time_CNN = 6

        args.time_out_dim = 64
        args.hidden_dim = 128
        args.space_out_dim = 64

        args.num_sensor = 12
        # classes
        args.n_class = 5  # PTB-XL-AR:5 PTB-MI:15 CPCS2018:9 HFHC:34
        args.model_name = 'STAR'
        return args


    args = args_config_PTBXL(args)
    print("=======================Start Train================================")
    print("Model:{}   Dataset:{}   Epoch:{}   GPU:{} ".format(args.model_name, args.dataset, args.epoch, args.gpu))
    args.save_path = f'./save/{args.dataset}/{args.model_name}/'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    results_acc = []
    results_auc = []
    for i in range(1):
        k = 1
        # seed = random.randint(1, 10000)
        seeds = [4010, 7466, 3726, 1756, 6303, 3678, 2567, 2276, 3545, 1267]
        seed = seeds[k - 1]
        print(seed)
        random.seed(seed)
        np.random.seed(seed)
        tr.manual_seed(seed)
        tr.cuda.manual_seed(seed)
        train = Train(args, k, seed)
        mean_acc, roc_auc = train.Train_model()
        results_acc.append(mean_acc)
        results_auc.append(roc_auc)
        print("=======================================================================")

    results = np.array(results_acc)
    results_new = np.array(results_auc)
    print(np.mean(results, 0))
    print(np.mean(results_new, 0))
    log_file = 'your_save_path/result_folds.txt'
    with open(log_file, 'a') as f:
        f.write(args.model_name)
        f.write('\n')
        f.write('10 Fold Accuracy:\n')
        f.write('result: ' + str(results))
        f.write('\n')
        f.write('mean: ' + str(np.mean(results, 0)))
        f.write('\n')
        f.write('std: ' + str(np.std(results, 0)))

        f.write('\n')
        f.write('10 Fold AUC:\n')
        f.write('result_new: ' + str(results_new))
        f.write('\n')
        f.write('mean_new: ' + str(np.mean(results_new, 0)))
        f.write('\n')
        f.write('std_new: ' + str(np.std(results_new, 0)))
