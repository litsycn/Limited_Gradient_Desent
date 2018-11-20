# -*- coding: UTF-8 -*-

from __future__ import print_function
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import parse
import utility as utility
# from models import *


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5, 1, padding=2,bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 5, 1, padding=2,bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2))


        self.dense = torch.nn.Sequential(
            torch.nn.Linear(32 * 7 * 7, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(100, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(100, 100),
            torch.nn.BatchNorm1d(100),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(100, 10)
        )

    def forward(self, x):
        conv_out = self.conv(x)
        res = conv_out.view(conv_out.size(0), -1)
        out = self.dense(res)
        return out

class Net2(torch.nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, 1, padding=2, bias=False)
        self.bn2d1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, padding=2, bias=False)
        self.bn2d2 = nn.BatchNorm2d(32)
        self.dense1 = nn.Linear(32 * 7 * 7, 100)
        self.bn1d1 = nn.BatchNorm1d(100)
        self.dense2 = nn.Linear(100, 100)
        self.bn1d2 = nn.BatchNorm1d(100)
        self.dense3 = nn.Linear(100, 100)
        self.bn1d3 = nn.BatchNorm1d(100)
        self.dense4 = nn.Linear(100, 10)


    def forward(self, x):
        conv1 = F.max_pool2d(F.relu(self.bn2d1(self.conv1(x))), 2)
        conv2 = F.max_pool2d(F.relu(self.bn2d2(self.conv2(conv1))), 2)
        res = conv2.view(conv2.size(0), -1)
        des1 = F.relu(self.bn1d1(self.dense1(res)))
        des2 = F.relu(self.bn1d2(self.dense2(des1)))
        des3 = F.relu(self.bn1d3(self.dense3(des2)))
        out = self.dense4(des3)
        return out



def train(args, model, device, trainset, optimizer, kwargs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=3)

    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()

    avg_accur = 1. * correct / len(train_loader.dataset)
    return loss.data.cpu(), avg_accur


def test(args, model, device, testset, kwargs):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                             shuffle=False, drop_last=False, num_workers=3)

    test_loss = 0
    correct = 0
    label_hat = torch.LongTensor().to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            label_hat = torch.cat([label_hat, pred], 0)
            correct += pred.eq(target.view_as(pred)).sum().item()


    test_loss /= len(test_loader.dataset)
    correct = float(correct)
    accur_rate = correct / len(test_loader.dataset)
    return accur_rate, label_hat.cpu().view(-1), test_loss

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        # m.weight.data.normal_(0, math.sqrt(2. / n))
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns
        m.weight.data.normal_(0.0, np.sqrt(2.0 / (fan_in + fan_out)))
    elif isinstance(m, nn.Linear):
        size = m.weight.size()
        fan_out = size[0] # number of rows
        fan_in = size[1] # number of columns
        m.weight.data.normal_(0.0, np.sqrt(2.0 / (fan_in + fan_out)))
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()



def main():
    # Training settings
    parser = parse.define_parser()
    args = parser.parse_args()
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 3, 'pin_memory': True} if use_cuda else {}

    print('==> Loading data set..')
    trainset = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    testset = datasets.MNIST('../data', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                    ]))

    # polluting trainset
    trainset_noisy = utility.polluting_dataset(trainset, args)

    obs_pred_accur = np.zeros(args.retrain_times)
    LoR_sel = np.zeros(args.retrain_times)
    test_acurr_selected = np.zeros(args.retrain_times)

    old_LoR_sel = 0

    plt.ion()

    for i in range(args.retrain_times):
        model = Net().to(device)

        # man-made reverse samples via label-shifting operation
        dataset_with_pure_error, reverse_set, leftover_set = \
            utility.prepare_reverse_samples_for_dataset(trainset_noisy, testset, args.beta, args.mix_samp)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        loss_rec = np.zeros(args.epoch_num_sel)
        avg_train_accur = np.zeros(args.epoch_num_sel)
        reverse_acurr = np.zeros(args.epoch_num_sel)
        leftover_acurr = np.zeros(args.epoch_num_sel)
        test_acurr = np.zeros(args.epoch_num_sel)
        LoR = np.zeros(args.epoch_num_sel)
        old_LoR = 0

        loss_R = np.zeros(args.epoch_num_sel)
        loss_L = np.zeros(args.epoch_num_sel)
        loss_LoR = np.zeros(args.epoch_num_sel)
        loss_test = np.zeros(args.epoch_num_sel)

        # LIMITED GRADIENT DESCENT
        for epoch in range(args.epoch_num_sel):
            tic = time.time()
            loss_rec[epoch], avg_train_accur[epoch] = train(args, model, device, dataset_with_pure_error, optimizer, kwargs)
            reverse_acurr[epoch], _, loss_R[epoch] = test(args, model, device, reverse_set, kwargs)
            leftover_acurr[epoch], _, loss_L[epoch] = test(args, model, device, leftover_set, kwargs)
            test_acurr[epoch], _, loss_test[epoch] = test(args, model, device, testset, kwargs)
            LoR[epoch] = leftover_acurr[epoch] / (reverse_acurr[epoch] + 0.00000001)

            torch.save({'avg_train_accur': avg_train_accur[0:epoch+1],
                        'reverse_acurr': reverse_acurr[0:epoch+1],
                        'leftover_acurr': leftover_acurr[0:epoch+1],
                        'test_acurr': test_acurr[0:epoch+1]}, 'save_train_curve_once_train.pt')

            # output the model according to maximum LoR
            if LoR[epoch] > old_LoR:
                old_LoR = LoR[epoch]
                model_output = copy.deepcopy(model)
                best_epoch_number = epoch
                test_acurr_selected[i] = test_acurr[epoch]
                LoR_sel[i] = LoR[epoch]

            toc = time.time()

            print('Epoch: {}     Loss:{:.4f}     train corr rate:{:.4f}     test corr rate: {:.4f}     pure error corr rate: {:.4f}     leftover corr rate: {:.4f}     LoR: {:.4f}     best_epoch_number: {}     time: {:.5f}'.format(
                   epoch,      loss_rec[epoch],  avg_train_accur[epoch],    test_acurr[epoch],        reverse_acurr[epoch],           leftover_acurr[epoch],       LoR[epoch],   best_epoch_number,         toc - tic))

            if args.disp_detail == 'True':
                plt.figure(1)
                plt.plot(np.arange(epoch+1), avg_train_accur[0:epoch+1], 'blue',
                         np.arange(epoch+1), test_acurr[0:epoch+1], 'red',
                         np.arange(epoch + 1), reverse_acurr[0:epoch + 1], 'green',
                         np.arange(epoch + 1), leftover_acurr[0:epoch + 1], 'pink'
                         )
                plt.title('Accuracies')

                plt.figure(2)
                plt.plot(np.arange(epoch+1), LoR[0:epoch+1],'blue')
                plt.title('LoR')
                plt.pause(0.1)

        if args.disp_detail == 'True':
            plt.close(1)
            plt.close(2)

        # cover trainset type to testset type for using test data DataLoader
        trainset_noisy_for_test = utility.trainset_to_testset(trainset_noisy, testset)
        accur_for_trainset_noisy, pred_labels,_ = test(args, model_output, device, trainset_noisy_for_test, kwargs)

        obs_pred_accur[i] = pred_labels.eq(trainset.train_labels).sum().float().div(len(trainset))
        trainset_noisy.train_labels = pred_labels

        print('Re-train times: {}  LoR sel:{:.4f}  trainset prdc accur(only observe):{:.4f}  test acurr rate selected:{:.4f}'.format(
            i, LoR_sel[i], obs_pred_accur[i], test_acurr_selected[i]))
        plt.figure(3)
        plt.plot( np.arange(i + 1), LoR_sel[0:i + 1], 'red')
        plt.title('LoR selected')
        plt.grid()
        plt.figure(4)
        plt.plot(np.arange(i + 1), obs_pred_accur[0:i + 1], 'blue',
                 np.arange(i + 1), test_acurr_selected[0:i + 1], 'red')
        plt.title('prediction accuracy of trainset and testset')
        plt.grid()
        plt.pause(0.01)

        if LoR_sel[i] > old_LoR_sel:
            old_LoR_sel = LoR_sel[i]
            torch.save(model_output, 'mnist_model.pt')
            test_accur_rec = test_acurr_selected[i]
            i_rec = i

        print('output test accur: %.4f  corresponding times %d' % (test_accur_rec, i_rec))

if __name__ == '__main__':
    main()