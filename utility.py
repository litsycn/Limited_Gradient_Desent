# encoding: utf-8
import copy
import torch
import numpy as np
import torch.utils.data as data
import time

def polluting_dataset(trainset, args):
    if args.pollution_type == 'choas':
        trainset_noisy = chaos_label_polluting(trainset, args.pollution_rate)
    elif args.pollution_type == 'fixed_rule':
        trainset_noisy = fixed_rule_label_polluting_dataset(trainset, args.pollution_rate)
    return trainset_noisy


def chaos_label_polluting(trainset, pollution_rate):
    dataset = copy.deepcopy(trainset)
    number = len(trainset)
    # shuffle the index
    shuffle = torch.randperm(number).long()
    po_ind = torch.LongTensor([number * pollution_rate])
    sel_ind = shuffle[0:po_ind]
    dataset.train_labels[sel_ind] = dataset.train_labels[sel_ind].random_(0,10)
    ind = dataset.train_labels[sel_ind].eq(trainset.train_labels[sel_ind])

    while ind.sum()>0:
        # print(ind.sum())
        dataset.train_labels[sel_ind[ind]] = dataset.train_labels[sel_ind[ind]].random_(0, 10)
        ind = dataset.train_labels[sel_ind].eq(trainset.train_labels[sel_ind])
        b=0
    # mm = dataset.train_labels.eq(trainset.train_labels).sum()
    return dataset


def fixed_rule_label_polluting_dataset(trainset, pollution_rate):
    class_seq = gen_random_class()
    class_seq = torch.from_numpy(class_seq).long()
    dataset = copy.deepcopy(trainset)
    number = len(trainset)
    # shuffle the index
    shuffle = torch.randperm(number).long()
    po_ind = torch.LongTensor([number * pollution_rate])
    sel_ind = shuffle[0:po_ind]
    tmp = dataset.train_labels[sel_ind].clone()
    for i in range(10):
        ind = dataset.train_labels[sel_ind].eq(i)
        tmp[ind] = class_seq[i]
    dataset.train_labels[sel_ind] = tmp
    # mm = dataset.train_labels.eq(trainset.train_labels).sum()
    return dataset


def prepare_reverse_samples_for_dataset(trainset_noisy, testset, beta, mix_samp):
    dataset = copy.deepcopy(trainset_noisy)
    number = len(trainset_noisy)
    pure_error_set = copy.deepcopy(testset)
    leftover_set = copy.deepcopy(testset)

    # shuffle the polluted data set
    if mix_samp == 'mixed':
        # shuffle the index
        shuffle = torch.randperm(number).long()
        po_ind = torch.LongTensor([number * beta])
        sel_ind = shuffle[0:po_ind]
        tmp = dataset.train_labels[sel_ind].add(1).fmod(10)
        ind = tmp.eq(10)
        tmp[ind] = 0
        dataset.train_labels[sel_ind] = tmp

        pure_error_set.test_data = dataset.train_data[sel_ind]
        pure_error_set.test_labels = dataset.train_labels[sel_ind]
        leftover_set.test_data = dataset.train_data[shuffle[po_ind:]]
        leftover_set.test_labels = dataset.train_labels[shuffle[po_ind:]]
    else:
        range = torch.range(0, number-1).long()  # 需要验证长度
        po_ind = torch.LongTensor([number * beta])
        sel_ind = range[0:po_ind]
        tmp = dataset.train_labels[sel_ind].add(1).fmod(10)
        ind = tmp.eq(10)
        tmp[ind] = 0
        dataset.train_labels[sel_ind] = tmp

        pure_error_set.test_data = dataset.train_data[sel_ind]
        pure_error_set.test_labels = dataset.train_labels[sel_ind]
        leftover_set.test_data = dataset.train_data[range[po_ind:]]
        leftover_set.test_labels = dataset.train_labels[range[po_ind:]]

    # mm = dataset.train_labels.eq(trainset_noisy.train_labels).sum()
    return dataset, pure_error_set, leftover_set


def trainset_to_testset(trainset, testset):
    dataset = copy.deepcopy(testset)
    dataset.test_data = trainset.train_data
    dataset.test_labels = trainset.train_labels
    return dataset


def gen_random_class():
    re_gen = True
    while re_gen:
        a = np.arange(10)
        x = np.zeros(10)
        for i in range(10):

            if i == 7:
                if a[0] == 6 and a[1] == 7 and a[2] == 8:
                    re_gen = True
                    break
                else:
                    re_gen = False


            if i == 8:
                if (a[0] == 7 and a[1] == 8) or (a[0] == 8 and a[1] == 9) or (a[0] == 7 and a[1] == 9):
                    re_gen = True
                    break
                else:
                    re_gen = False

            if i == 9:
                if a[0] == 9 or a[0] == 8 or a[0] == 0:
                    re_gen = True
                    break
                else:
                    re_gen = False


            if i - 1 == -1:
                left = 9
            else:
                left = i - 1

            if i + 1 == 10:
                right = 0
            else:
                right = i + 1

            while True:
                id = np.random.randint(0, 10-i)

                if a[id]!=left and a[id]!=i and a[id]!=right:
                    x[i] = a[id]
                    a = np.delete(a, id)
                    break
    return x

