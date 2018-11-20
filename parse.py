# encoding: utf-8
import argparse

def define_parser():

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--pollution-type', type=str, default='choas', metavar='N', #'fixed_rule'
                        help='label noise source type')
    parser.add_argument('--pollution-rate', type=float, default=0.8, metavar='N',
                        help='noise fraction')
    parser.add_argument('--beta', type=float, default=0.1, metavar='N',
                        help='manmade pure sample rate')
    parser.add_argument('--retrain-times', type=int, default=30, metavar='N', # 60
                        help='times of re-training model')
    parser.add_argument('--mix-samp', type=str, default='mixed', metavar='N', # 'none'
                        help='shuffle the whole training samples before re-training model')
    parser.add_argument('--disp-detail', action='store_true', default='False',   #True
                        help='display the details per re-training model')
    parser.add_argument('--epoch-num-sel', type=int, default=4, metavar='N',
                        help='epoch number in every re-training model'),
    return parser