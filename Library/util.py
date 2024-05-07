import argparse
import random

import keras
from component.CLIENT import CLIENT
from component.SCHEDULER import SCHEDULER
from sklearn.utils import shuffle
import numpy as np

DATA_LIST = ['mnist', 'cifar10']
MODEL_CHOICES = ['mcmahan2NN', 'mcmahanCNN', 'resnet50', 'resnet101', 'densenet121', 'VGG16']
OPTIMIZER_CHOICES = ['sgd','adam']
DATA_SPLIT_METHODS = ['random', 'iid', 'non_iid', 'diri']
CLIENT_MAPPING_METHODS = ['equal', 'diff']
DELAY_METHODS = ['Fixed', 'Range']
MODEL_AVG_METHODS = ['Equal', 'Acc', 'F1macro', 'F1micro', 'n_data', 'FedAT']
ADAPTIVE_AGG_METHOD = ['no_adapt', 'Epoch', 'Acc']
MODEL_DECAY_METHODS = ['Equal', 'Frac']

def arg_parsing():
    parser = argparse.ArgumentParser(description="Decentralized FL")

    # ----- Federated Learning ----- #
    parser.add_argument("-n_rounds", type=int, default=1000)
    parser.add_argument("-n_users", type=int, default=100)
    parser.add_argument("-n_epochs", type=int, default=1)
    parser.add_argument("-avg_method", type=str, default="Equal", help="How to average models", choices=MODEL_AVG_METHODS)

    # ----- Model Setting ----- #
    parser.add_argument("-model", type=str, default="densenet", choices=MODEL_CHOICES)
    parser.add_argument("-opt", type=str, default='sgd', choices=OPTIMIZER_CHOICES)
    parser.add_argument("-pre_trained", action='store_true')
    parser.add_argument("-lr", type=float, default=0.075)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr_decay", type=float, default=0.99)
    parser.add_argument("-lr_decay_round", type=int, default=5)

    # ----- Data Setting ----- #
    parser.add_argument("-dataset", type=str, default='cifar10', choices=DATA_LIST)
    parser.add_argument("-split", type=str, default='iid', choices=DATA_SPLIT_METHODS)
    parser.add_argument("-alpha", type=float, default=0.1)

    # ----- Debugging Setting ----- #
    parser.add_argument("-gpu_id", type=int)
    parser.add_argument("-eval_round", type=int, default=1)
    parser.add_argument("-debug", action='store_true')

    # ----- wandb.ai Setting ----- #
    parser.add_argument("-wandb", action='store_true')
    parser.add_argument("-wandb_id", type=str, default='create0327')
    parser.add_argument("-wandb_api", type=str, default='b2f21ce10a4365a21cfce06ad41f9a7f23d34639', help="check at https://wandb.ai/authorize")
    parser.add_argument("-exp_name", type=str)
    parser.add_argument("-group_name", type=str)

    args = parser.parse_args()
    print(args)
    return args

def compose_scheduler(args, model, clients, test_data):
    NUM_CLIENT = args.n_users
    avg_method = args.avg_method
    n_epochs = args.n_epochs
    
    return SCHEDULER(model, clients, NUM_CLIENT, test_data, avg_method, n_epochs)

def compose_client(args, model, datasets):
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    lr = args.lr
    opt = args.opt

    return CLIENT(opt, lr, model, datasets, n_epochs, batch_size)
