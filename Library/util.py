import argparse
import random

import keras
from component.USER import USER
from component.MEC import MEC
from component.SERVER import SERVER
from sklearn.utils import shuffle
import numpy as np

DATA_LIST = ['mnist', 'cifar10']
MODEL_CHOICES = ['resnet50', 'densenet']
DATA_SPLIT_METHODS = ['random', 'iid', 'non_iid', 'diri']
CLIENT_MAPPING_METHODS = ['equal', 'diff']
DELAY_METHODS = ['Fixed', 'Range']
GLOBAL_MODEL_AVG_METHODS = ['Equal', 'Acc', 'F1macro', 'F1micro', 'n_data', 'FedAT']
ADAPTIVE_AGG_METHOD = ['no_adapt', 'Epoch', 'Acc']
MODEL_DECAY_METHODS = ['Equal', 'Frac']

def arg_parsing():
    parser = argparse.ArgumentParser(description="Standalone training")

    # ----- Federated Learning ----- #
    parser.add_argument("-n_rounds", type=int, default=1000)
    parser.add_argument("-n_users", type=int, default=100)
    parser.add_argument("-n_epochs", type=int, default=1)

    # ----- Model Setting ----- #
    parser.add_argument("-model", type=str, default="densenet", choices=MODEL_CHOICES)
    parser.add_argument("-pre_trained", action='store_true')
    parser.add_argument("-lr", type=float, default=0.075)
    parser.add_argument("-batch_size", type=int, default=32)
    parser.add_argument("-lr_decay", type=float, default=0.99)
    parser.add_argument("-lr_decay_round", type=int, default=5)

    # ----- Hierarchical Setting ----- #
    parser.add_argument("-num_mec", type=int, default=10)
    parser.add_argument("-gmodel_avg_method", type=str, default="Equal", help="How to average MEC models", choices=GLOBAL_MODEL_AVG_METHODS)
    parser.add_argument("-adaptive_agg_method", type=str, default="no_adapt", choices=ADAPTIVE_AGG_METHOD)
    parser.add_argument("-adaptive_parameter", type=float, default=100)

    # ----- Data Setting ----- #
    parser.add_argument("-dataset", type=str, default='cifar10', choices=DATA_LIST)
    parser.add_argument("-split", type=str, default='iid', choices=DATA_SPLIT_METHODS)
    parser.add_argument("-alpha", type=float, default=0.1)
    parser.add_argument("-client_mapping", type=str, default='equal', choices=CLIENT_MAPPING_METHODS)

    # ----- Async Setting ----- #
    """
    def mec_delay_def(value):
        return list(map(int, value.split()))
    mec_default = '0 0 0 0 1 1 2 2 3 3'

    parser.add_argument("-mec_delays", help='MEC DELAYS', type=mec_delay_def, default=mec_default.split())
    # parser.add_argument("--mec_delays", type=str, help='MEC DELAYS', default=mec_default) #.split())
    parser.add_argument("--delay_method", type=str, default="Range", help="How to map delay to MEC",choices=DELAY_METHODS)
    parser.add_argument("--delay_epoch", type=int, default=0, help="the number of epoch in local client makes a delay")
    parser.add_argument("--delay_range", type=int, default=2, help="the number of range for MEC delay")
    parser.add_argument("--model_decay", type=str, default="Equal", help="How to decay according to delay",choices=MODEL_DECAY_METHODS)
    """

    # ----- Debugging Setting ----- #
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--debug", action='store_true')

    # ----- wandb.ai Setting ----- #
    parser.add_argument("--exp_name", type=str, default='model_decay')
    parser.add_argument("--group_name", type=str, required=True)

    args = parser.parse_args()
    # args.mec_delays = mec_delay_def(args.mec_delays)
    print(args)
    return args

def compose_server(args, model, mecs, data_test):
    delay_method = args.delay_method
    delay_range = args.delay_range
    MEC_delay = list(map(int, args.mec_delays))
    NUM_MEC = args.num_mec
    gmodel_avg_method = args.gmodel_avg_method
    delay_epoch = args.delay_epoch
    n_epochs = args.n_epochs
    model_decay = args.model_decay
    num_mec_datas = []
    for mec_mapping in mecs.mec_client_mapping.values():
        num_mec_data = 0
        for client_idx in mec_mapping:
            num_mec_data += len(mecs.clients.datasets[client_idx]['x'])
        num_mec_datas.append(num_mec_data)

    return SERVER(model, mecs, NUM_MEC, MEC_delay, data_test, delay_method, delay_range, gmodel_avg_method, delay_epoch, n_epochs,model_decay, num_mec_datas)

def compose_mec(args, model, edges):
    NUM_MEC = args.num_mec
    NUM_CLIENT = args.n_users

    mec_client_mapping = dict()
    clients = list(range(NUM_CLIENT))
    shuffle(clients)

    # todo: args에 따라 data distribute 방법 추가해야 함.
    if args.client_mapping == 'equal':
        num_client_per_mec = NUM_CLIENT // NUM_MEC
        for mec_idx in range(NUM_MEC):
            mec_client_mapping[mec_idx] = np.random.choice(range(NUM_CLIENT), num_client_per_mec, replace=False)
    elif args.client_mapping == 'diff':
        mec_client_mapping = [[] for _ in range(NUM_MEC)]
        for i in range(NUM_CLIENT):
            mec_idx = random.randint(0, NUM_MEC - 1)
            mec_client_mapping[mec_idx].append(i)
        for i, num_client in enumerate(mec_client_mapping):
            print(f"MEC {i}: {num_client}")

    return MEC(model, mec_client_mapping, edges)


def compose_user(args, model, splited_datasets):
    # todo: (sub) 이 부분 args에 따라 변경 필요.

    client = USER(args, model, datasets=splited_datasets, epochs=args.n_epochs, batch_size=args.batch_size,device='CUDA:0')

    return client