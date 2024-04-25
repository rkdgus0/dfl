import argparse
import random

import keras
from component.CLIENT import CLIENT
from component.MEC import MEC
from component.SERVER import SERVER
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
    parser.add_argument("-client_mapping", type=str, default='equal', choices=CLIENT_MAPPING_METHODS)

    # ----- Debugging Setting ----- #
    parser.add_argument("-eval_every", type=int, default=1)
    parser.add_argument("-debug", action='store_true')

    # ----- wandb.ai Setting ----- #
    parser.add_argument("-wandb_id", type=str, default='create0327')
    parser.add_argument("-wandb_api", type=str, default='b2f21ce10a4365a21cfce06ad41f9a7f23d34639', help="check at https://wandb.ai/authorize")
    parser.add_argument("-exp_name", type=str, required=True)
    parser.add_argument("-group_name", type=str, required=True)

    args = parser.parse_args()
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


def compose_client(args, model, splited_datasets):

    return CLIENT(args, model, datasets=splited_datasets, epochs=args.n_epochs, batch_size=args.batch_size)
