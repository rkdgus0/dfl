import argparse
import os
import copy
import random
import keras.src.utils

import tensorflow as tf
import numpy as np
import pandas as pd

from time import time
from datetime import datetime
from concurrent import futures

from Library.helper import *
from Library.datasets import *
from Library.util import *
from Library.models import *
from Library.Component import SERVER, MEC, USER

import wandb

# ----- Seed fix ----- #
seed_num = 0
np.random.seed(seed_num)
random.seed(seed_num)
tf.random.set_seed(seed_num)
keras.src.utils.set_random_seed(seed_num)

args = arg_parsing()

# ----- Wandb Setting ----- #
# Wandb Debug Setting
if args.debug:
    mode = 'disabled'
    group_name = 'DEBUG'
    run_name = f'Debug_{time()}'
else:
    mode = 'online'
    run_name = args.exp_name
    group_name = args.group_name

# Wandb login option
wandb_id = args.wandb_id
wandb_api = args.wandb_api
if wandb_api != None:
    wandb.login(key=f"{wandb_api}")

# Wandb init project & parameter
wandb.init(project="Gang_Test", mode=mode, group=group_name, entity=f'{wandb_id}', name=run_name)
wandb.config.update(args)

# ----- Model/Dataset Setting ----- #
print("Init Model & Dataset")
t1 = time()
model = define_model(args)
loss_fn = keras.losses.CategoricalCrossentropy()
if args.opt == "sgd":
    opt = keras.optimizers.SGD(learning_rate=args.lr)
elif args.opt == "adam":
    opt = keras.optimizers.Adam(learning_rate=args.lr)
data_train, data_test = load_dataset(args.dataset)
split_data = split_data(args.split, data_train, args.n_users, args.alpha)
t2 = time()
print(f'Time(sec): {round(t2-t1,2)}')

# ----- Client Setting ----- #
print("Init Models")
t1 = time()
EDGES = compose_user(args, model, split_data)
MECS = compose_mec(args, model, EDGES)
SERVER = compose_server(args, model, MECS, data_test)
t2 = time()
print(f'Time(sec): {round(t2-t1,2)}')

if __name__ == '__main__':