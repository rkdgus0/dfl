import argparse
import os
import copy
import random
import keras.src
import keras.src.utils

import tensorflow as tf
import numpy as np
import pandas as pd

from time import time
from datetime import datetime
from concurrent import futures

from Library.datasets import *
from Library.util import *
from Library.models import *

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
WANDB = args.wandb
if WANDB:
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
print(f"[MAIN] ===== Init Model(Opt: {args.opt}) & Dataset =====")
t1 = time()
model = define_model(args)
train_data, test_data = load_dataset(args.dataset)
split_data = split_data(args.split, train_data, args.n_users, args.alpha)
t2 = time()
print(f'[MAIN] ===== Time(sec): {round(t2-t1,2)}\n')

# ----- Client Setting ----- #
print("[MAIN] ===== Init Models =====")
t1 = time()
CLIENTS = compose_client(args, model, split_data)
SCHEDULER = compose_scheduler(args, model, CLIENTS, test_data)
t2 = time()
print(f'[MAIN] ===== Time(sec): {round(t2-t1,2)}\n')

# ----- Parameter Setting ----- #
USE_LR_DECAY = False
ROUND = args.n_rounds
NUM_CLIENT = args.n_users
eval_round = args.eval_round
dict_df = {'Round': [], 'Client': [], 'TestAcc': []}

if args.lr_decay:
    lr = args.lr
    USE_LR_DECAY=True

# ----- Global Round ----- #
print(f"[MAIN] ===== Global Round Start!")
print(f"[MAIN] ===== Total Round: {ROUND}")
print(f"[MAIN] ===== Total Client: {NUM_CLIENT}")
print(f"[MAIN] ===== Model: {args.model}(Pretrained : {args.pre_trained}, Opt: {args.opt})")
print(f"[MAIN] ===== Learning rate: {args.lr}, learning rate decay: {args.lr_decay}(round: {args.lr_decay_round})")
print(f"[MAIN] ===== Dataset: {args.dataset}, Data Split: {args.split}")
print(f"[MAIN] ===== Aggregation Method: {args.avg_method}")

for n_round in range(1, ROUND+1):
    SCHEDULER.train()
    test_df = {}

    # ----- Learning rate decay Setting ----- #
    if USE_LR_DECAY and args.lr_decay and not((n_round+1) % args.lr_decay_round):
        lr *= args.lr_decay
        SCHEDULER.set_lr(lr)
    
    # ----- Test Result upload ----- #
    if ((n_round-1) % eval_round == 0) or (n_round >= ROUND - 20):
        test_result = SCHEDULER.clients_test()
        
        for client_idx in range(NUM_CLIENT):
            dict_df['Round'].append(n_round)
            dict_df['Client'].append(client_idx)
            dict_df['TestAcc'].append(round(test_result[client_idx]['acc']*100, 2))
            test_df[f"{client_idx}Client_acc"]=round(test_result[client_idx]['acc']*100, 2)
            test_df[f"{client_idx}Client_loss"]=round(test_result[client_idx]['loss'], 2)
            print(f"[{client_idx} Client] Round: {n_round}, Loss: {test_result[client_idx]['loss']}, Acc: {test_result[client_idx]['acc']:.2%}")
            print(f"Test_df : {test_df}")
        if WANDB:
            wandb.log(test_df, step=n_round)
    
    '''
    if (n_round - 1) % 100 == 0:
        os.makedirs('./checkpoints', exist_ok=True)
        SCHEDULER.model.save(f'checkpoints/R:{n_round}_{args.exp_name}')
    '''

# ----- Result CSV load, wandb log-out ----- #
df = pd.DataFrame(dict_df)
os.makedirs('./csv_results', exist_ok=True)
f_name = f'{time()}_Mo{args.model}_Data{args.dataset}_Pre{args.pre_trained}_R{args.n_rounds}_N{args.n_users}_E{args.n_epochs}_Split{args.split}_Alp{args.alpha}_Delay_{args.delay_method}.csv'

df.to_csv(f'./csv_results/{f_name}')
if WANDB:
    wandb.save(f'./csv_results/{f_name}')
    #wandb.finish()