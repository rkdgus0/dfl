import random

import keras.src.utils

from component.USER import USER
import argparse

import tensorflow as tf
import pandas as pd
import os
from time import time

from Library.datasets import *
from Library.util import *
from Library.models import *
import wandb

seed_num = 0
np.random.seed(seed_num)
random.seed(seed_num)
tf.random.set_seed(seed_num)
keras.src.utils.set_random_seed(seed_num)

args = arg_parsing()
if args.debug:
    mode = 'disabled'
    group_name = 'SIMULATION'
    run_name = 'SIMULATION'
else:
    mode = 'online'
    run_name = args.exp_name
    group_name = args.group_name

wandb.init(project="Hier and Async FL", mode=mode, group=group_name, entity="hierfl-miil", name=run_name)
wandb.config.update(args)

model = define_model(args)
print("Init dataset")
t1 = time()
data_train, data_test = load_dataset(args.dataset)
split_data = split_data(args.split, data_train, args.n_users, args.alpha)
t2 = time()
print(f'Time(sec): {round(t2-t1,2)}')

print("Init Opt")
# loss_fn = keras.losses.CategoricalCrossentropy()
# opt = keras.optimizers.SGD(learning_rate=args.lr)#, weight_decay=0.0004)

print("Init Models")
t1 = time()
EDGES = compose_user(args, model, split_data)
MECS = compose_mec(args, model, EDGES)
SERVER = compose_server(args, model, MECS, data_test)
t2 = time()
print(f'Time(sec): {round(t2-t1,2)}')

ROUND = args.n_rounds
eval_every = args.eval_every
dict_df = {'Round': [], 'TestAcc': []}

USE_LR_DECAY = False

if args.lr_decay:
    lr = args.lr
    USE_LR_DECAY=True

if args.adaptive_agg_method != 'no_adapt':
    print("Adaptive Aggregation, starting with FedAvg, After,", args.adaptive_agg_method, args.adaptive_parameter, ", ", args.gmodel_avg_method)

for n_round in range(1, ROUND+1):
    SERVER.train()

    if USE_LR_DECAY and args.lr_decay and not((n_round+1) % args.lr_decay_round):
        lr *= args.lr_decay
        SERVER.set_lr(lr)


    if ((n_round-1) % eval_every == 0) or (n_round >= ROUND - 20):
        test_result = SERVER.test()
        print(f'Round: {n_round}, Loss: {test_result[0]}, Acc: {test_result[1]:.2%}')

        dict_df['Round'].append(n_round)
        dict_df['TestAcc'].append(round(test_result[1]*100, 2))

        wandb.log({
            "Test Acc": round(test_result[1]*100, 2),
            "Test Loss": round(test_result[0], 2),
        }, step=n_round)

    if args.adaptive_agg_method == "Epoch":
        if int(args.adaptive_parameter) > n_round:
            SERVER.gmodel_avg_method = 'Equal'
        else:
            SERVER.gmodel_avg_method = args.gmodel_avg_method
    elif args.adaptive_agg_method == "Acc":
        if float(args.adaptive_parameter)/100.0 > test_result[1]:
            SERVER.gmodel_avg_method = 'Equal'
        else:
            SERVER.gmodel_avg_method = args.gmodel_avg_method
    if (n_round - 1) % 100 == 0:
        os.makedirs('./checkpoints', exist_ok=True)
        SERVER.model.save(f'checkpoints/R:{n_round}_{args.exp_name}')

df = pd.DataFrame(dict_df)
os.makedirs('./csv_results', exist_ok=True)
f_name = f'{time()}_Mo{args.model}_Data{args.dataset}_Pre{args.pre_trained}_R{args.n_rounds}_N{args.n_users}_E{args.n_epochs}_Split{args.split}_Alp{args.alpha}_Delay_{args.delay_method}.csv'

df.to_csv(f'./csv_results/{f_name}')

# save_info = {'model_state_dict': handler.model.state_dict(),
#              'round': n_round}
wandb.save(f'./csv_results/{f_name}')

# source, _ = tff.simulation.datasets.emnist.load_data()
# def client_data(n):
#   return source.create_tf_dataset_for_client(source.client_ids[n]).map(
#       lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
#   ).repeat(10).batch(20)
#
# # Pick a subset of client devices to participate in training.
# train_data = [client_data(n) for n in range(3)]
#
# # Wrap a Keras model for use with TFF.
# def model_fn():
#   model = tf.keras.models.Sequential([
#       tf.keras.layers.Dense(10, tf.nn.softmax, input_shape=(784,),
#                             kernel_initializer='zeros')
#   ])
#   return tff.learning.from_keras_model(
#       model,
#       input_spec=train_data[0].element_spec,
#       loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#       metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
#
# # Simulate a few rounds of training with the selected client devices.
# edge = user()
# trainer = tff.learning.build_federated_averaging_process(
#   model_fn,
#   client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
# state = trainer.initialize()
# for _ in range(5):
#   state, metrics = trainer.next(state, train_data)
#   print(metrics['train']['loss'])