#!/bin/bash

source activate fl

cd ../

gpu_id=0
group_name='Gang_Test'
exp_name='resnet50_lr0.01_niid_u10_epoch1'
  python3 main.py -n_rounds 100 -n_users 10 -n_epochs 1 -eval_round 1 \
  -model resnet50 -opt sgd -lr 0.01 -lr_decay 0.99 -lr_decay_round 5 \
  -dataset cifar10 -batch_size 32 -split non_iid \
  -wandb -group_name $group_name -exp_name $exp_name \
  -gpu_id $gpu_id &

gpu_id=1
group_name='Gang_Test'
exp_name='resnet50_lr0.004_niid_u10_epoch1'
  python3 main.py -n_rounds 100 -n_users 10 -n_epochs 1 -eval_round 1 \
  -model resnet50 -opt sgd -lr 0.004 -lr_decay 1.1 -lr_decay_round 5 \
  -dataset cifar10 -batch_size 32 -split non_iid \
  -wandb -group_name $group_name -exp_name $exp_name \
  -gpu_id $gpu_id &


wait
echo "Experiments completed!"