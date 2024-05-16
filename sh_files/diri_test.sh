#!/bin/bash
cd ../

gpu_id=0
group_name='Gang_Test'
exp_name='resnet18_lr0.01_diri_a0.5_u10_b1024'
  python3 main.py -n_rounds 100 -n_users 10 -n_epochs 1 -eval_round 1 \
  -model resnet18 -opt sgd -lr 0.01 -lr_decay 0.99 -lr_decay_round 5 \
  -dataset cifar10 -batch_size 1024 -split diri -alpha 0.5 \
  -wandb -group_name $group_name -exp_name $exp_name \
  -gpu_id $gpu_id &

gpu_id=1
group_name='Gang_Test'
exp_name='resnet18_lr0.01_diri_a0.5_u10_b512'
  python3 main.py -n_rounds 100 -n_users 10 -n_epochs 1 -eval_round 1 \
  -model resnet18 -opt sgd -lr 0.075 -lr_decay 0.99 -lr_decay_round 5 \
  -dataset cifar10 -batch_size 512 -split diri -alpha 0.5 \
  -wandb -group_name $group_name -exp_name $exp_name \
  -gpu_id $gpu_id &


wait
echo "Experiments completed!"