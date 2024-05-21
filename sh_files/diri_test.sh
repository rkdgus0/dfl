#!/bin/bash

# Miniconda 초기화
. /home/gang/miniconda3/etc/profile.d/conda.sh

main_path="/home/gang/dfl/main.py"

# 가상 환경 활성화
conda activate fl

gpu_id=0
group_name='LeNet_Test'
exp_name='resnet18_lr0.075_diri_a0.5_u10_b1024'
python3 $main_path -n_rounds 1000 -n_users 10 -n_epochs 1 -eval_round 10 -model mcmahanCNN -opt sgd -lr 0.05 -avg_method n_data \
  -dataset cifar10 -batch_size 1024 -split diri -alpha 0.5 -wandb -group_name $group_name -exp_name $exp_name -gpu_id $gpu_id > log/$exp_name.log 2>&1 

gpu_id=0
group_name='LeNet_Test'
exp_name='resnet18_fl_lr0.05_diri_a0.5_u10_b256'
python3 $main_path -basic_fl -n_rounds 1000 -n_users 10 -n_epochs 1 -eval_round 10 -model mcmahanCNN -opt sgd -lr 0.075 -avg_method n_data \
  -dataset cifar10 -batch_size 1024 -split diri -alpha 0.5 -wandb -group_name $group_name -exp_name $exp_name -gpu_id $gpu_id > log/$exp_name.log 2>&1 &

wait
echo "Experiments completed!"
# -lr_decay 0.99 -lr_decay_round 5
# -avg_method n_data
# -pre_trained