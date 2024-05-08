cd ../
num_cpu=10

group_name='Gang_Test'
exp_name='Resnet50(pt)_lr0.002_niid_u10'
taskset -cpu-list $((gpu_id*num_cpu))-$((gpu_id*num_cpu+num_cpu-1)) \
  python3 main.py -n_round 50 -n_users 10 -n_epoch 2 -eval_round 1 \
  -model resnet50 -pre_trained -opt sgd -lr 0.002 -lr_decay 0.99 -lr_decay_round 5 \
  -dataset mnist -batch_size 32 -split non_iid \
  -wandb True -group_name $group_name -exp_name $exp_name \

wait
echo "Experiments completed!"