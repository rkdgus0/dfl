cd ../

gpu_id=0
group_name='Gang_Test'
exp_name='Resnet50(pt)_lr0.002_iid_u50'
  python3 main.py -n_rounds 100 -n_users 50 -n_epochs 2 -eval_round 1 \
  -model resnet50 -pre_trained -opt sgd -lr 0.002 -lr_decay 0.99 -lr_decay_round 5 \
  -dataset cifar10 -batch_size 32 -split iid \
  -wandb -group_name $group_name -exp_name $exp_name \
  -gpu_id $gpu_id

gpu_id=3
group_name='Gang_Test'
exp_name='Resnet50(pt)_lr0.002_iid_u100'
  python3 main.py -n_rounds 100 -n_users 100 -n_epochs 2 -eval_round 1 \
  -model resnet50 -pre_trained -opt sgd -lr 0.002 -lr_decay 0.99 -lr_decay_round 5 \
  -dataset cifar10 -batch_size 32 -split iid \
  -wandb -group_name $group_name -exp_name $exp_name \
  -gpu_id $gpu_id

wait
echo "Experiments completed!"