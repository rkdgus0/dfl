cd ../

gpu_id=0
group_name='Gang_Test'
exp_name='mcmahan2NN_lr0.075_iid_u10_epoch2'
  python3 main.py -n_rounds 50 -n_users 10 -n_epochs 2 -eval_round 1 \
  -model mcmahan2NN -opt sgd -lr 0.075 -lr_decay 0.99 -lr_decay_round 5 \
  -dataset cifar10 -batch_size 32 -split iid \
  -wandb -group_name $group_name -exp_name $exp_name \
  -gpu_id $gpu_id &

gpu_id=1
group_name='Gang_Test'
exp_name='mcmahan2NN_lr0.075_iid_u50_epoch2'
  python3 main.py -n_rounds 50 -n_users 50 -n_epochs 2 -eval_round 1 \
  -model mcmahan2NN -opt sgd -lr 0.075 -lr_decay 0.99 -lr_decay_round 5 \
  -dataset cifar10 -batch_size 32 -split iid \
  -wandb -group_name $group_name -exp_name $exp_name \
  -gpu_id $gpu_id &

wait
echo "Experiments completed!"