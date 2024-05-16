cd ../

gpu_id1=0
group_name1='Gang_Test'
exp_name1='mcmahan2NN_lr0.075_iid_u50_epoch1'
  python3 main.py -n_rounds 50 -n_users 10 -n_epochs 1 -eval_round 1 \
  -model mcmahan2NN -opt sgd -lr 0.005 -lr_decay 0.99 -lr_decay_round 5 \
  -dataset mnist -batch_size 32 -split iid \
  -wandb -group_name $group_name1 -exp_name $exp_name1 \
  -gpu_id $gpu_id1 &

gpu_id2=1
group_name2='Gang_Test'
exp_name2='mcmahan2NN_lr0.01_iid_u100_epoch2'
  python3 main.py -n_rounds 50 -n_users 10 -n_epochs 1 -eval_round 1 \
  -model mcmahan2NN -opt sgd -lr 0.01 -lr_decay 0.99 -lr_decay_round 5 \
  -dataset mnist -batch_size 32 -split iid \
  -wandb -group_name $group_name2 -exp_name $exp_name2 \
  -gpu_id $gpu_id2 &

wait
echo "Experiments completed!"