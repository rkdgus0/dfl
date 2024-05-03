cd ../

group_name='Gang_Test'
exp_name='resnet test'
  python main.py -n_round 5 -n_users 10 -n_epoch 3 -eval_round 1 \
  -model resnet50 -opt sgd -lr 0.002 -lr_decay 0.99 -lr_decay_round 5 \
  -dataset mnist -batch_size 32 -split iid \
  -wandb True -group_name $group_name -exp_name $exp_name 1> Logs/$exp_name.log 2>&1 &

wait
echo "Experiments completed!"