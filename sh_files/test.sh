 cd ../
num_cpu=10

gpu=0
group_name='Gang_Test'
exp_name='mcmahan test'
CUDA_VISIBLE_DEVICES=$((gpu)) taskset -cpu-list $((gpu*num_cpu))-$((gpu*num_cpu+num_cpu-1)) \
  nohup python main.py -n_round 5 -n_users 10 -n_epoch 3 -eval_round 1 \
  -model mcmahan2NN -opt sgd -lr 0.002 -lr_decay 0.99 -lr_decay_round 5 \
  -dataset mnist -batch_size 32 -split iid \
  -group_name $group_name -exp_name $exp_name 1> Logs/$exp_name.log 2>&1 &


CUDA_VISIBLE_DEVICES=0 nohup python main.py -n_round 5 -n_users 10 -n_epoch 3 -eval_round 1 -model resnet50 -opt sgd -lr 0.002 -lr_decay 0.99 -lr_decay_round 5 -dataset mnist -batch_size 32 -split iid

  
  