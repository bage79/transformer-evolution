pkill -f transformer-evolution
sleep 1

export PYTHONPATH=..
python -W ignore pretrain.py --epoch 100 --wandb True --config config_min.json
#nohup python pretrain.py --epoch 100 --wandb True --config config_half.json > pretrain.log 2>&1 &
