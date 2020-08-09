pkill -f transformer-evolution
sleep 1

export PYTHONPATH=..
python -W ignore train.py --epoch 100 --wandb True --config config_min.json
#nohup python train.py --epoch 100 --wandb True --config config_half.json > train.log 2>&1 &
