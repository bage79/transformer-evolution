pkill -f transformer-evolution
sleep 1

# kowiki.model(kowiki.vocab), ratings_train.json, ratings_test.json
export PYTHONPATH=..
nohup python train.py --epoch 100 --wandb True --config config_min.json >train.log 2>&1 &
