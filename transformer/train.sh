pkill -f transformer-evolution
sleep 1

# kowiki.model(kowiki.vocab), ratings_train.json, ratings_test.json
export PYTHONPATH=..
nohup python train.py --config config_half.json --epoch 100 --wandb True --batch 256 --gradient_accumulation 10 >train.log 2>&1 &
