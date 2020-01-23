pkill -f transformer-evolution
sleep 1

# kowiki.model(kowiki.vocab), ratings_train.json, ratings_test.json
export PYTHONPATH=..
python train.py --config config_half.json --epoch 2 --wandb False --batch 256 --gradient_accumulation 2
