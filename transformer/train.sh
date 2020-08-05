pkill -f transformer-evolution
sleep 1

# kowiki.model(kowiki.vocab), ratings_train.json, ratings_test.json
export PYTHONPATH=..
nohup python train.py --epoch 100 --wandb False --config config_min.json >train.log 2>&1 &

horovodrun --mpi -np 2 -H localhost:2 --log-hide-timestamp PYTHONPATH=.. python train.py --epoch 1 --data_dir ../data_sample
