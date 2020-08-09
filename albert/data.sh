pkill -f transformer-evolution
sleep 1

export PYTHONPATH=..
python data.py --config config_min.json
#nohup python data.py --config config_half.json > data.log 2>&1 &
