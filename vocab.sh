# kowiki.csv -> kowiki.txt, kowiki.vocab, kowiki.model
python vocab.py

# kowiki.model(kowiki.vocab) + ratings_train.txt -> ratings_train.json
# kowiki.model(kowiki.vocab) + ratings_test.txt -> ratings_test.json
python common_data.py
