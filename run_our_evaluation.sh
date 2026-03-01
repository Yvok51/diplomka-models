
venv/bin/python3 ./src/our_evaluation.py --input dataset.json --model-path finished_multiclass/ --encoder trainer_output/label_encoder.pkl --model-type canine --type multiclass --model-kind transformer --output our_evaluation/canine_multiclass.txt
venv/bin/python3 ./src/our_evaluation.py --input dataset.json --model-type canine --model-kind transformer --output our_evaluation/canine_multilabel.txt
venv/bin/python3 ./src/our_evaluation.py --input dataset.json --model-path finished_negative/ --encoder trainer_output/multilabel_encoder.pkl --model-type canine --model-kind transformer --output our_evaluation/canine_negative.txt
venv/bin/python3 ./src/our_evaluation.py --input dataset.json --type fasttext --output our_evaluation/fasttext.txt
venv/bin/python3 ./src/our_evaluation.py --input dataset.json --type gcld3 --output our_evaluation/gcld3.txt
venv/bin/python3 ./src/our_evaluation.py --input dataset.json --type glotlid --output our_evaluation/glotlid.txt
venv/bin/python3 ./src/our_evaluation.py --input dataset.json --type openlid --output our_evaluation/openlid.txt
venv/bin/python3 ./src/our_evaluation.py --input dataset.json --model-path models/nli_model_20260206_103101.pkl --encoder trainer_output/label_encoder.pkl --model-type canine --type multiclass --output our_evaluation/tf_idf_multiclass.txt --model-kind tfidf
venv/bin/python3 ./src/our_evaluation.py --input dataset.json --model-path models/nli_multilabel_model_20260214_044828.pkl --encoder trainer_output/multilabel_encoder.pkl --type multilabel --output our_evaluation/tf_idf_multilabel.txt --model-kind tfidf