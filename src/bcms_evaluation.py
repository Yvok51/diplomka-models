import urllib.request
import csv
import argparse
import sys
import logging
import os
import zipfile

import torch
from transformers import CanineForSequenceClassification
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import tqdm
import numpy as np
from sklearn.metrics import f1_score

from common import PROJECT_PATH, load_object, MODELS, ModelTypeT
from flores_evaluation import get_multiclass_model
from prediction import predict_multiclass, predict_multilabel
from multilabel import get_multilabel_model

BCMS_TO_OPENLID = {"sr": "srp_Cyrl", "hr": "hrv_Latn", "me": "mkd_Cyrl", "bs": "bos_Latn"}

BCMS_DATASET_URL = "https://zenodo.org/records/10998042/files/VarDial2024_DSL-ML_BCMS.zip?download=1"
FOLDER_NAME = "VarDial2024_DSL-ML_BCMS"
DATA_FOLDER = PROJECT_PATH / "data"

ENCODER_PATH = PROJECT_PATH / "trainer_output" / "label_encoder.pkl"
MODEL_PATH = PROJECT_PATH / "finished_multiclass"

def download_file(url: str, filename: str):
    """Download a source file

    :param url: URL to download from
    :param filename: The filename to save the file to
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if not os.path.exists(filename):
        logging.info("Downloading file %s...", url)
        urllib.request.urlretrieve(url, filename=f"{filename}.tmp")
        os.rename(f"{filename}.tmp", filename)


def unzip_file(filename: str, extract_to: str):
    with zipfile.ZipFile(filename, 'r') as zipped:
        zipped.extractall(extract_to)

def read_bcms(filename: str):
    bcms = []
    with open(filename, 'r', encoding="utf-8") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            bcms.append({ "languages": [BCMS_TO_OPENLID[lang] for lang in row[0].split(",")], "text": row[1]})
    return bcms


def compute_loose_accuracy(predicted, gold):
    correct = 0
    for prediction, y in zip(predicted, gold):
        # The paper says if there is an overlap, but the implementation at https://github.com/ltgoslo/slide/blob/main/src/evaluate.py seems to use issubset
        correct += len(set(prediction).intersection(set(y))) != 0

    return correct / len(gold)

def compute_exact_match_accuracy(predicted, gold):
    return sum([p == y for p, y in zip(predicted, gold)]) / len(gold)

def encode_labels(labels):
    return np.asarray(["srp_Cyrl" in labels, "hrv_Latn" in labels, "mkd_Cyrl" in labels, "bos_Latn" in labels])

def compute_f1_score(predicted, gold):
    predicted = np.asarray([encode_labels(labels) for labels in predicted])
    gold = np.asarray([encode_labels(labels) for labels in gold])

    return f1_score(predicted, gold, average=None, zero_division=0)

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation of language prediction using finetuned CANINE model")
    parser.add_argument("--model-type", type=ModelTypeT, choices=list(MODELS.keys()),
                        default="canine", help="The underlying model type to train")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH, help="Directory of the finetuned model")
    parser.add_argument("--type", choices=["multiclass", "multilabel"],
                        help="The model which we are using", default="multiclass")
    parser.add_argument("--encoder-path", type=str,
                        default=str(ENCODER_PATH), help="Path to the label encoder")
    parser.add_argument("--seed", type=int,
                        default=42, help="Path to the label encoder")
    parser.add_argument("--output", type=argparse.FileType('w'),
                        default=sys.stdout, help="The file to write the metrics to")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Using device %s", device)

    csv.field_size_limit(sys.maxsize)

    encoder: LabelEncoder | MultiLabelBinarizer = load_object(
        args.encoder_path)
    assert (isinstance(encoder, LabelEncoder) and args.type != "multilabel") or (
        isinstance(encoder, MultiLabelBinarizer) and args.type == "multilabel")

    download_file(BCMS_DATASET_URL, f"{DATA_FOLDER}/{FOLDER_NAME}.zip")
    unzip_file(f"{DATA_FOLDER}/{FOLDER_NAME}.zip", DATA_FOLDER)

    test_data = read_bcms(f"{DATA_FOLDER}/{FOLDER_NAME}/test.tsv")

    if args.type == "multiclass":
        model, tokenizer = get_multiclass_model(args.model_path, device)
        assert isinstance(model, CanineForSequenceClassification)

        def predict_func(sentence):
            prediction = predict_multiclass(
                sentence, model, tokenizer, encoder, device)
            return [prediction[0][0]]

    elif args.type == "multilabel":
        model, tokenizer = get_multilabel_model(args.model_path, device, args.model_type)

        def predict_func(sentence):
            prediction = predict_multilabel(
                sentence, model, tokenizer, encoder, device)
            predicted_langs = list(
                zip(*prediction))[0] if len(prediction) > 0 else []
            return list(predicted_langs)

    logging.info("Starting predictions...")
    predictions = []
    gold = []
    for item in tqdm.tqdm(test_data):
        predictions.append(predict_func(item["text"]))
        gold.append(item["languages"])

    print(f"Loose accuracy: {compute_loose_accuracy(predictions, gold)}", file=args.output)
    print(f"Exact match accuracy: {compute_exact_match_accuracy(predictions, gold)}", file=args.output)
    f1 = compute_f1_score(predictions, gold)
    for idx, lang in enumerate(list(BCMS_TO_OPENLID.keys())):
        print(f"F1,{lang},{f1[idx]}", file=args.output)



if __name__ == "__main__":
    main()
