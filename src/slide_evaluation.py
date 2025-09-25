import urllib.request
from typing import TypedDict, Optional
import json
import argparse
import sys
import logging

import torch
from transformers import CanineForSequenceClassification
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import tqdm
import numpy as np
from sklearn.metrics import f1_score

from common import PROJECT_PATH, load_object, ModelTypeT, MODELS
from flores_evaluation import get_multiclass_model
from prediction import predict_multiclass, predict_multilabel
from multilabel import get_multilabel_model

class SLIDEItem(TypedDict):
    text: str
    languages: list[str]
    original: str
    id: Optional[str]

SLIDE_TO_OPENLID = {"nb": "nob_Latn", "nn": "nno_Latn", "sv": "swe_Latn", "da": "dan_Latn"}

TEST_DATASET_URL = "https://raw.githubusercontent.com/ltgoslo/slide/refs/heads/main/test_data/test_other_2_new.jsonl"
VALIDATION_DATASET_URL = "https://raw.githubusercontent.com/ltgoslo/slide/refs/heads/main/validation_data/validation_annotated.jsonl"

ENCODER_PATH = PROJECT_PATH / "trainer_output" / "label_encoder.pkl"
MODEL_PATH = PROJECT_PATH / "finished_multiclass"

def download_file(url: str) -> str:
    logging.info("Downloading file from %s", url)
    with urllib.request.urlopen(url) as f:
        return f.read().decode('utf-8')

def decode_jsonl(contents: str):
    decoded = []
    for line in contents.split("\n"):
        if line:
            decoded.append(json.loads(line))
    return decoded

def transform(items: list[SLIDEItem]):
    slide_langs = set(SLIDE_TO_OPENLID.keys())
    filtered = [item for item in items if item["original"] in slide_langs]
    transform_labels = lambda labels: [SLIDE_TO_OPENLID[label] for label in labels if label in slide_langs]
    return [{**item, "languages": transform_labels(item["languages"])} for item in filtered]


def compute_loose_accuracy(predicted, gold):
    correct = 0
    for prediction, y in zip(predicted, gold):
        # The paper says if there is an overlap, but the implementation at https://github.com/ltgoslo/slide/blob/main/src/evaluate.py seems to use issubset
        correct += set(prediction).issubset(set(y))

    return correct / len(gold)

def compute_exact_match_accuracy(predicted, gold):
    return sum([p == y for p, y in zip(predicted, gold)]) / len(gold)

def encode_labels(labels):
    return np.asarray(["nob_Latn" in labels, "nno_Latn" in labels, "swe_Latn" in labels, "dan_Latn" in labels])

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

    encoder: LabelEncoder | MultiLabelBinarizer = load_object(
        args.encoder_path)
    assert (isinstance(encoder, LabelEncoder) and args.type != "multilabel") or (
        isinstance(encoder, MultiLabelBinarizer) and args.type == "multilabel")

    test_data: list[SLIDEItem] = transform(decode_jsonl(download_file(TEST_DATASET_URL)))

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
    for idx, lang in enumerate(list(SLIDE_TO_OPENLID.keys())):
        print(f"F1,{lang},{f1[idx]}", file=args.output)



if __name__ == "__main__":
    main()