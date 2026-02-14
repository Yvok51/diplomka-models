import argparse
import sys
import logging
from typing import TypedDict
import json

import torch
from transformers import CanineForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from components.common import PROJECT_PATH, load_object, GCLD_TO_OPENLID, MODELS
from components.prediction import predict_multiclass, predict_multilabel
from flores_evaluation import get_multiclass_model
from multilabel import get_multilabel_model

class Instance(TypedDict):
    text: str
    label: list[str]

Dataset = dict[str, dict[str, list[Instance]]]


DATA_FOLDER = PROJECT_PATH / "data"

ENCODER_PATH = PROJECT_PATH / "trainer_output" / "multilabel_encoder.pkl"
MODEL_PATH = PROJECT_PATH / "finished_multilabel"


def get_labels(instances: list[Instance]) -> set[str]:
    labels = set()
    for instance in instances:
        labels.update(instance["label"])
    return labels

def get_dataset(input_dataset: dict[str, list[Instance] | dict]) -> tuple[list[Instance], set[str]]:
    dataset = []
    labels = set()
    for section in input_dataset.values():
        if isinstance(section, dict):
            inner_dataset, inner_labels = get_dataset(section)
            dataset.extend(inner_dataset)
            labels.update(inner_labels)
        else:
            dataset.extend(section)
            labels.update(get_labels(section))

    return dataset, labels

def read_dataset(file: argparse.FileType) -> tuple[list[Instance], list[str]]:
    dataset: Dataset = json.load(file)
    return get_dataset(dataset)

def compute_loose_accuracy(predicted: list[list[str]], gold: list[list[str]]) -> float:
    correct = 0
    for prediction, y in zip(predicted, gold):
        # The paper says if there is an overlap, but the implementation at https://github.com/ltgoslo/slide/blob/main/src/evaluate.py seems to use issubset
        correct += len(set(prediction).intersection(set(y))) != 0

    return correct / len(gold)


def compute_exact_match_accuracy(predicted: list[list[str]], gold: list[list[str]]) -> float:
    return sum([set(p) == set(y) for p, y in zip(predicted, gold)]) / len(gold)


def compute_score(predicted: list[list[str]], gold: list[list[str]], encoder: MultiLabelBinarizer, metric):
    predicted = np.asarray(encoder.transform(predicted))
    gold = np.asarray(encoder.transform(gold))

    return metric(predicted, gold, average=None, zero_division=0)


def value_indices(arr, values):
    """Return indices in `arr` of the values in `values`. Returns the indices in the order of `values`."""
    sorter = np.argsort(arr)
    return sorter[np.searchsorted(arr, values, sorter=sorter)]
    # np.in1d(arr, values).nonzero()[0]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation of language prediction using finetuned model")
    parser.add_argument("--model-type", type=str, choices=list(MODELS.keys()), help="The underlying model type to train")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help="Directory of the finetuned model")
    parser.add_argument("--input", type=argparse.FileType('r'),
                        default=sys.stdin, help="Test dataset")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Instead of single file read an entire directory")
    parser.add_argument("--type", choices=["multiclass", "multilabel"],
                        help="The model which we are using", default="multilabel")
    parser.add_argument("--multilabel-encoder", type=str,
                        default=str(ENCODER_PATH), help="Path to the multilabel encoder")
    parser.add_argument("--encoder", type=str, default=str(ENCODER_PATH),
                        help="Encoder to use with the model")
    parser.add_argument("--seed", type=int,
                        default=42, help="Seed for the random number generator")
    parser.add_argument("--instances", action="store_true",
                        help="Print predictions for individual instances")
    parser.add_argument("--output", type=argparse.FileType('w'),
                        default=sys.stdout, help="The file to write the metrics to")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Using device %s", device)

    multilabel_encoder: MultiLabelBinarizer = load_object(
        args.multilabel_encoder)
    assert isinstance(multilabel_encoder, MultiLabelBinarizer)
    encoder: MultiLabelBinarizer | LabelEncoder = load_object(args.encoder)
    assert isinstance(multilabel_encoder, MultiLabelBinarizer)\
        if args.type == "multilabel" else isinstance(encoder, LabelEncoder)

    logging.info("Loading test data")
    test_data, labels = read_dataset(args.input)

    logging.info("Loading model: %s", args.type)
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
                sentence, model, tokenizer, multilabel_encoder, device)
            predicted_langs: list[str] = list(
                zip(*prediction))[0] if len(prediction) > 0 else []
            return list(predicted_langs)

    logging.info("Starting predictions...")
    predictions: list[list[str]] = []
    gold: list[list[str]] = []
    for item in tqdm.tqdm(test_data):
        predictions.append(predict_func(item["text"]))
        gold.append(item["label"])

    print(
        f"Loose accuracy: {compute_loose_accuracy(predictions, gold)}", file=args.output)
    print(
        f"Exact match accuracy: {compute_exact_match_accuracy(predictions, gold)}", file=args.output)

    print("=== Scores ===", file=args.output)
    f1 = compute_score(predictions, gold, multilabel_encoder, f1_score)
    precision = compute_score(predictions, gold, multilabel_encoder, precision_score)
    recall = compute_score(predictions, gold, multilabel_encoder, recall_score)
    indices = value_indices(
        multilabel_encoder.classes_, [GCLD_TO_OPENLID[l] for l in labels])
    for idx, label in zip(indices, labels):
        print(
            f"{label},{round(f1[idx], 4)},{round(precision[idx], 4)},{round(recall[idx], 4)}", file=args.output)

    if args.instances:
        print("=== Instances ===", file=args.output)
        for instance, prediction, correct in zip(test_data, predictions, gold):
            print(
                f"{instance['text'].strip()}\t{prediction}\t{correct}", file=args.output)


if __name__ == "__main__":
    main()
