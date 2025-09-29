import argparse
import sys
import logging
import glob
from pathlib import Path
import os

import torch
from transformers import CanineForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from components.common import PROJECT_PATH, load_object, GCLD_TO_OPENLID, ModelTypeT, MODELS
from flores_evaluation import get_multiclass_model
from prediction import predict_multiclass, predict_multilabel
from multilabel import get_multilabel_model


DATA_FOLDER = PROJECT_PATH / "data"

ENCODER_PATH = PROJECT_PATH / "trainer_output" / "multilabel_encoder.pkl"
MODEL_PATH = PROJECT_PATH / "finished_multilabel"


def read_directory(path):
    joint_dir = f"{path}/joint"
    single_dir = f"{path}/single"

    dataset = []

    label_set = set()

    for p in glob.iglob(f"{single_dir}/*.txt"):
        if os.path.getsize(p) == 0:
            continue

        labels = [Path(p).stem]
        label_set.update(labels)
        labels = [GCLD_TO_OPENLID[labels[0]]]
        with open(p, "r", encoding='utf-8') as f:
            for line in f:
                dataset.append({"languages": labels, "text": line.strip()})

    for p in glob.iglob(f"{joint_dir}/*.txt"):
        if os.path.getsize(p) == 0:
            continue

        labels = [label for label in Path(p).stem.split("-")]
        label_set.update(labels)
        labels = [GCLD_TO_OPENLID[l] for l in labels]
        with open(p, "r", encoding='utf-8') as f:
            for line in f:
                dataset.append({"languages": labels, "text": line.strip()})

    return dataset, list(label_set)


def read_file(file, labels: list[str]):
    instances = []
    for instance in file:
        instances.append(
            {"languages": [GCLD_TO_OPENLID[label]for label in labels], "text": instance.strip()})
    return instances


def compute_loose_accuracy(predicted, gold):
    correct = 0
    for prediction, y in zip(predicted, gold):
        # The paper says if there is an overlap, but the implementation at https://github.com/ltgoslo/slide/blob/main/src/evaluate.py seems to use issubset
        correct += len(set(prediction).intersection(set(y))) != 0

    return correct / len(gold)


def compute_exact_match_accuracy(predicted, gold):
    return sum([set(p) == set(y) for p, y in zip(predicted, gold)]) / len(gold)


def compute_score(predicted, gold, encoder: MultiLabelBinarizer, metric):
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
        description="Evaluation of language prediction using finetuned CANINE model")
    parser.add_argument("--model-type", type=ModelTypeT, choices=list(MODELS.keys()),
                        default="canine", help="The underlying model type to train")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help="Directory of the finetuned model")
    parser.add_argument("--labels", type=str, nargs="*",
                        help="The labels which the sentences are")
    parser.add_argument("--input", type=argparse.FileType('r'),
                        default=sys.stdin, help="Test sentences")
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

    if args.input_dir:
        test_data, labels = read_directory(args.input_dir)
    else:
        test_data = read_file(args.input, args.labels)
        labels = args.labels

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
            predicted_langs = list(
                zip(*prediction))[0] if len(prediction) > 0 else []
            return list(predicted_langs)

    logging.info("Starting predictions...")
    predictions = []
    gold = []
    for item in tqdm.tqdm(test_data):
        predictions.append(predict_func(item["text"]))
        gold.append(item["languages"])

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
