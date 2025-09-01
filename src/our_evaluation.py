import argparse
import sys
import logging

import torch
from transformers import CanineForSequenceClassification
from sklearn.preprocessing import MultiLabelBinarizer
import tqdm
import numpy as np
from sklearn.metrics import f1_score

from common import PROJECT_PATH, load_object
from flores_evaluation import get_multiclass_model, get_multilabel_model
from prediction import predict_multiclass, predict_multilabel
from multilabel import CanineForMultiLabelClassification

ISO_TO_OPENLID = {"es": "esp_Latn", "bg": "bul_Cyrl", "mk": "mkd_Cyrl", "cs": "ces_Latn",
                  "sk": "slk_Latn", "ca": "cat_Latn", "nl": "nld_Latn", "af": "afr_Latn", "no": "nob_Latn", "da": "dan_Latn", "pt": "por_Latn", "gl": "glg_Latn"}

DATA_FOLDER = PROJECT_PATH / "data"

ENCODER_PATH = PROJECT_PATH / "trainer_output" / "multilabel_encoder.pkl"
MODEL_PATH = PROJECT_PATH / "finished_multiclass"


def read_file(file, labels: list[str]):
    instances = []
    for instance in file:
        instances.append(
            {"languages": [ISO_TO_OPENLID[label]for label in labels], "text": instance.strip()})
    return instances


def compute_loose_accuracy(predicted, gold):
    correct = 0
    for prediction, y in zip(predicted, gold):
        # The paper says if there is an overlap, but the implementation at https://github.com/ltgoslo/slide/blob/main/src/evaluate.py seems to use issubset
        correct += len(set(prediction).intersection(set(y))) != 0

    return correct / len(gold)


def compute_exact_match_accuracy(predicted, gold):
    return sum([set(p) == set(y) for p, y in zip(predicted, gold)]) / len(gold)


def compute_f1_score(predicted, gold, encoder: MultiLabelBinarizer):
    predicted = np.asarray(encoder.transform(predicted))
    gold = np.asarray(encoder.transform(gold))

    return f1_score(predicted, gold, average=None, zero_division=0)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation of language prediction using finetuned CANINE model")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help="Directory of the finetuned model")
    parser.add_argument("--labels", type=str, nargs="+",
                        help="The labels which the sentences are")
    parser.add_argument("--input", type=argparse.FileType('r'),
                        default=sys.stdin, help="Test sentences")
    parser.add_argument("--type", choices=["multiclass", "multilabel"],
                        help="The model which we are using", default="multiclass")
    parser.add_argument("--multilabel-encoder", type=str,
                        default=str(ENCODER_PATH), help="Path to the multilabel encoder")
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

    encoder: MultiLabelBinarizer = load_object(
        args.multilabel_encoder)
    assert isinstance(encoder, MultiLabelBinarizer)

    test_data = read_file(args.input, args.labels)

    if args.type == "multiclass":
        model, tokenizer = get_multiclass_model(args.model_path, device)
        assert isinstance(model, CanineForSequenceClassification)

        def predict_func(sentence):
            prediction = predict_multiclass(
                sentence, model, tokenizer, encoder, device)
            return [prediction[0][0]]

    elif args.type == "multilabel":
        model, tokenizer = get_multilabel_model(args.model_path, device)
        assert isinstance(model, CanineForMultiLabelClassification)

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


    print(
        f"Loose accuracy: {compute_loose_accuracy(predictions, gold)}", file=args.output)
    print(
        f"Exact match accuracy: {compute_exact_match_accuracy(predictions, gold)}", file=args.output)
    print("=== Instances ===", file=args.output)
    for instance, prediction in zip(test_data, predictions):
        print(f"{instance['text'].strip()}\t{prediction}", file=args.output)


if __name__ == "__main__":
    main()
