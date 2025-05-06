import argparse
from collections import defaultdict
import sys
import logging

import datasets
import torch
from transformers import CanineForSequenceClassification, CanineTokenizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, f1_score, multilabel_confusion_matrix
import numpy as np
import tqdm

from common import load_object, PROJECT_PATH
from prediction import predict_multiclass, predict_multilabel
from multilabel import CanineForMultiLabelClassification

ENCODER_PATH = PROJECT_PATH / "trainer_output" / "label_encoder.pkl"
# Default path to your finetuned model
MODEL_PATH = PROJECT_PATH / "finetuned_epoch-2_samples-15000"


class FLORESDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

    def __len__(self):
        return len(self.sentences)


def get_test_dataset(known_sentences: list[str] | None = None):
    dataset = datasets.load_dataset(
        'facebook/flores', 'all', trust_remote_code=True)
    dataset = dataset['devtest']

    data: dict[str, list[str]] = defaultdict(lambda: [])
    for item in dataset:
        for k, v in item.items():
            if k.startswith("sentence"):
                language = k[9:]
                if not known_sentences or language in known_sentences:
                    data[language].append(v)

    return data


def get_multiclass_model(model_path, device):
    model = CanineForSequenceClassification.from_pretrained(
        model_path, local_files_only=True).to(device)
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    return model, tokenizer


def get_multilabel_model(model_path, device):
    model = CanineForMultiLabelClassification.from_pretrained(
        model_path).to(device)
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    return model, tokenizer


def get_rates_multiclass(predictions, gold):
    matrix = confusion_matrix(gold, predictions)
    FP = matrix.sum(axis=0) - np.diag(matrix)
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(matrix)
    TN = matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    return TP, FP, TN, FN


def get_rates_multilabel(predictions, gold):
    matrix = multilabel_confusion_matrix(gold, predictions)
    FP = matrix[:, 0, 1].astype(float)
    FN = matrix[:, 1, 0].astype(float)
    TP = matrix[:, 1, 1].astype(float)
    TN = matrix[:, 0, 0].astype(float)

    return TP, FP, TN, FN


def false_positive_rate(predictions, gold, get_rates):
    _, FP, TN, _ = get_rates(predictions, gold)

    return FP / (FP + TN) if FP + TN > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation of language prediction using finetuned CANINE model")
    parser.add_argument("--model-path", type=str,
                        default=str(MODEL_PATH), help="Directory of the finetuned model")
    parser.add_argument("--type", choices=["multiclass", "multilabel"],
                        help="The model which we are using", default="multiclass")
    parser.add_argument("--encoder-path", type=str,
                        default=str(ENCODER_PATH), help="Path to the label encoder")
    parser.add_argument("--seed", type=int,
                        default=42, help="Path to the label encoder")
    parser.add_argument("--max-length", type=int, default=512,
                        help="The max length of the tokenized input. The model maximum is 2048")
    parser.add_argument("--output", type=argparse.FileType('w'),
                        default=sys.stdout, help="The file to write the metrics to")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.info("Using device %s", device)

    encoder: LabelEncoder | MultiLabelBinarizer = load_object(
        args.encoder_path)
    assert (isinstance(encoder, LabelEncoder) and args.type == "multiclass") or (
        isinstance(encoder, MultiLabelBinarizer) and args.type == "multilabel")

    logging.info("Loading data...")
    data = get_test_dataset(list(encoder.classes_))

    logging.info("Loading model...")
    if args.type == "multiclass":
        model, tokenizer = get_multiclass_model(args.model_path, device)
        assert isinstance(model, CanineForSequenceClassification)
    else:
        model, tokenizer = get_multilabel_model(args.model_path, device)
        assert isinstance(model, CanineForMultiLabelClassification)

    predict_func = predict_multiclass if args.type == "multiclass" else predict_multilabel

    logging.info("Starting predictions...")
    predictions: dict[str, list[list[str]]] = defaultdict(list)
    for lang, sentences in tqdm.tqdm(data.items()):
        for sentence in sentences:
            prediction = predict_func(
                sentence, model, tokenizer, encoder, device)
            predicted_langs, _ = list(zip(*prediction))
            predictions[lang].append(
                predicted_langs if args.type == "multilabel" else predicted_langs[0])

    logging.info("Starting evaluations...")
    metrics: list = [
        (lambda predictions, gold: f1_score(
            y_pred=predictions,
            y_true=gold,
            average=None,
            zero_division=0), "F1"),
        (lambda predictions, gold: false_positive_rate(
            np.asarray(predictions),
            np.asarray(gold),
            get_rates_multiclass if args.type == "multiclass" else get_rates_multilabel
        ), "FPR")
    ]

    total_predictions = []
    total_labels = []
    for lang, predicted in tqdm.tqdm(predictions.items()):
        encoded_predicted = encoder.transform(predicted)
        correct = encoder.transform([lang])[0]
        if args.type == "multiclass":
            labels = np.full(shape=len(predicted),
                             fill_value=correct, dtype=int)
        else:
            labels = np.full(
                shape=(len(predicted), *correct.shape), fill_value=correct, dtype=int)

        total_predictions.extend(encoded_predicted)
        total_labels.extend(labels)

    total_labels = np.asarray(total_labels)
    total_predictions = np.asarray(total_predictions)

    for metric, name in metrics:
        values = metric(total_predictions, total_labels)
        for idx, lang in enumerate(encoder.classes_):
            print(f"{name},{lang},{values[idx]}",
                  file=args.output)

        print(f"{name},all,{np.average(values)}", file=args.output)


if __name__ == "__main__":
    main()
