import argparse
from collections import defaultdict
import sys
import logging

import datasets
import torch
from transformers import CanineForSequenceClassification, CanineTokenizer
import evaluate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import numpy as np
import tqdm

from common import load_object, PROJECT_PATH
from inference import get_logits

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


def get_model(model_path, device):
    model = CanineForSequenceClassification.from_pretrained(
        model_path, local_files_only=True).to(device)
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    return model, tokenizer


def get_rates(predictions, gold):
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


def false_positive_rate(predictions, gold):
    _, FP, TN, _ = get_rates(predictions, gold)

    FP = FP.sum()
    TN = TN.sum()

    return FP / (FP + TN) if FP + TN > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation of language prediction using finetuned CANINE model")
    parser.add_argument("--model-path", type=str,
                        default=str(MODEL_PATH), help="Directory of the finetuned model")
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

    encoder: LabelEncoder = load_object(args.encoder_path)
    label_list = np.arange(len(encoder.classes_))

    logging.info("Loading data...")
    data = get_test_dataset(list(encoder.classes_))

    logging.info("loading model...")
    model, tokenizer = get_model(args.model_path, device)

    logging.info("Starting predictions...")
    predictions = defaultdict(lambda: [])
    for lang, sentences in tqdm.tqdm(data.items()):
        for sentence in sentences:
            prediction, _ = get_logits(sentence, model, tokenizer, device)
            predictions[lang].append(prediction)

    logging.info("Starting evaluations...")
    F1_metric = evaluate.load("f1")
    metrics = [
        (lambda predictions, gold: F1_metric.compute(predictions=predictions, references=gold, average='macro')["f1"], {}, "F1"),
        (lambda predictions, gold: false_positive_rate(np.asarray(predictions), np.asarray(gold)), {}, "FPR")
    ]

    total_predictions = []
    total_labels = []
    for lang, predicted in tqdm.tqdm(predictions.items()):
        correct = encoder.transform([lang])[0]
        labels = np.full(shape=len(predicted),
                         fill_value=correct, dtype=int)

        for metric, results, name in metrics:
            results[lang] = metric(predicted, labels)
            print(f"{name},{lang},{results[lang]}", file=args.output)

        total_predictions.extend(predicted)
        total_labels.extend(labels)

    for metric, results, name in metrics:
        print(f"{name},all,{metric(total_predictions, total_labels)}", file=args.output)


if __name__ == "__main__":
    main()
