import argparse
from collections import defaultdict
import sys
import logging

import datasets
import torch
from transformers import CanineForSequenceClassification, CanineTokenizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, f1_score, multilabel_confusion_matrix, precision_score, recall_score
import numpy as np
import tqdm

from huggingface_hub import hf_hub_download

from components.common import load_object, PROJECT_PATH, FASTTEXT_TO_OPENLID, GLOT_TO_OPENLID, OPENLID_TO_OPENLID, GCLD_TO_OPENLID, ModelTypeT, MODELS
from components.prediction import predict_multiclass, predict_multilabel
from multilabel import get_multilabel_model

ENCODER_PATH = PROJECT_PATH / "trainer_output" / "multilabel_encoder.pkl"
# Default path to your finetuned model
MODEL_PATH = PROJECT_PATH / "finished_negative"


class FLORESDataset(torch.utils.data.Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

    def __len__(self):
        return len(self.sentences)


def get_test_dataset(known_languages: list[str] | None = None):
    dataset = datasets.load_dataset(
        'facebook/flores', 'all', trust_remote_code=True)
    dataset = dataset['devtest']

    data: dict[str, list[str]] = defaultdict(lambda: [])
    for item in dataset:
        for k, v in item.items():
            if k.startswith("sentence"):
                language = k[9:]
                if not known_languages or language in known_languages:
                    data[language].append(v)

    return data


def get_multiclass_model(model_path, device):
    model = CanineForSequenceClassification.from_pretrained(
        model_path, local_files_only=True).to(device)
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

    return FP / (FP + TN)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation of language prediction using finetuned CANINE model")
    parser.add_argument("--model-type", type=ModelTypeT, choices=list(MODELS.keys()),
                        default="canine", help="The underlying model type to train")
    parser.add_argument("--model-path", type=str,
                        default=str(MODEL_PATH), help="Directory of the finetuned model")
    parser.add_argument("--type", choices=["multiclass", "multilabel", "fasttext", "glotlid", "openlid", "gcld3"],
                        help="The model which we are using", default="multilabel")
    parser.add_argument("--encoder-path", type=str,
                        default=str(ENCODER_PATH), help="Path to the label encoder")
    parser.add_argument("--confusion-matrix",
                        action="store_true", help="Print out confusion matrix")
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

    logging.info("Loading data...")
    data = get_test_dataset(list(encoder.classes_))

    logging.info("Loading model...")

    TYPES_CONFIG = {
        "openlid": {"repo": "laurievb/OpenLID", "map": OPENLID_TO_OPENLID},
        "glotlid": {"repo": "cis-lmu/glotlid", "map": GLOT_TO_OPENLID},
        "fasttext": {"repo": "facebook/fasttext-language-identification", "map": FASTTEXT_TO_OPENLID}
    }

    if args.type == "multiclass":
        model, tokenizer = get_multiclass_model(args.model_path, device)
        assert isinstance(model, CanineForSequenceClassification)

        def predict_func(sentence):
            prediction = predict_multiclass(
                sentence, model, tokenizer, encoder, device)
            return prediction[0][0]

    elif args.type == "multilabel":
        model, tokenizer = get_multilabel_model(
            args.model_path, device, args.model_type)

        def predict_func(sentence):
            prediction = predict_multilabel(
                sentence, model, tokenizer, encoder, device)
            predicted_langs = list(
                zip(*prediction))[0] if len(prediction) > 0 else []
            return predicted_langs

    elif args.type == "gcld3":
        # imported here because it is trouble installing gcld3 in some environments (Metacentrum)
        from gcld3 import NNetLanguageIdentifier

        detector = NNetLanguageIdentifier(0, 512)

        def predict_func(sentence):
            prediction = detector.FindLanguage(sentence).language
            return GCLD_TO_OPENLID[prediction]

    else:
        # imported here because it is trouble installing fasttext in some environments (Metacentrum)
        import fasttext

        config = TYPES_CONFIG[args.type]
        model_path = hf_hub_download(
            repo_id=config["repo"], filename="model.bin")
        model = fasttext.load_model(model_path)

        def predict_func(sentence):
            prediction = model.predict(sentence)
            return config["map"][prediction[0][0]]

    logging.info("Starting predictions...")
    # The actual predicted labels for different languages
    predictions: dict[str, list[list[str]]] = defaultdict(list)
    for lang, sentences in tqdm.tqdm(data.items()):
        for sentence in sentences:
            prediction = predict_func(sentence)
            predictions[lang].append(prediction)

    def get_confusion_matrix(predictions, gold):
        if args.type == "multilabel":
            gold = np.asarray(encoder.inverse_transform(gold))
            predictions = encoder.inverse_transform(predictions)
            nonzero = np.asarray([len(p) > 0 for p in predictions])
            predictions = np.asarray([g if g[0] in p else [p[0]] for p, g, nonempty in zip(
                predictions, gold, nonzero) if nonempty])
            gold = gold[nonzero]
            final_langs = np.unique(np.concatenate((gold, predictions)))
        else:
            valid = predictions >= 0
            gold = gold[valid]
            predictions = predictions[valid]
            final_langs = encoder.inverse_transform(
                np.unique(np.concatenate((gold, predictions))))

        matrix = confusion_matrix(gold, predictions)
        return matrix, final_langs

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
            get_rates_multilabel if args.type == "multilabel" else get_rates_multilabel
        ), "FPR"),
        (lambda predictions, gold: precision_score(y_pred=predictions,
         y_true=gold, average=None, zero_division=0), "Precision"),
        (lambda predictions, gold: recall_score(y_pred=predictions,
         y_true=gold, average=None, zero_division=0), "Recall"),
    ]

    total_predictions = []
    total_labels = []
    for lang, predicted in tqdm.tqdm(predictions.items()):
        if args.type == "multilabel":
            encoded_predicted = encoder.transform(predicted)
        else:
            predicted = np.array(predicted)
            non_null = np.where(predicted != None)[0]
            non_null_encoded = encoder.transform(predicted[non_null])
            encoded_predicted = np.full(shape=predicted.shape, fill_value=-1)
            encoded_predicted[non_null] = non_null_encoded

        correct = encoder.transform(
            [[lang]] if args.type == "multilabel" else [lang])[0]
        labels = np.full(
            shape=(len(predicted), *
                   correct.shape) if args.type == "multilabel" else len(predicted),
            fill_value=correct,
            dtype=int
        )

        total_predictions.extend(encoded_predicted)
        total_labels.extend(labels)

    total_labels = np.asarray(total_labels)
    total_predictions = np.asarray(total_predictions)

    if args.type == "multilabel":
        tested_lang_idx = np.where((total_labels).sum(axis=0) != 0)[0]
    else:
        tested_lang_idx = np.unique(total_labels)
    tested_langs = np.asarray(encoder.classes_)[np.sort(tested_lang_idx)]

    for metric, name in metrics:
        values = metric(total_predictions, total_labels)[tested_lang_idx]
        for idx, lang in enumerate(tested_langs):
            print(f"{name},{lang},{values[idx]}", file=args.output)

        print(f"{name},all,{np.average(values)}", file=args.output)

    if args.confusion_matrix or True:
        print("=== Confusion matrix ===", file=args.output)
        matrix, langs = get_confusion_matrix(total_predictions, total_labels)
        print(9 * " " + " ".join(langs), file=args.output)
        for line, lang in zip(matrix, langs):
            print(f"{lang} " + "".join([str(n) + ((9 - len(str(n))) * " ")
                  for n in line]), file=args.output)


if __name__ == "__main__":
    main()
