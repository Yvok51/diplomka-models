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

from components.common import (
    PROJECT_PATH, load_object, GCLD_TO_OPENLID, MODELS,
    FASTTEXT_TO_OPENLID, GLOT_TO_OPENLID, OPENLID_TO_OPENLID
)
from components.prediction import predict_multiclass, predict_multilabel
from flores_evaluation import get_multiclass_model
from multilabel import get_multilabel_model
from components.tf_idf_model import NLIClassifier, MultilabelNLIClassifier

class Instance(TypedDict):
    text: str
    label: list[str]

Dataset = dict[str, dict[str, list[Instance]]]

# Predictions paired with gold labels
PredictionItem = tuple[list[str], list[str]]  # (prediction, gold)
PredictionDataset = dict[str, dict[str, list[PredictionItem]]]

DATA_FOLDER = PROJECT_PATH / "data"

ENCODER_PATH = PROJECT_PATH / "trainer_output" / "multilabel_encoder.pkl"
MODEL_PATH = PROJECT_PATH / "finished_multilabel"


def get_labels(instances: list[Instance]) -> set[str]:
    labels = set()
    for instance in instances:
        labels.update(instance["label"])
    return labels

def get_dataset(input_dataset: dict[str, list[Instance] | dict]) -> tuple[list[Instance], list[str]]:
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

    return dataset, list(labels)

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


def value_indices(arr: np.ndarray, values: np.ndarray):
    """Return indices in `arr` of the values in `values`. Returns the indices in the order of `values`."""
    sorter = np.argsort(arr)
    return sorter[np.searchsorted(arr, values, sorter=sorter)]
    # np.in1d(arr, values).nonzero()[0]


def read_dataset_hierarchical(file: argparse.FileType) -> Dataset:
    """Read dataset maintaining its hierarchical structure."""
    dataset: Dataset = json.load(file)
    return dataset


def make_predictions_hierarchical(
    dataset: Dataset,
    predict_func
) -> PredictionDataset:
    """Make predictions maintaining the hierarchical structure of the dataset.

    Handles variable depth in dataset structure:
    - 2-level: {level1: {level2: [instances]}}
    - 3-level: {level1: {level2: {level3: [instances]}}}
    """
    predictions: PredictionDataset = {}

    for level1_key, level1_value in dataset.items():
        if not isinstance(level1_value, dict):
            logging.warning("Skipping %s: not a dict, got %s", level1_key, type(level1_value))
            continue
        predictions[level1_key] = {}

        for level2_key, level2_value in level1_value.items():
            # Check if this is a list of instances (2-level structure)
            if isinstance(level2_value, list):
                predictions[level1_key][level2_key] = []
                for instance in tqdm.tqdm(level2_value, desc=f"{level1_key}/{level2_key}", leave=False):
                    if not isinstance(instance, dict):
                        logging.warning("Skipping instance in %s/%s: not a dict", level1_key, level2_key)
                        continue
                    pred = predict_func(instance["text"])
                    gold = instance["label"]
                    predictions[level1_key][level2_key].append((pred, gold))

            # Check if this is another dict level (3-level structure)
            elif isinstance(level2_value, dict):
                for level3_key, instances in level2_value.items():
                    if not isinstance(instances, list):
                        logging.warning("Skipping %s/%s/%s: not a list", level1_key, level2_key, level3_key)
                        continue
                    # Use level2_key/level3_key as the combined key
                    combined_key = f"{level2_key}/{level3_key}"
                    predictions[level1_key][combined_key] = []
                    for instance in tqdm.tqdm(instances, desc=f"{level1_key}/{combined_key}", leave=False):
                        if not isinstance(instance, dict):
                            logging.warning("Skipping instance in %s/%s: not a dict", level1_key, combined_key)
                            continue
                        pred = predict_func(instance["text"])
                        gold = instance["label"]
                        predictions[level1_key][combined_key].append((pred, gold))
            else:
                logging.warning("Skipping %s/%s: unexpected type %s", level1_key, level2_key, type(level2_value))

    return predictions


def collect_predictions(pred_items: list[PredictionItem]) -> tuple[list[list[str]], list[list[str]], set[str]]:
    """Extract predictions and gold labels from a list of prediction items."""
    if not pred_items:
        return [], [], set()

    predictions, gold = zip(*pred_items)
    labels = set()
    for g in gold:
        labels.update(g)

    return list(predictions), list(gold), labels


def collect_all_predictions(pred_dict: dict[str, list[PredictionItem]]) -> tuple[list[list[str]], list[list[str]], set[str]]:
    """Recursively collect all predictions from a dictionary subtree."""
    all_predictions = []
    all_gold = []
    all_labels = set()

    for value in pred_dict.values():
        preds, gold, labels = collect_predictions(value)
        all_predictions.extend(preds)
        all_gold.extend(gold)
        all_labels.update(labels)

    return all_predictions, all_gold, all_labels


def output_metrics(
    predictions: list[list[str]],
    gold: list[list[str]],
    labels: set[str],
    encoder: MultiLabelBinarizer,
    output_file,
    section_name: str
):
    """Compute and output all metrics for a given set of predictions."""
    if not predictions:
        return

    print(f"\n=== {section_name} ===", file=output_file)
    print(f"Sample count: {len(predictions)}", file=output_file)
    print(f"Loose accuracy: {compute_loose_accuracy(predictions, gold):.4f}", file=output_file)
    print(f"Exact match accuracy: {compute_exact_match_accuracy(predictions, gold):.4f}", file=output_file)

    # Compute per-language metrics
    f1 = compute_score(predictions, gold, encoder, f1_score)
    precision = compute_score(predictions, gold, encoder, precision_score)
    recall = compute_score(predictions, gold, encoder, recall_score)

    # Filter to labels present in this subset
    if labels:
        print("Per-language scores (language, F1, precision, recall):", file=output_file)
        sorted_labels = sorted(labels)
        indices = value_indices(encoder.classes_, np.asarray(sorted_labels))
        for idx, label in zip(indices, sorted_labels):
            print(f"  {label}: {round(f1[idx], 4)}, {round(precision[idx], 4)}, {round(recall[idx], 4)}", file=output_file)


def evaluate_hierarchical(
    pred_dataset: PredictionDataset,
    encoder: MultiLabelBinarizer,
    output_file,
    instances_flag: bool = False
):
    """Recursively evaluate predictions at all levels of the hierarchy."""

    # Evaluate at leaf level (level1/level2)
    for level1_key, level1_dict in pred_dataset.items():
        for level2_key, pred_items in level1_dict.items():
            predictions, gold, labels = collect_predictions(pred_items)
            section_name = f"{level1_key}/{level2_key}"
            output_metrics(predictions, gold, labels, encoder, output_file, section_name)

            if instances_flag:
                print(f"\n--- Instances for {section_name} ---", file=output_file)
                for (pred, g), idx in zip(pred_items, range(len(pred_items))):
                    print(f"  {idx}: predicted={pred}, gold={g}", file=output_file)

    # Evaluate at parent level (level1)
    for level1_key, level1_dict in pred_dataset.items():
        predictions, gold, labels = collect_all_predictions(level1_dict)
        section_name = f"{level1_key} (combined)"
        output_metrics(predictions, gold, labels, encoder, output_file, section_name)

    # Evaluate overall
    all_predictions = []
    all_gold = []
    all_labels = set()
    for level1_dict in pred_dataset.values():
        preds, gold, labels = collect_all_predictions(level1_dict)
        all_predictions.extend(preds)
        all_gold.extend(gold)
        all_labels.update(labels)

    output_metrics(all_predictions, all_gold, all_labels, encoder, output_file, "Overall")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluation of language prediction using finetuned model")
    parser.add_argument("--model-kind", type=str, choices=["transformer", "tfidf"],
                        help="Kind of model (transformer or tfidf), only needed for multiclass/multilabel types")
    parser.add_argument("--model-type", type=str, choices=list(MODELS.keys()), help="The underlying model type to train")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help="Directory of the finetuned model or path to TF-IDF pickle file")
    parser.add_argument("--input", type=argparse.FileType('r'),
                        default=sys.stdin, help="Test dataset")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Instead of single file read an entire directory")
    parser.add_argument("--type", choices=["multiclass", "multilabel", "fasttext", "glotlid", "openlid", "gcld3"],
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

    # Load encoders based on model type
    if args.type == "multilabel":
        # Multilabel models always need the multilabel encoder
        multilabel_encoder: MultiLabelBinarizer = load_object(
            args.multilabel_encoder)
        assert isinstance(multilabel_encoder, MultiLabelBinarizer)
    elif args.type in ["fasttext", "glotlid", "openlid", "gcld3"]:
        # These external models don't need encoders loaded, but we need multilabel_encoder for metrics
        multilabel_encoder: MultiLabelBinarizer = load_object(
            args.multilabel_encoder)
        assert isinstance(multilabel_encoder, MultiLabelBinarizer)
    else:
        # Multiclass and TF-IDF multiclass models
        multilabel_encoder: MultiLabelBinarizer = load_object(
            args.multilabel_encoder)
        assert isinstance(multilabel_encoder, MultiLabelBinarizer)
        if args.model_kind == "transformer":
            encoder: MultiLabelBinarizer | LabelEncoder = load_object(args.encoder)
            assert isinstance(encoder, LabelEncoder)

    logging.info("Loading test data")
    test_dataset = read_dataset_hierarchical(args.input)

    logging.info("Loading model: %s", args.type)

    # Check for external models first (they don't use model_kind)
    if args.type == "gcld3":
        # GCLD3 model evaluation
        from gcld3 import NNetLanguageIdentifier

        detector = NNetLanguageIdentifier(0, 512)

        def predict_func(sentence):
            prediction = detector.FindLanguage(sentence).language
            return [GCLD_TO_OPENLID[prediction]]

    elif args.type in ["fasttext", "glotlid", "openlid"]:
        # External models from HuggingFace (fasttext, glotlid, openlid)
        import fasttext
        from huggingface_hub import hf_hub_download

        TYPES_CONFIG = {
            "openlid": {"repo": "laurievb/OpenLID", "map": OPENLID_TO_OPENLID},
            "glotlid": {"repo": "cis-lmu/glotlid", "map": GLOT_TO_OPENLID},
            "fasttext": {"repo": "facebook/fasttext-language-identification", "map": FASTTEXT_TO_OPENLID}
        }

        config = TYPES_CONFIG[args.type]
        model_path = hf_hub_download(
            repo_id=config["repo"], filename="model.bin")
        model = fasttext.load_model(model_path)

        # Monkey-patch numpy.array to fix fasttext numpy 2.0 compatibility
        original_np_array = np.array
        def patched_array(object, dtype=None, copy=None, order='K', subok=False, ndmin=0, like=None):
            # If copy=False, change it to None to avoid numpy 2.0 error
            if copy is False:
                copy = None
            return original_np_array(object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin)

        np.array = patched_array

        def predict_func(sentence):
            prediction = model.predict(sentence)
            return [config["map"][prediction[0][0]]]

    # For multiclass/multilabel, check model_kind
    elif args.type in ["multiclass", "multilabel"]:
        if args.model_kind is None:
            parser.error("--model-kind is required when using --type multiclass or multilabel")

        if args.model_kind == "tfidf":
            # TF-IDF model evaluation
            if args.type == "multiclass":
                model = NLIClassifier.load_model(args.model_path)

                def predict_func(sentence):
                    result = model.predict_single(sentence)
                    return [result['prediction']]

            elif args.type == "multilabel":
                model = MultilabelNLIClassifier.load_model(args.model_path)

                def predict_func(sentence):
                    result = model.predict_single(sentence)
                    return result['predictions']

        elif args.model_kind == "transformer":
            # Transformer model evaluation
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
    prediction_dataset = make_predictions_hierarchical(test_dataset, predict_func)

    logging.info("Computing metrics at all hierarchy levels...")
    evaluate_hierarchical(
        prediction_dataset,
        multilabel_encoder,
        args.output,
        args.instances
    )


if __name__ == "__main__":
    main()
