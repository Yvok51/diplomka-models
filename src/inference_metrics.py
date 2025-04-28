import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

from multiclass import PROJECT_PATH

OUTPUT_FOLDER = PROJECT_PATH / 'output'

LABELS_MAP = {
    'en': 'eng_Latn',
    'eng': 'eng_Latn',
    'vi': 'vie_Latn',
    'vie': 'vie_Latn',
    'sk': 'slk_Latn',
    'slk': 'slk_Latn',
    'cs': 'ces_Latn',
    'ces': 'ces_Latn',
    'ru': 'rus_Cyrl',
    'rus': 'rus_Cyrl',
    'ua': 'ukr_Cyrl',
    'ukr': 'ukr_Cyrl',
    'mk': 'mkd_Cyrl',
    'mkd': 'mkd_Cyrl',
    'bg': 'bul_Cyrl',
    'bul': 'bul_Cyrl',
    'sr': 'srp_Cyrl',
    'srp': 'srp_Cyrl',
    'kk': 'kaz_Cyrl',
    'kaz': 'kaz_Cyrl',
}


def parse_prediction_file(file):
    predicted = []
    for line in file:
        [text, lang] = line.split("\t")
        predicted.append({"text": text.strip(), "languages": lang.strip().split(",")})
    return predicted


def accuracy(predicted: list[dict[str, str]], gold_label: str):
    correct = sum(
        (gold_label in item["languages"] for item in predicted))

    return correct / len(predicted), correct, len(predicted)


def file_accuracy(filename, gold_label):
    with open(filename, 'r', encoding='utf-8') as f:
        predicted = parse_prediction_file(f)
        return accuracy(predicted, gold_label)


def directory_accuracy(source_path, get_label: Callable[str, str] = lambda x: x, glob="*.txt"):
    for path in Path(source_path).rglob(glob):
        acc, correct, total = file_accuracy(
            path, LABELS_MAP[get_label(path.stem)])
        print(
            f"{path.parent}/{path.name}: {acc} ({correct} / {total})")


def main():
    parser = argparse.ArgumentParser(
        description="Language prediction using finetuned CANINE model")
    parser.add_argument("--directory", type=str, default=None, help="Directory to test all of the files")
    parser.add_argument("--input", type=argparse.FileType('r'), default=sys.stdin,
                        help="Path to the file of the model output")
    parser.add_argument("--most-common", type=int, default=5,
                        help="How many language counts to show")
    parser.add_argument("--correct", default=None, type=str,
                        help="The correct label for the sentences, prints out accuracy if provided")
    args = parser.parse_args()

    if args.directory:
        directory_accuracy(args.directory, get_label=lambda name: name.split("-")[-1][:3], glob="*.txt")
        return

    predicted = parse_prediction_file(args.input)

    counter = Counter((lang for item in predicted for lang in item["languages"]))
    print("=== Language counts ===")
    for lang, count in counter.most_common(args.most_common):
        print(f"{lang}: {count}")

    if args.correct:
        correct = sum(
            (args.correct in item["languages"] for item in predicted))
        print("=== Accuracy ===")
        print(
            f"Accuracy: {correct} / {len(predicted)} = {correct / len(predicted)}")


if __name__ == "__main__":
    main()
