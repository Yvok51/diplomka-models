import argparse
import sys
from collections import Counter

def main():
    parser = argparse.ArgumentParser(
        description="Language prediction using finetuned CANINE model")
    parser.add_argument("--input", type=argparse.FileType('r'), default=sys.stdin,
                        help="Path to the file of the model output")
    parser.add_argument("--most-common", type=int, default=5, help="How many language counts to show")
    parser.add_argument("--correct", default=None, type=str,
                        help="The correct label for the sentences, prints out accuracy if provided")
    args = parser.parse_args()

    predicted = []
    for line in args.input:
        [text, lang] = line.split("\t")
        predicted.append({"text": text.strip(), "language": lang.strip()})


    counter = Counter((item["language"] for item in predicted))
    print("=== Language counts ===")
    for lang, count in counter.most_common(args.most_common):
        print(f"{lang}: {count}")

    if args.correct:
        correct = sum(
            (item["language"] == args.correct for item in predicted))
        print("=== Accuracy ===")
        print(f"Accuracy: {correct} / {len(predicted)} = {correct / len(predicted)}")


if __name__ == "__main__":
    main()