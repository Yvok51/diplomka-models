import logging
import torch
import argparse
import sys
from collections import Counter

from transformers import CanineTokenizer, CanineForSequenceClassification

from common import load_object, PROJECT_PATH
from multiclass import ENCODER_PATH, MODEL_PATH
from multilabel import CanineForMultiLabelClassification
from prediction import predict_multiclass, predict_multilabel

MODEL_PATH = PROJECT_PATH / "finetuned_multilabel_epoch-2_samples-15000"
ENCODER_PATH = PROJECT_PATH / "trainer_output" / "multilabel_encoder.pkl"


def predict_from_file(predict, file, model, tokenizer, label_encoder, device):
    """Predict languages for each line in the given file."""
    results = []

    for idx, line in enumerate(file):
        line = line.strip()
        if not line:
            continue

        predictions = predict(
            line, model, tokenizer, label_encoder, device)
        languages, confidences = zip(*predictions) if len(predictions) > 0 else [], []
        results.append({"text": line, "languages": languages,
                        "confidences": confidences})

        if (idx + 1) % 1000 == 0:
            logging.info("Processed %s lines...", idx + 1)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Language prediction using finetuned CANINE model")
    parser.add_argument("--type", choices=["multiclass", "multilabel"], help="The model which we are using", default="multilabel")
    parser.add_argument("--input", type=argparse.FileType('r'), default=sys.stdin,
                        help="Path to input text file (one sentence per line)")
    parser.add_argument(
        "--output", type=argparse.FileType('w'), default=sys.stdout, help="Path to output file (default: input file with .pred extension)")
    parser.add_argument("--model-path", type=str,
                        default=str(MODEL_PATH), help="Path to the finetuned model")
    parser.add_argument("--encoder-path", type=str,
                        default=str(ENCODER_PATH), help="Path to the label encoder")
    parser.add_argument("--correct-label", default=None, type=str,
                        help="The correct label for the sentences, prints out accuracy if provided")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Using device: %s", device)

    logging.info("Loading label encoder from %s", args.encoder_path)
    label_encoder = load_object(args.encoder_path)

    # Load model and tokenizer
    logging.info("Loading model from %s", args.model_path)
    if args.type == "multiclass":
        model = CanineForSequenceClassification.from_pretrained(
            args.model_path).to(device)
    else:
        model = CanineForMultiLabelClassification.from_pretrained(args.model_path).to(device)
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")


    # Process input file
    logging.info("Processing input file: %s", args.input)

    predict_func = predict_multiclass if args.type == "multiclass" else predict_multilabel
    predicted = predict_from_file(
        predict_func, args.input, model, tokenizer, label_encoder, device
    )

    for item in predicted:
        print(item["text"] + "\t" + ",".join(item['languages']), file=args.output)

    counter = Counter((lang for item in predicted for lang in item["languages"]))
    print("=== Language counts ===")
    for lang, count in counter.most_common(5):
        print(f"{lang}: {count}")

    if args.correct_label:
        correct = sum(
            (args.correct_label in item["languages"] for item in predicted))
        print("=== Accuracy ===")
        print(
            f"Accuracy: {correct} / {len(predicted)} = {correct / len(predicted)}")


if __name__ == "__main__":
    main()
