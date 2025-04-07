import logging
import torch
import argparse
import sys
from collections import Counter

from transformers import CanineTokenizer, CanineForSequenceClassification

from main import load_label_encoder, ENCODER_PATH, MODEL_PATH, tokenize_dataset


def predict_language(text, model, tokenizer, label_encoder, device):
    """Predict the language of a given text."""
    inputs = tokenize_dataset(text, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Get prediction
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).cpu().numpy()[0]
    language = label_encoder.inverse_transform([prediction])[0]

    # Calculate confidence (softmax of logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probs[0][prediction].item()

    return language, confidence


def predict_from_file(file, model, tokenizer, label_encoder, device, output_file=None):
    """Predict languages for each line in the given file."""
    results = []

    for idx, line in enumerate(file):
        line = line.strip()
        if not line:
            continue

        language, confidence = predict_language(
            line, model, tokenizer, label_encoder, device)
        results.append({"text": line, "language": language,
                        "confidence": confidence})

        if (idx + 1) % 1000 == 0:
            logging.info(f"Processed {idx + 1} lines...")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Language prediction using finetuned CANINE model")
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

    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Set output file if not provided
    if not args.output:
        args.output = f"{args.input}.pred"

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load model and tokenizer
    logging.info(f"Loading model from {args.model_path}")
    model = CanineForSequenceClassification.from_pretrained(
        args.model_path).to(device)
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    # Load label encoder
    logging.info(f"Loading label encoder from {args.encoder_path}")
    label_encoder = load_label_encoder(args.encoder_path)

    # Process input file
    logging.info(f"Processing input file: {args.input}")
    predicted = predict_from_file(args.input, model, tokenizer,
                                  label_encoder, device, args.output)

    for item in predicted:
        print(item["text"] + "\t" + item['language'], file=args.output)

    counter = Counter((item["language"] for item in predicted))
    print("=== Language counts ===")
    for lang, count in counter.most_common(5):
        print(f"{lang}: {count}")

    if args.correct_label:
        correct = sum(
            (item["language"] == args.correct_label for item in predicted))
        print("=== Accuracy ===")
        print(f"Accuracy: {correct} / {len(predicted)} = {correct / len(predicted)}")



if __name__ == "__main__":
    main()
