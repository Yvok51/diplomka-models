import os
import logging
from pathlib import Path
import argparse

from dotenv import load_dotenv
from transformers import (
    CanineTokenizer,
    CanineForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
import datasets
import evaluate
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from common import (
    OpenLIDDataset,
    OnTheFlyTokenizationCollator,
    sample_dataset,
    create_language_dict,
    save_object,
    load_object,
    PROJECT_PATH,
    tokenize_dataset,
    EncodedOpenLIDDataset,
    ConcatenateEncodingCollator
)

ENCODER_PATH = PROJECT_PATH / "trainer_output" / "label_encoder.pkl"
MODEL_PATH = PROJECT_PATH / "finetuned"  # Default path to your finetuned model

SAMPLES_PER_LANGUAGE = 10_000


def load_dataset(samples_count: int | None, encoder_path: Path):
    dataset = datasets.load_dataset(
        'laurievb/OpenLID-v2', token=os.environ.get("HUGGINGFACE_TOKEN"),
        features=datasets.Features({  # Present because without it, the function throws an exception
            'text': datasets.Value('string'),
            'language': datasets.Value('string'),
            'source': datasets.Value('string'),
            '__index_level_0__': datasets.Value('int64')
        })
    )

    df = dataset['train']
    # df = df.select(range(10_000))

    if samples_count:
        logging.info("Randomly sampling dataset...")
        texts, labels = sample_dataset(create_language_dict(
            df['text'], df['language']), samples_count)
    else:
        texts, labels = df['text'], df['language']

    logging.info("Encoding the labels...")
    # Encode language labels
    if os.path.exists(encoder_path):
        label_encoder: LabelEncoder = load_object(encoder_path)
        assert isinstance(label_encoder, LabelEncoder)
        encoded_labels = label_encoder.transform(labels)
    else:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        save_object(label_encoder, encoder_path)

    logging.info("Splitting dataset...")
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts,
        encoded_labels,
        test_size=0.1,
    )

    return train_texts, eval_texts, train_labels, eval_labels, label_encoder


def finetune_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    pre_tokenize,
    output_dir='./finetuned',
    learning_rate=5e-5,
    batch_size=24,
    num_train_epochs=1,
    weight_decay=0.01,
    max_length=2048
):
    if pre_tokenize:
        collator = ConcatenateEncodingCollator(max_length)
    else:
        collator = OnTheFlyTokenizationCollator(
            tokenizer=tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="wandb",
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # processing_class=lambda text: tokenize_dataset(text, tokenizer),
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    trainer.save_model(output_dir)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Language prediction using finetuned CANINE model")
    parser.add_argument("--model-path", type=str,
                        default=str(MODEL_PATH), help="Directory to put final the finetuned model")
    parser.add_argument("--encoder-path", type=str,
                        default=str(ENCODER_PATH), help="Path to the label encoder")
    parser.add_argument("--seed", type=int,
                        default=42, help="Path to the label encoder")
    parser.add_argument("--samples-per-language", type=int, default=None,
                        help="The number of samples per language to use")
    parser.add_argument("--epochs", type=int, default=1,
                        help="The number of training epochs")
    parser.add_argument("--batch-size", type=int, default=96,
                        help="The batch size to use")
    parser.add_argument("--max-length", type=int, default=512,
                        help="The max length of the tokenized input. The model maximum is 2048")
    parser.add_argument("--pre-tokenize", action="store_true", default=False,
                        help="We should pre tokenize the entire dataset, by default the tokenization is on the fly")
    args = parser.parse_args()

    load_dotenv()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Using device: {device}")

    train_texts, eval_texts, train_labels, eval_labels, label_encoder = load_dataset(
        args.samples_per_language, Path(args.encoder_path))
    num_labels = len(label_encoder.classes_)

    model = CanineForSequenceClassification.from_pretrained(
        "google/canine-c", num_labels=num_labels).to(device)
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    if args.pre_tokenize:
        logging.info("Tokenizing dataset...")
        train_tokens = tokenize_dataset(
            train_texts, tokenizer, args.max_length)
        eval_tokens = tokenize_dataset(eval_texts, tokenizer, args.max_length)

        train_dataset = EncodedOpenLIDDataset(train_tokens, train_labels)
        eval_dataset = EncodedOpenLIDDataset(eval_tokens, eval_labels)
    else:
        train_dataset = OpenLIDDataset(train_texts, train_labels)
        eval_dataset = OpenLIDDataset(eval_texts, eval_labels)

    logging.info("Finetuning...")
    finetune_model(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        pre_tokenize=args.pre_tokenize,
        output_dir=args.model_path,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()
