import os
import logging
import argparse

from dotenv import load_dotenv
from transformers import (
    CanineTokenizer,
    CanineForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
import evaluate
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from common import (
    load_dataset,
    save_object,
    load_object,
    PROJECT_PATH,
    compute_eval_steps,
    get_checkpoint,
)
from LID_datasets import OpenLIDDataset
from collators import OnTheFlyTokenizationCollator

ENCODER_PATH = PROJECT_PATH / "trainer_output" / "label_encoder.pkl"
MODEL_PATH = PROJECT_PATH / "finetuned"  # Default path to your finetuned model

SAMPLES_PER_LANGUAGE = 10_000

EVAL_PHASES = 200
EVAL_STEPS = 200_000
LOG_STEPS = 100


def encode_multiclass(labels: list[str], encoder_path: str):
    if os.path.exists(encoder_path):
        label_encoder: LabelEncoder = load_object(encoder_path)
        assert isinstance(label_encoder, LabelEncoder)
        encoded_labels = label_encoder.transform(labels)
    else:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        save_object(label_encoder, encoder_path)

    return encoded_labels, label_encoder


def finetune_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    device,
    resume_from_checkpoint,
    output_dir='./finetuned',
    learning_rate=5e-5,
    batch_size=24,
    num_train_epochs=1,
    weight_decay=0.01,
    max_length=2048,
    warmup_ratio=0.0,
    no_report=False,
):
    collator = OnTheFlyTokenizationCollator(
        tokenizer=tokenizer, max_length=max_length, device=device)

    eval_steps = compute_eval_steps(
        train_dataset, batch_size, num_train_epochs, EVAL_PHASES)
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=LOG_STEPS,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none" if no_report else "wandb",
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

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logging.info("Saving the model...")

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
    parser.add_argument("--learning-rate", type=float,
                        default=5e-5, help="Starting learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.0,
                        help="Portion of the training dedicated to warming up the learning rate")
    parser.add_argument("--no-report", default=False, action="store_true", help="Report the learning progress to W&B")
    parser.add_argument("--max-length", type=int, default=512,
                        help="The max length of the tokenized input. The model maximum is 2048")
    parser.add_argument("--no-resume", default=False, action="store_true",
                        help="Don't attempt to resume from checkpoint, start training from scratch")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Specific checkpoint path to resume from (overrides automatic detection)")

    subparsers = parser.add_subparsers()
    existing_parser = subparsers.add_parser(
        "existing", help="Further finetuned an already finetuned model")
    existing_parser.add_argument(
        "path", type=str, help="Path to the directory containing the finetuned model")

    args = parser.parse_args()

    load_dotenv()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Using device: %s", device)

    train_texts, eval_texts, train_labels, eval_labels, = load_dataset(
        args.samples_per_language, test_size=0.001)
    label_encoder = load_object(args.encoder_path)
    num_labels = len(label_encoder.classes_)

    train_dataset = OpenLIDDataset(train_texts, train_labels, label_encoder)
    eval_dataset = OpenLIDDataset(eval_texts, eval_labels, label_encoder)

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    checkpoint_path = get_checkpoint(
        args.no_resume, args.checkpoint_path, args.model_path)

    if args.command == "existing":
        model = CanineForSequenceClassification.from_pretrained(
            args.path, num_labels=num_labels).to(device)

    else:
        if checkpoint_path is None:
            logging.info("Initializing new model...")
            model = CanineForSequenceClassification.from_pretrained(
                "google/canine-c", num_labels=num_labels).to(device)
        else:
            model = CanineForSequenceClassification.from_pretrained(
                checkpoint_path, num_labels=num_labels).to(device)

    if model is None:
        raise RuntimeError("Unable to load model. Shutting down.")

    logging.info("Finetuning...")
    finetune_model(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        device=device,
        resume_from_checkpoint=checkpoint_path,
        output_dir=args.model_path,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        no_report=args.no_report
    )


if __name__ == "__main__":
    main()
