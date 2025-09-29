"""Train multilabel model"""

import os
import logging
import argparse
from typing import Tuple

from dotenv import load_dotenv
from transformers import (
    CanineTokenizer,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)
import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

from components.common import (
    PROJECT_PATH,
    MODELS,
    ModelTypeT,
    load_dataset,
    load_object,
    save_object,
    compute_eval_steps,
    get_checkpoint
)
from components.models import CanineForMultiLabelClassification, CanineForMultiLabelClassificationConfig, LangIDMultiLabelClassification, LangIDMultiLabelClassificationConfig
from prediction import predict_multilabel
from LID_datasets import SyntheticOpenLIDDataset
from collators import OnTheFlyTokenizationCollator

ENCODER_PATH = PROJECT_PATH / "trainer_output" / "multilabel_encoder.pkl"
MODEL_PATH = PROJECT_PATH / "finetuned_multilabel"

SAMPLES_PER_LANGUAGE = 10_000
SYNTHETIC_LANGUAGE_SENTENCE_COUNT_CUTOFF = 100

EVAL_PHASES = 200
EVAL_STEPS = 200_000
LOG_STEPS = 100


def predict(dataset, model, tokenizer, encoder, device):
    predictions = []
    labels = []
    for text, label in dataset:
        results = predict_multilabel(text, model, tokenizer, encoder, device)
        languages, _ = zip(*results) if len(results) > 0 else [], []
        predictions.append(languages)

        text_label = encoder.inverse_transform([label])[0]
        labels.append(text_label)

    return {"labels": labels, "predictions": predictions}


def encode_multilabel(labels: list[str], encoder_path: str):
    """Encode the labels into a multilabel setup"""
    if os.path.exists(encoder_path):
        mlb = load_object(encoder_path)
        assert isinstance(mlb, MultiLabelBinarizer)
        encoded_labels = mlb.transform(([label] for label in labels))
    else:
        mlb = MultiLabelBinarizer()
        encoded_labels = mlb.fit_transform(([label] for label in labels))
        save_object(mlb, ENCODER_PATH)

    return encoded_labels, mlb


def finetune_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    device,
    resume_from_checkpoint,
    output_dir='./finetuned',
    learning_rate=5e-5,
    batch_size=16,
    num_train_epochs=3,
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
        metric_for_best_model="f1_macro",  # Use F1 macro for multi-label
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=LOG_STEPS,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="none" if no_report else "wandb",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Apply threshold for binary prediction
        predictions = (logits > 0).astype(np.float32)

        # Calculate metrics
        f1_micro = f1_score(labels, predictions, average='micro')
        f1_macro = f1_score(labels, average='macro', y_pred=predictions)
        precision_micro = precision_score(
            labels, predictions, average='micro', zero_division=0)
        recall_micro = recall_score(
            labels, predictions, average='micro', zero_division=0)

        return {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision": precision_micro,
            "recall": recall_micro
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # trainer.add_callback(progress_callback)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logging.info("Saving the model...")

    trainer.save_model(output_dir)

    return model


def get_multilabel_model(
    model_path: str | None,
    device: str,
    model_type: ModelTypeT,
    config: CanineForMultiLabelClassificationConfig | LangIDMultiLabelClassificationConfig | None = None
) -> Tuple[CanineForMultiLabelClassification | LangIDMultiLabelClassification, CanineTokenizer | AutoTokenizer]:
    if model_type == "canine":
        assert isinstance(
            config, CanineForMultiLabelClassificationConfig) or config is None

        model = CanineForMultiLabelClassification.from_pretrained(
            model_path, config=config) if model_path else CanineForMultiLabelClassification(config)
        tokenizer: CanineTokenizer = CanineTokenizer.from_pretrained(
            "google/canine-c")
    else:
        assert isinstance(
            config, LangIDMultiLabelClassificationConfig) or config is None

        model = LangIDMultiLabelClassification.from_pretrained(
            model_path, config=config) if model_path else LangIDMultiLabelClassification(config)
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(
            MODELS[model_type].type)

    return model.to(device), tokenizer


def load_model_from_checkpoint(
    checkpoint_path,
    device: str,
    model_type: ModelTypeT,
    config: CanineForMultiLabelClassificationConfig | LangIDMultiLabelClassificationConfig
):
    """
    Load model from a specific checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint directory
        config: Model configuration

    Returns:
        Loaded model or None if loading failed
    """
    try:
        logging.info("Loading model from checkpoint: %s", checkpoint_path)
        return get_multilabel_model(checkpoint_path, device, model_type=model_type, config=config)
    except Exception as e:
        logging.error(
            "Failed to load model from checkpoint %s: %s", checkpoint_path, e)
        return None


def get_config(
    mlb: MultiLabelBinarizer,
    negative_sampling: bool,
    model_type: ModelTypeT
) -> LangIDMultiLabelClassificationConfig | CanineForMultiLabelClassificationConfig:
    if model_type == "canine":
        return CanineForMultiLabelClassificationConfig(
            classes=mlb.classes_.tolist(),
            negative_sampling=negative_sampling
        )
    else:
        return LangIDMultiLabelClassificationConfig(model_type, mlb.classes_.tolist(), negative_sampling)


def main():
    parser = argparse.ArgumentParser(
        description="Language prediction using finetuned CANINE model")
    parser.add_argument("--model-type", type=str, choices=list(MODELS.keys()),
                        default="canine", help="The underlying model type to train")
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
    parser.add_argument("--batch-size", type=int, default=64,
                        help="The batch size to use")
    parser.add_argument("--learning-rate", type=float,
                        default=5e-5, help="Starting learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Portion of the training dedicated to warming up the learning rate")
    parser.add_argument("--no-report", default=False, action="store_true",
                        help="Report the learning progress to W&B")
    parser.add_argument("--max-length", type=int, default=512,
                        help="The max length of the tokenized input. The model maximum is 2048")
    parser.add_argument("--synthetic-proportion", type=float,
                        default=1., help="The proportion of synthetic data to use")
    parser.add_argument("--negative-sampling", default=False, action="store_true",
                        help="Whether to use negative sampling based on the languages proximity")
    parser.add_argument("--debug", default=False,
                        action="store_true", help="Print debug information")
    parser.add_argument("--no-resume", default=False, action="store_true",
                        help="Don't attempt to resume from checkpoint, start training from scratch")
    parser.add_argument("--checkpoint-path", type=str, default=None,
                        help="Specific checkpoint path to resume from (overrides automatic detection)")

    subparsers = parser.add_subparsers(dest="command")
    existing_parser = subparsers.add_parser(
        "existing", help="Further finetune an already finetuned model")
    existing_parser.add_argument(
        "path", type=str, help="Path to the directory containing the finetuned model")

    args = parser.parse_args()

    load_dotenv()

    logging.basicConfig(level=logging.INFO if not args.debug else logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    set_seed(args.seed)

    # tracemalloc.start()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Using device: %s", device)

    train_texts, eval_texts, train_labels, eval_labels = load_dataset(
        args.samples_per_language, test_size=0.001)
    mlb = load_object(args.encoder_path)

    train_dataset = SyntheticOpenLIDDataset(
        train_texts, train_labels, mlb, args.synthetic_proportion)
    eval_dataset = SyntheticOpenLIDDataset(
        eval_texts, eval_labels, mlb, args.synthetic_proportion)

    checkpoint_path = get_checkpoint(
        args.no_resume, args.checkpoint_path, args.model_path)

    config = get_config(mlb, args.negative_sampling, args.model_type)

    if checkpoint_path:
        logging.info("Restarting training from checkpoint %s...",
                     checkpoint_path)
        model, tokenizer = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path, device=device, config=config, model_type=args.model_type)

    elif args.command == "existing":
        logging.info("Further finetuning for model %s", args.path)
        model, tokenizer = get_multilabel_model(
            model_path=args.path, device=device, model_type=args.model_type)

    else:
        logging.info("Initializing new model...")
        model, tokenizer = get_multilabel_model(
            model_path=None, device=device, model_type=args.model_type, config=config)

    if model is None:
        raise RuntimeError("Unable to load model. Shutting down.")

    # display_top(tracemalloc.take_snapshot(), limit=10)

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
