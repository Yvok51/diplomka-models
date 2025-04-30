import os
import logging
from collections import defaultdict
import random
import argparse
from pathlib import Path

from dotenv import load_dotenv
from transformers import (
    CanineTokenizer,
    CanineModel,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PretrainedConfig,
    set_seed,
)
import datasets
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

from common import (
    PROJECT_PATH,
    sample_dataset,
    load_object,
    save_object,
    OpenLIDDataset,
    OnTheFlyTokenizationCollator,
    WandbPredictionProgressCallback
)
from prediction import predict_multilabel

ENCODER_PATH = PROJECT_PATH / "trainer_output" / "multilabel_encoder.pkl"
MODEL_PATH = PROJECT_PATH / "finetuned_multilabel"

SAMPLES_PER_LANGUAGE = 10_000
SYNTHETIC_LANGUAGE_SENTENCE_COUNT_CUTOFF = 100

EVAL_STEPS = 3_000
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


class CanineForMultiLabelClassificationConfig(PretrainedConfig):
    model_type = "CanineMultiLabelClassifier"
    num_labels = 2

    def __init__(self, num_labels=2, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels


class CanineForMultiLabelClassification(PreTrainedModel):
    config_class = CanineForMultiLabelClassificationConfig

    def __init__(self, config: CanineForMultiLabelClassificationConfig):
        super().__init__(config)
        self.config = config
        self.canine = CanineModel.from_pretrained("google/canine-c")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.canine.config.hidden_size, self.config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Quick fix for bug in transformers package
        kwargs.pop("num_items_in_batch", None)
        outputs = self.canine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
            loss = loss_fct(logits, labels.float())

            return {"loss": loss, "logits": logits}

        return {"logits": logits}


def prepare_multilabel_dataset(sample_count: int, dataset_path=None):
    """
    Prepare a multi-label dataset by:
    1. Loading original data
    2. Creating synthetic multi-language samples
    3. Encoding with MultiLabelBinarizer
    """
    logging.info("Loading dataset...")

    dataset = datasets.load_dataset(
        'laurievb/OpenLID-v2', token=os.environ.get("HUGGINGFACE_TOKEN"),
        features=datasets.Features({
            'text': datasets.Value('string'),
            'language': datasets.Value('string'),
            'source': datasets.Value('string'),
            '__index_level_0__': datasets.Value('int64')
        })
    )

    df = dataset['train']
    # df = df.select(range(10_000))

    texts_original = df['text']
    labels_original = df['language']

    if sample_count:
        texts_single, labels_single = sample_dataset(
            texts_original, labels_original, sample_count)
    else:
        texts_single, labels_single = texts_original, labels_original

    texts_multi, labels_multi = create_synthetic_data(
        texts_original, labels_original, len(texts_single))

    # Combine single-language and multi-language samples
    labels_single_as_lists = [[label] for label in labels_single]

    all_texts = texts_single + texts_multi
    all_labels = labels_single_as_lists + labels_multi

    # Encode multi-labels
    if os.path.exists(ENCODER_PATH):
        mlb = load_object(ENCODER_PATH)
        encoded_labels = mlb.transform(all_labels)
    else:
        mlb = MultiLabelBinarizer()
        encoded_labels = mlb.fit_transform(all_labels)
        save_object(mlb, ENCODER_PATH)

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        all_texts,
        encoded_labels,
        test_size=0.05,
    )

    return train_texts, eval_texts, train_labels, eval_labels, mlb


def create_synthetic_data(texts: list[str], labels: list, sample_count: int):
    texts_multi = []
    labels_multi = []

    # Group texts by language
    language_texts = defaultdict(list)
    for idx, label in enumerate(labels):
        language_texts[label].append(texts[idx])

    # Sample languages that have enough samples
    viable_languages = [lang for lang,
                        texts in language_texts.items() if len(texts) >= SYNTHETIC_LANGUAGE_SENTENCE_COUNT_CUTOFF]

    logging.info(
        "Creating %s synthetic multi-language samples...", sample_count)

    for _ in range(sample_count):
        # Select 2-3 random languages
        num_langs = random.randint(2, 3)
        selected_langs = random.sample(viable_languages, num_langs)

        # Sample a text from each language
        sample_texts = []
        for lang in selected_langs:
            text = random.choice(language_texts[lang])
            # Take a random fragment (50-100% of original)
            words = text.split()
            if len(words) > 3:  # Only fragment if enough words
                fragment_size = random.randint(
                    max(1, len(words)//2), len(words))
                start_idx = random.randint(0, len(words) - fragment_size)
                text = ' '.join(words[start_idx:start_idx + fragment_size])
            sample_texts.append(text)

        # Combine texts
        combined_text = ' '.join(sample_texts)
        texts_multi.append(combined_text)
        labels_multi.append(selected_langs)

    return texts_multi, labels_multi


def finetune_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    label_encoder,
    device,
    output_dir='./finetuned',
    learning_rate=5e-5,
    batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    max_length=2048
):
    collator = OnTheFlyTokenizationCollator(
        tokenizer=tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
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
        report_to="wandb",
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

    progress_callback = WandbPredictionProgressCallback(
        model=model,
        label_encoder=label_encoder,
        tokenizer=tokenizer,
        predict=predict,
        val_dataset=eval_dataset,
        device=device,
        num_samples=10,
        freq=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics
    )

    # trainer.add_callback(progress_callback)

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
    parser.add_argument("--batch-size", type=int, default=128,
                        help="The batch size to use")
    parser.add_argument("--max-length", type=int, default=512,
                        help="The max length of the tokenized input. The model maximum is 2048")
    args = parser.parse_args()

    load_dotenv()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Using device: %s", device)

    train_texts, eval_texts, train_labels, eval_labels, mlb = prepare_multilabel_dataset(
        args.samples_per_language, Path(args.encoder_path))
    num_labels = len(mlb.classes_)

    model = CanineForMultiLabelClassification(
        CanineForMultiLabelClassificationConfig(num_labels=num_labels)).to(device)
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    train_dataset = OpenLIDDataset(train_texts, train_labels)
    eval_dataset = OpenLIDDataset(eval_texts, eval_labels)

    logging.info("Finetuning...")
    finetune_model(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        device=device,
        label_encoder=mlb,
        output_dir=args.model_path,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()
