import os
import logging
import pickle
from pathlib import Path

from dotenv import load_dotenv
from transformers import (
    CanineTokenizer,
    CanineForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import datasets
import evaluate
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_PATH = Path(__file__).parent.parent.resolve()
ENCODER_PATH = PROJECT_PATH / "trainer_output" / "label_encoder.pkl"


class OpenLIDDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long).to(self.device)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)


def load_dataset():
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
    # df = df.select(range(100_000))

    logging.info("Encoding the labels...")
    # Encode language labels
    if os.path.exists(ENCODER_PATH):
        label_encoder = load_label_encoder(ENCODER_PATH)
        encoded_labels = label_encoder.transform(df['language'])
    else:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(df['language'])
        save_label_encoder(label_encoder, ENCODER_PATH)

    logging.info("Splitting dataset...")
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        df['text'],
        encoded_labels,
        test_size=0.1,
        random_state=42
    )

    return train_texts, eval_texts, train_labels, eval_labels, label_encoder


class OnTheFlyTokenizationCollator:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        # Extract texts and labels
        texts = [feature["text"] for feature in features]
        labels = [feature["label"] for feature in features]

        # Tokenize the texts
        batch_encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Add labels
        batch_encodings["labels"] = torch.stack(labels, dim=0)

        return batch_encodings


def finetune_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir='./finetuned',
    learning_rate=5e-5,
    per_device_train_batch_size=24,
    num_train_epochs=3,
    weight_decay=0.01,
):
    collator = OnTheFlyTokenizationCollator(tokenizer=tokenizer)

    training_args = TrainingArguments(
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        save_total_limit=3,
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=10,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
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


def tokenize_dataset(texts, tokenizer: CanineTokenizer, max_length=2048):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

def save_label_encoder(label_encoder: LabelEncoder, path: Path):
    os.makedirs(path.parent, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(label_encoder, f)

def load_label_encoder(path) -> LabelEncoder:
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Using device: {device}")

    train_texts, eval_texts, train_labels, eval_labels, label_encoder = load_dataset()
    num_labels = len(label_encoder.classes_)

    model = CanineForSequenceClassification.from_pretrained(
        "google/canine-c", num_labels=num_labels).to(device)
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    # logging.info("Tokenizing inputs...")
    # train_encodings = tokenize_dataset(train_texts, tokenizer)
    # eval_encodings = tokenize_dataset(eval_texts, tokenizer)

    train_dataset = OpenLIDDataset(train_texts, train_labels, tokenizer)
    eval_dataset = OpenLIDDataset(eval_texts, eval_labels, tokenizer)

    # labels = torch.tensor([1], dtype=torch.float)
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # loss = model(**inputs, labels=labels).loss

    logging.info("Finetuning...")
    finetune_model(model, tokenizer, train_dataset, eval_dataset)


if __name__ == "__main__":
    main()
