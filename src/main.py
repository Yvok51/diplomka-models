import os
import logging

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


class OpenLIDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encodings = {key: val.to(self.device)
                          for key, val in encodings.items()}
        self.labels = torch.tensor(labels, dtype=torch.float).to(self.device)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx], dtype=torch.long)
                for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def load_dataset():
    dataset = datasets.load_dataset(
        'laurievb/OpenLID-v2', token=os.environ.get("HUGGINGFACE_TOKEN"),
        features=datasets.Features({ # Present because without it, the function throws an exception
            'text': datasets.Value('string'),
            'language': datasets.Value('string'),
            'source': datasets.Value('string'),
            '__index_level_0__': datasets.Value('int64')
        })
    )

    df = dataset['train']
    # df = df.select(range(1_000))

    logging.info("Encoding the labels...")
    # Encode language labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df['language'])

    logging.info("Splitting dataset...")
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        df['text'],
        encoded_labels,
        test_size=0.1,
        random_state=42
    )

    return train_texts, eval_texts, train_labels, eval_labels, label_encoder


def finetune_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir='./finetuned',
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
):
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

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
        logging_steps=10
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


def main():
    load_dotenv()

    logging.basicConfig(level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_texts, eval_texts, train_labels, eval_labels, label_encoder = load_dataset()
    num_labels = len(label_encoder.classes_)

    model = CanineForSequenceClassification.from_pretrained(
        "google/canine-c", num_labels=num_labels).to(device)
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    logging.info("Tokenizing inputs...")
    train_encodings = tokenize_dataset(train_texts, tokenizer)
    eval_encodings = tokenize_dataset(eval_texts, tokenizer)

    train_dataset = OpenLIDDataset(train_encodings, train_labels)
    eval_dataset = OpenLIDDataset(eval_encodings, eval_labels)

    # labels = torch.tensor([1], dtype=torch.float)
    # inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    # loss = model(**inputs, labels=labels).loss

    logging.info("Finetuning...")
    finetune_model(model, tokenizer, train_dataset, eval_dataset)


if __name__ == "__main__":
    main()
