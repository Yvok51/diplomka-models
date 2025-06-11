import random
from collections import defaultdict
from pathlib import Path
import pickle
import os
import logging
from typing import Callable
import math

import pandas as pd
import datasets
import torch
from transformers import CanineTokenizer
from transformers.integrations import WandbCallback
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
import tqdm

from LID_datasets import OpenLIDDataset

PROJECT_PATH = Path(__file__).parent.parent.resolve()


def get_tokenized_inputs_path(max_length):
    return PROJECT_PATH / "trainer_output" / f"tokenized_inputs_{max_length}.pkl"


def create_language_dict(texts: list[str], labels: list[str]):
    languages: defaultdict[str, list[str]] = defaultdict(lambda: [])
    for idx, label in enumerate(labels):
        if isinstance(texts[idx], str):
            languages[label].append(texts[idx])

    return languages


def load_dataset(
    samples_count: int | None,
    encoder_path: Path,
    encode_labels: Callable[[list[str], str], tuple[torch.tensor, MultiLabelBinarizer | LabelEncoder]],
    test_size: float = 0.05
):
    """Load OpenLID dataset"""
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
    del dataset

    # df = df.select(range(3_000_000))
    df = df.filter(lambda d: isinstance(d['text'], str))

    logging.info("Splitting labels and texts...")
    if samples_count:
        texts, labels = sample_dataset(create_language_dict(
            df['text'], df['language']), samples_count)
    else:
        texts, labels = df['text'], df['language']
    del df

    logging.info("Encoding the labels...")
    # Encode language labels
    encoded_labels, encoder = encode_labels(labels, encoder_path)
    del labels

    logging.info("Splitting dataset...")
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts,
        encoded_labels,
        test_size=test_size,
        shuffle=False # Lower memory usage
    )

    return train_texts, eval_texts, train_labels, eval_labels, encoder


def sample_dataset(languages: dict[str, list[str]], samples_per_language: int):
    logging.info("Sampling %s samples per language", samples_per_language)
    new_texts = []
    new_labels = []
    for language, lang_texts in languages.items():
        if len(lang_texts) <= samples_per_language:
            new_texts.extend(lang_texts)
            new_labels.extend([language] * len(lang_texts))
        else:
            new_texts.extend(random.sample(lang_texts, k=samples_per_language))
            new_labels.extend([language] * samples_per_language)

    return new_texts, new_labels


def tokenize_input(texts: list[str], tokenizer: CanineTokenizer, max_length=512):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')


def tokenize_dataset(texts, tokenizer: CanineTokenizer, max_length=512):
    if os.path.exists(get_tokenized_inputs_path(max_length)):
        tokenized = load_object(get_tokenized_inputs_path(max_length))
    else:
        tokenized = [tokenize_input([text], tokenizer, max_length)
                     for text in tqdm.tqdm(texts) if isinstance(text, str)]
        save_object(tokenized, get_tokenized_inputs_path(max_length))

    return tokenized


def save_object(obj, path: Path):
    os.makedirs(path.parent, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_object(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_eval_steps(dataset: torch.utils.data.Dataset, batch_size, epochs, evals):
    steps = math.ceil(len(dataset) / batch_size) * epochs
    return math.floor(steps / evals)

class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset
          for generating predictions.
        num_samples (int, optional): Number of samples to select from
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, model, tokenizer, label_encoder, predict, val_dataset: OpenLIDDataset, device, num_samples=100, freq=2):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated
              with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from
              the validation dataset for generating predictions.
              Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.sample_dataset = val_dataset.random_subset(num_samples)
        self.predict = predict
        self.device = device
        self.freq = freq

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        # control the frequency of logging by logging the predictions
        # every `freq` epochs
        if state.epoch % self.freq == 0:
            # generate predictions
            predictions = self.predict(
                self.sample_dataset, self.model, self.tokenizer, self.label_encoder, self.device)

            # add predictions to a wandb.Table
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # log the table to wandb
            self._wandb.log({"sample_predictions": records_table})


def flores_to_iso(flores_label: str):
    return str(flores_label[:3])
