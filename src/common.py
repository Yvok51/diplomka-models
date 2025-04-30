import random
from collections import defaultdict
from pathlib import Path
import pickle
import os

import torch
import numpy as np
import pandas as pd
from transformers import CanineTokenizer
from transformers.integrations import WandbCallback
import tqdm

PROJECT_PATH = Path(__file__).parent.parent.resolve()

def get_tokenized_inputs_path(max_length):
    return PROJECT_PATH / "trainer_output" / f"tokenized_inputs_{max_length}.pkl"

class OpenLIDDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.texts = np.asarray(texts)
        self.labels = torch.tensor(labels, dtype=torch.long).to(self.device)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def random_subset(self, n=1):
        indices = np.asarray([np.random.randint(0, len(self.texts)) for _ in range(n)])
        texts = self.texts[indices]
        labels = self.labels[indices]

        return OpenLIDDataset(texts, labels)


class EncodedOpenLIDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long).to(self.device)

    def __getitem__(self, idx):
        return {"encodings": {k: v.to(self.device) for k, v in self.encodings[idx].items()}, "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

def create_language_dict(texts: list[str], labels: list[str]):
    languages: defaultdict[str, list[str]] = defaultdict(lambda: [])
    for idx, label in enumerate(labels):
        if isinstance(texts[idx], str):
            languages[label].append(texts[idx])

    return languages


def sample_dataset(languages: dict[str, list[str]], samples_per_language: int):
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


def tokenize_input(texts: list[str], tokenizer: CanineTokenizer, max_length=2048):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors='pt')

def tokenize_dataset(texts, tokenizer: CanineTokenizer, max_length=2048):
    if os.path.exists(get_tokenized_inputs_path(max_length)):
        tokenized = load_object(get_tokenized_inputs_path(max_length))
    else:
        tokenized = [tokenize_input([text], tokenizer, max_length) for text in tqdm.tqdm(texts) if isinstance(text, str)]
        save_object(tokenized, get_tokenized_inputs_path(max_length))

    return tokenized


class OnTheFlyTokenizationCollator:
    def __init__(self, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        texts = [feature["text"] for feature in features]
        labels = [feature["label"] for feature in features]

        batch_encodings = tokenize_input(
            texts, self.tokenizer, self.max_length)

        batch_encodings["labels"] = torch.stack(labels, dim=0)

        return batch_encodings


class ConcatenateEncodingCollator:
    def __init__(self, max_length):
        self.max_length = max_length

    def __call__(self, features):
        encodings = [feature["encodings"] for feature in features]
        labels = [feature["label"] for feature in features]

        maximum = max((x["input_ids"].shape[1] for x in encodings))
        to_pad = [maximum - x["input_ids"].shape[1] for x in encodings]

        batch = {}
        for k in encodings[0].keys():
            batch[k] = ConcatenateEncodingCollator.concatenate([encoding[k] for encoding in encodings], to_pad)

        batch["labels"] = torch.stack(labels, dim=0)

        return batch

    @classmethod
    def concatenate(self, tensors, to_pad):
        padded = [torch.nn.functional.pad(tensor, (0, pad)) for tensor, pad in zip(tensors, to_pad)]
        return torch.cat(padded)


def save_object(obj, path: Path):
    os.makedirs(path.parent, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_object(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


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
            predictions = self.predict(self.sample_dataset, self.model, self.tokenizer, self.label_encoder, self.device)

            # add predictions to a wandb.Table
            predictions_df = pd.DataFrame(predictions)
            predictions_df["epoch"] = state.epoch
            records_table = self._wandb.Table(dataframe=predictions_df)
            # log the table to wandb
            self._wandb.log({"sample_predictions": records_table})

