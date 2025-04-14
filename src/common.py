import random
from collections import defaultdict
from pathlib import Path
import pickle
import os

import torch
from transformers import CanineTokenizer
import tqdm

PROJECT_PATH = Path(__file__).parent.parent.resolve()

def get_tokenized_inputs_path(max_length):
    return PROJECT_PATH / "trainer_output" / f"tokenized_inputs_{max_length}.pkl"

class OpenLIDDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long).to(self.device)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)


class EncodedOpenLIDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long).to(self.device)

    def __getitem__(self, idx):
        return {"encodings": {k: v.to(self.device) for k, v in self.encodings[idx].items()}, "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)


def sample_dataset(texts: list[str], labels: list[str], samples_per_language: int):
    languages: defaultdict[str, list[str]] = defaultdict(lambda: [])
    for idx, label in enumerate(labels):
        if isinstance(texts[idx], str):
            languages[label].append(texts[idx])

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
