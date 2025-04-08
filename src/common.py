import random
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_PATH = Path(__file__).parent.parent.resolve()

class OpenLIDDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long).to(self.device)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

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

def save_label_encoder(label_encoder: LabelEncoder, path: Path):
    os.makedirs(path.parent, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(label_encoder, f)


def load_label_encoder(path) -> LabelEncoder:
    with open(path, 'rb') as f:
        return pickle.load(f)

