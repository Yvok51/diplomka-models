import random
import logging
from re import compile as re_compile
from unidecode import unidecode

import torch
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


_unicode_chr_splitter = re_compile(
    '(?s)((?:[\ud800-\udbff][\udc00-\udfff])|.)').split


class OpenLIDDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[str], encoder: LabelEncoder):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.texts = texts
        self.labels = labels
        self.encoder = encoder

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": torch.from_numpy(self.encoder.transform([self.labels[idx]])).to(self.device)}

    def __len__(self):
        return len(self.labels)


class EncodedOpenLIDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encodings = encodings
        self.labels = torch.from_numpy(labels).to(self.device)

    def __getitem__(self, idx):
        return {"encodings": {k: v.to(self.device) for k, v in self.encodings[idx].items()}, "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)


class SyntheticOpenLIDDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[str], encoder: MultiLabelBinarizer, synthetic_proportion: float = 1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.texts = texts
        # torch.from_numpy(labels.astype(np.uint8)).to(self.device)
        self.labels = labels
        self.encoder = encoder
        self.synthetic_proportion = synthetic_proportion
        self.synthetic_length = int(self.synthetic_proportion * len(self.labels))
        self.length = len(self.labels) + self.synthetic_length
        self.synthetic = [self.get_multilanguage_instance, self.get_transliterated_instance]

    def __getitem__(self, idx):
        logging.debug("Accessing index: %s", idx)
        if idx < len(self.labels):
            return {
                "text": self.texts[idx],
                "label": self.transform_label(self.labels[idx])
            }
        else:
            per_method = self.synthetic_length / len(self.synthetic)
            idx = (idx - len(self.labels)) // per_method
            # call one of the synthetic methods equally likely
            return self.synthetic[idx](self)

    def __len__(self):
        return self.length

    def get_multilanguage_instance(self):
        num_samples = np.random.randint(2, 3)
        indices = np.random.randint(len(self.labels), size=num_samples)

        label = [self.labels[i] for i in indices]
        label = torch.from_numpy(self.encoder.transform(
            ([l] for l in label))).to(self.device)
        label = torch.clamp(label.sum(dim=0), 0, 1)  # logical and
        # label = torch.clamp(self.labels[indices].sum(dim=0), 0, 1)  # logical and

        final_text = []
        for i in indices:
            text = self.texts[i]

            space_count = text.count(' ')
            if space_count > 0:
                words = text.split()
            else:
                # crude way to split chinese (and other) text
                words = [c for c in _unicode_chr_splitter(text) if c]

            if len(words) > 3:  # Only fragment if enough words
                fragment_size = np.random.randint(
                    max(1, len(words) // num_samples), len(words))
                start_idx = random.randint(0, len(words) - fragment_size)
                words = words[start_idx:start_idx + fragment_size]
            final_text.extend(words)

        return {"text": " ".join(final_text), "label": label}

    def get_transliterated_instance(self):
        idx = np.random.randint(len(self.labels))
        transliterated = unidecode(self.texts[idx])

        return {"text": transliterated, "label": self.transform_label(self.labels[idx])}

    def transform_label(self, label):
        return torch.from_numpy(self.encoder.transform([[label]])[0]).to(self.device)
