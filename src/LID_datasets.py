import random
import logging

import torch
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder


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
        self.length = int(len(self.labels) +
                          self.synthetic_proportion * len(self.labels))

    def __getitem__(self, idx):
        logging.debug("Accessing index: %s", idx)
        if idx < len(self.labels):
            return {
                "text": self.texts[idx],
                "label": torch.from_numpy(self.encoder.transform([[self.labels[idx]]])[0]).to(self.device)
            }
        else:
            num_samples = np.random.randint(2, 3)
            indices = np.random.randint(len(self.labels), size=num_samples)

            label = []
            for i in indices:
                label.append(self.labels[i])
            label = torch.from_numpy(self.encoder.transform(
                ([l] for l in label))).to(self.device)
            label = torch.clamp(label.sum(dim=0), 0, 1)  # logical and
            # label = torch.clamp(self.labels[indices].sum(dim=0), 0, 1)  # logical and

            final_text = []
            for i in indices:
                text = self.texts[i]
                words = text.split()
                if len(words) > 3:  # Only fragment if enough words
                    fragment_size = np.random.randint(
                        max(1, len(words) // num_samples), len(words))
                    start_idx = random.randint(0, len(words) - fragment_size)
                    words = words[start_idx:start_idx + fragment_size]
                final_text.extend(words)

            return {"text": " ".join(final_text), "label": label}

    def __len__(self):
        return self.length
