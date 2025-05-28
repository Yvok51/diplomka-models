import random

import torch
import numpy as np


class OpenLIDDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.texts = texts
        self.labels = torch.from_numpy(labels).to(self.device)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def random_subset(self, n=1):
        indices = np.asarray(
            [np.random.randint(0, len(self.texts)) for _ in range(n)])
        texts = np.asarray(self.texts)[indices]
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


class SyntheticOpenLIDDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, synthetic_proportion: float = 1):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.texts = texts
        self.labels = torch.from_numpy(labels.astype(np.uint8)).to(self.device)
        self.synthetic_proportion = synthetic_proportion
        self.length = int(len(self.labels) +
                          self.synthetic_proportion * len(self.labels))

    def __getitem__(self, idx):
        if idx < len(self.labels):
            return {"text": self.texts[idx], "label": self.labels[idx]}
        else:
            num_samples = np.random.randint(2, 3)
            indices = np.random.randint(len(self.labels), size=num_samples)

            # label = []
            # for i in indices:
            #     label.append(self.labels[i])
            # label = torch.clamp(torch.tensor(label, dtype=torch.long).to(
            #     self.device).sum(dim=0), 0, 1)  # logical and
            label = torch.clamp(self.labels[indices].sum(dim=0), 0, 1)  # logical and

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

    def random_subset(self, n=1):
        indices = np.asarray(
            [np.random.randint(0, len(self.texts)) for _ in range(n)])
        texts = np.asarray(self.texts)[indices]
        labels = self.labels[indices]

        return SyntheticOpenLIDDataset(texts, labels)
