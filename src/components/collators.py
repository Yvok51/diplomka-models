import torch

from .common import tokenize_input


class OnTheFlyTokenizationCollator:
    def __init__(self, tokenizer, max_length=2048, device="cpu"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device


    def __call__(self, features):
        texts = [feature["text"] for feature in features]
        labels = [feature["label"] for feature in features]

        batch_encodings = tokenize_input(
            texts, self.tokenizer, self.max_length).to(self.device)

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
            batch[k] = ConcatenateEncodingCollator.concatenate(
                [encoding[k] for encoding in encodings], to_pad)

        batch["labels"] = torch.stack(labels, dim=0)

        return batch

    @classmethod
    def concatenate(self, tensors, to_pad):
        padded = [torch.nn.functional.pad(
            tensor, (0, pad)) for tensor, pad in zip(tensors, to_pad)]
        return torch.cat(padded)
