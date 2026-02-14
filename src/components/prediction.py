import logging

import torch
import numpy as np
from transformers import CanineTokenizer, PreTrainedModel, PreTrainedTokenizer
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from .common import tokenize_input

MULTILABEL_THRESHOLD = 0.5


def get_logits(text: str, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: str) -> torch.Tensor:
    """Get logits from a model"""
    inputs = tokenize_input(text, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs['logits']


def predict_multilabel(
    text: str,
    model: PreTrainedModel,
    tokenizer: CanineTokenizer,
    label_encoder: MultiLabelBinarizer,
    device: str,
    threshold: float = MULTILABEL_THRESHOLD
) -> list[tuple[str, float]]:
    """Predict the language of a given text using a multilabel model."""
    try:
        logits = get_logits(text, model, tokenizer, device)
        probabilities: np.ndarray = torch.sigmoid(logits).cpu().numpy()[0]
        predicted_labels: np.ndarray = (probabilities > threshold)

        labels: np.ndarray = label_encoder.inverse_transform(
            np.asarray([predicted_labels.astype(int)]))[0]
        confidences = probabilities[predicted_labels]
        results = list(zip(labels, confidences))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
    except RuntimeError as e:
        # Sometimes the models seem to have trouble with single emoji length texts
        # Return empty prediction as the most logical option (predicting the input is not text)
        logging.warning("Model crashed on trying to classify text: '%s', message: '%s'", text, repr(e))
        return []


def predict_multiclass(
    text: str,
    model: PreTrainedModel,
    tokenizer: CanineTokenizer,
    label_encoder: LabelEncoder,
    device: str
) -> list[tuple[str, float]]: # always single item
    """Predict the language of a given text using a multiclass model."""
    logits = get_logits(text, model, tokenizer, device)
    prediction: np.ndarray = torch.argmax(logits, dim=-1).cpu().numpy()[0]

    language: str = label_encoder.inverse_transform([prediction])[0]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probs[0][prediction].item()

    return [(language, confidence)]
