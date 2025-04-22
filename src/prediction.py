
import torch
import numpy as np

from common import tokenize_input

MULTILABEL_THRESHOLD = 0.5

def get_logits(text, model, tokenizer, device):
    """Get logits from a model"""
    inputs = tokenize_input(text, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs['logits']


def predict_multilabel(text, model, tokenizer, label_encoder, device, threshold=MULTILABEL_THRESHOLD):
    """Predict the language of a given text using a multilabel model."""
    logits = get_logits(text, model, tokenizer, device)
    probabilities = torch.sigmoid(logits).cpu().numpy()[0]
    detected_indices = np.where(probabilities > threshold)[0]

    results = []
    for idx in detected_indices:
        language = label_encoder.inverse_transform([idx])[0]
        confidence = probabilities[idx]
        results.append((language, confidence))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def predict_multiclass(text, model, tokenizer, label_encoder, device):
    """Predict the language of a given text using a multiclass model."""
    logits = get_logits(text, model, tokenizer, device)
    prediction = torch.argmax(logits, dim=-1).cpu().numpy()[0]

    language = label_encoder.inverse_transform([prediction])[0]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    confidence = probs[0][prediction].item()

    return [(language, confidence)]

