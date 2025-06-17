import random
from collections import defaultdict
from pathlib import Path
import pickle
import os
import logging
import math
import glob

import datasets
import torch
from transformers import CanineTokenizer
from sklearn.model_selection import train_test_split
import tqdm

PROJECT_PATH = Path(__file__).parent.parent.resolve()
DATA_PATH = PROJECT_PATH / "trainer_output"


def get_tokenized_inputs_path(max_length):
    return PROJECT_PATH / "trainer_output" / f"tokenized_inputs_{max_length}.pkl"


def create_language_dict(texts: list[str], labels: list[str]):
    languages: defaultdict[str, list[str]] = defaultdict(lambda: [])
    for idx, label in enumerate(labels):
        if isinstance(texts[idx], str):
            languages[label].append(texts[idx])

    return languages


def get_data():
    text_path = DATA_PATH / "text.pkl"
    label_path = DATA_PATH / "label.pkl"

    if not os.path.exists(text_path) or not os.path.exists(label_path):
        dataset = datasets.load_dataset(
            'laurievb/OpenLID-v2', token=os.environ.get("HUGGINGFACE_TOKEN"),
            features=datasets.Features({  # Present because without it, the function throws an exception
                'text': datasets.Value('string'),
                'language': datasets.Value('string'),
                'source': datasets.Value('string'),
                '__index_level_0__': datasets.Value('int64')
            })
        )
        df = dataset["train"]
        del dataset

        # df = df.select(range(1_000_000))
        df = df.filter(lambda d: isinstance(d['text'], str))

        logging.info("Splitting labels and texts...")
        texts, labels = df['text'], df['language']
        save_object(texts, text_path)
        save_object(labels, label_path)

        return texts, labels

    else:
        logging.info("Loading data...")
        return load_object(text_path), load_object(label_path)


def load_dataset(
    samples_count: int | None,
    test_size: float = 0.05
):
    """Load OpenLID dataset"""
    texts, labels = get_data()
    if samples_count:
        texts, labels = sample_dataset(
            create_language_dict(texts, labels), samples_count)

    logging.info("Splitting dataset...")
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts,
        labels,
        test_size=test_size,
    )

    return train_texts, eval_texts, train_labels, eval_labels


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


def flores_to_iso(flores_label: str):
    return str(flores_label[:3])


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in the given directory.

    Args:
        checkpoint_dir: Directory to search for checkpoints

    Returns:
        Path to the latest checkpoint or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        return None

    # Extract step numbers and find the latest one
    def get_step_number(checkpoint_path):
        try:
            return int(os.path.basename(checkpoint_path).split('-')[1])
        except (IndexError, ValueError):
            return 0

    latest_checkpoint = max(checkpoints, key=get_step_number)

    # Verify the checkpoint is valid (contains required files)
    required_files = ['config.json', 'pytorch_model.bin', 'trainer_state.json']
    if all(os.path.exists(os.path.join(latest_checkpoint, f)) for f in required_files):
        return latest_checkpoint
    else:
        logging.warning("Checkpoint %s appears to be incomplete, ignoring", latest_checkpoint)
        return None

def get_checkpoint(no_resume: bool, checkpoint_path: str | None, model_path: str):
    """
    Get the checkpoint from which we start training

    Args:
        no_resume: Start from scratch
        checpoint_path: Specific checkpoint to start from
        model_path: The final path to save the model to

    Returns:
        The path to the checkpoint to start training from or None if we are to start from scratch
    """
    if no_resume:
        return None

    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            return checkpoint_path
        else:
            logging.warning("Specified checkpoint path does not exist: %s", checkpoint_path)
            return None

    return find_latest_checkpoint(model_path)


