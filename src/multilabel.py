import os
import logging
import random
import argparse
from pathlib import Path

from dotenv import load_dotenv
from transformers import (
    CanineTokenizer,
    CanineModel,
    TrainingArguments,
    Trainer,
    PreTrainedModel,
    PretrainedConfig,
    set_seed,
)
import numpy as np
import torch
import torch.nn as nn
import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, minmax_scale
from sklearn.metrics import f1_score, precision_score, recall_score
import lang2vec.lang2vec as l2v

from common import (
    PROJECT_PATH,
    sample_dataset,
    create_language_dict,
    load_object,
    save_object,
    WandbPredictionProgressCallback,
    flores_to_iso,
)
from prediction import predict_multilabel
from LID_datasets import SyntheticOpenLIDDataset
from collators import OnTheFlyTokenizationCollator

ENCODER_PATH = PROJECT_PATH / "trainer_output" / "multilabel_encoder.pkl"
MODEL_PATH = PROJECT_PATH / "finetuned_multilabel"

SAMPLES_PER_LANGUAGE = 10_000
SYNTHETIC_LANGUAGE_SENTENCE_COUNT_CUTOFF = 100

EVAL_STEPS = 5_000
LOG_STEPS = 100


def normalize_by_row(matrix: np.ndarray):
    """Normalize each row of a matrix to range 0 - 1"""
    return minmax_scale(matrix, (0, 1), axis=1, copy=True)
    # return np.nan_to_num(matrix / np.max(np.abs(matrix), axis=1)[:, None])


class NegativeSamplingBCELoss(nn.Module):

    @classmethod
    def calculate_similarity(cls, classes: list[str]):
        """
        Calculate the interclass similarity of the various languages.
        By default use the learned lang2vec vector representations and use the language family information as backup

        Args:
            classes: The languages to calculate the interclass similarity of
        """
        iso_names = np.asarray([flores_to_iso(c) for c in classes])

        learned_mask = np.asarray(
            [lang in l2v.available_learned_languages() for lang in iso_names])
        learned_feat = l2v.get_features(
            list(iso_names[learned_mask]), "learned")
        # Get the learned vector representations of languages and set the vectors to all zero if they are not available
        learned = np.asarray([learned_feat[lang] if lang in learned_feat else np.zeros(
            (512,)) for lang in iso_names])
        # Dot product of each language vector with all others
        learned_mask_mat = learned_mask @ learned_mask.T
        learned_mat = normalize_by_row(learned @ learned.T)

        family_feat = l2v.get_features(list(iso_names), "fam")
        family = np.asarray([family_feat[lang] for lang in iso_names])
        # Dot product of each language vector with all others
        family_mat = normalize_by_row(family @ family.T)

        # Use the learned similarity with the family similarity as backup
        return np.where(learned_mask_mat, learned_mat, family_mat)

    def __init__(self, classes, device="cuda" if torch.cuda.is_available() else "cpu", neg_sample_ratio=5.0):
        """
        Initialize negative sampling loss

        Args:
            num_classes: Total number of classes
            neg_sample_ratio: Ratio of negative samples to positive samples
        """
        super(NegativeSamplingBCELoss, self).__init__()
        self.neg_sample_ratio = neg_sample_ratio
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.similarity = torch.Tensor(
            NegativeSamplingBCELoss.calculate_similarity(classes)).to(device)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits: Model output logits of shape [batch_size, num_classes]
            targets: Binary target labels of shape [batch_size, num_classes]
        """
        positive_mask = targets.eq(1).float()
        negative_mask = targets.eq(0).float()

        num_positives = positive_mask.sum(dim=1, keepdim=True)

        num_negative_samples = torch.floor(torch.clip(
            num_positives, min=1).squeeze() * self.neg_sample_ratio)

        # For each instance, randomly select negative samples
        neg_sample_mask = torch.zeros_like(negative_mask)
        for i in range(logits.size(0)):
            neg_indices = torch.nonzero(negative_mask[i]).squeeze()

            if neg_indices.dim() == 0 and neg_indices.size(0) > 0:
                # Handle case where there's only one negative example
                neg_indices = neg_indices.unsqueeze(0)

            if neg_indices.size(0) > 0:
                samples_to_keep = min(
                    int(num_negative_samples[i].item()), neg_indices.size(0))

                # Average the similarity of the various languages in the instance
                average_similarity = torch.nan_to_num(torch.mean(
                    self.similarity[positive_mask[i].bool()], dim=0))
                # We actually want the least similar languages to be selected more often
                inverse_similarity = 1 - average_similarity

                similarity_negative_samples = inverse_similarity[negative_mask[i].bool(
                )]
                probabilities = similarity_negative_samples / similarity_negative_samples.sum()

                selected_indices = np.random.choice(
                    neg_indices.cpu().numpy(),
                    size=samples_to_keep,
                    replace=False,
                    p=probabilities.cpu().numpy()
                )

                # mark the negative samples to use
                neg_sample_mask[i, selected_indices] = 1.0

        final_mask = positive_mask + neg_sample_mask
        element_wise_loss = self.bce_loss(logits, targets)
        masked_loss = element_wise_loss * final_mask

        return masked_loss.sum() / final_mask.sum()


def predict(dataset, model, tokenizer, encoder, device):
    predictions = []
    labels = []
    for text, label in dataset:
        results = predict_multilabel(text, model, tokenizer, encoder, device)
        languages, _ = zip(*results) if len(results) > 0 else [], []
        predictions.append(languages)

        text_label = encoder.inverse_transform([label])[0]
        labels.append(text_label)

    return {"labels": labels, "predictions": predictions}


class CanineForMultiLabelClassificationConfig(PretrainedConfig):
    model_type = "CanineMultiLabelClassifier"
    classes: list[str] = ()
    negative_sampling = False

    def __init__(self, classes=(), negative_sampling=False, **kwargs):
        super().__init__(**kwargs)
        self.classes = classes
        self.negative_sampling = negative_sampling


class CanineForMultiLabelClassification(PreTrainedModel):
    config_class = CanineForMultiLabelClassificationConfig

    def __init__(self, config: CanineForMultiLabelClassificationConfig):
        super().__init__(config)
        self.config = config
        self.canine = CanineModel.from_pretrained("google/canine-c")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(
            self.canine.config.hidden_size, len(self.config.classes))
        self.loss = NegativeSamplingBCELoss(
            config.classes) if config.negative_sampling else nn.BCEWithLogitsLoss()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Quick fix for bug in transformers package
        kwargs.pop("num_items_in_batch", None)
        outputs = self.canine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss(logits, labels.float())

            return {"loss": loss, "logits": logits}

        return {"logits": logits}


def prepare_multilabel_dataset(sample_count: int, encoder_path=None):
    """
    Prepare a multi-label dataset by:
    1. Loading original data
    2. Creating synthetic multi-language samples
    3. Encoding with MultiLabelBinarizer
    """
    logging.info("Loading dataset...")

    dataset = datasets.load_dataset(
        'laurievb/OpenLID-v2', token=os.environ.get("HUGGINGFACE_TOKEN"),
        features=datasets.Features({
            'text': datasets.Value('string'),
            'language': datasets.Value('string'),
            'source': datasets.Value('string'),
            '__index_level_0__': datasets.Value('int64')
        })
    )

    df = dataset['train']
    # df = df.select(range(10_000))

    texts_original = df['text']
    labels_original = df['language']

    if sample_count:
        texts_single, labels_single = sample_dataset(
            create_language_dict(texts_original, labels_original), sample_count)
    else:
        texts_single, labels_single = texts_original, labels_original

    # Encode multi-labels
    if os.path.exists(encoder_path):
        mlb = load_object(encoder_path)
        assert isinstance(mlb, MultiLabelBinarizer)
        encoded_labels = mlb.transform(labels_single)
    else:
        mlb = MultiLabelBinarizer()
        encoded_labels = mlb.fit_transform(labels_single)
        save_object(mlb, ENCODER_PATH)

    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts_single,
        encoded_labels,
        test_size=0.05,
    )

    return train_texts, eval_texts, train_labels, eval_labels, mlb


def create_synthetic_data(languages: dict[str, list[str]], sample_count: int):
    texts_multi = []
    labels_multi = []

    # Sample languages that have enough samples
    viable_languages = [lang for lang,
                        texts in languages.items() if len(texts) >= SYNTHETIC_LANGUAGE_SENTENCE_COUNT_CUTOFF]

    logging.info(
        "Creating %s synthetic multi-language samples...", sample_count)

    for _ in range(sample_count):
        # Select random languages
        num_langs = random.randint(2, 3)
        selected_langs = random.sample(viable_languages, num_langs)

        # Sample a text from each language
        sample_texts = []
        for lang in selected_langs:
            text = random.choice(languages[lang])
            # Take a random fragment (50-100% of original)
            words = text.split()
            if len(words) > 3:  # Only fragment if enough words
                fragment_size = random.randint(
                    max(1, len(words)//2), len(words))
                start_idx = random.randint(0, len(words) - fragment_size)
                words = words[start_idx:start_idx + fragment_size]
            sample_texts.extend(words)

        combined_text = ' '.join(sample_texts)
        texts_multi.append(combined_text)
        labels_multi.append(selected_langs)

    return texts_multi, labels_multi


def finetune_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    label_encoder,
    device,
    output_dir='./finetuned',
    learning_rate=5e-5,
    batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    max_length=2048
):
    collator = OnTheFlyTokenizationCollator(
        tokenizer=tokenizer, max_length=max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=num_train_epochs,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",  # Use F1 macro for multi-label
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=LOG_STEPS,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        report_to="wandb",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Apply threshold for binary prediction
        predictions = (logits > 0).astype(np.float32)

        # Calculate metrics
        f1_micro = f1_score(labels, predictions, average='micro')
        f1_macro = f1_score(labels, average='macro', y_pred=predictions)
        precision_micro = precision_score(
            labels, predictions, average='micro', zero_division=0)
        recall_micro = recall_score(
            labels, predictions, average='micro', zero_division=0)

        return {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
            "precision": precision_micro,
            "recall": recall_micro
        }

    # pylint: disable=unused-variable
    progress_callback = WandbPredictionProgressCallback(
        model=model,
        label_encoder=label_encoder,
        tokenizer=tokenizer,
        predict=predict,
        val_dataset=eval_dataset,
        device=device,
        num_samples=10,
        freq=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # trainer.add_callback(progress_callback)

    trainer.train()

    logging.info("Saving the model...")

    trainer.save_model(output_dir)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Language prediction using finetuned CANINE model")
    parser.add_argument("--model-path", type=str,
                        default=str(MODEL_PATH), help="Directory to put final the finetuned model")
    parser.add_argument("--encoder-path", type=str,
                        default=str(ENCODER_PATH), help="Path to the label encoder")
    parser.add_argument("--seed", type=int,
                        default=42, help="Path to the label encoder")
    parser.add_argument("--samples-per-language", type=int, default=None,
                        help="The number of samples per language to use")
    parser.add_argument("--epochs", type=int, default=1,
                        help="The number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="The batch size to use")
    parser.add_argument("--max-length", type=int, default=512,
                        help="The max length of the tokenized input. The model maximum is 2048")
    parser.add_argument("--synthetic-proportion", type=float,
                        default=1., help="The proportion of synthetic data to use")
    parser.add_argument("--negative-sampling", default=False, action="store_true",
                        help="Whether to use negative sampling based on the languages proximity")
    args = parser.parse_args()

    load_dotenv()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info("Using device: %s", device)

    train_texts, eval_texts, train_labels, eval_labels, mlb = prepare_multilabel_dataset(
        args.samples_per_language, Path(args.encoder_path))

    model = CanineForMultiLabelClassification(
        CanineForMultiLabelClassificationConfig(
            classes=mlb.classes_.tolist(),
            negative_sampling=args.negative_sampling
        )
    ).to(device)
    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")

    train_dataset = SyntheticOpenLIDDataset(
        train_texts, train_labels, args.synthetic_proportion)
    eval_dataset = SyntheticOpenLIDDataset(
        eval_texts, eval_labels, args.synthetic_proportion)

    logging.info("Finetuning...")
    finetune_model(
        model,
        tokenizer,
        train_dataset,
        eval_dataset,
        device=device,
        label_encoder=mlb,
        output_dir=args.model_path,
        num_train_epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length
    )


if __name__ == "__main__":
    main()
