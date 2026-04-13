import pickle
import json
import os
import datetime
import argparse
import logging

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
)

from .common import load_dataset, PROJECT_PATH


class NLIClassifier:
    """
    Native Language Identification using sklearn's Logistic Regression
    """

    def __init__(self, ngram_size=6, C=100.0):
        """
        Args:
            ngram_size: Size of character n-grams (paper uses 6)
            C: Regularization parameter (paper uses 100.0)
        """
        self.ngram_size = ngram_size
        self.C = C

        # TF-IDF vectorizer with character n-grams
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(ngram_size, ngram_size),
            sublinear_tf=True,  # Uses 1+log(tf) as per paper
            norm='l2',          # L2 normalization as per paper
            min_df=5,           # Remove features appearing in < 5 docs
            max_df=0.5,         # Remove features appearing in > 50% docs
            lowercase=True
        )

        # Logistic Regression with LIBLINEAR solver
        self.classifier = LogisticRegression(
            C=C,
            solver='liblinear',
            multi_class='ovr',
            max_iter=1000,
            random_state=42
        )

        self.languages = None
        self.is_trained = False

    def fit(self, texts: list[str], labels: list[str]):
        """Train the model"""
        logging.debug("Extracting %s-character n-grams and computing TF-IDF...", self.ngram_size)
        X = self.vectorizer.fit_transform(texts)
        logging.debug("Feature dimension: %s", X.shape[1])

        self.languages = sorted(set(labels))
        logging.info("Training logistic regression...")

        self.classifier.fit(X, labels)
        self.is_trained = True

        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict native languages"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Get probability distributions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def predict_single(self, text: str) -> dict:
        """
        Predict native language for a single text with detailed output

        Returns:
            Dict with 'prediction', 'confidence', and 'all_probabilities'
        """
        probas = self.predict_proba([text])[0]
        predicted_idx = np.argmax(probas)
        predicted_lang = self.classifier.classes_[predicted_idx]

        # Create probability dictionary for all languages
        all_probs = {lang: float(prob) for lang, prob in zip(
            self.classifier.classes_, probas)}

        return {
            'prediction': predicted_lang,
            'confidence': float(probas[predicted_idx]),
            'all_probabilities': all_probs
        }

    def evaluate(self, texts: list[str], labels: list[str]) -> dict:
        """
        Evaluate model with detailed metrics

        Returns:
            Dict with accuracy, classification report, and confusion matrix
        """
        predictions = self.predict(texts)
        accuracy = accuracy_score(labels, predictions)

        report = classification_report(labels, predictions, output_dict=True)
        conf_matrix = confusion_matrix(
            labels, predictions, labels=self.languages)

        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }

    def save_model(self, filepath: str):
        """Save model to disk"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'languages': self.languages,
            'ngram_size': self.ngram_size,
            'C': self.C,
            'is_trained': self.is_trained
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to: {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create new instance
        model = cls(ngram_size=model_data['ngram_size'], C=model_data['C'])

        # Restore saved components
        model.vectorizer = model_data['vectorizer']
        model.classifier = model_data['classifier']
        model.languages = model_data['languages']
        model.is_trained = model_data['is_trained']

        print(f"Model loaded from: {filepath}")
        return model


class MultilabelNLIClassifier:
    """
    Multilabel Language Identification using TF-IDF and Logistic Regression.
    Allows predicting multiple languages per text (e.g., for code-mixed texts).
    """

    def __init__(self, ngram_size=6, C=100.0, threshold=0.5):
        """
        Args:
            ngram_size: Size of character n-grams (paper uses 6)
            C: Regularization parameter (paper uses 100.0)
            threshold: Probability threshold for predicting a label
        """
        self.ngram_size = ngram_size
        self.C = C
        self.threshold = threshold

        # TF-IDF vectorizer with character n-grams (same config as multiclass)
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(ngram_size, ngram_size),
            sublinear_tf=True,
            norm='l2',
            min_df=5,
            max_df=0.5,
            lowercase=True
        )

        # OneVsRest classifier with Logistic Regression for multilabel
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                solver='liblinear',
                max_iter=1000,
                random_state=42
            )
        )

        self.mlb = MultiLabelBinarizer()
        self.is_trained = False

    @property
    def languages(self):
        """Get list of languages the model was trained on."""
        if not self.is_trained:
            return None
        return list(self.mlb.classes_)

    def fit(self, texts: list[str], labels: list[list[str]]):
        """
        Train the model.

        Args:
            texts: List of text samples
            labels: List of label lists (each sample can have multiple labels)
        """
        logging.debug("Extracting %s-character n-grams and computing TF-IDF...", self.ngram_size)
        X = self.vectorizer.fit_transform(texts)
        logging.debug("Feature dimension: %s", X.shape[1])

        # Transform labels to binary matrix
        y = self.mlb.fit_transform(labels)
        logging.debug("Number of classes: %s", len(self.mlb.classes_))

        logging.info("Training multilabel logistic regression...")
        self.classifier.fit(X, y)
        self.is_trained = True

        return self

    def predict(self, texts: list[str]) -> list[list[str]]:
        """
        Predict languages for texts.

        Returns:
            List of predicted language lists for each text
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X = self.vectorizer.transform(texts)
        y_pred = self.classifier.predict(X)
        return [list(langs) for langs in self.mlb.inverse_transform(y_pred)]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """
        Get probability distributions for all languages.

        Returns:
            Array of shape (n_samples, n_classes) with probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def predict_with_threshold(self, texts: list[str], threshold: float = None) -> list[list[str]]:
        """
        Predict languages using a custom probability threshold.

        Args:
            texts: List of texts to classify
            threshold: Probability threshold (uses self.threshold if None)

        Returns:
            List of predicted language lists for each text
        """
        if threshold is None:
            threshold = self.threshold

        probas = self.predict_proba(texts)
        predictions = (probas >= threshold).astype(int)
        return [list(langs) for langs in self.mlb.inverse_transform(predictions)]

    def predict_single(self, text: str, threshold: float = None) -> dict:
        """
        Predict languages for a single text with detailed output.

        Returns:
            Dict with 'predictions', 'confidences', and 'all_probabilities'
        """
        if threshold is None:
            threshold = self.threshold

        probas = self.predict_proba([text])[0]

        # Get all probabilities as dict
        all_probs = {lang: float(prob) for lang, prob in zip(self.mlb.classes_, probas)}

        # Get predictions above threshold
        predicted_mask = probas >= threshold
        predicted_langs = self.mlb.classes_[predicted_mask]
        predicted_probs = probas[predicted_mask]

        return {
            'predictions': list(predicted_langs),
            'confidences': {lang: float(prob) for lang, prob in zip(predicted_langs, predicted_probs)},
            'all_probabilities': all_probs
        }

    def evaluate(self, texts: list[str], labels: list[list[str]]) -> dict:
        """
        Evaluate model with multilabel metrics.

        Returns:
            Dict with F1 scores, precision, recall, and hamming loss
        """
        X = self.vectorizer.transform(texts)
        y_true = self.mlb.transform(labels)
        y_pred = self.classifier.predict(X)

        return {
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'hamming_loss': hamming_loss(y_true, y_pred),
        }

    def save_model(self, filepath: str):
        """Save model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'mlb': self.mlb,
            'ngram_size': self.ngram_size,
            'C': self.C,
            'threshold': self.threshold,
            'is_trained': self.is_trained
        }

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Multilabel model saved to: {filepath}")

    @classmethod
    def load_model(cls, filepath: str):
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        model = cls(
            ngram_size=model_data['ngram_size'],
            C=model_data['C'],
            threshold=model_data.get('threshold', 0.5)
        )

        model.vectorizer = model_data['vectorizer']
        model.classifier = model_data['classifier']
        model.mlb = model_data['mlb']
        model.is_trained = model_data['is_trained']

        print(f"Multilabel model loaded from: {filepath}")
        return model


# ============================================================================
# Training Script
# ============================================================================

def train_model(
    output_dir: str,
    samples_per_language: int | None = 10000,
    test_size: float = 0.001,
    ngram_size: int = 6,
    C: float = 100.0,
):
    """Complete training pipeline using OpenLID dataset"""

    logging.info("LANGUAGE IDENTIFICATION - TRAINING")

    logging.info("N-gram size: %s", ngram_size)
    logging.info("Regularization C: %s", C)

    train_texts, eval_texts, train_labels, eval_labels = load_dataset(
        samples_per_language, test_size=test_size)

    logging.debug("Training samples: %s", len(train_texts))
    logging.debug("Eval samples: %s", len(eval_texts))

    # Train model
    logging.info("Training model...")
    model = NLIClassifier(ngram_size=ngram_size, C=C)
    model.fit(train_texts, train_labels)

    # Evaluate on test set
    logging.info("Evaluating on eval set...")
    test_results = model.evaluate(eval_texts, eval_labels)
    logging.info("Test Accuracy: %s (%s%%)", test_results['accuracy'], test_results['accuracy']*100)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"nli_model_{timestamp}.pkl")
    model.save_model(model_path)

    # Save results
    results_path = os.path.join(output_dir, f"results_{timestamp}.json")
    results = {
        'timestamp': timestamp,
        'dataset': 'OpenLID',
        'ngram_size': ngram_size,
        'C': C,
        'num_train': len(train_texts),
        'num_test': len(eval_texts),
        'test_accuracy': float(test_results['accuracy']),
        'model_path': model_path
    }

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logging.info("Results saved to: %s", results_path)

    return model, results


def train_multilabel_model(
    output_dir: str,
    samples_per_language: int | None = 10000,
    test_size: float = 0.001,
    ngram_size: int = 6,
    C: float = 100.0,
    threshold: float = 0.5,
):
    """Complete training pipeline for multilabel model using OpenLID dataset.

    Note: The OpenLID dataset has single labels per sample. This function wraps
    each label as a list to create a multilabel setup. For true multilabel data
    (e.g., code-mixed texts), you would need a different data source.
    """

    logging.info("MULTILABEL LANGUAGE IDENTIFICATION - TRAINING")

    logging.info("N-gram size: %s", ngram_size)
    logging.info("Regularization C: %s", C)
    logging.info("Threshold: %s", threshold)

    train_texts, eval_texts, train_labels, eval_labels = load_dataset(
        samples_per_language, test_size=test_size)

    # Convert single labels to multilabel format (list of lists)
    train_labels_ml = [[label] for label in train_labels]
    eval_labels_ml = [[label] for label in eval_labels]

    logging.debug("Training samples: %s", len(train_texts))
    logging.debug("Eval samples: %s", len(eval_texts))

    # Train model
    logging.info("Training multilabel model...")
    model = MultilabelNLIClassifier(ngram_size=ngram_size, C=C, threshold=threshold)
    model.fit(train_texts, train_labels_ml)

    # Evaluate on test set
    logging.info("Evaluating on eval set...")
    test_results = model.evaluate(eval_texts, eval_labels_ml)
    logging.info("Test F1 Micro: %.4f", test_results['f1_micro'])
    logging.info("Test F1 Macro: %.4f", test_results['f1_macro'])
    logging.info("Test Hamming Loss: %.4f", test_results['hamming_loss'])

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(output_dir, f"nli_multilabel_model_{timestamp}.pkl")
    model.save_model(model_path)

    # Save results
    results_path = os.path.join(output_dir, f"multilabel_results_{timestamp}.json")
    results = {
        'timestamp': timestamp,
        'dataset': 'OpenLID',
        'model_type': 'multilabel',
        'ngram_size': ngram_size,
        'C': C,
        'threshold': threshold,
        'num_train': len(train_texts),
        'num_test': len(eval_texts),
        'f1_micro': float(test_results['f1_micro']),
        'f1_macro': float(test_results['f1_macro']),
        'precision_micro': float(test_results['precision_micro']),
        'recall_micro': float(test_results['recall_micro']),
        'hamming_loss': float(test_results['hamming_loss']),
        'model_path': model_path
    }

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    logging.info("Results saved to: %s", results_path)

    return model, results


# ============================================================================
# Inference Functions
# ============================================================================


def predict_text(model: NLIClassifier, text: str, verbose: bool = True) -> dict:
    """
    Predict native language for a single text

    Args:
        model: Trained NLI model
        text: Text to classify
        verbose: Whether to print results

    Returns:
        Dictionary with prediction results
    """
    result = model.predict_single(text)

    if verbose:
        print("=" * 80)
        print("PREDICTION RESULT")
        print("=" * 80)
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print()
        print(f"Predicted Language: {result['prediction']}")
        print(
            f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print()
        print("All Probabilities:")
        sorted_probs = sorted(
            result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for lang, prob in sorted_probs:
            print(f"  {lang}: {prob:.4f} ({prob*100:.2f}%)")
        print("=" * 80)

    return result


def predict_batch(model: NLIClassifier, texts: list[str]) -> list[dict]:
    """
    Predict native languages for multiple texts

    Args:
        model: Trained NLI model
        texts: List of texts to classify

    Returns:
        List of prediction dictionaries
    """
    results = []
    for text in texts:
        result = model.predict_single(text)
        results.append(result)
    return results


def predict_from_file(model_path: str, text: str):
    """
    Load model and predict from file

    Args:
        model_path: Path to saved model
        text: Text to classify
    """
    model = NLIClassifier.load_model(model_path)
    return predict_text(model, text)


def predict_text_multilabel(
    model: MultilabelNLIClassifier, text: str, verbose: bool = True, top_k: int = 10
) -> dict:
    """
    Predict languages for a single text using multilabel model.

    Args:
        model: Trained multilabel NLI model
        text: Text to classify
        verbose: Whether to print results
        top_k: Number of top probabilities to show

    Returns:
        Dictionary with prediction results
    """
    result = model.predict_single(text)

    if verbose:
        print("=" * 80)
        print("MULTILABEL PREDICTION RESULT")
        print("=" * 80)
        print(f"Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print()
        print(f"Predicted Languages: {', '.join(result['predictions']) if result['predictions'] else 'None'}")
        if result['confidences']:
            print("Confidences:")
            for lang, prob in result['confidences'].items():
                print(f"  {lang}: {prob:.4f} ({prob*100:.2f}%)")
        print()
        print(f"Top {top_k} Probabilities:")
        sorted_probs = sorted(
            result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)[:top_k]
        for lang, prob in sorted_probs:
            print(f"  {lang}: {prob:.4f} ({prob*100:.2f}%)")
        print("=" * 80)

    return result


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='N-gram Language Identification'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility")
    train_parser.add_argument("--samples-per-language", type=int,
                              default=None, help="The number of samples per language to use (default: 10000)")
    train_parser.add_argument("--test-size", type=float,
                              default=0.001, help="Proportion of data to use for evaluation")
    train_parser.add_argument('--output', type=str, default='models',
                              help='Output directory for model and results')
    train_parser.add_argument('--ngram-size', type=int, default=6,
                              help='Size of character n-grams')
    train_parser.add_argument('--C', type=float, default=100.0,
                              help='Regularization parameter')
    train_parser.add_argument('--multilabel', action='store_true',
                              help='Train a multilabel model instead of multiclass')
    train_parser.add_argument('--threshold', type=float, default=0.5,
                              help='Probability threshold for multilabel predictions')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions')
    predict_parser.add_argument('--model', type=str, required=True,
                                help='Path to saved model')
    predict_parser.add_argument('--text', type=str,
                                help='Text to classify')
    predict_parser.add_argument('--file', type=str,
                                help='File containing text to classify')
    predict_parser.add_argument('--multilabel', action='store_true',
                                help='Load as multilabel model')

    args = parser.parse_args()

    if args.command == 'train':
        np.random.seed(args.seed)
        if args.multilabel:
            train_multilabel_model(
                output_dir=args.output,
                samples_per_language=args.samples_per_language,
                test_size=args.test_size,
                ngram_size=args.ngram_size,
                C=args.C,
                threshold=args.threshold,
            )
        else:
            train_model(
                output_dir=args.output,
                samples_per_language=args.samples_per_language,
                test_size=args.test_size,
                ngram_size=args.ngram_size,
                C=args.C,
            )

    elif args.command == 'predict':
        if args.multilabel:
            model = MultilabelNLIClassifier.load_model(args.model)
            predict_fn = predict_text_multilabel
        else:
            model = NLIClassifier.load_model(args.model)
            predict_fn = predict_text

        if args.text:
            predict_fn(model, args.text)
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
            predict_fn(model, text)
        else:
            print("Provide either --text or --file")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
