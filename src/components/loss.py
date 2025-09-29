import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import minmax_scale
import lang2vec.lang2vec as l2v

from .common import flores_to_iso

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

