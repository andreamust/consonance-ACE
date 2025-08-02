"""
Loss functions for training the ACE models.
"""

import torch
import torch.nn.functional as F
from torch import nn


class DecomposedLoss(nn.Module):
    def __init__(
        self,
        root_weight: float = 1.0,
        bass_weight: float = 1.0,
        chord_weight: float = 1.0,
        min_notes_weight: float = 0.5,
        activation_threshold: float = 0.5,
        min_notes_threshold: float = 3.5,
    ):
        super().__init__()

        # Just make the weights learnable parameters
        self.chord_activation_threshold = activation_threshold
        self.min_notes_threshold = min_notes_threshold

        self.root_weight = root_weight
        self.bass_weight = bass_weight
        self.chord_weight = chord_weight
        self.min_notes_weight = min_notes_weight

        # Loss functions
        self.root_loss = nn.CrossEntropyLoss()
        self.bass_loss = nn.CrossEntropyLoss()
        self.chord_loss = nn.BCEWithLogitsLoss()

    def cardinality_loss(self, chord_probs, chord_label):
        """Encourage predicting the right number of notes based on chord type"""
        # Count active notes in ground truth
        target_note_count = chord_label.sum(dim=-1)  # (B, T)

        # Count predicted active notes
        pred_note_count = (
            (chord_probs > self.chord_activation_threshold).sum(dim=-1).float()
        )

        # L1 loss between predicted and target counts
        cardinality_loss = F.l1_loss(pred_note_count, target_note_count)

        return cardinality_loss

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
        epoch: int = 0,
    ) -> torch.Tensor:
        # turn off deterministic algorithms for this forward pass
        prev_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        # Root loss
        root_pred = outputs["root"]
        root_label = labels["root"]
        root_loss = self.root_loss(root_pred, root_label)

        # Bass loss
        bass_pred = outputs["bass"]
        bass_label = labels["bass"]
        bass_loss = self.bass_loss(bass_pred, bass_label)

        # Chord loss (binary cross entropy)
        # For BCE with logits, both inputs should have same dimensions
        chord_pred = outputs["onehot"]
        chord_label = labels["onehot"].float()  # Ensure labels float for BCE
        chord_loss = self.chord_loss(chord_pred, chord_label)

        # Cardinality loss
        cardinality_loss = self.cardinality_loss(chord_pred, chord_label)
        # root_w, bass_w, chord_w, min_w = self.get_weights()
        # root_w, bass_w, chord_w, min_w = self.curriculum_weights(epoch)
        root_w = self.root_weight
        bass_w = self.bass_weight
        chord_w = self.chord_weight
        min_w = self.min_notes_weight

        total_loss = (
            root_w * root_loss
            + bass_w * bass_loss
            + chord_w * chord_loss
            + min_w * cardinality_loss
        )

        return total_loss


class ConsonanceDecomposedLoss(nn.Module):
    def __init__(
        self,
        root_weight: float = 1.0,
        bass_weight: float = 1.0,
        chord_weight: float = 8.0,
        min_notes_weight: float = 4.0,
        activation_threshold: float = 0.5,
        smoothing_alpha: float = 0.1,
    ):
        super().__init__()
        self.chord_activation_threshold = activation_threshold
        self.smoothing_alpha = smoothing_alpha

        self.root_weight = root_weight
        self.bass_weight = bass_weight
        self.chord_weight = chord_weight
        self.min_notes_weight = min_notes_weight

        self.chord_loss = nn.BCEWithLogitsLoss()

        # Consonance vector (0 is most consonant, 7 is most dissonant)
        consonance_vector = torch.tensor(
            [0, 7, 5, 1, 1, 2, 3, 1, 2, 2, 4, 6], dtype=torch.float32
        )

        # Pre-compute and register consonance/similarity matrix as buffer
        self._precompute_similarity_matrix(consonance_vector)

    def _precompute_similarity_matrix(self, consonance_vector):
        """Pre-compute the similarity matrix once during initialization"""

        smoothing_alpha = self.smoothing_alpha
        # Create consonance matrix (13x13)
        consonance_matrix = torch.zeros(13, 13, dtype=torch.float32)

        # Fill consonance matrix for pitch classes (0-11)
        for i in range(12):
            for j in range(12):
                interval = (j - i) % 12
                consonance_matrix[i, j] = consonance_vector[interval]

        # Handle silence (index 12)
        consonance_matrix[12, :12] = 7.0
        consonance_matrix[:12, 12] = 7.0
        consonance_matrix[12, 12] = 0.0

        # Convert consonance to similarity and apply temperature scaling
        similarity_matrix = 1.0 - (consonance_matrix / 7.0)
        temperature = 1.0 / (1.0 - smoothing_alpha + 1e-8)
        similarity_matrix = torch.pow(similarity_matrix, temperature)

        # Register as buffer so it moves with the model to correct device
        self.register_buffer("similarity_matrix", similarity_matrix)

    def create_smoothed_labels_vectorized(self, labels):
        """
        Vectorized version of create_smoothed_labels - much faster!

        Args:
            labels: (B, T) tensor with class indices (0-12)

        Returns:
            smoothed_labels: (B, T, 13) tensor with smoothed probabilities
        """
        smoothing_alpha = self.smoothing_alpha
        batch_size, seq_len = labels.shape
        device = labels.device

        # Create mask for valid labels (< 13)
        valid_mask = labels < 13

        # Initialize smoothed labels
        smoothed_labels = torch.zeros(batch_size, seq_len, 13, device=device)

        if not valid_mask.any():
            return smoothed_labels

        # Get valid labels and their positions
        valid_labels = labels[valid_mask]  # (N,) where N is number of valid labels

        # Use advanced indexing to get similarities for all valid labels at once
        similarities = self.similarity_matrix[valid_labels]  # type: ignore

        # Apply smoothing vectorized
        smoothed_probs = similarities * smoothing_alpha  # (N, 13)

        # Add the extra probability mass to true classes
        true_class_indices = torch.arange(len(valid_labels), device=device)
        smoothed_probs[true_class_indices, valid_labels] += 1.0 - smoothing_alpha

        # Normalize
        smoothed_probs = smoothed_probs / smoothed_probs.sum(dim=-1, keepdim=True)

        # Place back into the full tensor
        smoothed_labels[valid_mask] = smoothed_probs

        return smoothed_labels

    def cross_entropy_with_smoothed_labels(self, logits, smoothed_labels):
        """
        Compute cross entropy loss with smoothed labels.

        Args:
            logits: (B, T, 13) tensor of logits
            smoothed_labels: (B, T, 13) tensor of smoothed probabilities

        Returns:
            loss: scalar tensor
        """
        # Apply log_softmax to logits along the class dimension
        log_probs = F.log_softmax(logits, dim=-1)  # (B, 13. T)
        log_probs = log_probs.permute(0, 2, 1)  # (B, T, 13)

        # Compute cross entropy: -sum(p_true * log(p_pred))
        loss = -(smoothed_labels * log_probs).sum(dim=-1).mean()

        return loss

    def cardinality_loss(self, chord_probs, chord_label):
        """Encourage predicting the right number of notes based on chord type"""
        # Count active notes in ground truth
        target_note_count = chord_label.sum(dim=-1)  # (B, T)

        # Count predicted active notes
        pred_note_count = (
            (chord_probs > self.chord_activation_threshold).sum(dim=-1).float()
        )

        # L1 loss between predicted and target counts
        cardinality_loss = F.l1_loss(pred_note_count, target_note_count)

        return cardinality_loss

    def forward(
        self, outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Root loss with consonance-based smoothing
        root_pred = outputs["root"]  # (B, T, 13)
        root_label = labels["root"]  # (B, T)
        smoothed_root_labels = self.create_smoothed_labels_vectorized(root_label)
        root_loss = self.cross_entropy_with_smoothed_labels(
            root_pred, smoothed_root_labels
        )

        # Bass loss with consonance-based smoothing
        bass_pred = outputs["bass"]  # (B, T, 13)
        bass_label = labels["bass"]  # (B, T)
        smoothed_bass_labels = self.create_smoothed_labels_vectorized(bass_label)
        bass_loss = self.cross_entropy_with_smoothed_labels(
            bass_pred, smoothed_bass_labels
        )

        # Chord loss (binary cross entropy)
        chord_pred = outputs["onehot"]
        chord_label = labels["onehot"].float()
        chord_loss = self.chord_loss(chord_pred, chord_label)

        # Cardinality loss
        chord_probs = torch.sigmoid(chord_pred)
        cardinality_loss = self.cardinality_loss(chord_probs, chord_label)

        # root_w, bass_w, chord_w, min_w = self.get_weights()

        root_w = self.root_weight
        bass_w = self.bass_weight
        chord_w = self.chord_weight
        min_w = self.min_notes_weight

        # Combine losses with learnable weights
        total_loss = (
            root_w * root_loss
            + bass_w * bass_loss
            + chord_w * chord_loss
            + min_w * cardinality_loss
        )

        return total_loss
