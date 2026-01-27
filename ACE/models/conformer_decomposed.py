import gin
import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchaudio.models import Conformer
from torchmetrics.classification import BinaryAccuracy
from pathlib import Path

from ACE.losses import ConsonanceDecomposedLoss, DecomposedLoss
from ACE.mir_evaluation import evaluate_batch_decomposed
from ACE.utils import PositionalEncoding

@gin.configurable
class ConformerDecomposedModel(L.LightningModule):
    def __init__(
        self,
        vocabularies: dict,  # This dict is passed in the trainer from the dataloader
        loss: str = "decomposed",
        input_dim: int = 144,  # CQT feature dimension
        conformer_dim: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        num_layers: int = 4,
        depthwise_conv_kernel_size: int = 31,
        use_group_norm: bool = False,
        convolution_first: bool = True,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        root_weight: float = 1.0,
        bass_weight: float = 1.0,
        chord_weight: float = 2.0,
        min_notes_weight: float = 2.0,
        smoothing_alpha: float = 0.1,
        positional_encoding: bool = False,
        vocab_path: str | Path = "./ACE/chords_vocab.joblib",
        mir_eval_on_validation: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss"])
        # store path to chords vocabulary (used by mir_evaluation -> convert_ground_truth)
        self.vocab_path = vocab_path

        # Loss function
        assert loss in ["decomposed", "consonance_decomposed"], (
            f"Unsupported loss type: {loss}. Supported types: 'decomposed', "
            " 'consonance_decomposed'."
        )
        if loss == "decomposed":
            self.loss = DecomposedLoss(
                root_weight=root_weight,
                bass_weight=bass_weight,
                chord_weight=chord_weight,
                min_notes_weight=min_notes_weight,
            )
        elif loss == "consonance_decomposed":
            self.loss = ConsonanceDecomposedLoss(
                root_weight=root_weight,
                bass_weight=bass_weight,
                chord_weight=chord_weight,
                min_notes_weight=min_notes_weight,
                # Use a fixed smoothing alpha for consonance decomposed loss
                smoothing_alpha=smoothing_alpha,
            )

        # Learning rate
        self.learning_rate = learning_rate

        # Vocabulary configuration
        self.vocabularies = vocabularies
        self.num_classes_root = self.vocabularies["root"]
        self.num_classes_bass = self.vocabularies["bass"]
        self.num_classes_chord = self.vocabularies["onehot"]

        # Positional encodings
        self.positional_encoding = positional_encoding
        # self.positional_encodings = nn.Parameter(
        #     torch.randn(1, 108), requires_grad=True
        # )
        self.positional_encodings = PositionalEncoding(hidden_size=conformer_dim)

        # Input projection
        self.input_projection = nn.Linear(input_dim, conformer_dim)

        # Conformer layers
        self.conformer = Conformer(
            input_dim=conformer_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
            convolution_first=convolution_first,
        )

        # Chord prediction (primary task)
        self.output_chord = nn.Linear(conformer_dim, self.num_classes_chord)
        # Bass prediction from chord + features
        self.bass_fusion = nn.Sequential(
            nn.Linear(conformer_dim + self.num_classes_chord, conformer_dim),
            nn.LayerNorm(conformer_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.output_bass = nn.Linear(conformer_dim, self.num_classes_bass)

        # Root prediction from chord + features
        self.root_fusion = nn.Sequential(
            nn.Linear(conformer_dim + self.num_classes_chord, conformer_dim),
            nn.LayerNorm(conformer_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.output_root = nn.Linear(conformer_dim, self.num_classes_root)

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()

        # Storage for validation and test predictions
        self.mir_eval_on_validation = mir_eval_on_validation

        self.validation_predictions_root = []
        self.validation_predictions_bass = []
        self.validation_predictions_chord = []
        self.validation_onsets = []
        self.validation_labels = []

        self.test_predictions_root = []
        self.test_predictions_bass = []
        self.test_predictions_chord = []
        self.test_onsets = []
        self.test_labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
        x = x.squeeze(1)  # x shape: [batch, features, time]
        # Define lengths for Conformer
        lengths = torch.full(
            (x.shape[0],), x.shape[2], dtype=torch.long, device=x.device
        )
        # Prepare for conformer [batch, time, features]
        x = x.transpose(1, 2)
        # Apply input projection
        x = self.input_projection(x)
        # Apply positional encodings
        if self.positional_encoding:
            x = self.positional_encodings(x)
        # Apply conformer -> Conformer expects [batch, time, features]
        x, _ = self.conformer(x, lengths=lengths)

        # Chord prediction first
        chord_logits = self.output_chord(x)

        # Bass prediction conditioned on chord
        bass_input = torch.cat([x, chord_logits], dim=-1)
        bass_features = self.bass_fusion(bass_input)
        bass_logits = self.output_bass(bass_features)

        # Root prediction conditioned on chord
        root_input = torch.cat([x, chord_logits], dim=-1)
        root_features = self.root_fusion(root_input)
        root_logits = self.output_root(root_features)
        return {"root": root_logits, "bass": bass_logits, "onehot": chord_logits}

    def _shared_step(
        self,
        batch: tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch_idx: int,
        step_type: str,
    ) -> torch.Tensor | dict:
        """Shared step for training and validation"""
        audio, labels = batch
        logits = self(audio)

        # Get batch size
        batch_size = audio.shape[0]

        # Fix dimensions
        # 1. Root and bass: [batch, time, classes] -> [batch, classes, time]
        logits["root"] = logits["root"].permute(0, 2, 1)
        logits["bass"] = logits["bass"].permute(0, 2, 1)
        # 2. Root and bass labels: subtract 1 to match the classes
        labels["root"] = labels["root"] - 1
        labels["bass"] = labels["bass"] - 1
        labels["onsets"] = labels["onsets"].squeeze(-1)  # shape: (batch, time)

        # Compute loss
        loss = self.loss(logits, labels)

        # Get predictions
        predictions_root = logits["root"].argmax(dim=1)  # (batch, time)
        predictions_bass = logits["bass"].argmax(dim=1)  # (batch, time)
        # predictions chord are multi-hot encoded (binary)
        predictions_chord = logits["onehot"]  # (batch, time, 13)

        # Calculate accuracy (always)
        # Update metrics
        metric = getattr(self, f"{step_type}_accuracy")
        metric(predictions_chord, labels["onehot"])
        self.log(
            f"{step_type}_accuracy",
            metric,
            on_step=False,
            on_epoch=True,
            prog_bar=(step_type == "train"),
        )

        # Log loss
        self.log(
            f"{step_type}_loss", loss, on_step=False, on_epoch=True, prog_bar=False
        )

        # if in validation or test step, compute mir_eval metrics
        if step_type in ["val", "test"]:
            # For validation/test, store data for chord recognition metrics
            # Convert to numpy and store
            preds_root_np = predictions_root.detach().cpu().numpy()  # (batch, time)
            preds_bass_np = predictions_bass.detach().cpu().numpy()  # (batch, time)
            preds_chord_np = (
                predictions_chord.detach().cpu().float().numpy()
            )  # (batch, time, 13)
            label_onset = labels["onsets"].detach().cpu().numpy()  # (batch, time)
            label_original = labels["original"].detach().cpu().numpy()  # (batch, time)

            # Store predictions for each item in the batch
            for i in range(batch_size):
                if step_type == "val":
                    self.validation_predictions_root.append(preds_root_np[i])  # (time,)
                    self.validation_predictions_bass.append(preds_bass_np[i])  # (time,)
                    self.validation_predictions_chord.append(
                        preds_chord_np[i]
                    )  # (time, 13)
                    self.validation_onsets.append(label_onset[i])
                    self.validation_labels.append(label_original[i])
                elif step_type == "test":
                    self.test_predictions_root.append(preds_root_np[i])  # (time,)
                    self.test_predictions_bass.append(preds_bass_np[i])  # (time,)
                    self.test_predictions_chord.append(preds_chord_np[i])  # (time, 13)
                    self.test_onsets.append(label_onset[i])
                    self.test_labels.append(label_original[i])

        return {"loss": loss, "labels": labels}

    def training_step(
        self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor | dict:
        """Training step for the model."""
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor | dict:
        """Validation step for the model."""
        return self._shared_step(batch, batch_idx, "val")

    def test_step(
        self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor | dict:
        """Test step for the model."""
        return self._shared_step(batch, batch_idx, "test")

    def on_validation_epoch_end(self) -> None:
        """Calculate chord recognition metrics at the end of validation epoch."""
        if self.mir_eval_on_validation:
            if len(self.validation_predictions_root) > 0:
                # try:
                # Convert list of predictions to batch format
                batched_predictions_root = np.stack(
                    self.validation_predictions_root
                )  # (B, T, D)
                batched_predictions_bass = np.stack(
                    self.validation_predictions_bass
                )  # (B, T, D)
                batched_predictions_chord = np.stack(
                    self.validation_predictions_chord
                )  # (B, T, D)
                batched_labels = np.stack(self.validation_labels)  # (B, T)
                batched_onsets = np.stack(self.validation_onsets)  # (B, T)

                # Calculate chord recognition metrics
                scores = evaluate_batch_decomposed(
                    batched_predictions_root=batched_predictions_root,
                    batched_predictions_bass=batched_predictions_bass,
                    batched_predictions_chord=batched_predictions_chord,
                    batched_onsets=batched_onsets,  # type: ignore
                    batched_gt_labels=batched_labels,  # type: ignore
                    segment_duration=20.0,  # 20 seconds segments
                    vocab_path=self.vocab_path,  # <-- forward vocab path
                )

                # Log all metrics
                for metric_name, score in scores.items():
                    self.log(f"val_{metric_name}", score, on_epoch=True, prog_bar=False)

                # Clear stored data
                self.validation_predictions_root.clear()
                self.validation_predictions_bass.clear()
                self.validation_predictions_chord.clear()
                self.validation_onsets.clear()
                self.validation_labels.clear()

    def on_test_epoch_end(self) -> None:
        """Calculate chord recognition metrics at the end of test epoch."""
        if len(self.test_predictions_root) > 0:
            # Convert list of predictions to batch format
            batched_predictions_root = np.stack(self.test_predictions_root)
            batched_predictions_bass = np.stack(self.test_predictions_bass)
            batched_predictions_chord = np.stack(
                self.test_predictions_chord
            )  # (B, T, D)
            batched_labels = np.stack(self.test_labels)
            batched_onsets = np.stack(self.test_onsets)

            # Calculate chord recognition metrics
            scores = evaluate_batch_decomposed(
                batched_predictions_root=batched_predictions_root,
                batched_predictions_bass=batched_predictions_bass,
                batched_predictions_chord=batched_predictions_chord,
                batched_onsets=batched_onsets,  # type: ignore
                batched_gt_labels=batched_labels,  # type: ignore
                segment_duration=20.0,  # 20 seconds segments
                vocab_path=self.vocab_path,  # <-- forward vocab path
            )

            # Log all metrics
            for metric_name, score in scores.items():
                self.log(f"test_{metric_name}", score, on_epoch=True, prog_bar=True)

            # Clear stored data
            self.test_predictions_root.clear()
            self.test_predictions_bass.clear()
            self.test_predictions_chord.clear()
            self.test_onsets.clear()
            self.test_labels.clear()

    def configure_optimizers(self) -> dict:  # type: ignore
        """Configure optimizers and learning rate scheduler."""
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=15,  # initial restart interval (epochs)
            T_mult=2,  # factor to increase T_0 after each restart
            eta_min=1e-6,  # minimum learning rate
            last_epoch=-1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "learning_rate",
            },
        }
