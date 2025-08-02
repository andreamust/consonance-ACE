import gin
import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torchaudio.models import Conformer
from torchmetrics.classification import Accuracy

from ACE.mir_evaluation import evaluate_batch
from ACE.utils import PositionalEncoding


@gin.configurable
class ConformerModel(L.LightningModule):
    def __init__(
        self,
        vocabularies: dict,  # This dict is passed in the trainer from the dataloader
        input_dim: int = 144,  # CQT feature dimension
        hidden_dim: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        num_layers: int = 4,
        depthwise_conv_kernel_size: int = 31,
        use_group_norm: bool = False,
        convolution_first: bool = True,
        dropout: float = 0.1,
        vocab_type: str = "majmin",
        learning_rate: float = 1e-4,
        positional_encoding: bool = False,
        vocab_path: str = "./ACE/chords_vocab.joblib",  # Path to the vocabulary file
        mir_eval_on_validation: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        # Vocabulary configuration
        self.vocabularies = vocabularies
        self.vocab_type = vocab_type
        self.num_classes = self.vocabularies[self.vocab_type]
        self.vocab_path = vocab_path

        # Positional encodings
        self.positional_encoding = positional_encoding
        self.positional_encodings = PositionalEncoding(hidden_size=hidden_dim)

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Conformer layers
        self.conformer = Conformer(
            input_dim=hidden_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            use_group_norm=use_group_norm,
            convolution_first=convolution_first,
        )

        # Output projection
        self.output_projection = nn.Linear(hidden_dim, self.num_classes)

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

        # Storage for validation and test predictions
        self.mir_eval_on_validation = mir_eval_on_validation

        self.validation_predictions = []
        self.validation_onsets = []
        self.validation_labels = []

        self.test_predictions = []
        self.test_onsets = []
        self.test_labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

        # Apply output projection
        logits = self.output_projection(x)

        return logits

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

        # Get labels
        label_onset = labels["onsets"]
        label_onset = label_onset.squeeze(-1)  # shape: (batch, time)
        label_original = labels["original"]
        labels = labels[self.vocab_type]

        # Adjust labels for loss calculation
        if self.vocab_type in ["root", "bass", "mode"]:
            labels = labels - 1

        # Reshape for loss calculation
        labels_view = labels.view(-1)
        logits_view = logits.view(-1, self.num_classes)

        # Calculate loss
        loss = F.cross_entropy(logits_view, labels_view)

        # Calculate accuracy
        predictions = logits.argmax(dim=-1)

        # Calculate accuracy (always)
        # Update metrics
        metric = getattr(self, f"{step_type}_accuracy")
        metric(predictions, labels)
        self.log(
            f"{step_type}_accuracy",
            metric,
            on_step=False,
            on_epoch=True,
            prog_bar=(step_type == "train"),
        )

        # if in validation or test step, compute mir_eval metrics
        if step_type in ["val", "test"]:
            # For validation/test, store data for chord recognition metrics
            # Convert to numpy and store
            preds_np = predictions.detach().cpu().numpy()  # (batch, time, num_classes)
            label_onset = label_onset.detach().cpu().numpy()  # (batch, time)
            label_original = label_original.detach().cpu().numpy()  # (batch, time)

            # Store predictions for each item in the batch
            for i in range(batch_size):
                if step_type == "val":
                    self.validation_predictions.append(
                        preds_np[i]
                    )  # (time, num_classes)
                    self.validation_onsets.append(label_onset[i])
                    self.validation_labels.append(label_original[i])
                elif step_type == "test":
                    self.test_predictions.append(preds_np[i])  # (time, num_classes)
                    self.test_onsets.append(label_onset[i])
                    self.test_labels.append(label_original[i])

        # Log loss
        self.log(
            f"{step_type}_loss", loss, on_step=False, on_epoch=True, prog_bar=False
        )

        return {"loss": loss, "predictions": predictions}

    def training_step(
        self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor | dict:
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor | dict:
        return self._shared_step(batch, batch_idx, "val")

    def test_step(
        self, batch: tuple[torch.Tensor, dict[str, torch.Tensor]], batch_idx: int
    ) -> torch.Tensor | dict:
        return self._shared_step(batch, batch_idx, "test")

    def on_validation_epoch_end(self) -> None:
        """Calculate chord recognition metrics at the end of validation epoch."""
        if self.mir_eval_on_validation:
            if len(self.validation_predictions) > 0:
                # try:
                # Convert list of predictions to batch format
                batched_predictions = np.stack(self.validation_predictions)  # (B, T, D)
                batched_labels = np.stack(self.validation_labels)  # (B, T)
                batched_onsets = np.stack(self.validation_onsets)  # (B, T)

                # Calculate chord recognition metrics
                scores = evaluate_batch(
                    batched_predictions=batched_predictions,
                    batched_onsets=batched_onsets,  # type: ignore
                    batched_gt_labels=batched_labels,  # type: ignore
                    vocabulary=self.vocab_type,
                    segment_duration=20.0,  # 30 seconds segments
                    vocab_path=self.vocab_path,
                )

                # Log all metrics
                for metric_name, score in scores.items():
                    self.log(f"val_{metric_name}", score, on_epoch=True, prog_bar=False)

                # Clear stored data
                self.validation_predictions.clear()
                self.validation_onsets.clear()
                self.validation_labels.clear()

    def on_test_epoch_end(self) -> None:
        """Calculate chord recognition metrics at the end of test epoch."""
        if len(self.test_predictions) > 0:
            # Convert list of predictions to batch format
            batched_predictions = np.stack(self.test_predictions)
            batched_labels = np.stack(self.test_labels)
            batched_onsets = np.stack(self.test_onsets)

            # Calculate chord recognition metrics
            scores = evaluate_batch(
                batched_predictions=batched_predictions,
                batched_onsets=batched_onsets,  # type: ignore
                batched_gt_labels=batched_labels,  # type: ignore
                vocabulary=self.vocab_type,
                segment_duration=20.0,  # 30 seconds segments
                vocab_path=self.vocab_path,
            )

            # Log all metrics
            for metric_name, score in scores.items():
                self.log(f"test_{metric_name}", score, on_epoch=True, prog_bar=True)

            # Clear stored data
            self.test_predictions.clear()
            self.test_onsets.clear()
            self.test_labels.clear()

    def configure_optimizers(self) -> dict:  # type: ignore
        """Configure optimizers and learning rate scheduler."""
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=25,  # total number of epochs (adjust to your trainer setting)
            eta_min=1e-6,  # final LR value (minimum)
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # step the scheduler every epoch
                "frequency": 1,
            },
        }
