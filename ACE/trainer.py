"""
Training script for all the models.

Supports both CLI and programmatic usage.

Example (CLI):
    python -m ACE.trainer --model conformer --name my_run --max_epochs 50

Example (API):
    from ACE.trainer import main
    main(
        model_name="conformer",
        run_name="experiment_01",
        params={
            "train.max_epochs": 10,
            "train.accelerator": "gpu",
            "ModelCheckpoint.dirpath": "checkpoints",
            "EarlyStopping.patience": 5
        }
    )
"""

import argparse
from pathlib import Path

import gin
import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks.early_stopping import EarlyStopping  # type: ignore
from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint  # type: ignore
from lightning.pytorch.loggers import WandbLogger

from .dataloader import ChocoAudioDataModule


@gin.configurable
def wandb_logger(project: str, name: str, group: str | None = None) -> WandbLogger:
    """Initialize WandB logger with gin configuration."""
    return WandbLogger(project=project, name=name, log_model=False, group=group)


@gin.configurable
class ModelCheckpoint(ModelCheckpoint):  # type: ignore
    """Model checkpoint callback configurable with gin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@gin.configurable
class EarlyStopping(EarlyStopping):  # type: ignore
    """Early stopping callback configurable with gin."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


@gin.configurable
def train(
    model_class: type[L.LightningModule],
    data_path: str | Path,
    run_name: str = "default_run",
    max_epochs: int = 100,
    precision: str = "16-mixed",
    accelerator: str = "gpu",
    devices: int = 1,
    log_every_n_steps: int = 10,
    check_val_every_n_epoch: int = 10,
    accumulate_grad_batches: int = 1,
    gradient_clip_val: float = 0.0,
    vocab_path: str | Path = "./ACE/chords_vocab.joblib",
):
    """Main training function configurable with gin."""
    # Set torch precision
    torch.set_float32_matmul_precision("medium")

    # Initialize callbacks
    checkpoint_callback = gin.get_configurable(ModelCheckpoint)()
    early_stop_callback = gin.get_configurable(EarlyStopping)()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Initialize logger
    logger = wandb_logger(name=run_name)  # type: ignore

    # Initialize data module
    datamodule = ChocoAudioDataModule(data_path=data_path)

    # Print datamodule information
    print(f"Data path: {datamodule.data_path}")

    # Initialize model
    model = gin.get_configurable(model_class)(
        vocabularies=datamodule.vocabularies, vocab_path=vocab_path
    )

    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,  # type: ignore
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        deterministic=True,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
    )

    # Train the model
    trainer.fit(model, datamodule=datamodule)

    # Test the model
    trainer.test(model, datamodule=datamodule)

    # Close wandb run
    wandb.finish()


def bind_overrides(params: dict | None):
    """Bind a dictionary of overrides to gin parameters."""
    if params is None:
        return
    for key, value in params.items():
        if value is not None:
            gin.bind_parameter(key, value)



@gin.configurable
def main(model_name: str, run_name: str, params: dict | None = None):
    """
    Minimal main function for both CLI and API.

    params: optional dictionary of gin parameter overrides.
    """
    # Load default gin config files
    gin.parse_config_file("ACE/trainer.gin")

    # Import all possible models
    from ACE.models.conformer import ConformerModel
    from ACE.models.conformer_decomposed import ConformerDecomposedModel

    # Create a registry of models
    model_registry = {
        "conformer": ConformerModel,
        "conformer_decomposed": ConformerDecomposedModel,
    }

    # Get the model class from the registry
    ModelClass = model_registry.get(model_name)
    if ModelClass is None:
        raise ValueError(f"Unknown model: {model_name}")

    # Apply parameter overrides
    bind_overrides(params)

    # Call train
    train(model_class=ModelClass, run_name=run_name)


# -----------------------------
# CLI entrypoint
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--vocab_path", type=str)
    parser.add_argument("--accelerator", type=str, choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--max_epochs", type=int)
    parser.add_argument("--precision", type=str)
    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--checkpoint_save_top_k", type=int)
    parser.add_argument("--checkpoint_monitor", type=str)
    parser.add_argument("--earlystop_patience", type=int)
    parser.add_argument("--earlystop_monitor", type=str)
    args = parser.parse_args()

    # Map CLI args to gin fully-qualified parameters
    cli_overrides = {
        "train.data_path": args.data_path,
        "train.vocab_path": args.vocab_path,
        "train.max_epochs": args.max_epochs,
        "train.precision": args.precision,
        "train.accelerator": args.accelerator,
        "ModelCheckpoint.dirpath": args.checkpoint_dir,
        "ModelCheckpoint.save_top_k": args.checkpoint_save_top_k,
        "ModelCheckpoint.monitor": args.checkpoint_monitor,
        "EarlyStopping.patience": args.earlystop_patience,
        "EarlyStopping.monitor": args.earlystop_monitor,
    }

    # Call main with CLI overrides
    main(model_name=args.model, run_name=args.name, params=cli_overrides)
