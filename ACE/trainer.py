"""
Training script for the all the models.

Running example:
    python -m ACE.trainer

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
    model: L.LightningModule,
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
    model = gin.get_configurable(model)(vocabularies=datamodule.vocabularies)

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


@gin.configurable
def main(model_name: str, run_name: str, cache_name: str = "cqt_augment_long"):
    """Main function to run training.
    Args:
        model_name: Name of the model to train.
        data_path: Path to the data.
    """
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
        raise ValueError(f"Model {model_name} not found in registry")

    # Initialize gin
    gin.parse_config_file(Path(f"ACE/models/{model_name}.gin").__str__())

    # Run training
    data_path = Path("/home/must/Documents/marl_data/cache/") / cache_name
    train(model=ModelClass, data_path=data_path, run_name=run_name)


if __name__ == "__main__":
    gin.parse_config_file("ACE/trainer.gin")
    parser = argparse.ArgumentParser(description="Train a model with ACE.")
    parser.add_argument(
        "--model", type=str, required=True, help="Name of the model to train."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the run for logging purposes.",
    )
    main(
        model_name=parser.parse_args().model,
        run_name=parser.parse_args().name,
    )
