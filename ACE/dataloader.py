"""
Dataloader for the Choco dataset to be used for chord recognition tasks.
"""

import logging
from collections import defaultdict
from pathlib import Path

import gin
import lightning as L
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ChocoAudioDataset(Dataset):
    """
    Dataset for loading audio data from the Choco dataset.
    """

    def __init__(
        self,
        data_path: Path,
        track_ids: list,
        data_dict: dict,
        augmentation: bool = False,
    ):
        super().__init__()

        # Data paths
        self.data_path = data_path

        # Augmentation
        self.augmentation = augmentation

        # Data dictionary that maps track IDs to their corresponding files
        self.data_dict = data_dict

        # Data list
        self.data_list = self._get_data_list(track_ids)

    def __len__(self):
        """
        Return the number of excerpts in the dataset.
        """
        return len(self.data_list)

    def _get_data_list(self, track_ids: list) -> list:
        """Cached version of get_data_list"""

        data_list = []
        # Get the list of files for each track ID
        for track_id in track_ids:
            data_list.extend(self.data_dict[track_id])

        # Filter out augmented files if augmentation is disabled
        if self.augmentation is False:
            # if not augmentation, filter out augmented files (pitch transposed)
            data_list = [x for x in data_list if x.stem.endswith("_p+0")]
            # also, do not include overlapping files (only multiple of 20)
            data_list = [
                x
                for x in data_list
                if int(x.stem.split("_")[-2].replace("t", "")) % 20
                == 0  # TODO: make this configurable
            ]

        return data_list

    def __getitem__(self, index):
        """Now uses cached loading"""
        return torch.load(
            self.data_path / self.data_list[index],
            weights_only=False,
            map_location="cpu",
        )


@gin.configurable
class ChocoAudioDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for loading audio data from the Choco dataset.
    """

    def __init__(
        self,
        data_path: str | Path,
        batch_size: int = 64,
        num_workers: int = 0,
        augmentation: bool = False,
        train_ratio: float | None = 0.6,
        val_ratio: float | None = 0.5,
        random_seed: int | None = None,
    ):
        super().__init__()

        # Data paths
        self.data_path = Path(data_path)

        # Data parameters
        self.batch_size = batch_size

        # vocab size
        self.vocabularies = {
            "simplified": 50,
            "root": 13,
            "bass": 13,
            "mode": 8,
            "majmin": 26,
            "onehot": 12,
            "complete": 170,
        }

        # Data loaders parameters
        self.num_workers = num_workers
        self.augmentation = augmentation
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed

        # Get list of all files in the dataset
        isophonics_files = list(self.data_path.glob("Isophonics*.pt"))
        billboard_files = list(self.data_path.glob("Billboard*.pt"))
        marl_files = list(self.data_path.glob("MARL*.pt"))

        # Combine files and create train and test datasets
        self.train_list = isophonics_files + billboard_files
        self.test_list = marl_files

        # Precompute data dictionary
        self.data_dict_train = self._precompute_data_dict(self.train_list)
        self.data_dict_test = self._precompute_data_dict(self.test_list)

        # Log the number of files found
        print(
            f"Found {len(self.train_list)} training files "
            f"and {len(self.test_list)} test files."
        )
        # Log IDs of the training and test datasets
        print(f"Training IDs: {list(self.data_dict_train.keys())[:5]}... ")
        print(f"Test IDs: {list(self.data_dict_test.keys())[:5]}... ")

    def _precompute_data_dict(self, data_list: list) -> dict:
        """
        Precompute the data dictionary by creating a dictionary of track IDs and their
        corresponding files.

        Returns:
            dict: Dictionary of track IDs and their corresponding files as a list
        """
        data_dict = defaultdict(list)

        for file_path in data_list:
            track_id = "_".join(file_path.stem.split("_")[:-2])
            data_dict[track_id].append(file_path)

        return data_dict

    def _split_dataset(self):
        """
        Split dataset ensuring all chunks and augmentations from the same song stay
        together.

        Args:
            train_ratio (float): Ratio of data to use for training (0.0 to 1.0)
            seed (int): Random seed for reproducibility

        Returns:
            tuple: Lists of train, validation, and test files
        """
        # Convert to list for splitting
        song_ids = list(self.data_dict_train.keys())

        # Split song IDs into train/temp and then temp into val/test
        train_songs, val_songs = train_test_split(
            song_ids,
            train_size=self.train_ratio,
            random_state=self.random_seed,
            shuffle=True,
        )

        test_songs = list(self.data_dict_test.keys())

        # val_songs, test_songs = train_test_split(
        #     temp_songs, train_size=self.val_ratio, random_state=self.random_seed
        # )

        logger.info(
            f"Train set: {len(train_songs)} songs, "
            f"Validation set: {len(val_songs)} songs, "
            f"Test set: {len(test_songs)} songs"
        )

        return train_songs, val_songs, test_songs

    def prepare_data(self):
        self.train_track_ids, self.val_track_ids, self.test_track_ids = (
            self._split_dataset()
        )

    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = ChocoAudioDataset(
                self.data_path,
                self.train_track_ids,
                self.data_dict_train,
                augmentation=self.augmentation,
            )
            self.val_dataset = ChocoAudioDataset(
                self.data_path,
                self.val_track_ids,
                self.data_dict_train,
                augmentation=False,
            )

        if stage == "test":
            self.test_dataset = ChocoAudioDataset(
                self.data_path,
                self.test_track_ids,
                self.data_dict_test,
                augmentation=False,
            )

    def train_dataloader(self):
        """
        Return the training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """
        Return the validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        """
        Return the test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


if __name__ == "__main__":
    data_path = "/home/must/Documents/marl_data/cache/cqt_augment_fix"
    data_module = ChocoAudioDataModule(
        data_path,
        batch_size=16,
        num_workers=8,
        random_seed=8031,
        train_ratio=0.85,
        augmentation=True,
    )
    # Logger in debug mode
    logging.basicConfig(level=logging.DEBUG)

    # prepare data
    data_module.prepare_data()
    # a, b, c = data_module._split_dataset()
    data_module.setup("fit")
    print(len(data_module.train_dataset))
    print(len(data_module.val_dataset))

    data_module.setup("test")
    print(len(data_module.test_dataset))
