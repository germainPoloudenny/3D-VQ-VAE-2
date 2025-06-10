import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from typing import Optional

class AmplitudeDataset(Dataset):
    def __init__(self, h5_path: str, data_key: str = "amplitudes", hkl_max_index: int = 10):
        self.h5_path = h5_path
        self.data_key = data_key
        self.hkl_max_index = hkl_max_index
        with h5py.File(h5_path, "r") as f:
            self.length = len(f[data_key])

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> torch.Tensor:
        with h5py.File(self.h5_path, "r") as f:
            amplitude = f[self.data_key][idx]

        amplitude = torch.tensor(amplitude, dtype=torch.float32)

        side = 2 * self.hkl_max_index + 1

        base = side ** 2
        num_valid_slices = amplitude.numel() // base
        amplitude = amplitude.reshape(side, side, num_valid_slices)

        amplitude = amplitude.unsqueeze(0)
        print("amplitude:", amplitude.shape) 

        return amplitude, num_valid_slices
class AmplitudeDataModule(pl.LightningDataModule):
    """PyTorch Lightning datamodule wrapping :class:`AmplitudeDataset`."""

    def __init__(
        self,
        path: str,
        batch_size: int = 16,
        train_frac: float = 0.95,
        num_workers: int = 6,
        data_key: str = "amplitudes",
        hkl_max_index: int = 10,
    ) -> None:
        super().__init__()
        assert 0 <= train_frac <= 1
        self.path = path
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.num_workers = num_workers
        self.data_key = data_key
        self.hkl_max_index = hkl_max_index

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = AmplitudeDataset(
            self.path,
            data_key=self.data_key,
            hkl_max_index=self.hkl_max_index,
        )
        train_len = int(len(dataset) * self.train_frac)
        val_len = len(dataset) - train_len
        train_split, val_split = random_split(dataset, [train_len, val_len])
        self.train_dataset = train_split
        self.val_dataset = val_split

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )