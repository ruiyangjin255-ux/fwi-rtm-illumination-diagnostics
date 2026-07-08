from __future__ import annotations

from pathlib import Path
from typing import Any

from fwi_visionfm.datasets import discover_npz_samples, load_npz_sample
from fwi_visionfm.torch_backend import require_torch_backend


class NPZTorchDataset:
    def __init__(self, data_dir_or_paths: str | Path | list[str | Path]) -> None:
        self.torch = require_torch_backend()
        if isinstance(data_dir_or_paths, list):
            self.paths = [Path(path) for path in data_dir_or_paths]
        else:
            self.paths = discover_npz_samples(data_dir_or_paths)
        if not self.paths:
            raise ValueError(f"no npz samples found in {data_dir_or_paths}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = load_npz_sample(self.paths[index])
        return {
            "records": self.torch.as_tensor(sample.records, dtype=self.torch.float32),
            "velocity": self.torch.as_tensor(sample.velocity, dtype=self.torch.float32),
            "source_positions": self.torch.as_tensor(sample.source_positions, dtype=self.torch.float32),
            "path": str(self.paths[index]),
        }


def build_torch_dataloader(
    data_dir_or_paths: str | Path | list[str | Path],
    *,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 0,
) -> Any:
    torch = require_torch_backend()
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        NPZTorchDataset(data_dir_or_paths),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )
