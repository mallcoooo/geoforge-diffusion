from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader


class CyclingDataLoader:
    """Repeat a deterministic dataset while exposing resumable iterator state."""

    def __init__(self, loader: DataLoader[Any]) -> None:
        if len(loader) == 0:
            raise ValueError("CyclingDataLoader requires at least one batch")
        if not hasattr(loader.dataset, "set_epoch"):
            raise TypeError("CyclingDataLoader dataset must implement set_epoch(epoch)")
        self.loader = loader
        self.epoch = 0
        self.batches_consumed = 0
        self._iterator: Any = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._reset_iterator()
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.epoch += 1
            self.batches_consumed = 0
            self._reset_iterator()
            batch = next(self._iterator)
        self.batches_consumed += 1
        return batch

    def state_dict(self) -> dict[str, int]:
        return {
            "epoch": self.epoch,
            "batches_consumed": self.batches_consumed,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        epoch = int(state["epoch"])
        batches_consumed = int(state["batches_consumed"])
        if epoch < 0 or not 0 <= batches_consumed <= len(self.loader):
            raise ValueError("Invalid cycling loader state")
        self.epoch = epoch
        self.batches_consumed = 0
        self._reset_iterator()
        for _ in range(batches_consumed):
            next(self._iterator)
            self.batches_consumed += 1

    def _reset_iterator(self) -> None:
        self.loader.dataset.set_epoch(self.epoch)
        self._iterator = iter(self.loader)
