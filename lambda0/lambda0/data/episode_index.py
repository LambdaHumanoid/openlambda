"""Low-memory random access to compact episode metadata stored in Parquet."""

from __future__ import annotations

import bisect
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


class EpisodeIndex:
    """Read individual episode records without materializing the full index."""

    def __init__(self, path: str | Path, cache_size: int = 2) -> None:
        self.path = Path(path)
        self.cache_size = int(cache_size)
        if self.cache_size < 1:
            raise ValueError("Episode-index cache_size must be positive.")
        if not self.path.is_file():
            raise FileNotFoundError(f"Episode index does not exist: {self.path}")
        self._parquet: pq.ParquetFile | None = None
        self._row_group_starts: tuple[int, ...] = ()
        self._length = -1
        self._cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_parquet"] = None
        state["_cache"] = OrderedDict()
        return state

    def _open(self) -> pq.ParquetFile:
        if self._parquet is None:
            parquet = pq.ParquetFile(self.path, memory_map=True)
            starts = []
            offset = 0
            for index in range(parquet.num_row_groups):
                starts.append(offset)
                offset += parquet.metadata.row_group(index).num_rows
            self._parquet = parquet
            self._row_group_starts = tuple(starts)
            self._length = offset
        return self._parquet

    def __len__(self) -> int:
        self._open()
        return self._length

    def _load_row_group(self, index: int) -> list[dict[str, Any]]:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        rows = self._open().read_row_group(index).to_pylist()
        self._cache[index] = rows
        self._cache.move_to_end(index)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return rows

    def __getitem__(self, index: int) -> dict[str, Any]:
        length = len(self)
        index = int(index)
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError(index)
        group = bisect.bisect_right(self._row_group_starts, index) - 1
        row = self._load_row_group(group)[
            index - self._row_group_starts[group]
        ]
        return {key: value for key, value in row.items() if value is not None}


__all__ = ["EpisodeIndex"]
