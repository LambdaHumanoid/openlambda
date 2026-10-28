# SPDX-License-Identifier: Apache-2.0
"""Durable record stream used by the G1-local Episode agent."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import zlib
from pathlib import Path
from typing import Dict, Iterator, Optional

import msgpack

FILE_MAGIC = b"G1EP\x02\x00\x00\x00"
RECORD_HEADER = struct.Struct("!II")
MAX_RECORD_BYTES = 4 * 1024 * 1024


class EpisodeWireError(ValueError):
    """A corrupt or invalid local Episode record stream."""


class RecordWriter:
    """Append checksummed MessagePack objects to a newly-created file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stream = path.open("xb", buffering=256 * 1024)
        self.stream.write(FILE_MAGIC)
        self.records = 0
        self.payload_bytes = 0
        self.closed = False

    def write(self, value: Dict[str, object]) -> None:
        if self.closed:
            raise EpisodeWireError("cannot write a closed Episode stream")
        payload = msgpack.packb(value, use_bin_type=True)
        if not 0 < len(payload) <= MAX_RECORD_BYTES:
            raise EpisodeWireError("Episode record size is outside the supported range")
        checksum = zlib.crc32(payload) & 0xFFFFFFFF
        self.stream.write(RECORD_HEADER.pack(len(payload), checksum))
        self.stream.write(payload)
        self.records += 1
        self.payload_bytes += len(payload)

    def close(self, *, durable: bool = True) -> None:
        if self.closed:
            return
        self.stream.flush()
        if durable:
            os.fsync(self.stream.fileno())
        self.stream.close()
        self.closed = True


def read_records(path: Path) -> Iterator[Dict[str, object]]:
    """Yield every complete, checksummed MessagePack record."""
    with path.open("rb") as stream:
        if stream.read(len(FILE_MAGIC)) != FILE_MAGIC:
            raise EpisodeWireError(
                "Episode stream has an invalid header: {}".format(path)
            )
        index = 0
        while True:
            header = stream.read(RECORD_HEADER.size)
            if not header:
                return
            if len(header) != RECORD_HEADER.size:
                raise EpisodeWireError("Episode stream has a truncated record header")
            length, expected_checksum = RECORD_HEADER.unpack(header)
            if not 0 < length <= MAX_RECORD_BYTES:
                raise EpisodeWireError("Episode stream record length is invalid")
            payload = stream.read(length)
            if len(payload) != length:
                raise EpisodeWireError("Episode stream has a truncated record payload")
            checksum = zlib.crc32(payload) & 0xFFFFFFFF
            if checksum != expected_checksum:
                raise EpisodeWireError("Episode stream record checksum mismatch")
            try:
                value = msgpack.unpackb(payload, raw=False, strict_map_key=True)
            except (TypeError, ValueError, msgpack.ExtraData) as exc:
                raise EpisodeWireError(
                    "Episode stream record is not MessagePack"
                ) from exc
            if not isinstance(value, dict):
                raise EpisodeWireError("Episode stream record root must be an object")
            index += 1
            yield value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(
    path: Path, value: Dict[str, object], *, durable: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(".{}.{}.tmp".format(path.name, os.getpid()))
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        if durable:
            os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def directory_size(path: Optional[Path]) -> int:
    if path is None or not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
