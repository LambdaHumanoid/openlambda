"""Timestamp-aware video decoding shared by dataset adapters."""
from fractions import Fraction
from pathlib import Path
import av
import numpy as np

def decode(path: Path, indices: list[int] | None = None) -> list[np.ndarray]:
    container = av.open(str(path), options={"threads": "2"})
    try:
        if indices is None:
            frames = [
                frame.to_ndarray(format="rgb24")
                for frame in container.decode(video=0)
            ]
        else:
            wanted = [int(value) for value in indices]
            if not wanted:
                raise ValueError("source frame indices must not be empty")
            if any(value < 0 for value in wanted):
                raise IndexError("source frame indices must be non-negative")
            stream = container.streams.video[0]
            if stream.average_rate is None or stream.time_base is None:
                raise ValueError(
                    f"Video lacks timing metadata required for seeking: {path}"
                )
            rate = Fraction(stream.average_rate)
            time_base = Fraction(stream.time_base)
            start_time = int(stream.start_time or 0)
            first_index = min(wanted)
            last_index = max(wanted)
            seek_index = max(0, first_index - 1)
            seek_pts = start_time + int(
                Fraction(seek_index, 1) / rate / time_base
            )
            container.seek(
                seek_pts,
                stream=stream,
                backward=True,
                any_frame=False,
            )
            target = set(wanted)
            selected: dict[int, np.ndarray] = {}
            for frame in container.decode(video=0):
                if frame.pts is None:
                    raise ValueError(f"Decoded frame lacks a timestamp: {path}")
                index = round(
                    Fraction(int(frame.pts) - start_time, 1)
                    * time_base
                    * rate
                )
                if index < first_index:
                    continue
                if index in target:
                    selected[index] = frame.to_ndarray(format="rgb24")
                if index >= last_index:
                    break
            missing = sorted(target.difference(selected))
            if missing:
                raise IndexError(
                    f"source frame index {missing[0]} is outside video {path}"
                )
            frames = [selected[value] for value in wanted]
    finally:
        container.close()
    if not frames:
        raise ValueError(f"Video contains no frames: {path}")
    return frames
