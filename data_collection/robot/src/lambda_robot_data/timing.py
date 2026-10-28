# SPDX-License-Identifier: Apache-2.0
"""MPEG-TS PTS extraction and robot-ingress timestamp sidecar."""

from __future__ import annotations
import struct
from dataclasses import dataclass
from typing import Optional

MPEG_TS_PACKET_SIZE = 188
MPEG_CLOCK_HZ = 90_000
MPEG_PTS_WRAP = 1 << 33
PROTOCOL_VERSION = 1
_FRAME_MAGIC = b"G1FT"
_FRAME_STRUCT = struct.Struct("!4sB3xQQQQQ")
FRAME_TIMESTAMP_PACKET_SIZE = _FRAME_STRUCT.size


@dataclass(frozen=True)
class FrameIngressTimestamp:
    epoch: int
    sequence: int
    pts_90khz: int
    g1_monotonic_ns: int
    g1_realtime_ns: int


def pack_frame_timestamp(value: FrameIngressTimestamp) -> bytes:
    return _FRAME_STRUCT.pack(
        _FRAME_MAGIC,
        PROTOCOL_VERSION,
        value.epoch,
        value.sequence,
        value.pts_90khz,
        value.g1_monotonic_ns,
        value.g1_realtime_ns,
    )


def unpack_frame_timestamp(payload: bytes) -> FrameIngressTimestamp:
    if len(payload) != _FRAME_STRUCT.size:
        raise ValueError("invalid frame timestamp packet size")
    magic, version, epoch, sequence, pts, monotonic_ns, realtime_ns = (
        _FRAME_STRUCT.unpack(payload)
    )
    if magic != _FRAME_MAGIC or version != PROTOCOL_VERSION:
        raise ValueError("invalid frame timestamp protocol header")
    if pts >= MPEG_PTS_WRAP or monotonic_ns <= 0 or realtime_ns <= 0:
        raise ValueError("invalid frame timestamp values")
    return FrameIngressTimestamp(
        epoch=epoch,
        sequence=sequence,
        pts_90khz=pts,
        g1_monotonic_ns=monotonic_ns,
        g1_realtime_ns=realtime_ns,
    )


def decode_pts_90khz(data: bytes) -> int:
    if len(data) != 5:
        raise ValueError("PTS field must contain five bytes")
    return (
        ((data[0] >> 1) & 0x07) << 30
        | data[1] << 22
        | (data[2] >> 1) << 15
        | data[3] << 7
        | data[4] >> 1
    )


class MpegTsPtsExtractor:
    """Extract video PES presentation timestamps from MPEG-TS datagrams."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._last_pts: Optional[int] = None

    def feed(self, payload: bytes) -> tuple[int, ...]:
        self._buffer.extend(payload)
        values = []
        while len(self._buffer) >= MPEG_TS_PACKET_SIZE:
            if self._buffer[0] != 0x47:
                sync = self._buffer.find(0x47, 1)
                if sync < 0:
                    del self._buffer[:-1]
                    break
                del self._buffer[:sync]
                continue
            packet = bytes(self._buffer[:MPEG_TS_PACKET_SIZE])
            del self._buffer[:MPEG_TS_PACKET_SIZE]
            pts = self._packet_pts(packet)
            if pts is not None and pts != self._last_pts:
                self._last_pts = pts
                values.append(pts)
        return tuple(values)

    @staticmethod
    def _packet_pts(packet: bytes) -> Optional[int]:
        if len(packet) != MPEG_TS_PACKET_SIZE or packet[0] != 0x47:
            return None
        if packet[1] & 0x80 or not packet[1] & 0x40:
            return None
        adaptation_control = (packet[3] >> 4) & 0x03
        if adaptation_control in (0, 2):
            return None
        payload_start = 4
        if adaptation_control == 3:
            payload_start += 1 + packet[4]
        if payload_start + 14 > len(packet):
            return None
        payload = packet[payload_start:]
        if payload[:3] != b"\x00\x00\x01":
            return None
        stream_id = payload[3]
        if not 0xE0 <= stream_id <= 0xEF:
            return None
        if not payload[7] & 0x80 or payload[8] < 5:
            return None
        return decode_pts_90khz(payload[9:14])
