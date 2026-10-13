import os
import queue
import sys
import threading

import av
import numpy as np


class VideoWriter:
    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        codec: str = "h264",
        buffer_size: int = 50,
    ):
        self.output_path = output_path
        self._first_frame = True

        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self.queue = queue.Queue(maxsize=buffer_size)
        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream(codec, rate=fps)
        self.stream.width = width
        self.stream.height = height
        self._closed = False  # set by stop()/cancel() so worker exits cleanly
        self._thread = threading.Thread(target=self._writer_worker, daemon=True)
        self._thread.start()

    def _assert_dimensions(self, frame: np.ndarray) -> None:
        assert (
            frame.shape[1] == self.stream.width and frame.shape[0] == self.stream.height
        ), (
            f"Incorrect frame dimensions. Input dimensions: {frame.shape[1]}x{frame.shape[0]}. "
            f"Expected dimensions: {self.stream.width}x{self.stream.height}"
        )

    def add_frame(self, frame: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError(f"video writer is already closed: {self.output_path}")
        self._assert_dimensions(frame)
        self.queue.put(frame)

    def _writer_worker(self) -> None:
        while True:
            frame = self.queue.get()
            # None is the shutdown sentinel; also break if container was closed.
            if frame is None or self._closed:
                break
            self._assert_dimensions(frame)
            frame = av.VideoFrame.from_ndarray(frame, format="bgr24")

            if self._first_frame:
                stderr_fd = sys.stderr.fileno()
                old_stderr = os.dup(stderr_fd)
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, stderr_fd)
                try:
                    packets = self.stream.encode(frame)
                    for packet in packets:
                        self.container.mux(packet)
                finally:
                    os.dup2(old_stderr, stderr_fd)
                    os.close(old_stderr)
                    os.close(devnull)
                    self._first_frame = False
            else:
                packets = self.stream.encode(frame)
                for packet in packets:
                    self.container.mux(packet)

    def _flush_stream(self) -> None:
        packets = self.stream.encode()
        for packet in packets:
            self.container.mux(packet)

    def stop(self) -> str:
        """Drain queued frames, flush the codec, and close the container."""
        if self._closed:
            return self.output_path

        # The sentinel is queued after every pending frame, so joining the worker
        # is the only reliable indication that encoding has finished.  Checking
        # Queue.empty() races with the worker after it dequeues the last frame.
        self.queue.put(None)
        self._thread.join()
        self._flush_stream()
        self._closed = True
        self.container.close()
        return self.output_path

    def cancel(self) -> None:
        """Immediately stops the worker, closes the container, and deletes the file.
        Use this for 'discard' so the partial mp4 doesn't linger on disk."""
        if self._closed:
            return
        self._closed = True
        self.queue.put(None)  # wake worker; it sees _closed and breaks
        self._thread.join(timeout=1.0)
        try:
            self.container.close()
        except Exception:
            pass
        if os.path.exists(self.output_path):
            try:
                os.remove(self.output_path)
            except OSError as e:
                print(f"[VideoWriter] failed to remove {self.output_path}: {e}")

    def __del__(self) -> None:
        if not getattr(self, "_closed", True):
            try:
                self.container.close()
            except Exception:
                pass
