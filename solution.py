from __future__ import annotations

from multiprocessing import shared_memory
from typing import TypeAlias

import numpy as np


__all__ = ["SharedBuffer"]

RingView: TypeAlias = tuple[memoryview, memoryview | None, int, bool]


class SharedBuffer(shared_memory.SharedMemory):
    """
    A cross-process shared-memory ring buffer.

    Single writer, multiple independent readers. Bounded storage with
    reusable space after readers advance.
    """

    _NO_READER = -1

    # Header layout (int64 slots):
    #   0: write_pos
    #   1: buffer_size
    #   2: num_readers
    #   3-5: reserved
    # readers:
    #   6 + i*3:     reader[i] position
    #   6 + i*3 + 1: reader[i] active flag
    #   6 + i*3 + 2: reader[i] reserved

    _GLOBAL_SLOTS = 6
    _SLOTS_PER_READER = 3
    _SLOT_BYTES = 8  # np.int64

    def __init__(
        self,
        name: str,
        create: bool,
        size: int,
        num_readers: int,
        reader: int,
        cache_align: bool = False,
        cache_size: int = 64,
    ):
        #validation
        if cache_align and (cache_size <= 0 or (cache_size & (cache_size - 1)) != 0):
            raise ValueError(
                f"cache_size must be a power of two when cache_align is True, got {cache_size}"
            )

        if reader != self._NO_READER and (reader < 0 or reader >= num_readers):
            raise ValueError(
                f"reader index {reader} out of range [0, {num_readers})"
            )

        #compute sizes
        num_header_slots = self._GLOBAL_SLOTS + num_readers * self._SLOTS_PER_READER
        header_bytes = num_header_slots * self._SLOT_BYTES
        total_size = header_bytes + size

        #init shared memory
        super().__init__(name=name, create=create, size=total_size)

        #local fields
        self.buffer_size = size
        self.num_readers = num_readers
        self.reader = reader
        self._header_bytes = header_bytes
        self._num_header_slots = num_header_slots

        #shared header as int64 array
        self.header = np.ndarray(
            num_header_slots, dtype=np.int64, buffer=self.buf[:header_bytes]
        )

        #payload region
        self._payload = self.buf[header_bytes : header_bytes + size]

        #initialize header on creation
        if create:
            self.header[:] = 0
            self.header[1] = size
            self.header[2] = num_readers

        #cache reader slot index
        self._slot = (self._GLOBAL_SLOTS + reader * self._SLOTS_PER_READER) if reader != self._NO_READER else -1

        #cache for writable amount
        self._cached_writable = size

    def close(self) -> None:
        try:
            super().close()
        except Exception:
            pass

    def __enter__(self) -> "SharedBuffer":
        if self.reader != self._NO_READER: #if not writer
            self.set_reader_active(True)
        return self

    def __exit__(self, *_):
        if self.reader != self._NO_READER: #if not writer
            self.set_reader_active(False)
        self.close()

    def _reader_slot(self) -> int:
        return self._GLOBAL_SLOTS + self.reader * self._SLOTS_PER_READER

    def _assert_reader(self) -> None:
        if self.reader == self._NO_READER: #if writer
            raise RuntimeError("Operation not allowed on a writer-only instance")

    def int_to_pos(self, value: int) -> int:
        return value % self.buffer_size #ring buffer wraparound

    def update_reader_pos(self, new_reader_pos: int) -> None:
        self._assert_reader()
        self.header[self._slot] = new_reader_pos

    def set_reader_active(self, active: bool) -> None:
        self._assert_reader()
        self.header[self._slot + 1] = 1 if active else 0

    def is_reader_active(self) -> bool:
        self._assert_reader()
        return bool(self.header[self._slot + 1])

    def update_write_pos(self, new_writer_pos: int) -> None:
        self.header[0] = new_writer_pos

    def inc_writer_pos(self, inc_amount: int) -> None:
        self.header[0] += inc_amount

    def inc_reader_pos(self, inc_amount: int) -> None:
        self._assert_reader()
        self.header[self._slot] += inc_amount

    def get_write_pos(self) -> int:
        return int(self.header[0])

    def compute_max_amount_writable(self, force_rescan: bool = False) -> int:
        write_pos = int(self.header[0])
        min_reader_pos = None

        for i in range(self.num_readers):
            slot = self._GLOBAL_SLOTS + i * self._SLOTS_PER_READER
            active = bool(self.header[slot + 1])
            if active:
                rpos = int(self.header[slot])
                if min_reader_pos is None or rpos < min_reader_pos:
                    min_reader_pos = rpos

        if min_reader_pos is None:
            self._cached_writable = self.buffer_size
        else:
            used = write_pos - min_reader_pos
            self._cached_writable = self.buffer_size - used

        return self._cached_writable

    def jump_to_writer(self) -> None:
        self._assert_reader()
        self.header[self._slot] = self.header[0]

    def calculate_pressure(self) -> int:
        write_pos = int(self.header[0])
        min_reader_pos = None

        for i in range(self.num_readers):
            slot = self._GLOBAL_SLOTS + i * self._SLOTS_PER_READER
            active = bool(self.header[slot + 1])
            if active:
                rpos = int(self.header[slot])
                if min_reader_pos is None or rpos < min_reader_pos:
                    min_reader_pos = rpos

        if min_reader_pos is None:
            return 0

        used = write_pos - min_reader_pos
        return int(used * 100 / self.buffer_size)

    def _make_ring_view(self, start_pos: int, actual: int) -> RingView:
        if actual == 0:
            empty = memoryview(bytearray(0))
            return (empty, None, 0, False)

        phys_start = self.int_to_pos(start_pos)
        end = phys_start + actual

        #if the view does not wrap around the end of the buffer, return a single contiguous view
        if end <= self.buffer_size:
            mv1 = self._payload[phys_start:end]
            return (mv1, None, actual, False)
        #if the view wraps around, return two views for the two contiguous segments
        else:
            first_len = self.buffer_size - phys_start
            mv1 = self._payload[phys_start:self.buffer_size]
            mv2 = self._payload[0:actual - first_len]
            return (mv1, mv2, actual, True)

    def expose_writer_mem_view(self, size: int) -> RingView:
        actual = min(size, self._cached_writable)
        write_pos = int(self.header[0])
        return self._make_ring_view(write_pos, actual)

    def expose_reader_mem_view(self, size: int) -> RingView:
        self._assert_reader()

        write_pos = int(self.header[0])
        reader_pos = int(self.header[self._slot])
        available = write_pos - reader_pos

        # Reader has fallen behind beyond retention — resync
        if available > self.buffer_size:
            self.header[self._slot] = write_pos
            reader_pos = write_pos
            available = 0

        actual = min(size, available)
        return self._make_ring_view(reader_pos, actual)

    def simple_write(self, writer_mem_view: RingView, src: object) -> None:
        mv1, mv2, actual, split = writer_mem_view
        if actual == 0:
            return

        src_bytes = bytes(src)
        copy_len = min(len(src_bytes), actual)

        if not split or mv2 is None:
            mv1[:copy_len] = src_bytes[:copy_len]
        else:
            first_len = len(mv1)
            mv1[:first_len] = src_bytes[:first_len]
            remaining = copy_len - first_len
            if remaining > 0:
                mv2[:remaining] = src_bytes[first_len:first_len + remaining]

    def simple_read(self, reader_mem_view: RingView, dst: object) -> None:
        mv1, mv2, actual, split = reader_mem_view
        if actual == 0:
            return

        dst_view = memoryview(dst).cast("B")
        copy_len = min(len(dst_view), actual)

        if not split or mv2 is None:
            dst_view[:copy_len] = mv1[:copy_len]
        else:
            first_len = min(len(mv1), copy_len)
            dst_view[:first_len] = mv1[:first_len]
            remaining = copy_len - first_len
            if remaining > 0:
                dst_view[first_len:first_len + remaining] = mv2[:remaining]

    def write_array(self, arr: np.ndarray) -> int:
        nbytes = arr.nbytes
        writable = self.compute_max_amount_writable()
        if writable < nbytes:
            return 0

        view = self.expose_writer_mem_view(nbytes)
        self.simple_write(view, memoryview(arr).cast("B"))
        self.inc_writer_pos(nbytes)
        return nbytes

    def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
        view = self.expose_reader_mem_view(nbytes)
        if view[2] < nbytes:
            return np.array([], dtype=dtype)

        dst = bytearray(nbytes)
        self.simple_read(view, dst)
        self.inc_reader_pos(nbytes)
        return np.frombuffer(bytes(dst), dtype=dtype)


def _release_views(*views: memoryview | None) -> None:
    for v in views:
        if v is None:
            continue
        try:
            v.release()
        except Exception:
            pass