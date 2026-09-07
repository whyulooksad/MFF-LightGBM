"""把多个零散、乱序、重复、部分重叠的 TCP 包，还原成尽可能正确的连续字节流。"""

from __future__ import annotations

from dataclasses import dataclass, field


def _unwrap(seq: int, reference: int) -> int:
    """Map a 32-bit TCP sequence near an already unwrapped reference."""
    base = reference & ~0xFFFFFFFF
    candidates = (base + seq, base + seq - (1 << 32), base + seq + (1 << 32))
    return min(candidates, key=lambda value: abs(value - reference))


@dataclass
class TCPStream:
    """Collect TCP segments and return only contiguous, non-duplicated chunks.

    Overlapping bytes keep the first observed value.  A capture gap creates a
    separate chunk so a TLS parser never silently joins bytes that were absent.
    """

    segments: list[tuple[int, bytes]] = field(default_factory=list)
    reference: int | None = None

    def add(self, seq: int, payload: bytes, syn: bool = False) -> None:
        if not payload:
            return
        data_seq = (int(seq) + (1 if syn else 0)) & 0xFFFFFFFF
        if self.reference is None:
            self.reference = data_seq
            unwrapped = data_seq
        else:
            unwrapped = _unwrap(data_seq, self.reference)
            self.reference = max(self.reference, unwrapped + len(payload))
        self.segments.append((unwrapped, bytes(payload)))

    def contiguous_chunks(self) -> list[bytes]:
        if not self.segments:
            return []
        chunks: list[bytes] = []
        current = bytearray()
        end: int | None = None
        for start, data in sorted(self.segments, key=lambda item: item[0]):
            if end is None or start > end:
                if current:
                    chunks.append(bytes(current))
                current = bytearray(data)
                end = start + len(data)
                continue
            overlap = end - start
            if overlap < len(data):
                current.extend(data[overlap:])
                end += len(data) - overlap
        if current:
            chunks.append(bytes(current))
        return chunks
