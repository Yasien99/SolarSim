"""Input adapters: turning recordings into the canonical motion contract."""

from .recording import (
    MocapSpec,
    PowerSpec,
    Recording,
    RecordingSpec,
    read_recording,
)

__all__ = [
    "MocapSpec",
    "PowerSpec",
    "Recording",
    "RecordingSpec",
    "read_recording",
]
