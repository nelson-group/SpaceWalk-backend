"""Enum for different parts."""
from enum import Enum


class PartType(Enum):
    """Enum for part type."""

    GAS = "PartType0"
    BLACKHOLE = "PartType5"

    @property
    def filename(self) -> str:
        """Return filename for enum."""
        return _FILENAMES[self.value]


_FILENAMES = {PartType.GAS.value: "", PartType.BLACKHOLE.value: "Blackhole_"}
