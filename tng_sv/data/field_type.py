"""Enum for different fields in PartType0."""
from enum import Enum
from typing import Tuple


class FieldType(Enum):
    """Enum for field type."""

    VELOCITY = "Velocities"
    MAGNETIC = "MagneticField"
    DENSITY = "Density"
    METALLICITY = "GFM_Metallicity"
    PARTICLEIDS = "ParticleIDs"
    ALL = "All"

    @property
    def dim(self) -> Tuple:
        """get dim"""
        res = _DIMS[self.value]
        if res is None:
            raise ValueError(f"{self.value} should not be used with dims")
        return tuple(res)


_DIMS = {
    FieldType.VELOCITY.value: [0, 3],
    FieldType.MAGNETIC.value: [0, 3],
    FieldType.DENSITY.value: [0],
    FieldType.METALLICITY.value: [0],
    FieldType.PARTICLEIDS.value: [0],
    FieldType.ALL.value: None,
}
