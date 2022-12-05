"""Enum for different fields in PartType0."""
from enum import Enum
from typing import Tuple


class FieldType(Enum):
    """Enum for field type."""

    VELOCITY = "Velocities"
    MAGNETIC = "MagneticField"
    DENSITY = "Density"
    METALLICITY = "GFM_Metallicity"

    @property
    def dim(self) -> Tuple:
        """get dim"""
        return tuple(_DIMS[self.value])


_DIMS = {
    FieldType.VELOCITY.value: [0, 3],
    FieldType.MAGNETIC.value: [0, 3],
    FieldType.DENSITY.value: [0],
    FieldType.METALLICITY.value: [0],
}
