"""Enum for different fields in PartType0."""
from enum import Enum


class FieldType(Enum):
    """Enum for field type."""

    VELOCITY = "Velocities"
    MAGNETIC = "MagneticField"
