"""Module to generate data dir paths."""

from pathlib import Path

from tng_sv.data import DATADIR


def get_snapshot_index_path(simulation_name: str, snapshot_idx: int) -> Path:
    """Return path to snapshot dir given simulation_name and snapshot_idx."""
    return DATADIR.joinpath(f"{simulation_name}/{snapshot_idx:03d}/")
