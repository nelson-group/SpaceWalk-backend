"""Module to generate data dir paths."""

import glob
from pathlib import Path

from tng_sv.data import DATADIR


def get_snapshot_index_path(simulation_name: str, snapshot_idx: int) -> Path:
    """Return path to snapshot dir given simulation_name and snapshot_idx."""
    return DATADIR.joinpath(f"{simulation_name}/{snapshot_idx:03d}/")


def get_snapshot_combination_index_path(simulation_name: str, snapshot_idx: int) -> Path:
    """Return path to a combined simulation snapshot."""
    _dir = DATADIR.joinpath(f"{simulation_name}/{snapshot_idx:03d}/")
    return Path(glob.glob(f"{_dir}/combined*.hdf5")[0])
