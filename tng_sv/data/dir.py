"""Module to generate data dir paths."""

import contextlib
import glob
import logging
import os
from pathlib import Path
from typing import Optional

from tng_sv.data import DATADIR
from tng_sv.data.field_type import FieldType

logger = logging.getLogger(__name__)


def get_snapshot_index_path(simulation_name: str, snapshot_idx: int) -> Path:
    """Return path to snapshot dir given simulation_name and snapshot_idx."""
    return DATADIR.joinpath(f"{simulation_name}/{snapshot_idx:03d}/")


def get_snapshot_combination_index_path(simulation_name: str, snapshot_idx: int, field_type: FieldType) -> Path:
    """Return path to a combined simulation snapshot."""
    _dir = DATADIR.joinpath(f"{simulation_name}/{snapshot_idx:03d}/")
    return _dir.joinpath(f"combined_{field_type.value}_{simulation_name.lower()}_{snapshot_idx:03d}.hdf5")


def get_delaunay_path(simulation_name: str, snapshot_idx: int, field_type: FieldType) -> Path:
    """Return path to the delaunay output file."""
    _dir = DATADIR.joinpath(f"{simulation_name}/{snapshot_idx:03d}/")
    return _dir.joinpath(f"combined_{field_type.value}_{simulation_name.lower()}_{snapshot_idx:03d}_delaunay.pvd")


def path_to_resampled_file(simulation_name: str, snapshot_idx: int, field_type: FieldType) -> Optional[Path]:
    """Return path to the delaunay output file."""
    logger.warning("path_to_resampled_file is deprecated. Use get_path_to_resampled_file instead!")
    _dir = DATADIR.joinpath(f"{simulation_name}/{snapshot_idx:03d}/")
    with contextlib.suppress(IndexError):
        return Path(glob.glob(f"{_dir}/combined_{field_type.value}*_{snapshot_idx:03d}_resampled_delaunay.pvd")[0])
    return None


def get_resampled_delaunay_path(simulation_name: str, snapshot_idx: int, field_type: FieldType) -> Path:
    """Return path to the delaunay output file."""
    _dir = DATADIR.joinpath(f"{simulation_name}/{snapshot_idx:03d}/")
    return _dir.joinpath(
        f"combined_{field_type.value}_{simulation_name.lower()}_{snapshot_idx:03d}_resampled_delaunay.pvd"
    )


def get_scalar_field_experiment_path(
    simulation_name: str, snapshot_idx: int, experiment_name: str, field_type_1: FieldType, field_type_2: FieldType
) -> Path:
    """Return path to the experiment output file."""
    return get_snapshot_index_path(simulation_name, snapshot_idx).joinpath(
        f"combined_{field_type_1.value}_{field_type_2.value}_{experiment_name}_{simulation_name}_{snapshot_idx}.pvd"
    )


def get_delaunay_time_symlink_path(simulation_name: str, snapshot_idx: int, field_type: FieldType) -> Path:
    """Get the path to a symlink target for a specific time of a  delaunay."""
    _dir = DATADIR.joinpath(f"{simulation_name}/{field_type.value}_delaunay_time_data/")
    if not _dir.exists():
        os.makedirs(_dir)

    return _dir.joinpath(f"combined_{field_type.value}_delaunay.{snapshot_idx:03d}.pvd")


def get_resampled_delaunay_time_symlink_path(simulation_name: str, snapshot_idx: int, field_type: FieldType) -> Path:
    """Get the path to a symlink target for a specific time of a delaunay."""
    _dir = DATADIR.joinpath(f"{simulation_name}/{field_type.value}_resampled_delaunay_time_data/")
    if not _dir.exists():
        os.makedirs(_dir)

    return _dir.joinpath(f"combined_{field_type.value}_resampled_delaunay.{snapshot_idx:03d}.pvd")


def get_scalar_field_experiment_symlink_path(
    simulation_name: str, snapshot_idx: int, experiment_name: str, field_type_1: FieldType, field_type_2: FieldType
) -> Path:
    """Get the path to a symlink target for a specific time of a scalar field experiment."""
    _dir = DATADIR.joinpath(f"{simulation_name}/{field_type_1.value}_{field_type_2.value}_{experiment_name}_time_data")
    if not _dir.exists():
        os.makedirs(_dir)

    return _dir.joinpath(
        f"combined_{field_type_1.value}_{field_type_2.value}_{experiment_name}_{simulation_name}.{snapshot_idx}.pvd"
    )
