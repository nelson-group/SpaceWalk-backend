"""Utilities to operate on data."""

import logging
import os
from pathlib import Path

import h5py
import numpy as np

from tng_sv.data.dir import (
    get_delaunay_path,
    get_delaunay_time_symlink_path,
    get_resampled_delaunay_time_symlink_path,
    get_snapshot_index_path,
    path_to_resampled_file,
)
from tng_sv.data.field_type import FieldType

logger = logging.getLogger(__name__)


def combine_snapshot(simulation_name: str, snapshot_idx: int, field_type: FieldType) -> Path:
    """Combine a snapshot."""
    _dir = get_snapshot_index_path(simulation_name, snapshot_idx)

    file_names = list(_dir.glob("snap*.hdf5"))
    if len(file_names) == 0:
        raise ValueError(f"No files found to combine. Searched at: {_dir}")

    coordinates = np.zeros((0, 3))
    values = np.zeros((0, 3))

    for file_name in file_names:
        with h5py.File(file_name, "r") as f_in:
            try:
                coordinates = np.concatenate((coordinates, f_in["PartType0"]["Coordinates"]), axis=0)
                values = np.concatenate((values, f_in["PartType0"][field_type.value]), axis=0)
            except KeyError:
                continue

    f_out = h5py.File(
        _dir.joinpath(f"combined_{field_type.value}_{simulation_name.lower()}_{snapshot_idx:03d}.hdf5"), "w"
    )
    f_out.create_group("PartType0")
    f_out["PartType0"].create_dataset("Coordinates", coordinates.shape, float, coordinates)
    f_out["PartType0"].create_dataset(field_type.value, values.shape, float, values)

    file_name = Path(f_out.filename)
    f_out.close()
    return file_name


def create_delaunay_symlink(simulation_name: str, snapshot_idx: int, field_type: FieldType) -> None:
    """Create symlink for combined delauny for loading of timed data."""
    os.symlink(
        get_delaunay_path(simulation_name, snapshot_idx, field_type),
        get_delaunay_time_symlink_path(simulation_name, snapshot_idx, field_type),
    )


def create_resampled_delaunay_symlink(simulation_name: str, snapshot_idx: int, field_type: FieldType) -> None:
    """Create symlink for resampled delauny for loading of timed data."""
    path = path_to_resampled_file(simulation_name, snapshot_idx, field_type)
    if path is None:
        logger.error(
            "Couldn't create symlink for %(name)s, %(idx)d, %(type)s, because the target file doesn't exist.",
            {"name": simulation_name, "idx": snapshot_idx, "type": field_type.value},
        )
        return
    os.symlink(
        path,
        get_resampled_delaunay_time_symlink_path(simulation_name, snapshot_idx, field_type),
    )
