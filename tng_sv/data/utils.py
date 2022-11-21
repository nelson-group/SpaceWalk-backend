"""Utilities to operate on data."""

import os
from pathlib import Path

import h5py
import numpy as np

from tng_sv.data.dir import get_delaunay_path, get_delaunay_time_symlink_path, get_snapshot_index_path


def combine_snapshot(simulation_name: str, snapshot_idx: int) -> Path:
    """Combine a snapshot."""
    _dir = get_snapshot_index_path(simulation_name, snapshot_idx)

    file_names = list(_dir.glob("snap*.hdf5"))
    if len(file_names) == 0:
        raise ValueError(f"No files found to combine. Searched at: {_dir}")

    coordinates = np.zeros((0, 3))
    velocities = np.zeros((0, 3))

    for file_name in file_names:
        with h5py.File(file_name, "r") as f_in:
            try:
                coordinates = np.concatenate((coordinates, f_in["PartType0"]["Coordinates"]), axis=0)
                velocities = np.concatenate((velocities, f_in["PartType0"]["Velocities"]), axis=0)
            except KeyError:
                continue

    f_out = h5py.File(_dir.joinpath(f"combined_{simulation_name.lower()}.hdf5"), "w")
    f_out.create_group("PartType0")
    f_out["PartType0"].create_dataset("Coordinates", coordinates.shape, float, coordinates)
    f_out["PartType0"].create_dataset("Velocities", velocities.shape, float, velocities)

    file_name = Path(f_out.filename)
    f_out.close()
    return file_name


def create_delaunay_symlink(simulation_name: str, snapshot_idx: int) -> None:
    """Create symlink for combined delauny for loading of timed data."""
    os.symlink(
        get_delaunay_path(simulation_name, snapshot_idx), get_delaunay_time_symlink_path(simulation_name, snapshot_idx)
    )
