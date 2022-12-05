"""Utilities to operate on data."""

import logging
import os
import platform
from pathlib import Path

import h5py
import numpy as np

from tng_sv.data.dir import (
    get_bound_info_file,
    get_delaunay_path,
    get_delaunay_time_symlink_path,
    get_resampled_delaunay_path,
    get_resampled_delaunay_time_symlink_path,
    get_scalar_field_experiment_path,
    get_scalar_field_experiment_symlink_path,
    get_snapshot_index_path,
)
from tng_sv.data.field_type import FieldType
from tng_sv.data.part_type import PartType

logger = logging.getLogger(__name__)


def combine_snapshot(simulation_name: str, snapshot_idx: int, part_type: PartType, field_type: FieldType) -> Path:
    """Combine a snapshot."""
    _dir = get_snapshot_index_path(simulation_name, snapshot_idx)

    file_names = list(_dir.glob("snap*.hdf5"))
    if len(file_names) == 0:
        raise ValueError(f"No files found to combine. Searched at: {_dir}")

    coordinates = np.zeros((0, 3))
    values = np.zeros(field_type.dim)

    for file_name in file_names:
        with h5py.File(file_name, "r") as f_in:
            try:
                coordinates = np.concatenate((coordinates, f_in[part_type.value]["Coordinates"]), axis=0)
                if field_type.dim[-1] == 3:
                    values = np.concatenate((values, f_in[part_type.value][field_type.value]), axis=0)
                else:
                    values = np.concatenate((values, f_in[part_type.value][field_type.value]))
            except KeyError:
                logger.warning(
                    "No data of type %(_type)s in file %(_file)s", {"_type": part_type.value, "_file": file_name}
                )
                continue

    f_out = h5py.File(
        _dir.joinpath(
            f"combined_{part_type.filename}{field_type.value}_{simulation_name.lower()}_{snapshot_idx:03d}.hdf5"
        ),
        "w",
    )

    bound_info_file = get_bound_info_file(simulation_name)
    if part_type == PartType.GAS and not bound_info_file.exists():
        zeros = np.zeros((2, 3))
        zeros[0] = np.min(coordinates, axis=0)
        zeros[1] = np.max(coordinates, axis=0)
        if not bound_info_file.exists():  # Double check if someone wrote in in the meantime
            np.save(bound_info_file, zeros)

    f_out.create_group(part_type.value)
    f_out[part_type.value].create_dataset("Coordinates", coordinates.shape, float, coordinates)
    f_out[part_type.value].create_dataset(field_type.value, values.shape, float, values)

    file_name = Path(f_out.filename)
    f_out.close()
    return file_name


def create_delaunay_symlink(
    simulation_name: str, snapshot_idx: int, part_type: PartType, field_type: FieldType
) -> None:
    """Create symlink for combined delauny for loading of timed data."""
    os.symlink(
        get_delaunay_path(simulation_name, snapshot_idx, part_type, field_type),
        get_delaunay_time_symlink_path(simulation_name, snapshot_idx, part_type, field_type),
    )


def create_resampled_delaunay_symlink(
    simulation_name: str, snapshot_idx: int, part_type: PartType, field_type: FieldType
) -> None:
    """Create symlink for resampled delauny for loading of timed data."""
    path = get_resampled_delaunay_path(simulation_name, snapshot_idx, part_type, field_type)
    if not path.exists():
        logger.error(
            "Couldn't create symlink for %(name)s, %(idx)d, %(type)s, because the target file doesn't exist.",
            {"name": simulation_name, "idx": snapshot_idx, "type": field_type.value},
        )
        return
    os.symlink(
        path,
        get_resampled_delaunay_time_symlink_path(simulation_name, snapshot_idx, part_type, field_type),
    )


def create_scalar_field_experiment_symlink(
    simulation_name: str, snapshot_idx: int, experiment_name: str, field_type_1: FieldType, field_type_2: FieldType
) -> None:
    """Create symlink for a specific time of a scalar field experiment."""
    data_path = get_scalar_field_experiment_path(
        simulation_name, snapshot_idx, experiment_name, field_type_1, field_type_2
    )
    symlink_path = get_scalar_field_experiment_symlink_path(
        simulation_name, snapshot_idx, experiment_name, field_type_1, field_type_2
    )

    if symlink_path.is_symlink():
        return

    if platform.system() == "Windows":
        logger.warning("Unable to create symlink due to os!")
        return

    os.symlink(data_path, symlink_path)
