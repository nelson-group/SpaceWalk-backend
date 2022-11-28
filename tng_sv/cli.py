"""CLI entrypoint."""


import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple, cast

import numpy as np
import typer

from tng_sv.api.download import download_snapshot, get_snapshot_amount
from tng_sv.data.dir import (
    get_delaunay_time_symlink_path,
    get_resampled_delaunay_path,
    get_resampled_delaunay_time_symlink_path,
    get_snapshot_index_path,
)
from tng_sv.data.field_type import FieldType
from tng_sv.data.utils import combine_snapshot, create_delaunay_symlink, create_resampled_delaunay_symlink
from tng_sv.preprocessing import run_delaunay, run_resample_delaunay
from tng_sv.preprocessing.two_field_operations import scalar_product, vector_angle

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command(name="download")
def download_cmd(simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0) -> None:
    """Download a snapshot."""
    download_snapshot(simulation_name, snapshot_idx)


@app.command()
def combine(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx: int = 0,
    field_type: FieldType = cast(FieldType, "Velocities"),
) -> None:
    """Combine a snapshot."""
    combine_snapshot(simulation_name, snapshot_idx, field_type)


@app.command()
def delaunay(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx: int = 0,
    field_type: FieldType = cast(FieldType, "Velocities"),
) -> None:
    """Download a snapshot."""
    run_delaunay(simulation_name, snapshot_idx, field_type)


@app.command()
def resample(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx: int = 0,
    field_type: FieldType = cast(FieldType, "Velocities"),
) -> None:
    """Download a snapshot."""
    run_resample_delaunay(simulation_name, snapshot_idx, field_type)


@app.command()
def scalar_field_experiments_for_one_idx(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx: int = 0,
    field_type_1: FieldType = cast(FieldType, "Velocities"),
    field_type_2: FieldType = cast(FieldType, "MagneticField"),
) -> None:
    """Run the scalar field experiments for a given simulation snapshot."""
    scalar_product(simulation_name, snapshot_idx, field_type_1, field_type_2)
    vector_angle(simulation_name, snapshot_idx, field_type_1, field_type_2)
    print(f"Did scalar field experiments for snapshot idx {snapshot_idx}")


def _scalar_field_experiments_for_one_idx_wrapper(args):
    """Wrapper for running the scalar field experiments with multiple snapshots"""
    scalar_field_experiments_for_one_idx(*args)


@app.command()
def run_scalar_field_experiments(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx_step_size: int = 100,
    field_type_1: FieldType = cast(FieldType, "Velocities"),
    field_type_2: FieldType = cast(FieldType, "MagneticField"),
):
    """Run the scalar field experiments for multiple simulation snapshots."""
    amount = get_snapshot_amount(simulation_name)
    _range = np.arange(0, amount, snapshot_idx_step_size)

    if _range[-1] != amount:
        _range = np.append(_range, amount - 1)

    print(_range)

    args = [(simulation_name, i, field_type_1, field_type_2) for i in _range]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        pool.map(_scalar_field_experiments_for_one_idx_wrapper, args)


@app.command()
def run(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx_step_size: int = 100,
    field_type: FieldType = cast(FieldType, "Velocities"),
) -> None:
    """Run the whole pipeline."""
    amount = get_snapshot_amount(simulation_name)
    _range = np.arange(0, amount, snapshot_idx_step_size)

    if _range[-1] != amount:
        _range = np.append(_range, amount - 1)

    args = [(simulation_name, i, field_type) for i in _range]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        pool.map(_run, args)


def _run(simulation: Tuple[str, int, FieldType]) -> None:
    """Do things, exit early if already done."""
    try:
        simulation_name, snapshot_idx, field_type = simulation

        if not get_resampled_delaunay_path(simulation_name, snapshot_idx, field_type).exists():
            if len(list(get_snapshot_index_path(simulation_name, snapshot_idx).glob("*.*.hdf5"))) == 0:
                download_snapshot(simulation_name, snapshot_idx)

            combine_snapshot(simulation_name, snapshot_idx, field_type)
            run_delaunay(simulation_name, snapshot_idx, field_type)
            resample(simulation_name, snapshot_idx, field_type)

        if not get_delaunay_time_symlink_path(simulation_name, snapshot_idx, field_type).exists():
            create_delaunay_symlink(simulation_name, snapshot_idx, field_type)

        if not get_resampled_delaunay_time_symlink_path(simulation_name, snapshot_idx, field_type).exists():
            create_resampled_delaunay_symlink(simulation_name, snapshot_idx, field_type)
    except Exception as exc:
        logger.exception("Failed job: %(job)s with exc: %(exc)s", {"job": simulation, "exc": exc})
        raise exc from exc


def cli() -> int:
    """Run the main function with typer."""
    app()
    return 0


if __name__ == "__main__":
    cli()
