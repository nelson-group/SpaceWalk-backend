"""CLI entrypoint."""


import os
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np
import typer

from tng_sv.api.download import download_snapshot, get_snapshot_amount
from tng_sv.data.dir import (
    get_delaunay_time_symlink_path,
    get_resampled_delaunay_time_symlink_path,
    get_snapshot_index_path,
    path_to_resampled_file,
)
from tng_sv.data.field_type import FieldType
from tng_sv.data.utils import combine_snapshot, create_delaunay_symlink, create_resampled_delaunay_symlink
from tng_sv.preprocessing import run_delaunay, run_resample_delaunay

app = typer.Typer()


@app.command()
def download(simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0) -> None:
    """Download a snapshot."""
    download_snapshot(simulation_name, snapshot_idx)


@app.command()
def combine(
    simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0, field_type: FieldType = FieldType.VELOCITY
) -> None:
    """Combine a snapshot."""
    combine_snapshot(simulation_name, snapshot_idx, field_type)


@app.command()
def delaunay(
    simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0, field_type: FieldType = FieldType.VELOCITY
) -> None:
    """Download a snapshot."""
    run_delaunay(simulation_name, snapshot_idx, field_type)


@app.command()
def resample(
    simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0, field_type: FieldType = FieldType.VELOCITY
) -> None:
    """Download a snapshot."""
    run_resample_delaunay(simulation_name, snapshot_idx, field_type)


@app.command()
def run(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx_step_size: int = 100,
    field_type: FieldType = FieldType.VELOCITY,
) -> None:
    """Run the whole pipeline."""
    amount = get_snapshot_amount(simulation_name)
    _range = np.arange(0, amount, snapshot_idx_step_size)

    if _range[-1] != amount:
        _range = np.append(_range, amount)

    args = [(simulation_name, i, field_type) for i in _range]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        pool.map(_run, args)


def _run(simulation: Tuple[str, int, FieldType]) -> None:
    """Do things, exit early if already done."""
    simulation_name, snapshot_idx, field_type = simulation

    if len(list(get_snapshot_index_path(simulation_name, snapshot_idx).glob("*.*.hdf5"))) == 0:
        download_snapshot(simulation_name, snapshot_idx)

    if path_to_resampled_file(simulation_name, snapshot_idx, field_type) is None:
        combine_snapshot(simulation_name, snapshot_idx, field_type)
        run_delaunay(simulation_name, snapshot_idx, field_type)
        resample(simulation_name, snapshot_idx, field_type)

    if not get_delaunay_time_symlink_path(simulation_name, snapshot_idx, field_type).exists():
        create_delaunay_symlink(simulation_name, snapshot_idx, field_type)

    if not get_resampled_delaunay_time_symlink_path(simulation_name, snapshot_idx, field_type).exists():
        create_resampled_delaunay_symlink(simulation_name, snapshot_idx, field_type)


def cli() -> int:
    """Run the main function with typer."""
    app()
    return 0


if __name__ == "__main__":
    cli()
