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

logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command(name="download")
def download_cmd(simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0) -> None:
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
    field_type: FieldType = cast(FieldType, "Velocities"),
    download: bool = True,
) -> None:
    """Run the whole pipeline."""
    amount = get_snapshot_amount(simulation_name)
    _range = np.arange(0, amount, snapshot_idx_step_size)

    if _range[-1] != amount:
        _range = np.append(_range, amount - 1)

    args = [(simulation_name, i, field_type, download) for i in _range]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        pool.map(_run, args)


def _run(simulation: Tuple[str, int, FieldType, bool]) -> None:
    """Do things, exit early if already done."""
    try:
        simulation_name, snapshot_idx, field_type, download = simulation

        if len(list(get_snapshot_index_path(simulation_name, snapshot_idx).glob("*.*.hdf5"))) == 0 and download:
            download_snapshot(simulation_name, snapshot_idx)

        if not get_resampled_delaunay_path(simulation_name, snapshot_idx, field_type).exists():
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
