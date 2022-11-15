"""CLI entrypoint."""


import os
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

import numpy as np
import typer

from tng_sv.api.download import download_snapshot, get_snapshot_amount
from tng_sv.data.dir import path_to_resampled_file
from tng_sv.data.utils import combine_snapshot
from tng_sv.preprocessing import run_delaunay, run_resample_delaunay

app = typer.Typer()


@app.command()
def download(simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0) -> None:
    """Download a snapshot."""
    download_snapshot(simulation_name, snapshot_idx)


@app.command()
def combine(simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0) -> None:
    """Combine a snapshot."""
    combine_snapshot(simulation_name, snapshot_idx)


@app.command()
def delaunay(simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0) -> None:
    """Download a snapshot."""
    run_delaunay(simulation_name, snapshot_idx)


@app.command()
def resample(simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0) -> None:
    """Download a snapshot."""
    run_resample_delaunay(simulation_name, snapshot_idx)


@app.command()
def run(simulation_name: str = "TNG50-4-Subbox2", snapshot_idx_step_size: int = 100) -> None:
    """Run the whole pipeline."""
    amount = get_snapshot_amount(simulation_name)
    _range = np.arange(0, amount, snapshot_idx_step_size)

    if _range[-1] != amount:
        _range = np.append(_range, amount)

    args = [(simulation_name, i) for i in _range]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        pool.map(_run, args)


def _run(simulation: Tuple[str, int]) -> None:
    """Do things, exit early if already done."""
    simulation_name, snapshot_idx = simulation
    if path_to_resampled_file(simulation_name, snapshot_idx) is None:
        download_snapshot(simulation_name, snapshot_idx)
        combine_snapshot(simulation_name, snapshot_idx)
        run_delaunay(simulation_name, snapshot_idx)
        resample(simulation_name, snapshot_idx)


def cli() -> int:
    """Run the main function with typer."""
    app()
    return 0


if __name__ == "__main__":
    cli()
