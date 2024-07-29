"""CLI entrypoint."""


import logging
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, cast

import numpy as np
import typer
import uvicorn

from tng_sv.api.download import (
    download_halo,
    download_snapshot,
    download_subhalos,
    download_webapp_groups,
    download_webapp_snapshot,
    get_snapshot_amount,
    get_subhalos_from_subbox,
)
from tng_sv.data.dir import (
    get_delaunay_time_symlink_path,
    get_halo_dir,
    get_resampled_delaunay_path,
    get_resampled_delaunay_time_symlink_path,
    get_scalar_field_experiment_path,
    get_snapshot_index_path,
    get_subhalo_dir,
    get_subhalo_info_json,
)
from tng_sv.data.field_type import FieldType
from tng_sv.data.part_type import PartType
from tng_sv.data.utils import (
    bounds,
    combine_snapshot,
    create_delaunay_copy,
    create_delaunay_symlink,
    create_resampled_delaunay_copy,
    create_resampled_delaunay_symlink,
    create_scalar_field_experiment_symlink,
)
from tng_sv.plot import plot_subhalo_com_against_cob
from tng_sv.preprocessing import _run_center, _run_delaunay, run_delaunay, run_resample_delaunay, webapp
from tng_sv.preprocessing.two_field_operations import scalar_product, vector_angle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = typer.Typer()

halo_app = typer.Typer()
subhalo_app = typer.Typer()
plot_app = typer.Typer()
web_app = typer.Typer()


@app.command(name="download")
def download_cmd(simulation_name: str = "TNG50-4-Subbox2", snapshot_idx: int = 0) -> None:
    """Download a snapshot."""
    download_snapshot(simulation_name, snapshot_idx)


@app.command()
def combine(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx: int = 0,
    part_type: PartType = cast(PartType, "PartType0"),
    field_type: FieldType = cast(FieldType, "Velocities"),
) -> None:
    """Combine a snapshot."""
    combine_snapshot(simulation_name, snapshot_idx, part_type, field_type)


@app.command()
def delaunay(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx: int = 0,
    part_type: PartType = cast(PartType, "PartType0"),
    field_type: FieldType = cast(FieldType, "Velocities"),
) -> None:
    """Download a snapshot."""
    run_delaunay(simulation_name, snapshot_idx, part_type, field_type)


@app.command()
def resample(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx: int = 0,
    part_type: PartType = cast(PartType, "PartType0"),
    field_type: FieldType = cast(FieldType, "Velocities"),
) -> None:
    """Download a snapshot."""
    run_resample_delaunay(simulation_name, snapshot_idx, part_type, field_type)


@app.command(name="bounds")
def bounds_cmd(
    simulation_name: str = "TNG50-1",
    part_type: PartType = cast(PartType, "PartType0"),
) -> None:
    """Generate bound.pvd for simulation."""
    bounds(simulation_name, part_type)


@app.command()
def scalar_field_experiments_for_one_idx(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx: int = 0,
    field_type_1: FieldType = cast(FieldType, "Velocities"),
    field_type_2: FieldType = cast(FieldType, "MagneticField"),
    force_override: bool = False,
) -> None:
    """Run the scalar field experiments for a given simulation snapshot."""
    if (
        not force_override
        and get_scalar_field_experiment_path(
            simulation_name, snapshot_idx, "scalar_product", field_type_1, field_type_2
        ).exists()
        and get_scalar_field_experiment_path(
            simulation_name, snapshot_idx, "vector_angle", field_type_1, field_type_2
        ).exists()
    ):
        print(f"[snapshot idx {snapshot_idx}] experiments already computed")
    elif (
        get_resampled_delaunay_path(simulation_name, snapshot_idx, PartType.GAS, FieldType.VELOCITY).exists()
        and get_resampled_delaunay_path(simulation_name, snapshot_idx, PartType.GAS, FieldType.MAGNETIC).exists()
    ):
        scalar_product(simulation_name, snapshot_idx, field_type_1, field_type_2)
        vector_angle(simulation_name, snapshot_idx, field_type_1, field_type_2)
        create_scalar_field_experiment_symlink(
            simulation_name, snapshot_idx, "scalar_product", field_type_1, field_type_2
        )
        create_scalar_field_experiment_symlink(
            simulation_name, snapshot_idx, "vector_angle", field_type_1, field_type_2
        )
        print(f"[snapshot idx {snapshot_idx}] did scalar field experiments :)")
    else:
        print(f"[snapshot idx {snapshot_idx}] missing image data!")


def _scalar_field_experiments_for_one_idx_wrapper(args):
    """Wrapper for running the scalar field experiments with multiple snapshots"""
    try:
        scalar_field_experiments_for_one_idx(*args)
    except Exception as exception:  # pylint: disable=broad-except
        print(f"[snapshot idx {args[1]}] failed: {exception}")

        if hasattr(sys, "gettrace") and sys.gettrace() is not None:
            # if used in debugger print the whole exception
            traceback.print_exc()


@app.command()
def run_scalar_field_experiments(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx_step_size: int = 100,
    field_type_1: FieldType = cast(FieldType, "Velocities"),
    field_type_2: FieldType = cast(FieldType, "MagneticField"),
    force_override: bool = False,
):
    """Run the scalar field experiments for multiple simulation snapshots."""
    amount = get_snapshot_amount(simulation_name)
    _range = np.arange(0, amount, snapshot_idx_step_size)

    if _range[-1] != amount:
        _range = np.append(_range, amount - 1)

    args = [(simulation_name, i, field_type_1, field_type_2, force_override) for i in _range]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        pool.map(_scalar_field_experiments_for_one_idx_wrapper, args)


@app.command()
def timeseries(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx_step_size: int = 100,
    part_type: PartType = cast(PartType, "PartType0"),
    field_type: FieldType = cast(FieldType, "Velocities"),
):
    """
    Create copy of time steps instead of symlink.
    """
    amount = get_snapshot_amount(simulation_name)
    _range = np.arange(0, amount, snapshot_idx_step_size)

    if _range[-1] != amount:
        _range = np.append(_range, amount - 1)

    if field_type == FieldType.ALL:
        field_types = [enum for enum in FieldType if enum != FieldType.ALL]
        args = []
        for i in _range:
            for _field_type in field_types:
                args.append((simulation_name, i, part_type, _field_type))
    else:
        args = [(simulation_name, i, part_type, field_type) for i in _range]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        pool.map(_timeseries, args)


def _timeseries(args: Tuple[str, int, PartType, FieldType]):
    simulation_name, snapshot_idx, part_type, field_type = args

    if not get_delaunay_time_symlink_path(simulation_name, snapshot_idx, part_type, field_type).exists():
        create_delaunay_copy(simulation_name, snapshot_idx, part_type, field_type)

    if not get_resampled_delaunay_time_symlink_path(simulation_name, snapshot_idx, part_type, field_type).exists():
        create_resampled_delaunay_copy(simulation_name, snapshot_idx, part_type, field_type)


@app.command()
def run(
    simulation_name: str = "TNG50-4-Subbox2",
    snapshot_idx_step_size: int = 100,
    part_type: PartType = cast(PartType, "PartType0"),
    field_type: FieldType = cast(FieldType, "Velocities"),
) -> None:
    """Run the whole pipeline."""
    amount = get_snapshot_amount(simulation_name)
    _range = np.arange(0, amount, snapshot_idx_step_size)

    if _range[-1] != amount:
        _range = np.append(_range, amount - 1)

    if field_type == FieldType.ALL:
        field_types = [enum for enum in FieldType if enum != FieldType.ALL]
        args = []
        for i in _range:
            for _field_type in field_types:
                args.append((simulation_name, i, part_type, _field_type))
    else:
        args = [(simulation_name, i, part_type, field_type) for i in _range]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        pool.map(_run, args)


def _run(simulation: Tuple[str, int, PartType, FieldType]) -> None:
    """Do things, exit early if already done."""
    try:
        simulation_name, snapshot_idx, part_type, field_type = simulation

        if not get_resampled_delaunay_path(simulation_name, snapshot_idx, part_type, field_type).exists():
            if len(list(get_snapshot_index_path(simulation_name, snapshot_idx).glob("*.*.hdf5"))) == 0:
                download_snapshot(simulation_name, snapshot_idx)

            combine_snapshot(simulation_name, snapshot_idx, part_type, field_type)
            run_delaunay(simulation_name, snapshot_idx, part_type, field_type)
            resample(simulation_name, snapshot_idx, part_type, field_type)

        if not get_delaunay_time_symlink_path(simulation_name, snapshot_idx, part_type, field_type).exists():
            create_delaunay_symlink(simulation_name, snapshot_idx, part_type, field_type)

        if not get_resampled_delaunay_time_symlink_path(simulation_name, snapshot_idx, part_type, field_type).exists():
            create_resampled_delaunay_symlink(simulation_name, snapshot_idx, part_type, field_type)
    except Exception as exc:
        logger.exception("Failed job: %(job)s with exc: %(exc)s", {"job": simulation, "exc": exc})
        raise exc from exc


@subhalo_app.command(name="download")
def download_subhalos_cmd(
    simulation_name: str = "TNG50-1",
    snapshot_idx: int = 0,
    subhalo_idx: int = 0,
    step_size: int = 1,
    parent_halo: bool = False,
) -> None:
    """Download a list of subhalos based on the start parameters."""
    download_subhalos(simulation_name, snapshot_idx, subhalo_idx, step_size, parent_halo)


@subhalo_app.command(name="delaunay")
def delaunay_subhalos_cmd(simulation_name: str = "TNG50-1", snapshot_idx: int = 0, subhalo_idx: int = 0) -> None:
    """Run delaunay on a list of subhalos in parallel."""
    _dir = get_subhalo_dir(simulation_name, snapshot_idx, subhalo_idx)
    files = _dir.glob("cutout*.hdf5*")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        pool.map(_delaunay_subhalos_cmd, files)


def _delaunay_subhalos_cmd(in_path: Path) -> None:
    """Run delauny for one subhalo file."""
    try:
        out_path = Path(str(in_path).replace("hdf5", "pvd"))
        if out_path.exists():
            return
        _run_delaunay(in_path, out_path, PartType.GAS, FieldType.ALL)
    except Exception as exc:
        logger.exception("Failed job: %(path)s with exc: %(exc)s", {"path": in_path, "exc": exc})
        raise exc from exc


@subhalo_app.command(name="center")
def center_subhalos_cmd(simulation_name: str = "TNG50-1", snapshot_idx: int = 0, subhalo_idx: int = 0) -> None:
    """Run center on a list of subhalos in parallel."""
    _dir = get_subhalo_dir(simulation_name, snapshot_idx, subhalo_idx)
    files = _dir.glob("cutout*.pvd*")
    info_json = get_subhalo_info_json(simulation_name, snapshot_idx, subhalo_idx)
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        pool.map(_center_subhalos_cmd, [(_file, info_json) for _file in files])


def _center_subhalos_cmd(args: Tuple[Path, Dict[str, Any]]) -> None:
    """Run center for one subhalo file."""
    in_path, info_json = args
    try:
        out_path = Path(str(in_path).replace("cutout", "centered_cutout"))
        if out_path.exists():
            return
        com = (
            info_json[in_path.parts[-1].replace("pvd", "hdf5")]["cm_x"],
            info_json[in_path.parts[-1].replace("pvd", "hdf5")]["cm_y"],
            info_json[in_path.parts[-1].replace("pvd", "hdf5")]["cm_z"],
        )
        vel_disp = (
            info_json[in_path.parts[-1].replace("pvd", "hdf5")]["vel_x"],
            info_json[in_path.parts[-1].replace("pvd", "hdf5")]["vel_y"],
            info_json[in_path.parts[-1].replace("pvd", "hdf5")]["vel_z"],
        )
        _run_center(in_path, out_path, com, vel_disp)

    except Exception as exc:
        logger.exception("Failed center job: %(path)s with exc: %(exc)s", {"path": in_path, "exc": exc})
        raise exc from exc


@subhalo_app.command(name="find")
def find_subhalos_cmd(
    simulation_name: str = "TNG50-1", subbox_idx: int = 1, snapshot_idx: int = 0, min_mass_stars: float = 10.0
) -> None:
    """Find subhalos command."""
    get_subhalos_from_subbox(simulation_name, subbox_idx, snapshot_idx, min_mass_stars)


@plot_app.command(name="subhalo-com")
def cmd_plot_subhalo_com_against_cob(
    simulation_name: str = "TNG50-1", snapshot_idx: int = 0, subhalo_idx: int = 0
) -> None:
    """Create plot for distance between com and cob."""
    plot_subhalo_com_against_cob(simulation_name, snapshot_idx, subhalo_idx)


@halo_app.command(name="download")
def download_halo_cmd(simulation_name: str = "TNG50-1", snapshot_idx: int = 0, halo_idx: int = 0) -> None:
    """Download a halo."""
    download_halo(simulation_name, snapshot_idx, halo_idx)


@halo_app.command(name="delaunay")
def delaunay_halos_cmd(simulation_name: str = "TNG50-1", snapshot_idx: int = 0, halo_idx: int = 0) -> None:
    """Run delaunay on a list of halos in parallel."""
    _dir = get_halo_dir(simulation_name, snapshot_idx, halo_idx)
    files = _dir.glob("cutout*.hdf5*")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        pool.map(_delaunay_subhalos_cmd, files)


@web_app.command(name="download")
def download_web_cmd(simulation_name: str = "TNG50-1", snapshot_idx: int = 0) -> None:
    """Download snapshot for simulation webapp."""
    download_webapp_snapshot(simulation_name, snapshot_idx)
    download_webapp_groups(simulation_name, snapshot_idx)


@web_app.command(name="preprocess")
def preprocess_web_cmd(simulation_name: str = "TNG50-1", snapshot_idx: int = 0, filter_out_percentage: float = 0.95, data_path: Optional[Path] = None) -> None:
    """Preprocess snapshot for simulation webapp."""
    if not data_path:
        download_webapp_snapshot(simulation_name, snapshot_idx)
        download_webapp_groups(simulation_name, snapshot_idx)
        download_webapp_snapshot(simulation_name, snapshot_idx + 1)
        download_webapp_groups(simulation_name, snapshot_idx + 1)
    webapp.preprocess_snap(simulation_name, snapshot_idx, filter_out_percentage=filter_out_percentage, data_path=data_path)


@web_app.command(name="batch-preprocess")
def batch_preprocess_web_cmd(
    simulation_name: str = "TNG50-1", snapshot_idx: int = 0, end_snapshot_idx: Optional[int] = None, filter_out_percentage: float = 0.95, data_path: Optional[Path] = None
) -> None:
    """Sequentiall preprocess for the whole simulation."""
    amount = end_snapshot_idx or get_snapshot_amount(simulation_name)
    _range = np.arange(snapshot_idx, amount, 1)

    for snap in _range:
        preprocess_web_cmd(simulation_name, snap, filter_out_percentage=filter_out_percentage, data_path=data_path)


@web_app.command(name="batch-download")
def batch_download_web_cmd(simulation_name: str = "TNG50-1", snapshot_idx: int = 0) -> None:
    """Sequentiall download for the whole simulation."""
    amount = get_snapshot_amount(simulation_name)
    _range = np.arange(snapshot_idx, amount, 1)

    for snap in _range:
        download_web_cmd(simulation_name, snap)


@web_app.command(name="serve")
def start_preprocess_webservice(host: str = "127.0.0.1", port: int = 8999) -> None:
    """Preprocess snapshot for simulation webapp."""
    uvicorn.run("tng_sv.serve:app", host=host, port=port, log_level="info")


def cli() -> int:
    """Run the main function with typer."""
    app.add_typer(halo_app, name="halos")
    app.add_typer(subhalo_app, name="subhalos")
    app.add_typer(plot_app, name="plot")
    app.add_typer(web_app, name="web")
    app()
    return 0


if __name__ == "__main__":
    cli()
