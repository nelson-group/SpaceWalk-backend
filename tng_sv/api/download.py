"""Download logic."""


import csv
import json
import os
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple

import h5py
from tqdm import tqdm

from tng_sv.api import BASEURL
from tng_sv.api.utils import get_file, get_index, get_json, get_json_list
from tng_sv.data.dir import (
    get_halo_dir,
    get_simulation_dir,
    get_snapshot_index_path,
    get_subhalo_dir,
    get_subhalo_info_path,
)


def download_snapshot(simulation_name: str, snapshot_idx: int) -> List[str]:
    """Download a specific snapshot."""
    simulations: List[Dict[str, Any]] = get_json(BASEURL)["simulations"]
    _, simulation_meta = get_index(simulations, "name", simulation_name)
    simulation = get_json(simulation_meta["url"])
    snapshots: List[Dict[str, Any]] = get_json_list(simulation["snapshots"])

    snapshot_url = snapshots[snapshot_idx]["url"]
    number = snapshots[snapshot_idx]["number"]
    assert snapshot_idx == number, ValueError("idx doesn't match snapshot number")

    snapshot = get_json(snapshot_url)
    files_meta = get_json(snapshot["files"]["snapshot"])["files"]

    _dir = get_snapshot_index_path(simulation_name, number)
    if not _dir.exists():
        os.makedirs(_dir)

    return [get_file(file_url, pre_dir=_dir) for file_url in tqdm(files_meta)]


def get_snapshot_amount(simulation_name: str) -> int:
    """Get the number of snapshots for a given simulation_name."""
    simulations: List[Dict[str, Any]] = get_json(BASEURL)["simulations"]
    _, simulation_meta = get_index(simulations, "name", simulation_name)
    simulation = get_json(simulation_meta["url"])
    return len(get_json_list(simulation["snapshots"]))


def download_subhalos(
    simulation_name: str, begin_snapshot: int, begin_idx: int, step_size: int, parent_halo: bool
) -> None:
    """Download a list of subhalos that merge."""
    # pylint: disable=too-many-locals
    _dir = get_subhalo_dir(simulation_name, begin_snapshot, begin_idx)
    if not _dir.exists():
        os.makedirs(_dir)

    total_snapshots = get_snapshot_amount(simulation_name)

    _range = _inclusive_range(begin_snapshot, total_snapshots, step_size)

    simulations: List[Dict[str, Any]] = get_json(BASEURL)["simulations"]
    _, simulation_meta = get_index(simulations, "name", simulation_name)
    snapshots = get_json_list(get_json(simulation_meta["url"])["snapshots"])
    subhalo_meta = get_json(get_json(snapshots[begin_snapshot]["url"])["subhalos"] + f"{begin_idx}/")

    with open(
        get_subhalo_info_path(simulation_name, begin_snapshot, begin_idx), mode="w+", encoding="utf-8"
    ) as info_file:
        writer = csv.writer(info_file, delimiter=";")

        count = 0
        for _ in tqdm(range(total_snapshots)):
            next_subhalo = subhalo_meta["related"]["sublink_descendant"]
            cutout_url = subhalo_meta["cutouts"]["subhalo"]
            if subhalo_meta["snap"] == _range[count]:
                filename = f"cutout_{begin_snapshot}_{begin_idx}_{subhalo_meta['snap']}.hdf5"
                get_file(cutout_url, pre_dir=_dir, override_filename=filename)
                writer.writerow([filename, json.dumps(subhalo_meta)])
                count += 1

                # Download parent halo
                if parent_halo:
                    parent_halo_url = subhalo_meta["citouts"]["parent_halo"]
                    filename = f"cutout_parent_halo_{begin_snapshot}_{begin_idx}_{subhalo_meta['snap']}.hdf5"
                    get_file(parent_halo_url, pre_dir=_dir, override_filename=filename)
            if next_subhalo is None:
                break
            subhalo_meta = get_json(next_subhalo)


def download_halo(simulation_name: str, snapshot_idx: int, halo_idx: int) -> None:
    """Download a halo."""
    # pylint: disable=too-many-locals
    _dir = get_halo_dir(simulation_name, snapshot_idx, halo_idx)
    if not _dir.exists():
        os.makedirs(_dir)

    simulations: List[Dict[str, Any]] = get_json(BASEURL)["simulations"]
    _, simulation_meta = get_index(simulations, "name", simulation_name)
    snapshots = get_json_list(get_json(simulation_meta["url"])["snapshots"])
    halo_url = snapshots[snapshot_idx]["url"] + f"halos/{halo_idx}/cutout.hdf5"
    get_file(halo_url.replace("http://", "https://"), pre_dir=_dir)


def _inclusive_range(begin: int, end: int, step_size: int) -> List[int]:
    """Return list of inclusive range."""
    _range = list(range(begin, end, step_size))

    if _range[-1] != end:
        _range.append(end - 1)

    return _range


def find_subhalo_recursive(args: Tuple[str, int]) -> None:
    """Find subhalo recursively."""
    url, wanted_snapshot = args
    subhalo_meta = get_json(url)
    if int(subhalo_meta["primary_flag"]) == 0 or subhalo_meta["mass_stars"] < 10:
        return

    if subhalo_meta["snap"] == wanted_snapshot:
        print(f"Match: {subhalo_meta['id']}")
        return

    prev_subhalo = subhalo_meta["related"]["sublink_progenitor"]
    find_subhalo_recursive((prev_subhalo, wanted_snapshot))


def get_subhalos_from_subbox(simulation_name: str, subbox_idx: int, snapshot_idx: int) -> None:
    """Get subhalos in subbox with correct attributes."""

    simulation_dir = get_simulation_dir(simulation_name)
    if not simulation_dir.exists():
        os.makedirs(simulation_dir)

    simulations: List[Dict[str, Any]] = get_json(BASEURL)["simulations"]
    _, simulation_meta = get_index(simulations, "name", simulation_name)
    simulation = get_json(simulation_meta["url"])
    subbox_subhalo_list_url = simulation["files"][f"subbox_subhalo_list_{subbox_idx}"]

    filename = get_file(subbox_subhalo_list_url, pre_dir=simulation_dir)
    hdf5_file = h5py.File(filename, "r")

    entry_snapshot = int(simulation["num_snapshots"]) - 1
    snapshots = get_json_list(get_json(simulation_meta["url"])["snapshots"])
    subhalos_base_url = get_json(snapshots[entry_snapshot]["url"])["subhalos"]

    args = [(subhalos_base_url + f"{subhalo}/", snapshot_idx) for subhalo in hdf5_file["SubhaloIDs"]]
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.map(find_subhalo_recursive, args)
