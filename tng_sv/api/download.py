"""Download logic."""


import csv
import json
import os
from typing import Any, Dict, List

from tqdm import tqdm

from tng_sv.api import BASEURL
from tng_sv.api.utils import get_file, get_index, get_json, get_json_list
from tng_sv.data.dir import get_snapshot_index_path, get_subhalo_dir, get_subhalo_info_path


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


def download_subhalos(simulation_name: str, begin_snapshot: int, begin_idx: int, step_size: int) -> None:
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
            if next_subhalo is None:
                break
            subhalo_meta = get_json(next_subhalo)


def _inclusive_range(begin: int, end: int, step_size: int) -> List[int]:
    """Return list of inclusive range."""
    _range = list(range(begin, end, step_size))

    if _range[-1] != end:
        _range.append(end - 1)

    return _range
