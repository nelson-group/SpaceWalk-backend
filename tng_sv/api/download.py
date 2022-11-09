"""Download logic."""


import os
from typing import Any, Dict, List

from tqdm import tqdm

from tng_sv.api import BASEURL
from tng_sv.api.utils import get_file, get_index, get_json, get_json_list
from tng_sv.data import DATADIR


def download_snapshot(simulation_name: str, snapshot_idx: int) -> List[str]:
    """Download a specific snapshot."""

    simulations: List[Dict[str, Any]] = get_json(BASEURL)["simulations"]
    _, simulation_meta = get_index(simulations, "name", simulation_name)
    simulation = get_json(simulation_meta["url"])
    snapshots: List[Dict[str, Any]] = get_json_list(simulation["snapshots"])
    snapshot_url = snapshots[snapshot_idx]["url"]
    snapshot = get_json(snapshot_url)
    files_meta = get_json(snapshot["files"]["snapshot"])["files"]

    pre_dir = f"{simulation_name}/{snapshot_idx:03d}/"

    _dir = DATADIR.joinpath(pre_dir)
    if not _dir.exists():
        os.makedirs(_dir)

    return [get_file(file_url, pre_dir=_dir) for file_url in tqdm(files_meta)]
