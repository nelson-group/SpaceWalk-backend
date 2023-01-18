"""Module for plotting on downloaded data."""

import matplotlib.pyplot as plt
import numpy as np

from tng_sv.data.dir import get_subhalo_info_json


def plot_subhalo_com_against_cob(simulation_name: str, snapshot_idx: int, subhalo_idx: int) -> None:
    """."""
    data = get_subhalo_info_json(simulation_name, snapshot_idx, subhalo_idx)
    snap_data = {}
    for value in data.values():
        snap_data[value["snap"]] = value

    sorted_keys = sorted(snap_data.keys())

    distances = np.zeros(len(sorted_keys))
    for idx, key in enumerate(sorted_keys):
        com = np.array(
            [
                snap_data[key]["cm_x"],
                snap_data[key]["cm_y"],
                snap_data[key]["cm_z"],
            ]
        )
        cob = np.array(
            [
                snap_data[key]["pos_x"],
                snap_data[key]["pos_y"],
                snap_data[key]["pos_z"],
            ]
        )
        distances[idx] = np.linalg.norm(com - cob)

    plt.ylabel("Distance [ckpc?]")
    plt.xlabel("Local Snap IDX")
    plt.plot([int(key) for key in sorted_keys], distances)
    plt.show()
