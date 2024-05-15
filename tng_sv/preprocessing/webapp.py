"""Module for the webapp preprocessing."""

import logging
import os.path as Path
from datetime import datetime
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from numba import jit
from tqdm import tqdm

import illustris_python as il
from tng_sv.data.dir import get_webapp_base_path

logger = logging.getLogger(__name__)

IDX = 0

def filter_snapshots(all_loaded_snaps: dict[str, Any], fields: list[str], percentage: float = 0.99, sort_field: str = "Density"):
    """Remove given precentage of data which is of lower sort field."""
    for i in range(2):
        data = all_loaded_snaps[i]["snapData"][sort_field]
        arg_data = np.argsort(data)
        idx = int(len(data) * percentage)

        relevant_idxs = arg_data[idx:]

        for field in fields:
            all_loaded_snaps[i]["snapData"][field] = all_loaded_snaps[i]["snapData"][field][relevant_idxs]

    return all_loaded_snaps




def get_same_particle_in_two_data_sets(  # pylint: disable=too-many-locals
    snapshot0: Dict[str, Any], snapshot1: Dict[str, Any], data_types: List[str]
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Get the same particles in two data sets."""
    id_0 = np.array(snapshot0["ParticleIDs"])
    id_1 = np.array(snapshot1["ParticleIDs"])

    _max = np.max((np.max(id_0), np.max(id_1)))
    _min = np.min((np.min(id_0), np.min(id_1)))

    _len = int(_max - _min) + 1

    mask1 = np.zeros(_len)
    mask1[np.array(id_0 - _min, dtype=int)] = 1

    mask2 = np.zeros(_len)
    mask2[np.array(id_1 - _min, dtype=int)] = 1
    mask = (mask1 * mask2).astype(bool)

    all_combined_attributes = {}
    all_combined_attributes["Mask"] = mask
    for data_type in data_types:
        attributes_dense_0 = snapshot0[data_type]
        attributes_dense_1 = snapshot1[data_type]
        if attributes_dense_1.ndim == 1:
            dimension_tuple = (_len,)
        else:
            dimension_tuple = (_len, 3)  # type: ignore

        attributes_sparse_0 = np.zeros(dimension_tuple)
        attributes_sparse_1 = np.zeros(dimension_tuple)

        attributes_sparse_0[np.array(id_0 - _min, dtype=int)] = attributes_dense_0
        attributes_sparse_1[np.array(id_1 - _min, dtype=int)] = attributes_dense_1
        all_combined_attributes[data_type] = (attributes_sparse_0[mask], attributes_sparse_1[mask])  # type: ignore

    return all_combined_attributes  # type: ignore


def load_datasets(base_path: str, snap_idx: int, fields: List[str]):  # pylint: disable=c-extension-no-member
    """Load two datasets."""
    # pylint: disable=c-extension-no-member
    all_loaded_snaps = []

    # Load snapshot n and n + 1
    for i in range(snap_idx, snap_idx + 2):
        snap_dict = {}
        snap_dict["snapData"] = il.snapshot.loadSubset(base_path, i, "gas", fields=fields)  # type: ignore[attr-defined]
        snap_dict["snapInfo"] = il.groupcat.loadHeader(base_path, i)  # type: ignore[attr-defined]
        all_loaded_snaps.append(snap_dict)

    return all_loaded_snaps


def _generate_octree(coordinates: np.ndarray, max_depth: int = 5) -> o3d.geometry.Octree:
    """Generate octree helper."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(coordinates))
    oct_tree = o3d.geometry.Octree(max_depth=max_depth)
    oct_tree.convert_from_point_cloud(pcd)
    return oct_tree


@jit(nopython=True)
def spline_calculation(y: np.ndarray, dydx: np.ndarray) -> np.ndarray:  # pylint: disable=invalid-name
    """Calculate spline."""
    # pylint: disable=invalid-name
    x = np.array([0.0, 1.0])
    dxr = np.array([[1]])

    slope = (y[1] - y[0]) / dxr
    t = (dydx[:-1] + dydx[1:] - 2 * slope) / dxr

    c = np.empty((4, len(x) - 1) + y.shape[1:], dtype=t.dtype)
    c[0] = t / dxr
    c[1] = (slope - dydx[:-1]) / dxr - t
    c[2] = dydx[:-1]
    c[3] = y[:-1]
    return c


def generate_new_octree(  # pylint: disable=too-many-locals
    size_per_leaf: int,
    box_size: int,
    all_combined_attributes: Dict[str, Tuple[np.ndarray, np.ndarray]],
    initial_sort_field: str,
) -> Tuple[o3d.geometry.Octree, List[np.ndarray]]:
    """Generate new octree."""
    # approximation der tiefe bei size_per_leaf Angabe => sizeLeaf ~ sizeBox / 2^x ==> formel vorne
    max_depth = np.ceil(np.log2(box_size / size_per_leaf)).astype(int)
    logger.info("Max Depth %(max_depth)s", {"max_depth": max_depth})

    coordinates = np.vstack(all_combined_attributes["Coordinates"])
    oct_tree = _generate_octree(coordinates, max_depth)

    dims_of_sort_field = all_combined_attributes[initial_sort_field][0].ndim

    if dims_of_sort_field == 1:
        fields_array = np.hstack(all_combined_attributes[initial_sort_field])
    elif dims_of_sort_field == 3:
        three_dims_fields = np.vstack(all_combined_attributes[initial_sort_field])
        fields_array = np.linalg.norm(three_dims_fields, axis=0)
    else:
        raise ValueError(f"Either 1 or 3 dims, instead found {dims_of_sort_field} dims")

    offset = len(all_combined_attributes["Coordinates"][0])
    particle_ids = np.hstack(all_combined_attributes["ParticleIDs"]).astype(np.int64)

    indices_for_octree: List[np.ndarray] = []

    def change_ids_with_list_id(
        node: o3d.geometry.OctreeNode, _node_info: o3d.geometry.OctreeNodeInfo
    ) -> Optional[bool]:
        """Traverse octree."""
        if isinstance(node, o3d.geometry.OctreeLeafNode):

            global IDX  # pylint: disable=global-statement
            _, indices = np.unique(particle_ids[node.indices], return_index=True)
            all_indices_without_duplicates = np.array(node.indices)[indices]
            all_indices_without_duplicates[all_indices_without_duplicates >= offset] -= offset

            fields_in_leaf = fields_array[all_indices_without_duplicates]
            sorted_indices = np.array(np.argsort(fields_in_leaf)[::-1])
            indices_for_octree.append(np.array(all_indices_without_duplicates[sorted_indices]))
            node.indices = [IDX]
            IDX += 1
            return True

        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            node.indices = []

        return None

    logger.info("start octree calc")
    now = datetime.now()
    global IDX  # pylint: disable=global-statement
    IDX = 0
    oct_tree.traverse(change_ids_with_list_id)
    finish = datetime.now()
    logger.info("Octree calculated. Duration: %(duration)s", {"duration": (finish - now).total_seconds()})
    return oct_tree, indices_for_octree


def _write_particle_of_leafs_array(
    file_name: str, particle_list_of_leafs: List[np.ndarray], file_name_sufix: str
) -> None:
    """Write particle of leafs array and scan."""
    empty: List[np.ndarray] = [np.array([])]
    empty.extend(particle_list_of_leafs)
    particle_list_of_leafs = empty

    scan = np.cumsum(np.array([len(element) for element in particle_list_of_leafs]))[:-1]

    particle_list_of_leafs = np.array(list(chain(*particle_list_of_leafs)))  # type: ignore

    np.save(file_name + file_name_sufix, particle_list_of_leafs)
    np.save(file_name + "_scan" + file_name_sufix, scan)
    logger.info("Saved: %(file_name)s", {"file_name": file_name})


def write_particle_of_leafs_arrays(
    file_path: str, sort_fields: List[str], all_combined_attributes, indices_for_octree: List[np.ndarray]
) -> None:
    """Write particle of leafs array for all fields."""
    file_name_prefix = "particle_list_of_leafs"
    file_name_sufix = ".npy"
    _write_particle_of_leafs_array(
        file_path + f"{file_name_prefix}_{sort_fields[0]}", indices_for_octree, file_name_sufix
    )
    if len(sort_fields) > 1:
        for sortfield in sort_fields[1 : len(sort_fields)]:
            dims_of_sort_field = all_combined_attributes[sortfield][0].ndim
            if dims_of_sort_field == 1:
                fields_array = np.hstack(all_combined_attributes[sortfield])
            else:
                three_dims_fields = np.vstack(all_combined_attributes[sortfield])
                fields_array = np.linalg.norm(three_dims_fields, axis=0)
            new_indices_for_octree = []
            for indices in indices_for_octree:
                fields_in_leaf = fields_array[indices]
                sorted_indices = np.array(np.argsort(fields_in_leaf)[::-1])
                new_indices_for_octree.append(np.array(fields_in_leaf[sorted_indices]))
            _write_particle_of_leafs_array(
                file_path + f"{file_name_prefix}_{sortfield}{file_name_sufix}", new_indices_for_octree, file_name_sufix
            )


def calculate_edge_size(masses: np.ndarray, densities: np.ndarray) -> np.ndarray:
    """Calculate edge size."""
    return np.power(masses / densities, 1 / 3)  # 4/3 · π · r3


def preprocess_snap(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    simulation_name: str,
    snap_idx: int,
    fields: Optional[List[str]] = None,
    size_per_leaf=350,
    sort_fields: Optional[List[str]] = None,
) -> None:
    """Preprocess a given snap.

    Assumes that the data is downloaded.
    """
    base_path = str(get_webapp_base_path(simulation_name))
    necessary_fields = [
        "Coordinates",
        "ParticleIDs",
        "Density",
        "Velocities",
        "Masses",
    ]

    # this fields are neccesarry for the calculations
    if not fields:
        fields = []

    if not sort_fields:
        sort_fields = ["Density"]

    for field in necessary_fields:
        if field not in fields:
            fields.append(field)

    # load two snapshots n and n+1
    all_loaded_snaps = load_datasets(base_path, snap_idx, fields)

    all_loaded_snaps = filter_snapshots(all_loaded_snaps, fields)

    # Filter out 97% of particles
    # take top 3 % of density, remove rest

    # Load
    snapshot_n = 0

    all_combined_attributes = get_same_particle_in_two_data_sets(
        all_loaded_snaps[snapshot_n]["snapData"], all_loaded_snaps[snapshot_n + 1]["snapData"], fields
    )

    edge_size = calculate_edge_size(
        np.hstack(all_combined_attributes["Masses"]), np.hstack(all_combined_attributes["Density"])
    )

    # snapdir path
    snapdir_path = base_path + "/snapdir_" + str(snap_idx + snapshot_n).zfill(3) + "/"

    box_size = all_loaded_snaps[0]["snapInfo"]["BoxSize"]
    (oct_tree, indices_for_octree) = generate_new_octree(size_per_leaf, box_size, all_combined_attributes, "Density")

    # --------------Spline Array bauen ---------------
    cubic_hermite_splines = None
    file_name = snapdir_path + "splines.npy"
    if Path.isfile(file_name):
        logger.info("Spline calc skipped. Already there!")
    else:
        logger.info("start spline calc")
        coord_0 = np.array(all_combined_attributes["Coordinates"][0])
        coord_1 = np.array(all_combined_attributes["Coordinates"][1])
        vel_0 = np.array(all_combined_attributes["Velocities"][0]) * 3.154e7 / 3.086e16  # calc km/s to kpc/a
        vel_1 = np.array(all_combined_attributes["Velocities"][1]) * 3.154e7 / 3.086e16  # calc km/s to kpc/a

        zipped = zip(coord_0, coord_1, vel_0, vel_1)

        cubic_hermite_splines = np.ones((len(coord_0), 4, 3))
        for idx_, value in enumerate(tqdm(zipped, total=len(coord_0))):
            _coord_0, _coord_1, _vel_0, _vel_1 = value
            cubic_hermite_splines[idx_, :] = spline_calculation(
                np.array([_coord_0, _coord_1]), np.array([_vel_0, _vel_1])
            ).squeeze()
        logger.info("Splines calculated")

    file_name = snapdir_path + "o3dOctree.json"
    logger.info("Safe Octree to %(file_name)s", {"file_name": file_name})

    # Try to write octree and fields to disk
    if o3d.io.write_octree(file_name, oct_tree):
        logger.info("Object successfully saved to %(file_name)s, Saving additional data:", {"file_name": file_name})

        write_particle_of_leafs_arrays(snapdir_path, sort_fields, all_combined_attributes, indices_for_octree)

        file_name = snapdir_path + "splines.npy"
        if not Path.isfile(file_name):
            if cubic_hermite_splines is not None:
                np.save(file_name, cubic_hermite_splines)
                logger.info("Saved: %(file_name)s", {"file_name": file_name})
            else:
                logger.info("Cubic hermite splines are None, did not calculate them.")

        # Save all fields
        for field in fields:
            file_name = snapdir_path + field + ".npy"
            if np.array(all_combined_attributes[field]).ndim == 1:
                np.save(file_name, np.hstack(all_combined_attributes[field]))
            else:
                np.save(file_name, np.vstack(all_combined_attributes[field]))
            logger.info("Saved: %(file_name)s", {"file_name": file_name})

        file_name = snapdir_path + "densities_quantiles.npy"
        densities = np.hstack(all_combined_attributes["Density"])
        np.save(file_name, np.quantile(densities.flatten(), np.linspace(0, 1, 100)))

        file_name = snapdir_path + "voronoi_diameter_extended" + ".npy"
        np.save(file_name, edge_size)
        logger.info("Saved: %(file_name)s", {"file_name": file_name})
    else:
        logger.info(
            "Object was not saved to %(file_name)s. Please check if the path is sane.", {"file_name": file_name}
        )
