import logging
import multiprocessing
import os.path as Path
import pickle
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple

import illustris_python as il
import numpy as np
import open3d as o3d
import scipy

from webScripts.preprocessing.gridGeneration import get_same_particle_in_two_data_sets

logger = logging.getLogger(__name__)


def loadDatasets(base_path: str, snap_idx: int, fields: List[str]):
    allLoadedSnap = []

    # Load snapshot n and n + 1
    for i in range(snap_idx, snap_idx + 1):
        snapDict = {}
        snapDict["snapData"] = il.snapshot.loadSubset(base_path, i, "gas", fields=fields)
        snapDict["snapInfo"] = il.groupcat.loadHeader(base_path, i)
        allLoadedSnap.append(snapDict)

    return allLoadedSnap


def generateOctree(coordinates: np.ndarray, max_depth: int = 5) -> o3d.geometry.Octree:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(coordinates))
    octTree = o3d.geometry.Octree(max_depth=max_depth)
    octTree.convert_from_point_cloud(pcd)
    return octTree


def spline_calculation(c0: np.ndarray, c1: np.ndarray, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    spline = scipy.interpolate.CubicHermiteSpline([0, 1], [c0, c1], [v0, v1])
    return spline.c.squeeze()


idx = 0


def generate_new_octree(
    size_per_leaf: int, box_size: int, all_combined_attributes, initialSortField: str
) -> Tuple[o3d.geometry.Octree, List[np.ndarray]]:
    # approximation der tiefe bei size_per_leaf Angabe => sizeLeaf ~ sizeBox / 2^x ==> formel vorne
    maxDepth = np.ceil(np.log2(box_size / size_per_leaf)).astype(int)
    logger.info("Max Depth %(maxDepth)s", {"maxDepth": maxDepth})

    coordinates = np.vstack(all_combined_attributes["Coordinates"])
    octTree = generateOctree(coordinates, maxDepth)

    indicesForOctree: List[np.ndarray] = []
    dimsOfSortField = all_combined_attributes[initialSortField][0].ndim

    if dimsOfSortField == 1:
        fields_array = np.hstack(all_combined_attributes[initialSortField])
    elif dimsOfSortField == 3:
        threeDimsFields = np.vstack(all_combined_attributes[initialSortField])
        fields_array = np.linalg.norm(threeDimsFields, axis=0)
    else:
        raise Exception(f"Either 1 or 3 dims, instead found {dimsOfSortField} dims")

    offset = len(all_combined_attributes["Coordinates"][0])
    particleIds = np.hstack(all_combined_attributes["ParticleIDs"]).astype(np.int64)

    def changeIdsWithListId(node: o3d.geometry.OctreeNode, _node_info: o3d.geometry.OctreeNodeInfo):
        if isinstance(node, o3d.geometry.OctreeLeafNode):
            global idx

            _, indices = np.unique(particleIds[node.indices], return_index=True)
            allIndicesWithoutDuplicates = np.array(node.indices)[indices]
            allIndicesWithoutDuplicates[allIndicesWithoutDuplicates >= offset] -= offset

            fields_in_leaf = fields_array[allIndicesWithoutDuplicates]
            sorted_indices = np.array(np.argsort(fields_in_leaf)[::-1])
            indicesForOctree.append(np.array(allIndicesWithoutDuplicates[sorted_indices]))
            node.indices = [idx]
            idx += 1
            return True

        if isinstance(node, o3d.geometry.OctreeInternalPointNode):
            node.indices = []

    logger.info("start octree calc")
    octTree.traverse(changeIdsWithListId)
    logger.info("Octree calculated")
    return octTree, indicesForOctree


def write_particle_of_leafs_array(file_name: str, indicesForOctree: List[np.ndarray]) -> None:
    with open(file_name, "wb") as file:
        pickle.dump(indicesForOctree, file)
    logger.info("Saved: %(file_name)s", {"file_name": file_name})


def write_particle_of_leafs_arrays(
    file_path: str, sortFields: List[str], allCombinedAttributes, indicesForOctree: List[np.ndarray]
) -> None:
    file_name_prefix = "particleListOfLeafs"
    file_name_sufix = ".obj"
    write_particle_of_leafs_array(file_path + f"{file_name_prefix}_{sortFields[0]}{file_name_sufix}", indicesForOctree)
    if len(sortFields) > 1:
        for sortfield in sortFields[1 : len(sortFields)]:
            dimsOfSortField = allCombinedAttributes[sortfield][0].ndim
            if dimsOfSortField == 1:
                fields_array = np.hstack(allCombinedAttributes[sortfield])
            else:
                threeDimsFields = np.vstack(allCombinedAttributes[sortfield])
                fields_array = np.linalg.norm(threeDimsFields, axis=0)
            new_indices_for_octree = []
            for indices in indicesForOctree:
                fields_in_leaf = fields_array[indices]
                sorted_indices = np.array(np.argsort(fields_in_leaf)[::-1])
                new_indices_for_octree.append(np.array(fields_in_leaf[sorted_indices]))
            write_particle_of_leafs_array(
                file_path + f"{file_name_prefix}_{sortfield}{file_name_sufix}", new_indices_for_octree
            )


def calculate_edge_size(masses: np.ndarray, densities: np.ndarray) -> np.ndarray:
    return np.power(masses / densities, 1 / 3)  # 4/3 · π · r3


def preprocess_snap(
    base_path: str,
    snap_idx: int,
    fields: List[str],
    size_per_leaf=100,
    sortFields: List[str] = ["Density"],
) -> None:
    global idx

    necessary_fields = [
        "Coordinates",
        "ParticleIDs",
        "Density",
        "Velocities",
        "Masses",
    ]

    # this fields are neccesarry for the calculations
    for field in necessary_fields:
        if field not in fields:
            fields.append(field)

    # load two snapshots n and n+1
    all_loaded_snaps = loadDatasets(base_path, snap_idx, fields)

    # Load
    n = 0

    all_combined_attributes = get_same_particle_in_two_data_sets(
        all_loaded_snaps[n]["snapData"], all_loaded_snaps[n + 1]["snapData"], fields
    )

    edge_size = calculate_edge_size(
        np.hstack(all_combined_attributes["Masses"]), np.hstack(all_combined_attributes["Density"])
    )

    # snapdir path
    snapdir_path = base_path + "snapdir_" + str(snap_idx + n).zfill(3) + "/"

    file_name = snapdir_path + "o3dOctree.json"
    if Path.isfile(file_name):
        logger.info("Octree available: load data")
        octTree = o3d.io.read_octree(file_name)
        with open(snapdir_path + f"particleListOfLeafs_{sortFields[0]}.obj", "rb") as objFile:
            indicesForOctree = pickle.load(objFile)
    else:
        box_size = all_loaded_snaps[0]["snapInfo"]["BoxSize"]
        (octTree, indicesForOctree) = generate_new_octree(size_per_leaf, box_size, all_combined_attributes, "Density")

    # --------------Spline Array bauen ---------------
    file_name = snapdir_path + "splines.npy"
    if Path.isfile(file_name):
        logger.info("Spline calc skipped. Already there!")
    else:
        logger.info("start spline calc")
        c0 = np.array(all_combined_attributes["Coordinates"][0])
        c1 = np.array(all_combined_attributes["Coordinates"][1])
        v0 = np.array(all_combined_attributes["Velocities"][0]) * 3.154e7 / 3.086e16  # calc km/s to kpc/a
        v1 = np.array(all_combined_attributes["Velocities"][1]) * 3.154e7 / 3.086e16  # calc km/s to kpc/a

        zipped = zip(c0, c1, v0, v1)

        with Pool(processes=int(multiprocessing.cpu_count() // 2)) as pool:
            c = pool.starmap(spline_calculation, zipped)
        logger.info("Splines calculated")

    file_name = snapdir_path + "o3dOctree.json"
    logger.info("Safe Octree to %(file_name)s", {"file_name": file_name})

    # Try to write octree and fields to disk
    if o3d.io.write_octree(file_name, octTree):
        logger.info("Object successfully saved to %(file_name)s, Saving additional data:", {"file_name": file_name})

        write_particle_of_leafs_arrays(snapdir_path, sortFields, all_combined_attributes, indicesForOctree)

        file_name = snapdir_path + "splines.npy"
        if not Path.isfile(file_name):
            np.save(file_name, c)
            logger.info("Saved: %(file_name)s", {"file_name": file_name})

        # Save all fields
        for field in fields:
            file_name = snapdir_path + field + ".npy"
            if np.array(all_combined_attributes[field]).ndim == 1:
                np.save(file_name, np.hstack(all_combined_attributes[field]))
            else:
                np.save(file_name, np.vstack(all_combined_attributes[field]))
            logger.info("Saved: %(file_name)s", {"file_name": file_name})

        file_name = snapdir_path + "voronoi_diameter_extended" + ".npy"
        np.save(file_name, edge_size)
        logger.info("Saved: %(file_name)s", {"file_name": file_name})
    else:
        logger.info(
            "Object was not saved to %(file_name)s. Please check if the path is sane.", {"file_name": file_name}
        )


def main():
    snap_idx = 75
    base_path = "D:/VMShare/Documents/data/"
    base_path = "/home/mulc/Documents/data/tng/webapp/TNG50-4/"
    fields = ["Coordinates", "ParticleIDs", "Density", "Velocities", "Masses", "GFM_Metallicity"]

    preprocess_snap(base_path, snap_idx, fields, size_per_leaf=350, sortFields=["Density", "Masses"])


if __name__ == "__main__":
    main()
