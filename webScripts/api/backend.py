import pickle
import re
import time
from os import listdir
from os.path import isdir, join
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d
from dataclasses import dataclass
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import illustris_python as il
from webScripts.octree.OctreeTraversal import OctreeTraversal, ViewBox

O3D_OCTREE = "o3dOctree.json"
BASE = Path("D:/VMShare/Documents/data/")
BASE = Path("/home/tng/Documents/data/tng/webapp/")
# BASE = Path("~/Documents/data/tng/manual_download/")

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://94.16.31.82",
    "http://94.16.31.82:8080",
    "http://94.16.31.82:9998",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CameraInformation(BaseModel):
    x: float
    y: float
    z: float
    size: float

    def to_viewbox(self) -> ViewBox:
        """Return camera position as Viewbox."""
        boxMin = np.array([self.x - self.size // 2, self.y - self.size // 2, self.z - self.size // 2])
        boxMax = np.array([self.x + self.size // 2, self.y + self.size // 2, self.z + self.size // 2])
        return ViewBox(boxMin, boxMax)


class ClientState(BaseModel):
    node_indices: list[int]
    level_of_detail: dict[int, int]
    batch_size_lod: int  # Number of particles per leaf to be loaded
    camera_information: CameraInformation


def data_basedir(simulation: str, snap_id: int) -> Path:
    """Return path to relevant data basedir."""
    return Path(BASE.joinpath(f"{simulation}/snapdir_{str(snap_id).zfill(3)}/")).expanduser()


@dataclass
class ListOfLeafs:
    list_of_leafs: np.array
    list_of_leafs_scan: np.array

    def __getitem__(self, key: int) -> np.array:
        begin = self.list_of_leafs_scan[key]
        end = self.list_of_leafs_scan[key + 1] if key < len(self.list_of_leafs_scan) else -1
        return self.list_of_leafs[begin:end]


class DataCache:
    def __init__(self) -> None:
        self._cache = dict()

    def get_data(self, simulation, snap_id):
        dictkey = simulation + str(snap_id)
        if dictkey in self._cache:
            return self._cache[dictkey]

        basedir = data_basedir(simulation, snap_id)
        octree = o3d.io.read_octree(str(basedir.joinpath(O3D_OCTREE)))

        splines = np.load(basedir.joinpath("splines.npy"))
        velocities = np.load(basedir.joinpath("Velocities.npy"))
        densities = np.load(basedir.joinpath("Density.npy"))
        coordinates = np.load(basedir.joinpath("Coordinates.npy"))

        leafs = np.load(basedir.joinpath("particle_list_of_leafs_Density.npy"))
        leafs_scan = np.load(basedir.joinpath("particle_list_of_leafs_Density_scan.npy"))
        voronoi_diameter_extended = np.load(basedir.joinpath("voronoi_diameter_extended.npy"))

        # type: list[list[int]]
        # particle_list_of_leafs = pickle.load(basedir.joinpath("particleListOfLeafs.obj").open(mode="rb"))
        particle_list_of_leafs = ListOfLeafs(leafs, leafs_scan)
        density_quantiles = np.quantile(densities.flatten(), np.linspace(0, 1, 100))

        self._cache[dictkey] = {
            "particle_list_of_leafs": particle_list_of_leafs,
            "octree": octree,
            "splines": splines,
            "velocities": velocities,
            "densities": densities,
            "coordinates": coordinates,
            "density_quantiles": density_quantiles.tolist(),
            "voronoi_diameter_extended": voronoi_diameter_extended
        }
        return self._cache[dictkey]


cache = DataCache()


def get_init_data(simulation: str) -> Optional[dict[str, float | list[float]]]:
    return_data = {}
    only_dirs = [f for f in listdir(BASE / simulation) if isdir(join(BASE / simulation, f))]
    return_data["all_possible_snaps"] = []

    for dir in only_dirs:
        if re.search(r"snapdir_", dir):
            dir_splitted = dir.split("_")
            return_data["all_possible_snaps"].append(float(dir_splitted[-1]))
            return_data["all_possible_snaps"].sort()

        if re.search(r"groups_", dir) and not "BoxSize" in return_data:
            dir_splitted = dir.split("_")
            return_data["BoxSize"] = il.groupcat.loadHeader(str(BASE / simulation), float(dir_splitted[-1]))["BoxSize"]

    return_data["all_possible_snaps"].pop(-1)
    if "BoxSize" in return_data and len(return_data["all_possible_snaps"]) > 0:
        return return_data

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)


@app.get("/v1/get/init/{simulation}/{snap_id}")
async def get_init(simulation: str, snap_id: int) -> JSONResponse:
    data = cache.get_data(simulation, snap_id)
    init_data = get_init_data(simulation)

    return JSONResponse(
        {
            "density_quantiles": data["density_quantiles"],
            "n_quantiles": len(data["density_quantiles"]),
            "available_snaps": init_data["all_possible_snaps"],
            "BoxSize": init_data["BoxSize"],
        }
    )


@app.post("/v1/get/splines/{simulation}/{snap_id}")
async def get_splines(client_state: ClientState, simulation: str, snap_id: int) -> JSONResponse:
    """Return splines for a specific camera position and snap_id."""

    data = cache.get_data(simulation, snap_id)
    octree = data["octree"]
    splines = data["splines"]
    velocities = data["velocities"]
    densities = data["densities"]
    coordinates = data["coordinates"]
    voronoi_diameter_extended = data["voronoi_diameter_extended"]

    octree_traversal = OctreeTraversal(client_state.camera_information.to_viewbox())
    octree.traverse(octree_traversal.getIntersectingNodes)
    node_indices = np.array(octree_traversal.particleArrIds)

    # Load list of leaves with list of particlIdx's per leaf
    particle_list_of_leafs: list[list[int]] = data["particle_list_of_leafs"]
    # Get number of particles per leaf (=total number of particles to load in one level of detail)
    lod_indices_per_leaf = client_state.batch_size_lod
    client_node_indices = np.array(client_state.node_indices)
    client_level_of_detail = client_state.level_of_detail

    # List of numbers of total particles per leaf
    length_particles_in_leafs = {}
    for node_idx in node_indices:
        length_particles_in_leafs[int(node_idx)] = len(particle_list_of_leafs[node_idx])
    # length_particles_in_leafs = {node: len(node) for node in particle_list_of_leafs[node_idx]}

    # Get current level of detail only for current leaf
    level_of_detail = {lod: 0 for lod in node_indices}
    node_indices_in_old_and_current_state = np.in1d(client_node_indices, node_indices)
    for lod in client_node_indices[node_indices_in_old_and_current_state]:
        level_of_detail[lod] = client_level_of_detail.get(lod)

    lod_indices_start = {lod: level_of_detail.get(lod) * lod_indices_per_leaf for lod in node_indices}
    lod_indices_end = {lod: (level_of_detail.get(lod) + 1) * lod_indices_per_leaf for lod in node_indices}

    relevant_ids = list()
    # Out of bounds check
    for leaf in node_indices:
        # Check if there are nnot enough particles in every leaf, if so reset end of indices to last element in this leaf
        if length_particles_in_leafs[leaf] < lod_indices_end[leaf]:
            lod_indices_end[leaf] = length_particles_in_leafs[leaf] - 1

        # Create flattened array of all relevant particleIdx's
        relevant_ids.extend(particle_list_of_leafs[leaf][lod_indices_start[leaf] : lod_indices_end[leaf]+1])

    # Increase Level of details
    level_of_detail = {str(lod): int(level_of_detail.get(lod) + 1) for lod in level_of_detail}
    splines = splines[relevant_ids]
    splines_a = splines[:, 0].flatten()
    splines_b = splines[:, 1].flatten()
    splines_c = splines[:, 2].flatten()
    splines_d = splines[:, 3].flatten()

    min_den = 0
    max_den = 0
    nParticles = len(relevant_ids)
    if nParticles > 0:
        min_den = np.min(np.array(densities.T[relevant_ids]))
        max_den = np.max(np.array(densities.T[relevant_ids]))

    client_level_of_detail.update(level_of_detail)

    return JSONResponse(
        {
            "level_of_detail": client_level_of_detail,
            "relevant_ids": np.array(relevant_ids).tolist(),
            "node_indices": list(client_level_of_detail.keys()),
            "coordinates": coordinates[relevant_ids].tolist(),
            "velocities": velocities[relevant_ids].tolist(),
            "densities": densities.T[relevant_ids].flatten().tolist(),
            "voronoi_diameter_extended": voronoi_diameter_extended[relevant_ids].tolist(),
            "splines_a": splines_a.tolist(),
            "splines_b": splines_b.tolist(),
            "splines_c": splines_c.tolist(),
            "splines_d": splines_d.tolist(),
            "min_density": min_den,
            "max_density": max_den,
            "nParticles": nParticles,
            "density_quantiles": data["density_quantiles"],
            "snapnum": snap_id,
        }
    )
