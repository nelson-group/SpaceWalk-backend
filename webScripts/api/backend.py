from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import open3d as o3d
import pickle
import numpy as np
from webScripts.octree.OctreeTraversal import OctreeTraversal, ViewBox

O3D_OCTREE = "o3dOctree.json"
BASE = Path("~/Documents/data/tng/manual_download/")

app = FastAPI()

class ClientState(BaseModel):
    node_indices: list[int]
    level_of_detail: dict[int, int]
    batch_size_lod: int # Number of particles per leaf to be loaded


class CameraInformation(BaseModel):
    x: int
    y: int
    z: int
    size: int
    
    def to_viewbox(self) -> ViewBox:
        """Return camera position as Viewbox."""
        boxMin = np.array([self.x-self.size//2, self.y-self.size//2, self.z-self.size//2])
        boxMax = np.array([self.x+self.size//2, self.y+self.size//2, self.z+self.size//2])
        return ViewBox(boxMin, boxMax)


def data_basedir(_simulation: str, snap_id: int) -> Path:
    """Return path to relevant data basedir."""
    return Path(BASE.joinpath(f"snapdir_{str(snap_id).zfill(3)}/")).expanduser()


class DataCache:
    def __init__(self) -> None:
        self._cache = dict()


    def get_data(self, simulation, snap_id):
        dictkey = simulation + snap_id
        if dictkey in self._cache:
            return self._cache[dictkey]
        
        basedir = data_basedir(simulation, snap_id)
        octree = o3d.io.read_octree(str(basedir.joinpath(O3D_OCTREE)))

        splines = np.load(basedir.joinpath("splines.npy"))
        velocities = np.load(basedir.joinpath("Velocities.npy"))
        densities = np.load(basedir.joinpath("Density.npy"))
        coordinates = np.load(basedir.joinpath("Coordinates.npy"))

        particle_list_of_leafs = pickle.load(basedir.joinpath("particleListOfLeafs.obj").open(mode="rb"))

        self._cache[dictkey] = {"particle_list_of_leafs": particle_list_of_leafs, "octree": octree, "splines": splines, "velocities": velocities, "densities": densities, "coordinates": coordinates}
        return self._cache[dictkey]


cache = DataCache()


@app.post("/v1/get/splines/{simulation}/{snap_id}")
async def get_splines(
    camera_information: CameraInformation,
    client_state: ClientState,
    simulation: str,
    snap_id: int
) -> JSONResponse:
    """Return splines for a specific camera position and snap_id."""

    data = cache.get_data(simulation, snap_id)
    octree = data["octree"]
    splines = data["splines"]
    velocities = data["velocities"]
    densities = data["densities"]
    coordinates = data["coordinates"]

    octree_traversal = OctreeTraversal(camera_information.to_viewbox())
    octree.traverse(octree_traversal.getIntersectingNodes)
    node_indices = np.array(octree_traversal.particleArrIds)

    # Load list of leaves with list of particlIdx's per leaf
    particle_list_of_leafs: list[list[int]] = data["particle_list_of_leafs"]

    # Get number of particles per leaf (=total number of particles to load in one level of detail) 
    lod_indices_per_leaf = client_state.batch_size_lod
    client_node_indices = np.array(client_state.node_indices)
    client_level_of_detail = np.array(client_state.level_of_detail)

    # List of numbers of total particles per leaf
    length_particles_in_leafs = {node: len(node) for node in particle_list_of_leafs[node_indices]}


    # Define Level of detail per leaf, in first run they should all be zero
    if len(client_node_indices) == 0:
        level_of_detail = {lod: 0 for lod in node_indices}
    else:
        # Get current level of detail only for current leafs
        node_indices_in_old_and_current_state = np.in1d(client_node_indices, node_indices)
        lods_to_be_incremeted = client_node_indices[node_indices_in_old_and_current_state]
        level_of_detail = {lod: client_level_of_detail.get(lod) for lod in lods_to_be_incremeted}


    lod_indices_start = {lod: level_of_detail.get(lod) * lod_indices_per_leaf for lod in node_indices}
    lod_indices_end = {lod: (level_of_detail.get(lod) + 1) * lod_indices_per_leaf for lod in node_indices}

    relevant_ids = list()
    # Out of bounds check
    for leaf in node_indices:
        # Check if there are nnot enough particles in every leaf, if so reset end of indices to last element in this leaf
        if length_particles_in_leafs[leaf] < lod_indices_end[leaf]:
            lod_indices_end[leaf] = length_particles_in_leafs[leaf] - 1
        
        # Create flattened array of all relevant particleIdx's
        relevant_ids.extend(particle_list_of_leafs[leaf][lod_indices_start[leaf]:lod_indices_end[leaf]])


    # Increase Level of details
    level_of_detail = {lod: level_of_detail.get(lod)+1 for lod in level_of_detail}

    return JSONResponse({
        "level_of_detail": level_of_detail,
        "relevant_ids": relevant_ids.tolist(),
        "node_indices": node_indices.tolist(),
        "coordinates": coordinates[relevant_ids].tolist(),
        "velocities": velocities[relevant_ids].tolist(),
        "densities": densities.T[relevant_ids].tolist(),
        "splines": splines[relevant_ids].tolist(),
    })

