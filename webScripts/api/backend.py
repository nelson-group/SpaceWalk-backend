from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
import open3d as o3d
import pickle
import numpy as np

O3D_OCTREE = "o3dOctree.json"
BASE = Path("~/Documents/data/tng/manual_download/")

app = FastAPI()

class CameraPosition(BaseModel):
    x: int
    y: int
    z: int
    
    def to_point(self) -> np.ndarray:
        """Return camera position as point."""
        return np.array([self.x, self.y, self.z])


def data_basedir(_simulation: str, snap_id: int) -> Path:
    """Return path to relevant data basedir."""
    return Path(BASE.joinpath(f"snapdir_{str(snap_id).zfill(3)}/")).expanduser()


@app.post("/get/splines/{simulation}/{snap_id}")
async def get_splines(
    camera_position: CameraPosition,
    simulation: str,
    snap_id: int
) -> JSONResponse:
    """Return splines for a specific camera position and snap_id."""
    basedir = data_basedir(simulation, snap_id)

    octree = o3d.io.read_octree(str(basedir.joinpath(O3D_OCTREE)))

    splines = np.load(basedir.joinpath("splines.npy"))
    velocities = np.load(basedir.joinpath("Velocities.npy"))
    densities = np.load(basedir.joinpath("Density.npy"))
    coordinates = np.load(basedir.joinpath("Coordinates.npy"))

    leaf_id, _ = octree.locate_leaf_node(camera_position.to_point())
    particle_list_of_leafs = pickle.load(basedir.joinpath("particleListOfLeafs.obj").open(mode="rb"))
    relevant_ids = particle_list_of_leafs[leaf_id.indices[0]]

    return JSONResponse({
        "relevant_ids": relevant_ids.tolist(),
        "coordinates": coordinates[relevant_ids].tolist(),
        "velocities": velocities[relevant_ids].tolist(),
        "densities": densities.T[relevant_ids].tolist(),
        "splines": splines[relevant_ids].tolist(),
    })
