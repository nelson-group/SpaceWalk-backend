"""Check distance."""
import h5py
import numpy as np
from tqdm import tqdm

pathToFiles = "c:/Users/rr3072/Documents/data/tng/TNG50-4/"
startId = 41
endId = 46
allmeans = list()
allvars = list()
for id in tqdm(range(startId, endId)):
    f41 = h5py.File(pathToFiles + f"0{id}/combined_All_tng50-4_0{id}.hdf5")
    f42 = h5py.File(pathToFiles + f"0{id+1}/combined_All_tng50-4_0{id+1}.hdf5")
    id_41 = np.array(f41["PartType0"]["ParticleIDs"])
    id_42 = np.array(f42["PartType0"]["ParticleIDs"])
    coords_41 = np.array(f41["PartType0"]["Coordinates"])
    coords_42 = np.array(f42["PartType0"]["Coordinates"])
    vel_41 = np.array(f41["PartType0"]["Velocities"]) * np.sqrt(1 / 2.41)
    vel_42 = np.array(f42["PartType0"]["Velocities"]) * np.sqrt(1 / 2.36)

    f41.close()
    f42.close()

    _max = np.max((np.max(id_41), np.max(id_42)))
    _min = np.min((np.min(id_41), np.min(id_42)))

    _len = int(_max - _min) + 1

    coords1 = np.zeros((_len, 3))
    coords2 = np.zeros((_len, 3))

    coords1[np.array(id_41 - _min, dtype=int)] = coords_41
    coords2[np.array(id_42 - _min, dtype=int)] = coords_42

    mask1 = np.zeros(_len)
    mask1[np.array(id_41 - _min, dtype=int)] = 1

    mask2 = np.zeros(_len)
    mask2[np.array(id_42 - _min, dtype=int)] = 1
    mask = (mask1 * mask2).astype(int)

    dist = np.linalg.norm(coords1 - coords2, axis=1)

    allmeans.append(np.mean(dist[mask]))
    allvars.append(np.var(dist[mask]))
    print(_min, _max)
    print(np.mean(vel_41, axis=0), np.mean(vel_42, axis=0))
    print(np.mean(np.linalg.norm(vel_41, axis=1)), np.mean(np.linalg.norm(vel_42, axis=1)))
print(allmeans)
print(allvars)
