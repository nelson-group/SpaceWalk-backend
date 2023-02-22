"""Check distance."""
import h5py
import numpy as np

f41 = h5py.File("041/combined_All_tng50-4_041.hdf5")
f42 = h5py.File("042/combined_All_tng50-4_042.hdf5")
id_41 = np.array(f41["PartType0"]["ParticleIDs"])
id_42 = np.array(f42["PartType0"]["ParticleIDs"])
coords_41 = np.array(f41["PartType0"]["Coordinates"])
coords_42 = np.array(f42["PartType0"]["Coordinates"])

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

print(np.mean(dist[mask]))
print(np.var(dist[mask]))
