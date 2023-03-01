import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

load_saves = 1
if load_saves:
    with open('test.npy', 'rb') as f:
        variancesPos = np.load(f, allow_pickle=True)
        variancesDist = np.load(f, allow_pickle=True)

    fig = plt.figure()
    plt.plot(variancesDist)
    plt.plot(variancesPos)
    plt.show()


pathToFiles = "c:/Users/rr3072/Documents/data/tng/TNG50-4/"
startId = 41
endId = 42

pos = np.zeros((0, 3))
variancesPos = np.zeros(0)
oldPos = np.zeros(0)
oldIds = np.zeros(0)
oldValues = {}
variancesDist = np.zeros(0)
meanNorm = np.zeros(0)
varNorm = np.zeros(0)

for id in range(startId, endId + 1):
    with h5py.File(pathToFiles + f"0{id}/combined_All_tng50-4_0{id}.hdf5", mode="r") as combined:
        # print(combined.keys())
        pos = np.array(combined["PartType0"]["Coordinates"])

        var = np.var(pos, axis=0)
        variancesPos = np.append(variancesPos, var, axis=0)
        if len(oldValues) != 0:
            currentIds = np.array(combined["PartType0"]["ParticleIDs"])
            distances = np.empty((len(pos), 3))
            for x, val in tqdm(enumerate(currentIds)):
                oldCoord = oldValues.get(val, None)
                if oldCoord is not None:
                    distances[x] = oldCoord - pos[x]

            variancesDist = np.append(variancesDist, np.var(distances, axis=0))
            norms = np.linalg.norm(distances, axis=1)
            print(norms.shape)
            meanNorm = np.append(meanNorm, np.mean(norms))
            varNorm = np.append(varNorm, np.var(norms))
            print(variancesDist)
            print(meanNorm)
            print(varNorm)

        oldValues = {}
        parIds = np.array((combined["PartType0"]["ParticleIDs"]))
        oldCoords = np.array(combined["PartType0"]["Coordinates"]).reshape(-1, 3)[:]
        print(oldCoords)
        print(parIds.min(), parIds.max(), parIds.shape)
        for idx, val in enumerate(oldCoords):
            oldValues[parIds[idx]] = val


variancesPos = variancesPos.reshape((-1, 3))
variancesDist = variancesDist.reshape((-1, 3))
print(variancesDist.reshape((-1, 3)))
print(variancesPos.reshape((-1, 3)))
with open('test.npy', 'wb') as f:
    np.save(f, variancesPos)
    np.save(f, variancesDist)

fig = plt.figure()
plt.plot(variancesDist.T)
plt.plot(variancesPos.T)
plt.show()

# for i in range(0, 11):
#     with h5py.File(f"fof_subhalo_tab_099.{i}.hdf5", mode="r")  as f:
#         group_pos = np.concatenate((group_pos, f["Group"]["GroupPos"]), axis=0)
#         group_vel = np.concatenate((group_vel, f["Group"]["GroupVel"]), axis=0)
#         group_mass = np.concatenate((group_vel, f["Group"]["GroupMass"]), axis=0)
#         subhalo_pos = np.concatenate((subhalo_pos, f["Subhalo"]["SubhaloPos"]), axis=0)
#         subhalo_vel = np.concatenate((subhalo_vel, f["Subhalo"]["SubhaloVel"]), axis=0)
#         # subhalo_mass = np.concatenate((subhalo_vel, f["Subhalo"]["SubhaloMass"]), axis=0)
#
# _group.create_dataset("GroupPos", group_pos.shape, float, group_pos)
# _group.create_dataset("GroupVel", group_vel.shape, float, group_vel)
# _group.create_dataset("GroupMass", group_mass.shape, float, group_mass)
# _subhalo.create_dataset("SubhaloPos", subhalo_pos.shape, float, subhalo_pos)
# _subhalo.create_dataset("SubhaloVel", subhalo_vel.shape, float, subhalo_vel)
# _subhalo.create_dataset("SubhaloMass", subhalo_mass.shape, float, subhalo_mass)

# combined.close()
