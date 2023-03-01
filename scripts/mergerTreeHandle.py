import illustris_python as il
import scipy
import numpy as np
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
from tqdm import tqdm

baseSnapId = 99
basePath = 'D:/VMShare/Documents/data/'

fields = ['Masses', 'Coordinates', 'Density']
snapshot = il.snapshot.loadSubset(basePath,baseSnapId,'gas',fields=fields)
fields = ['SubhaloMass', 'SubhaloSFRinRad']
subhalos = il.groupcat.loadSubhalos(basePath, baseSnapId, fields=fields)
subhalosAll = il.groupcat.loadSubhalos(basePath, baseSnapId)
header = il.groupcat.loadHeader(basePath, baseSnapId)
hubble = header["HubbleParam"] * 100 # see docs
o0 = header["Omega0"]
boxSize = header["BoxSize"]
extends = (0, boxSize)
GroupFirstSub = il.groupcat.loadHalos(basePath, baseSnapId, fields=['GroupFirstSub'])

fields = ['SubhaloPos', 'SubfindID', 'SnapNum', 'SubhaloVel', "SubhaloVelDisp"]

subhaloIds = np.linspace(0, 3000, 100, dtype=int)
print(GroupFirstSub.shape)


cosmo = FlatLambdaCDM(H0=hubble * u.km / u.s / u.Mpc, Om0=o0) # passt bei tng50-4 für ersten und letzen redshift
# print(cosmo.age(20.05))

fig = plt.figure()
plt.ion()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim(extends)
ax.set_ylim(extends)
ax.set_zlim(extends)
travelCoef = np.zeros(len(subhaloIds))
anzSnapnums = np.zeros(len(subhaloIds))
for idx, subhaloId in tqdm(enumerate(subhaloIds)):
    tree = il.sublink.loadTree(basePath, baseSnapId, GroupFirstSub[subhaloId], fields=fields, onlyMPB=True)
    subhaloSnapNums = np.flip(np.array(tree["SnapNum"]), axis=0)
    anzSnapnums[idx] = len(subhaloSnapNums)
    subhaloPos = np.flip(np.array(tree["SubhaloPos"]), axis=0)
    subhaloVel = np.flip(np.array(tree["SubhaloVel"]), axis=0) # km sqrt(a) / s (spatial vel, see doc) 3.2408e-11 = km -> kpc
    subhaloVel = subhaloVel * 3.154e+7 / 3.086e+16 # calc km/s to kpc/a
    # print(tree["SubhaloVelDisp"] * 3.154e+7 / 3.086e+16)
    traveledDistance = 0
    for snapId in range(len(subhaloSnapNums)-1):
        piecewiseSnapNums = subhaloSnapNums[snapId:snapId+2]

        headerSnapNum0 = il.groupcat.loadHeader(basePath, piecewiseSnapNums[0])
        headerSnapNum1 = il.groupcat.loadHeader(basePath, piecewiseSnapNums[1])
        timeDif = (headerSnapNum0["Redshift"] - headerSnapNum1["Redshift"]) * 1e9
        piecewisePos = subhaloPos[snapId:snapId+2]
        piecewiseVelOld = np.array(subhaloVel[snapId:snapId+2], dtype=np.float64)
        piecewiseVel = piecewiseVelOld
        spline = scipy.interpolate.CubicHermiteSpline(piecewiseSnapNums, piecewisePos, piecewiseVel)
        c = spline.c
        x = spline.x
        k = c.shape[0] - 1
        s = np.zeros((0, 3))
        interpolationSteps = np.linspace(0, 1, 100)
        for i in interpolationSteps:
            terms = [c[m, 0] * i ** (k - m) for m in range(k+1)]
            sTmp = np.sum(terms, axis=0).reshape((1, -1))

            s = np.concatenate((s, sTmp))

        difPos = np.linalg.norm(np.diff(piecewisePos, axis=0))
        tempo = np.linalg.norm(np.sum(piecewiseVelOld, axis=0) / 2)
        expectedTime = difPos / tempo
        # print(f"{snapId} for subhalo {subhaloId}: {expectedTime / timeDif}")

        traveledDistance += np.sum(np.linalg.norm(np.diff(s,axis=0), axis=1))
        # dif0 = s[0] - piecewisePos[0]
        # dif1 = s[-1] - piecewisePos[1]
        plt.title(f"All Subhalos; ")
        ax.scatter(s[:, 0], s[:, 1], s[:, 2],s=0.2)
        ax.scatter(piecewisePos[:, 0], piecewisePos[:, 1], piecewisePos[:, 2],marker='x')
    travelCoef[idx] = traveledDistance

print(travelCoef)
print(travelCoef / boxSize)
print(travelCoef / (30 * anzSnapnums))
print(np.median(travelCoef))
print(np.median(travelCoef / boxSize))
print(np.median(travelCoef / np.linalg.norm([boxSize]*3)))
print(np.median(travelCoef / (30 * anzSnapnums)))

plt.show(block=True) # if they break down and have strange ways, thats based on the data: https://www.tng-project.org/data/docs/scripts/
