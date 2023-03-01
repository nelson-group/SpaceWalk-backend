import illustris_python as il
import scipy
import numpy as np
import matplotlib.pyplot as plt

baseSnapId = 99
basePath = 'D:/VMShare/Documents/data/'

fields = ['Masses', 'Coordinates', 'Density']
snapshot = il.snapshot.loadSubset(basePath,baseSnapId,'gas',fields=fields)
minCoord = np.min(snapshot["Coordinates"],axis=0)
maxCoord = np.max(snapshot["Coordinates"],axis=0)
print(minCoord)
print(maxCoord)
fields = ['SubhaloMass', 'SubhaloSFRinRad']
subhalos = il.groupcat.loadSubhalos(basePath, baseSnapId, fields=fields)
GroupFirstSub = il.groupcat.loadHalos(basePath, baseSnapId, fields=['GroupFirstSub'])

fields = ['SubhaloPos', 'SubfindID', 'SnapNum', 'SubhaloVel']

subhaloIds = [0, 100, 500, 1000, 3000]
print(GroupFirstSub.shape)

fig = plt.figure()
plt.ion()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim(minCoord[0], maxCoord[0])
ax.set_ylim(minCoord[1], maxCoord[1])
ax.set_zlim(minCoord[2], maxCoord[2])

for subhaloId in subhaloIds:
    tree = il.sublink.loadTree(basePath, baseSnapId, GroupFirstSub[subhaloId], fields=fields, onlyMPB=True)
    subhaloSnapNums = np.flip(np.array(tree["SnapNum"]), axis=0)
    subhaloPos = np.flip(np.array(tree["SubhaloPos"]), axis=0)
    subhaloVel = np.flip(np.array(tree["SubhaloVel"]), axis=0) # km sqrt(a) / s  3.2408e-11 = km -> kpc
    subhaloVel = subhaloVel * 3.154e+7 / 3.086e+16 # calc km/s to kpc/a
    for snapId in range(len(subhaloSnapNums)-1):
        piecewiseSnapNums = subhaloSnapNums[snapId:snapId+2]
        piecewisePos = subhaloPos[snapId:snapId+2]
        piecewiseVel = subhaloVel[snapId:snapId+2]
        spline = scipy.interpolate.CubicHermiteSpline(piecewiseSnapNums, piecewisePos, piecewiseVel)
        c = spline.c
        x = spline.x
        k = c.shape[0] - 1
        s = np.zeros((0, 3))
        interpolationSteps = np.linspace(0, 1, 10)
        for i in interpolationSteps:
            terms = [c[m, 0] * i ** (k - m) for m in range(k+1)]
            sTmp = np.sum(terms, axis=0).reshape((1, -1))

            s = np.concatenate((s, sTmp))

        # dif0 = s[0] - piecewisePos[0]
        # dif1 = s[-1] - piecewisePos[1]
        plt.title(f"All Subhalos; ")
        ax.scatter(s[:, 0], s[:, 1], s[:, 2],s=0.1)
        ax.scatter(piecewisePos[:, 0], piecewisePos[:, 1], piecewisePos[:, 2],marker='x')

plt.show(block=True)# if they break down and have strange ways, thats based on the data: https://www.tng-project.org/data/docs/scripts/
