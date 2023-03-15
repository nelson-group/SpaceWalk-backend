import numpy as np

particleIds = np.arange(100,200)
print(len(particleIds))
particleIds = np.hstack((particleIds,particleIds))
print(len(particleIds))
nodesIndices = np.array([3,5,6, 103,104,105])
offset = particleIds.max() - particleIds.min() + 1
u, inds = np.unique(particleIds[nodesIndices], return_index=True)
print(particleIds[nodesIndices])
print(u, inds)
noDuplicates = nodesIndices[inds] # ==> 3,5,6,104
print(noDuplicates)
print(offset)
noDuplicates[noDuplicates >= offset] -= offset
print(noDuplicates)
