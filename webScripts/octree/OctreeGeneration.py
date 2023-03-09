import illustris_python as il
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import pickle

from tng_sv.webScripts.preprocessing.gridGeneration import getSameParticleInTwoDataSets


def loadDatasets(basePath, baseSnapId, fields, nSnapsToLoad, loadHeader=False):
    allLoadedSnap = list()
    for i in range(baseSnapId, baseSnapId+nSnapsToLoad):
        snapDict = {}
        snapDict["snapData"] = il.snapshot.loadSubset(basePath, baseSnapId, 'gas', fields=fields)
        snapDict["snapInfo"] = None
        if loadHeader:
            snapDict["snapInfo"] = il.groupcat.loadHeader(basePath, baseSnapId)

        allLoadedSnap.append(snapDict)
    return allLoadedSnap


def generateOctree(coordinates, max_depth = 5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(coordinates))
    octTree = o3d.geometry.Octree(max_depth=max_depth)
    octTree.convert_from_point_cloud(pcd)
    return octTree


def preprocessSnaps(basePath, baseSnapId, fields, nSnapsToLoad, sizePerLeaf=100,safeOctree=False):
    global idx

    loadHeader = True
    allLoadedSnaps = loadDatasets(basePath, baseSnapId, fields, nSnapsToLoad, loadHeader)
    maxDepth = np.ceil(np.log2(allLoadedSnaps[0]["snapInfo"]["BoxSize"] / sizePerLeaf)).astype(int) # approximation der tiefe bei sizePerLeaf Angabe => sizeLeaf ~ sizeBox / 2^x ==> formel vorne
    print(maxDepth)
    allOctrees = list()


    for i in tqdm(range(nSnapsToLoad-1),total=nSnapsToLoad-1):
        allCombinedAttributes = getSameParticleInTwoDataSets(allLoadedSnaps[i]["snapData"], allLoadedSnaps[i+1]["snapData"], fields)
        coordinates = np.vstack(allCombinedAttributes['Coordinates'])
        # densities = np.hstack(allCombinedAttributes['Density'])
        octTree = generateOctree(coordinates, maxDepth)
        indicesForOctree = []
        idx = 0

        def changeIdsWithListId(node, node_info):
            if isinstance(node, o3d.geometry.OctreeLeafNode):
                global idx
                indicesForOctree.append(np.array(node.indices))
                node.indices = [idx]
                idx += 1
                return True
            if isinstance(node, o3d.geometry.OctreeInternalPointNode):
                node.indices = []

        octTree.traverse(changeIdsWithListId)
        sum = 0
        for idx, val in enumerate(indicesForOctree):
            sum += len(val)
        print(sum / len(indicesForOctree))

        if not safeOctree:
            allOctrees.append(octTree)
        else:
            file_name = basePath + "snapdir_" + str(baseSnapId + i).zfill(3)+"/o3dOctree.json"
            print("Safe Octree to " + file_name)
            if o3d.io.write_octree(file_name, octTree):
                print(f'Object successfully saved to "{file_name}"')
            else:
                print(f'Object was not saved to "{file_name}"')

    return allOctrees



def main():
    baseSnapId = 75
    basePath = 'D:/VMShare/Documents/data/'
    fields = ['Coordinates', 'ParticleIDs']
    nSnapsToLoad = 10

    preprocessSnaps(basePath, baseSnapId, fields, nSnapsToLoad, sizePerLeaf=350, safeOctree=True)

if __name__ == "__main__":
    main()