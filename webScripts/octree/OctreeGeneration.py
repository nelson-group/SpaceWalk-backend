import multiprocessing
from multiprocessing import Pool

import illustris_python as il
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
import pickle
import scipy

from webScripts.preprocessing.gridGeneration import getSameParticleInTwoDataSets


def loadDatasets(basePath, baseSnapId, fields, nSnapsToLoad, loadHeader=False):
    allLoadedSnap = list()
    for i in range(baseSnapId, baseSnapId+nSnapsToLoad):
        snapDict = {}
        snapDict["snapData"] = il.snapshot.loadSubset(basePath, i, 'gas', fields=fields)
        snapDict["snapInfo"] = None
        if loadHeader:
            snapDict["snapInfo"] = il.groupcat.loadHeader(basePath, i)

        allLoadedSnap.append(snapDict)
    return allLoadedSnap


def generateOctree(coordinates, max_depth = 5):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(coordinates))
    octTree = o3d.geometry.Octree(max_depth=max_depth)
    octTree.convert_from_point_cloud(pcd)
    return octTree


def f(c0, c1, v0, v1):
    spline = scipy.interpolate.CubicHermiteSpline([0, 1],
                                                  [c0, c1],
                                                  [v0, v1])
    return spline.c.squeeze()

def preprocessSnaps(basePath, baseSnapId, fields, nSnapsToLoad, sizePerLeaf=100,safeOctree=False, sortField = "Density"):
    global idx

    loadHeader = True
    allLoadedSnaps = loadDatasets(basePath, baseSnapId, fields, nSnapsToLoad, loadHeader)
    maxDepth = np.ceil(np.log2(allLoadedSnaps[0]["snapInfo"]["BoxSize"] / sizePerLeaf)).astype(int) # approximation der tiefe bei sizePerLeaf Angabe => sizeLeaf ~ sizeBox / 2^x ==> formel vorne
    print(maxDepth)
    allOctrees = list()


    for i in tqdm(range(nSnapsToLoad-1),total=nSnapsToLoad-1):
        allCombinedAttributes = getSameParticleInTwoDataSets(allLoadedSnaps[i]["snapData"], allLoadedSnaps[i+1]["snapData"], fields)
        coordinates = np.vstack(allCombinedAttributes['Coordinates'])
        octTree = generateOctree(coordinates, maxDepth)
        indicesForOctree = []
        idx = 0
        dimsOfSortField = allCombinedAttributes[sortField][0].ndim
        if dimsOfSortField == 1:
            fields_array = np.hstack(allCombinedAttributes[sortField])
        elif dimsOfSortField == 3:
            threeDimsFields = np.vstack(allCombinedAttributes[sortField])
            fields_array = np.linalg.norm(threeDimsFields, axis=0)
        else:
            raise Exception(f"Either 1 or 3 dims, instead found {dimsOfSortField} dims")

        offset = len(allCombinedAttributes["Coordinates"][0])
        particleIds = np.hstack(allCombinedAttributes['ParticleIDs']).astype(np.int64)

        def changeIdsWithListId(node, node_info):
            if isinstance(node, o3d.geometry.OctreeLeafNode):
                global idx

                _,indices = np.unique(particleIds[node.indices], return_index=True)
                allIndicesWithoutDuplicates = np.array(node.indices)[indices]
                allIndicesWithoutDuplicates[allIndicesWithoutDuplicates >= offset] -= offset

                fields_in_leaf = fields_array[allIndicesWithoutDuplicates]
                sorted_indices = np.array(np.argsort(fields_in_leaf)[::-1])
                indicesForOctree.append(np.array(allIndicesWithoutDuplicates[sorted_indices]))
                node.indices = [idx]
                idx += 1
                return True

            if isinstance(node, o3d.geometry.OctreeInternalPointNode):
                node.indices = []
        print("start octree calc")
        octTree.traverse(changeIdsWithListId)
        print("Octree calculated")

        sum = 0
        for idx, val in enumerate(indicesForOctree):
            sum += len(val)
        print(sum / len(indicesForOctree))

        # --------------Spline Array bauen ---------------
        print("start spline calc")
        c0 = np.array(allCombinedAttributes["Coordinates"][0])
        c1 = np.array(allCombinedAttributes["Coordinates"][1])
        v0 = np.array(allCombinedAttributes["Velocities"][0])
        v1 = np.array(allCombinedAttributes["Velocities"][1])
        zipped = zip(c0, c1, v0, v1)
        with Pool(processes=int(multiprocessing.cpu_count())) as pool:
            c = pool.starmap(f, zipped)
        print("Splines calculated")

        if not safeOctree:
            allOctrees.append(octTree)
        else:
            file_path = basePath + "snapdir_" + str(baseSnapId + i).zfill(3)+"/"
            file_name = file_path + "o3dOctree.json"
            print("Safe Octree to " + file_name)
            if o3d.io.write_octree(file_name, octTree):
                print(f'Object successfully saved to "{file_name}", Saving additional data:')
                file_name = file_path + "particleListOfLeafs.obj"
                with open(file_name, 'wb') as file:
                    pickle.dump(indicesForOctree, file)
                print(f'Saved: "{file_name}"')

                file_name = file_path + "splines.npy"
                np.save(file_name, c)
                print(f'Saved: "{file_name}"')

                for field in fields:
                    file_name = file_path + field + ".npy"
                    if np.array(allCombinedAttributes[field]).ndim == 1:
                        np.save(file_name, np.hstack(allCombinedAttributes[field]))
                    else:
                        np.save(file_name, np.vstack(allCombinedAttributes[field]))
                    print(f'Saved: "{file_name}"')
            else:
                print(f'Object was not saved to "{file_name}"')

    return allOctrees

def main():
    baseSnapId = 75
    basePath = 'D:/VMShare/Documents/data/'
    fields = ['Coordinates', 'ParticleIDs', 'Density', 'Velocities']
    nSnapsToLoad = 5

    preprocessSnaps(basePath, baseSnapId, fields, nSnapsToLoad, sizePerLeaf=350, safeOctree=True)

if __name__ == "__main__":
    main()