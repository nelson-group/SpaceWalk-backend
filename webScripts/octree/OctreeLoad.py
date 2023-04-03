import pickle

import illustris_python as il
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm
from os import listdir
from os.path import isfile, join
import re
import time

def loadOctreeWithData(basePath, baseSnapId, fields):
    file_path = basePath + "snapdir_" + str(baseSnapId).zfill(3)
    file_name = file_path + "/o3dOctree.json"
    snapData = {}
    start = time.time()
    # snapData["oct"] = o3d.io.read_octree(file_name)
    end = time.time()
    duration = end - start
    print(f"duration octree:{duration}")

    start = time.time()
    onlyfiles = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    end = time.time()
    duration = end - start
    print(f"duration filenames:{duration}")

    start = time.time()
    for file in onlyfiles:
        if re.search(r".*obj", file):
            file_name = file_path + f"/{file}"
            with open(file_name, 'rb') as objFile:
                snapData["particleListOfLeaf"] = pickle.load(objFile)
                break

    end = time.time()
    duration = end - start
    print(f"duration obj:{duration}")

    start = time.time()
    for file in onlyfiles:
        if re.search(r".*splines.*", file):
            file_name = file_path + f"/{file}"
            snapData["splines"] = np.load(file_path + f"/{file}")
            break

    end = time.time()
    duration = end - start
    print(f"duration splines:{duration}")

    if not snapData["particleListOfLeaf"]:
        print("Error while loading particleListOfLeaf!")
        return snapData

    start = time.time()
    for field in fields:
        for file in onlyfiles:
            if field.lower() in file.lower():
                snapData[field] = np.load(file_path + f"/{file}")
    end = time.time()
    duration = end - start
    print(f"duration data:{duration}")

    return snapData




def main():
    baseSnapId = 75
    basePath = 'D:/VMShare/Documents/data/'
    fields = ['Coordinates', 'ParticleIDs', 'Density']
    times = []
    allData = {}
    for i in range(10):
        start = time.time()
        allData[i] = loadOctreeWithData(basePath, baseSnapId+i, fields)
        print(allData[0]['Coordinates'].shape)
        end = time.time()
        duration = end - start
        print(duration)
        times.append(duration)

    print(np.mean(times))
    print("test")
    # print(allData[1]['ParticleIDs'])
    print(np.all(np.equal(allData[0]['ParticleIDs'][0], allData[0]['ParticleIDs'][1])))
    # print(allData[1]['Density'])
    ids = np.in1d(allData[0]['ParticleIDs'][0], allData[1]['ParticleIDs'][0])
    sameIds = np.sum(ids)
    print(len(ids))

    print(len(allData[0]['ParticleIDs'][0]))
    print(len(allData[1]['ParticleIDs'][0]))
    # print(len(allData[3]['ParticleIDs'][0]))
    # print(len(allData[4]['ParticleIDs'][0]))
    # print(len(allData[5]['ParticleIDs'][0]))
    print(sameIds / allData[0]['ParticleIDs'].shape[1])
    print(sameIds / allData[1]['ParticleIDs'].shape[1])

    print(np.equal(np.sum([allData[0]['splines'][5,m, :] * 1 ** (3 - m) for m in range(3+1)], axis=0), allData[1]['Coordinates'][5]))
    # print(allData[0]['ParticleIDs'][ids[5]] == allData[1]['ParticleIDs'][ids[5]])
    nLost = np.sum(np.invert(ids))
    nNewOnes = np.sum(np.invert(np.in1d(allData[1]['ParticleIDs'][0], allData[0]['ParticleIDs'][0])))
    print(nLost + nNewOnes == allData[0]['ParticleIDs'].shape[1] - sameIds)
    print(nLost + nNewOnes)
    print(allData[1]['ParticleIDs'].shape[1] - sameIds)
    print((nLost + nNewOnes) / allData[0]['ParticleIDs'].shape[1])
    print((nLost + nNewOnes) / allData[1]['ParticleIDs'].shape[1])
    print((allData[1]['ParticleIDs'].shape[1] - sameIds) / allData[1]['ParticleIDs'].shape[1])

    max0 = np.argmax(allData[0]["Density"][0])
    max1 = np.argmax(allData[1]["Density"][0])
    print(allData[0]["Density"][0][max0], allData[1]["Density"][0][max1])
    print(allData[0]["ParticleIDs"][0][max0], allData[1]["ParticleIDs"][0][max1])
    densitiyIdOfHighest1in0 = np.where(allData[0]["ParticleIDs"][0] == allData[1]["ParticleIDs"][0][max1])
    densitiyIdOfHighest0in1 = np.where(allData[1]["ParticleIDs"][0] == allData[0]["ParticleIDs"][0][max0])
    print(densitiyIdOfHighest1in0, densitiyIdOfHighest0in1)
    print(allData[0]["Density"][0][densitiyIdOfHighest1in0], allData[1]["Density"][0][max1])
    print(allData[0]["Density"][0][max0], allData[1]["Density"][0][densitiyIdOfHighest0in1])
    for i in range(10):
        tmp = np.where(allData[i]["ParticleIDs"][0] == allData[1]["ParticleIDs"][0][max1])
        print(allData[i]["Coordinates"][tmp], allData[i]["Density"][0][tmp], np.max(allData[i]["Density"][0]))

if __name__ == "__main__":
    main()

# np.sum([allData[0]['splines'][0,m, :] * 1 ** (3 - m) for m in range(3+1)], axis=0) == allData[1]['Coordinates'][0]
# gleiche abs 36617478
# gleiche rel np.sum(np.in1d(allData[0]['ParticleIDs'], allData[1]['ParticleIDs']))