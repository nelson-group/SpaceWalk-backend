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
    snapData["oct"] = o3d.io.read_octree(file_name)
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
    fields = ['Coordinates', 'ParticleIDs']
    times = []
    for i in range(10):
        start = time.time()
        snapData = loadOctreeWithData(basePath, baseSnapId, fields)
        end = time.time()
        duration = end - start
        print(duration)
        times.append(duration)

    print(np.mean(times))

if __name__ == "__main__":
    main()