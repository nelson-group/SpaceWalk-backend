import illustris_python as il
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tqdm import tqdm


def loadOctreeWithData(basePath, baseSnapId, fields):
    file_name = basePath + "snapdir_" + str(baseSnapId).zfill(3) + "/o3dOctree.json"
    snapData = {}
    snapData["oct"] = o3d.io.read_octree(file_name)
    return snapData




def main():
    baseSnapId = 75
    basePath = 'D:/VMShare/Documents/data/'
    fields = ['Coordinates', 'ParticleIDs']
    snapData = loadOctreeWithData(basePath, baseSnapId, fields)

if __name__ == "__main__":
    main()