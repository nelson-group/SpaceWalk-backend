import illustris_python as il
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

idx = 0
def main():
    baseSnapId = 75
    basePath = 'D:/VMShare/Documents/data/'

    fields = ['Masses', 'Coordinates', 'Density', 'Velocities', 'ParticleIDs']
    snapshot0 = il.snapshot.loadSubset(basePath, baseSnapId, 'gas', fields=fields)
    header0 = il.groupcat.loadHeader(basePath, baseSnapId)

    numberOfParticlesPerBox = np.ceil(snapshot0['count'] / header0['Nsubgroups_Total'])
    print(numberOfParticlesPerBox)


    snapshot1 = il.snapshot.loadSubset(basePath, baseSnapId + 1, 'gas', fields=fields)
    header1 = il.groupcat.loadHeader(basePath, baseSnapId + 1)

    allCombinedAttributes = getSameParticleInTwoDataSets(snapshot0, snapshot1, fields)
    coordinates = np.vstack(allCombinedAttributes['Coordinates'])
    densities = np.hstack(allCombinedAttributes['Density'])
    selection = np.floor(np.linspace(0, len(densities)-1, len(densities))).astype(int)
    print("preprocessing")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(coordinates[selection]))
    print("loaded pcd")

    maxDens = densities[selection].max()
    color = np.array((np.linspace(0, 0.7, len(selection)),) * 3).T
    pcd.colors = o3d.utility.Vector3dVector(color)
    # o3d.visualization.draw_geometries([pcd])
    # print(pcd.get_center(), pcd.get_max_bound(), pcd.get_min_bound())

    oct = o3d.geometry.Octree(max_depth=5)
    oct.convert_from_point_cloud(pcd)
    print("loaded oct")

    viewBox = dict(Min=coordinates[0]-10000, Max=coordinates[0]+10000)
    lod = 100
    lodMax = 10000


    def getIntersectingNodes(node, node_info):
        global idx
        early_stop = True

        if idx >= lodMax:
            return early_stop

        # if len(particleArrIds) >= lodMax:
        #     return  early_stop

        if not boxIntersect(node_info.origin, node_info.origin+node_info.size, viewBox["Min"], viewBox["Max"]):
            return early_stop

        if isinstance(node, o3d.geometry.OctreeLeafNode):
            rangeLod = min(lod, len(node.indices), lodMax - idx)
            particleArrIds[idx:idx+rangeLod] = node.indices[0:rangeLod]
            idx += rangeLod
            # rangeLod = min(lod, len(node.indices))
            # for ids in range(rangeLod):
            #     particleArrIds.append(node.indices[ids])

            return early_stop

        return not early_stop

    import time
    global idx
    times = []
    for i in range(10):
        start = time.time()

        idx = 0
        particleArrIds = np.zeros(lodMax)
        # particleArrIds = []
        oct.traverse(getIntersectingNodes)
        end = time.time()
        print(len(particleArrIds), particleArrIds)
        times.append(end - start)

    print(np.mean(times))




    # o3d.visualization.draw_geometries([oct])
    # print(oct.get_center(), oct.get_max_bound(), oct.get_min_bound())
    # print(coordinates[0])

    # leafnode = oct.locate_leaf_node(coordinates[0]+1000)
    # print(leafnode)
    #
    # grid = o3d.geometry.VoxelGrid()
    # grid.create_from_octree(oct)
    # o3d.visualization.draw_geometries([oct.to_voxel_grid()])

    # print(grid.get_center(), grid.get_max_bound(), grid.get_min_bound())

def boxIntersect(minBox,maxBox,minCamera,maxCamera):
        dx = min(maxBox[0], maxCamera[0]) - max(minBox[0], minCamera[0])
        dy = min(maxBox[1], maxCamera[1]) - max(minBox[1], minCamera[1])
        dz = min(maxBox[2], maxCamera[2]) - max(minBox[2], minCamera[2])
        if (dx >= 0 and dy >= 0 and dz >= 0):
            return True
        return False




# def f_traverse(node, node_info):
#     early_stop = False
#
#     if isinstance(node, o3d.geometry.OctreeInternalNode):
#         if isinstance(node, o3d.geometry.OctreeInternalPointNode):
#             n = 0
#             for child in node.children:
#                 if child is not None:
#                     n += 1
#             print(
#                 "{}{}: Internal node at depth {} has {} children and {} points ({})"
#                 .format('    ' * node_info.depth,
#                         node_info.child_index, node_info.depth, n,
#                         len(node.indices), node_info.origin))
#
#             # we only want to process nodes / spatial regions with enough points
#             early_stop = len(node.indices) < 250
#     elif isinstance(node, o3d.geometry.OctreeLeafNode):
#         if isinstance(node, o3d.geometry.OctreePointColorLeafNode):
#             print("{}{}: Leaf node at depth {} has {} points with origin {}".
#                   format('    ' * node_info.depth, node_info.child_index,
#                          node_info.depth, len(node.indices), node_info.origin))
#     else:
#         raise NotImplementedError('Node type not recognized!')
#
#     # early stopping: if True, traversal of children of the current node will be skipped
#     return early_stop



def getSameParticleInTwoDataSets(snapshot0, snapshot1, dataTypes):
    id_0 = np.array(snapshot0["ParticleIDs"])
    id_1 = np.array(snapshot1["ParticleIDs"])

    _max = np.max((np.max(id_0), np.max(id_1)))
    _min = np.min((np.min(id_0), np.min(id_1)))

    _len = int(_max - _min) + 1

    mask1 = np.zeros(_len)
    mask1[np.array(id_0 - _min, dtype=int)] = 1

    mask2 = np.zeros(_len)
    mask2[np.array(id_1 - _min, dtype=int)] = 1
    mask = (mask1 * mask2).astype(bool)
    allCombinedAttributes = dict()
    for dataType in dataTypes:
        attributesDense0 = snapshot0[dataType]
        attributesDense1 = snapshot1[dataType]
        dimensionTuple = (_len, 3)
        if attributesDense1.ndim == 1:
            dimensionTuple = (_len,)

        attributesSparse0 = np.zeros(dimensionTuple)
        attributesSparse1 = np.zeros(dimensionTuple)

        attributesSparse0[np.array(id_0 - _min, dtype=int)] = attributesDense0
        attributesSparse1[np.array(id_1 - _min, dtype=int)] = attributesDense1
        allCombinedAttributes[dataType] = (attributesSparse0[mask], attributesSparse1[mask])

    return allCombinedAttributes

if __name__ == "__main__":
    main()