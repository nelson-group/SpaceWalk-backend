import numpy as np
import open3d as o3d
from dataclasses import dataclass, field


@dataclass
class ViewBox:
    boxMin: np.ndarray
    boxMax: np.ndarray

@dataclass
class OctreeTraversal:
    viewBox: ViewBox
    particleArrIds: list[int] = field(default_factory=list)


    def getIntersectingNodes(self, node, node_info):
        early_stop = True

        if not _boxIntersect(node_info.origin, node_info.origin+node_info.size, self.viewBox.boxMin, self.viewBox.boxMax):
            return early_stop

        if isinstance(node, o3d.geometry.OctreeLeafNode):
            self.particleArrIds.append(node.indices[0])

            return early_stop

        return not early_stop


def _boxIntersect(minBox,maxBox,minCamera,maxCamera):
    dx = min(maxBox[0], maxCamera[0]) - max(minBox[0], minCamera[0])
    dy = min(maxBox[1], maxCamera[1]) - max(minBox[1], minCamera[1])
    dz = min(maxBox[2], maxCamera[2]) - max(minBox[2], minCamera[2])
    if (dx >= 0 and dy >= 0 and dz >= 0):
        return True
    return False
