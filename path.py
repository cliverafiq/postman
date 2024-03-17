from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
import numpy as np
from typing import List
from adjacency import AdjMatrix

def test_dijkstra(am):
    #g = csr_matrix(am)
    #print(g)
    dist_matrix, pred = dijkstra(csgraph=am, directed=False, indices=None, return_predecessors=True)
    print(dist_matrix)
    print(pred)

def _cost(dist_matrix: np.ndarray, start_idx, end_idx):
    return dist_matrix[start_idx, end_idx]

def _path(pred: np.ndarray, start_idx: int, end_idx: int):
    p = []
    cur = end_idx
    while cur != start_idx:
        p.insert(0, cur)
        cur = pred[start_idx, cur]
    p.insert(0, start_idx) 
    return p


class PathMatrix:
    def __init__(self, am: AdjMatrix) -> None:
        dist_matrix, pred = dijkstra(csgraph=am, directed=False, indices=None, return_predecessors=True)
        self.dm = dist_matrix
        self.pred = pred
    
    def cost(self, start_idx, end_idx):
        return _cost(self.dm, start_idx, end_idx)
    
    def path(self, start_idx: int, end_idx: int):
        return _path(self.pred, start_idx, end_idx)
    
    def path_cost(self, path: List[int]):
        if len(path) in [0, 1]:
            return 0
        cost = 0
        for i in range(1, len(path)):
            cost = cost + self.cost(path[i-1], path[i])
        return cost


def test_pathmatrix(am):
    pm = PathMatrix(am)
    print(pm.cost(0, 7))
    print(pm.path(0, 7))
    print(pm.cost(3, 6))
    print(pm.path(3, 6))
    print(pm.path(6, 3))
