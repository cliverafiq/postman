from adjacency import AdjMatrix
from typing import Tuple
import numpy as np


sample1: AdjMatrix = [  
    [ 0, 50, 50, 50,  0,  0,  0,  0],
    [50,  0,  0, 50, 70, 50,  0,  0],
    [50,  0,  0, 70,  0,  0, 70, 120],
    [50, 50, 70,  0,  0, 60,  0,  0],
    [ 0, 70,  0,  0,  0, 70,  0,  0],
    [ 0, 50,  0, 60, 70,  0,  0, 60],
    [ 0,  0, 70,  0,  0,  0,  0, 70],
    [ 0,  0,  0,  0,  0, 60, 70,  0]
]




def gen_lattice(size: Tuple[int, int], length: int, dropout: float):

    NEIGHBORS_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def add_coord(coord1, coord2):
        return coord1[0] + coord2[0], coord1[1]+coord2[1]

    def in_range(coord):
        return coord[0] >= 0 and coord[0]< size[0] and coord[1]>=0 and coord[1]< size[1]

    def neighbors(coord):
        return [add_coord(coord, delta) for delta in NEIGHBORS_DELTA]

    def valid_neighbors(coord):
        return filter(in_range, neighbors(coord))

    def coord_to_mat_idx(dim_len, coord):
        return (dim_len*coord[1], coord[0])

    dim_len = size[0]*size[1]
    mat= np.zeros(shape=(dim_len, dim_len), dtype=int)
    for y in range(size[1]):
        for x in range(size[0]):
            ns = valid_neighbors((x,y))
            for n in ns:
                mat[y*size[0]+x, n[1]*size[0]+n[0]] = length
    print(f'number of edges: {np.count_nonzero(mat)/2}') # divide by 2 as the matrix is symmetric
    return mat
