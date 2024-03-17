from typing import List, Tuple

AdjMatrix = List[List[float]]
Edge = Tuple[int, int]
VertexPath = List[int]


def adjmatrix_to_edges(am: AdjMatrix)-> List[Edge]:
    num_vertices = len(am[0])
    edges = []
    for v1 in range(num_vertices):
        for v2 in range(v1, num_vertices):
            cost = am[v1][v2]
            if cost > 0:
                edges.append((v1, v2))
    return edges

def vertex_id_to_letter(id: int) -> chr:
    return chr(65+id)

def pretty_edge_list(edges: List[Edge]):
    p_edges = []
    for elem in edges:
        p_edges.append(
            (vertex_id_to_letter(elem[0]), vertex_id_to_letter(elem[1]))
        )
    return p_edges

def pretty_vertices(vertices: List[int]):
    return [vertex_id_to_letter(v) for v in vertices]

def contiguous_vertexpath_to_edges(path: VertexPath) -> List[Edge]:
    edges = []
    for i in range(len(path)-1):
        edges.append((path[i], path[i+1]))
    return edges
