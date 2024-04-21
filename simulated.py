import random
from typing import List, Any, Callable
from adjacency import Edge, AdjMatrix, adjmatrix_to_edges, pretty_edge_list, pretty_vertices
from path import PathMatrix
import random 
from copy import copy
import math

Path = List[Any]
P_PREFIX = "P"

def postmen_edges_list_to_path(p_edge_lists: List[List[Edge]]):
    path = []
    for p in range(len(p_edge_lists)):
        # genome.append(P_PREFIX + str(p+1))
        path.append(P_PREFIX)
        path = path + p_edge_lists[p]
    return path

def random_solution(edges: List[Edge], k: int) -> Path:
    """
    Given all the edges in a graph, allocates randomly and in equal number to k postmen
    """

    idx_list = list(range(len(edges)))
    edges_per_postman = len(edges) // k
    remainder = len(edges) % k
    p_edge_lists = []

    for p in range(k):
        p_list = []
        for _ in range(edges_per_postman):
            idx = random.randint(0, len(idx_list)-1)
            edge = edges[idx_list[idx]]
            p_list.append(edge)
            del idx_list[idx]
        p_edge_lists.append(p_list)
    # distribute the remainder across as many postmen as possible
    for p in range(remainder):
        idx = random.randint(0, len(idx_list)-1)
        edge = edges[idx_list[idx]]
        p_edge_lists[p].append(edge)
    
    # make path
    return postmen_edges_list_to_path(p_edge_lists)

def path_to_postmen_edge_lists(path: Path) -> List[List[Edge]]:
    """Converts a path to a list of edges list
    The path encodes the edges allocated to each postman
    This function walks the path and reconstitute the edge list for each postman

    :param path: a path
    :type path: Path
    :return: a list of list of edges (one list of edges for each postman)
    :rtype: List[List[Edge]]
    """

    # find the location of the first "P"
    first_p_idx = path.index(P_PREFIX)

    p_edges = []
    edges = []
    start = True
    for i in range(first_p_idx, len(path)+first_p_idx):
        idx = i % len(path)
        if path[idx] == P_PREFIX:
            if not start:
                p_edges.append(edges)
                edges = []
            start = False
        else:
            edges.append(path[idx]) 
    if len(edges) > 0:
        p_edges.append(edges)      
        
    return p_edges


def path_fitness(path: Path, pm: PathMatrix, start_idx: int = 0) -> float:
    """Compute the fitness of a path

    For each postman, examine the edge list:
    - we start from start_idx
    - we look at the first edge and determine which side of the edge is closest (shortest path)
    - we build the path using the start, intermediate points and edge
    - we continue with the next edge

    note: a shorter implementation could be to just compute the cumulative cost and not build the full path

    :param path: a path
    :type path: Path
    :param pm: a path matrix to calculate costs and path using Dijkstra
    :type pm: PathMatrix
    :param start_idx: the depot vertex, defaults to 0
    :type start_idx: int, optional
    :param which: knowing if we should return the max or the total solution
    :return: the fitness value
    :rtype: float
    """
    p_edge_lists = path_to_postmen_edge_lists(path)
    p_costs = []
    vertices_lists = []
    for p_edges in p_edge_lists:
        vertices = [start_idx]
        for edge in p_edges:
            c1 = pm.cost(vertices[-1], edge[0])
            c2 = pm.cost(vertices[-1], edge[1])

            if c1 <= c2:
                path = pm.path(vertices[-1], edge[0])
                vertices = vertices + path[1:]
                vertices.append(edge[1])
            else:
                path = pm.path(vertices[-1], edge[1])
                vertices = vertices + path[1:]
                vertices.append(edge[0])

        #last part - return to base
        if vertices[-1] != start_idx:
            path = pm.path(vertices[-1], start_idx)
            vertices = vertices + path[1:]
        # compute total path cost
        # print(f'edges: {pretty_edge_list(p_edges)} vertices: {pretty_vertices(vertices)} cost: {pm.path_cost(vertices)}')
        p_costs.append(pm.path_cost(vertices))
        vertices_lists.append(vertices)

    return max(p_costs), p_costs, vertices_lists
 
def get_neighbor(p: Path) -> Path:
    """Mutate a path by flipping two random elements
    (note that we operate on a copy, and do not alter the original p)

    :param p: path
    :type p: Path
    :return: the mutated path
    :rtype: Path
    """
    idx1 = random.randint(0, len(p)-1)
    idx2 = random.randint(0, len(p)-1)
    while idx2 == idx1:
        idx2 = random.randint(0, len(p)-1)

    p_c = copy(p)
    val = p_c[idx1]
    p_c[idx1] = p_c[idx2]
    p_c[idx2] = val
    return p_c

def compare_cost(initial_cost: int, neighbor_cost: int, temperature: int, same_solution: int, same_cost_diff: int):
    
    """
    Compare the inital cost to the neighbor cost

    :param initial_cost: the cost for the initial path
    :type initial_cost: int
    :param neighbor_cost: the cost for the neighbors path
    :type neigbhor_cost: int
    :param temperature: the temperature
    :type temperature: int
    :param same_solution: counter of times the same solution was found
    :type same_solution: int
    :param same_cost_diff: counter of times the same cost was found
    :type same_cost_diff: int
    """

    
    cost_diff = neighbor_cost - initial_cost

    if cost_diff < 0:
        return_cost = neighbor_cost
        same_solution = 0 
        same_cost_diff = 0
    
    elif cost_diff == 0:
        return_cost = neighbor_cost
        same_solution = 0 
        same_cost_diff += 1
    
    else:
        p = math.exp(-(cost_diff) / temperature)
        r = random.random()

        if r <= p:
            return_cost = neighbor_cost
            same_solution = 0
            same_cost_diff = 0
        
        else:
            return_cost = initial_cost
            same_solution += 1
            same_cost_diff += 1
    return return_cost, same_solution, same_cost_diff



def sa_loop(am: AdjMatrix, k_postmen: int, temperature: int, alpha: int, start_idx, same_solution_max: int, same_cost_diff_max: int):
    """Main SA loop

    :param am: graph adjacency matrix
    :type am: AdjMatrix
    :param k_postmen: number of postmen
    :type k_postmen: int
    :param temperature: initial temperature
    :type temperature: int
    :param alpha: temperature decrease rate
    :type alpha: int
    :param start_idx: depot vertex id
    :type start_idx: _type_
    :param same_solution_max: maximum amount of same solution attempts
    :type same_solution_max: int
    :param same_cost_diff_max: 
    
    :type same_cost_diff_max: int
    """
    edges = adjmatrix_to_edges(am)
    pm = PathMatrix(am)

    current_path = random_solution(edges, k_postmen)
    
    same_solution = 0
    same_cost_diff = 0


    while (same_solution < same_solution_max and same_cost_diff < same_cost_diff_max and temperature > (1*10**-50)):

        current_minmax_cost, current_costs, current_vertices = path_fitness(current_path, pm, start_idx)
        neighbor_path = get_neighbor(current_path)
        neighbor_minmax_cost, neighbor_costs, neighbor_vertices = path_fitness(neighbor_path, pm, start_idx)

        current_minmax_cost, same_solution, same_cost_diff = compare_cost(current_minmax_cost, neighbor_minmax_cost, temperature, same_solution, same_cost_diff)
        if current_minmax_cost == neighbor_minmax_cost:
            current_vertices = neighbor_vertices
            current_costs = neighbor_costs
            current_path = neighbor_path

        temperature = temperature*alpha

        pretty_paths = [pretty_vertices(p) for p in current_vertices]
        print(f'best solution costs {current_costs} paths: {pretty_paths} ')
        print("temperature: ", temperature)
