from typing import List, Any, Callable
from adjacency import Edge, AdjMatrix, adjmatrix_to_edges, pretty_edge_list, pretty_vertices
from path import PathMatrix
import random 
from copy import copy
import math

Genome = List[Any] # contains both the string "P" and edges (tuple of vertex)

P_PREFIX = "P"

def postmen_edges_list_to_genome(p_edge_lists: List[List[Edge]]):
    genome = []
    for p in range(len(p_edge_lists)):
        # genome.append(P_PREFIX + str(p+1))
        genome.append(P_PREFIX)
        genome = genome + p_edge_lists[p]
    return genome

def random_solution(edges: List[Edge], k: int) -> Genome:
    """
    Given all the edges in a graph, allocates randomly and in equal number to k postmen

    :param edges: all the edges in the graph
    :type edges: List[Edge]
    :param k: number of postmen
    :type k: int
    :return: a random solution as a genome
    :rtype: GraphGenome
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
    
    # make genome
    return postmen_edges_list_to_genome(p_edge_lists)

def initial_population(edges: List[Edge], k: int, pop_size: int) -> List[Genome]:
    """Generates an initial random population

    :param edges: the list of edges in the graph
    :type edges: List[Edge]
    :param k: _description_
    :type k: number of postmen
    :param pop_size: the population size to generate
    :type pop_size: int
    """

    population = []
    for _ in range(pop_size):
        population.append(random_solution(edges, k))
    return population

def genome_to_postmen_edge_lists(genome: Genome) -> List[List[Edge]]:
    """Converts a genome to a list of edges list
    The genome encodes the edges allocated to each postman
    This function walks the genome and reconstitute the edge list for each postman

    :param genome: a genome
    :type genome: Genome
    :return: a list of list of edges (one list of edges for each postman)
    :rtype: List[List[Edge]]
    """

    # find the location of the first "P"
    first_p_idx = genome.index(P_PREFIX)

    p_edges = []
    edges = []
    start = True
    for i in range(first_p_idx, len(genome)+first_p_idx):
        idx = i % len(genome)
        if genome[idx] == P_PREFIX:
            if not start:
                p_edges.append(edges)
                edges = []
            start = False
        else:
            edges.append(genome[idx]) 
    if len(edges) > 0:
        p_edges.append(edges)      
        
    return p_edges

def genome_fitness(genome: Genome, pm: PathMatrix, start_idx: int = 0) -> float:
    """Compute the fitness of a genome

    For each postman, examine the edge list:
    - we start from start_idx
    - we look at the first edge and determine which side of the edge is closest (shortest path)
    - we build the path using the start, intermediate points and edge
    - we continue with the next edge

    note: a shorter implementation could be to just compute the cumulative cost and not build the full path

    :param genome: a genome
    :type genome: Genome
    :param pm: a path matrix to calculate costs and path using Dijkstra
    :type pm: PathMatrix
    :param start_idx: the depot vertex, defaults to 0
    :type start_idx: int, optional
    :return: the fitness value
    :rtype: float
    """
    p_edge_lists = genome_to_postmen_edge_lists(genome)
    p_costs = []
    vertices_lists = []
    for p_edges in p_edge_lists:
        vertices = [start_idx]
        for edge in p_edges:
            c1 = pm.cost(vertices[-1], edge[0])
            c2 = pm.cost(vertices[-1], edge[1])

            if c1 <= c2:
                genome = pm.path(vertices[-1], edge[0])
                vertices = vertices + genome[1:]
                vertices.append(edge[1])
            else:
                genome = pm.path(vertices[-1], edge[1])
                vertices = vertices + genome[1:]
                vertices.append(edge[0])

        #last part - return to base
        if vertices[-1] != start_idx:
            genome = pm.path(vertices[-1], start_idx)
            vertices = vertices + genome[1:]
        # compute total genome cost
        # print(f'edges: {pretty_edge_list(p_edges)} vertices: {pretty_vertices(vertices)} cost: {pm.genome_cost(vertices)}')
        p_costs.append(pm.path_cost(vertices))
        vertices_lists.append(vertices)

    return max(p_costs), p_costs, vertices_lists
    

def fitness_selection(genomes: List[Genome], pm: PathMatrix, fit_size: int, start_idx: int = 0) -> List[Genome]:
    """Select the top fittest genomes

    :param genomes: a list of genomes
    :type genomes: List[Genome]
    :param pm: the path matrix calculator
    :type pm: PathMatrix
    :param fit_size: size of the fittest list
    :type fit_size: int
    :param start_idx: depot vertex, defaults to 0
    :type start_idx: int, optional
    :return: the list of fittest genomes
    :rtype: List[Genome]
    """
    fitnesses = [genome_fitness(genome, pm, start_idx)[0] for genome in genomes]
    print(f'best fitness: {min(fitnesses)} worst: {max(fitnesses)} avg: {sum(fitnesses)/len(fitnesses)}')
    
    sorted_pairs = sorted(zip(fitnesses, genomes), key=lambda x:x[0])
    sorted_genomes = [item for _, item in sorted_pairs]

    s_fit = [genome_fitness(s, pm, start_idx)[0] for s in sorted_genomes[0:fit_size]]
    # print(f'selected best fitness: {min(s_fit)} worst: {max(s_fit)} avg: {sum(s_fit)/len(s_fit)}')
    return sorted_genomes[0:fit_size]

    #inverse_weights = [ 1/ (num + 0.0001) for num in fitnesses]
    #total_weight = sum(inverse_weights)
    #normalized_weights = [w / total_weight for w in inverse_weights]
    #selected = random.choices(genomes, weights=normalized_weights, k=fit_size)
    #s_fit = [genome_fitness(s, pm, start_idx) for s in selected]
    #print(f'selected best fitness: {min(s_fit)} worst: {max(s_fit)} avg: {sum(s_fit)/len(s_fit)}')
    #return selected

def ordered_cross_over(p1: Genome, p2: Genome) -> Genome:
    """Crossover of two parents

    We choose two indices idx1 and idx2 
    We select the substring of parent 1 from idx1 to idx2
    We remove the substring values from a copy of parent 2
    We then add the remaining values of parent 2 in front and at the back of the substring
    We return the result as the child
    
    :param p1: parent 1
    :type p1: Genome
    :param p2: parent 2
    :type p2: Genome
    :return: child genome
    :rtype: Genome
    """
    #randint includes both endpoints
    idx1 = random.randint(0, len(p1)-2) # cannot be the last element of p1
    idx2 = random.randint(idx1+1, len(p1)-1)
    substring = p1[idx1:idx2+1] # right side is not inclusive
    
    p2_c = copy(p2)
    for s in substring:
        p2_c.remove(s)

    child = p2_c[0:idx1] + substring + p2_c[idx1:]
    assert len(child) == len(p2)
    return child

def mutate(p: Genome) -> Genome:
    """Mutate a genome by flipping two random elements
    (note that we operate on a copy, and do not alter the original p)

    :param p: genome
    :type p: Genome
    :return: the mutated genome
    :rtype: Genome
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

def evolve(fit_genomes: List[Genome], pop_size: int, prob_mutation: float) -> List[Genome]:
    """Evolves the fittest genomes into a new population

    :param fit_genomes: the pool of genomes to start from
    :type fit_genomes: List[Genome]
    :param pop_size: the number of children
    :type pop_size: int
    :param prob_mutation: probabilty of mutation
    :type prob_mutation: float
    :return: a population of genomes of size pop_size
    :rtype: List[Genome]
    """
    children = []
    for _ in range(pop_size):
        p1 = random.choice(fit_genomes)
        p2 = random.choice(fit_genomes)
        while p1 == p2:
            p2 = random.choice(fit_genomes)
        child = ordered_cross_over(p1, p2)
        children.append(child)
    
    num_mutations = int(pop_size * prob_mutation)
    for _ in range(num_mutations):
        idx = random.randint(0, pop_size-1)
        child = children[idx]
        mut_child = mutate(child)
        children[idx] = mut_child
    
    return children

def ga_loop(am: AdjMatrix, k_postmen: int, pop_size: int, fit_size: int, start_idx, loops: int, prob_mutation: float):
    """Main GA loop

    :param am: graph adjacency matrix
    :type am: AdjMatrix
    :param k_postmen: number of postmen
    :type k_postmen: int
    :param pop_size: population size
    :type pop_size: int
    :param fit_size: number of selected fit individuals
    :type fit_size: int
    :param start_idx: depot vertex id
    :type start_idx: _type_
    :param loops: number of loops
    :type loops: int
    :param prob_mutation: probability of mutation
    :type prob_mutation: float
    """
    edges = adjmatrix_to_edges(am)
    pm = PathMatrix(am)
    genomes = initial_population(edges, k_postmen, pop_size)

    for _ in range(loops):
        fit_genomes = fitness_selection(genomes, pm, fit_size, start_idx)
        _, costs, vertices = genome_fitness(fit_genomes[0], pm, start_idx)
        pretty_paths = [pretty_vertices(p) for p in vertices]
        print(f'best solution costs {costs} genome {fit_genomes[0]} paths: {pretty_paths} ')
        genomes = evolve(fit_genomes, pop_size, prob_mutation)
    
