from samples import sample1
from genetic import random_solution, postmen_edges_list_to_genome, genome_to_postmen_edge_lists
from adjacency import adjmatrix_to_edges, pretty_edge_list

def test_random_solution():
    edges = adjmatrix_to_edges(sample1)
    print(pretty_edge_list(edges))
    print(edges)
    print(random_solution(edges, 1 ))
    print(random_solution(edges, 2))
    print(random_solution(edges, 3))
    print(random_solution(edges, 4))

def test_genome_conversions():

    g1 = ["P", 1, 2, "P", 4, 5, 6, "P", 7, 8, 9, 10, 11, "P", 12, 13, 14]
    el1 = genome_to_postmen_edge_lists(g1)
    print(el1)
    print(postmen_edges_list_to_genome(el1))

    g2 = [12, 13, 14, "P", 1, 2, "P", 4, 5, 6, "P", 7, 8, 9, 10, 11, "P"]
    el2 = genome_to_postmen_edge_lists(g2)
    print(el2)
    print(postmen_edges_list_to_genome(el2))

    g3 = [12, 13, 14, "P", "P", 4, 5, 6, "P", 7, 8, 9, 10, 11, "P"]
    el3 = genome_to_postmen_edge_lists(g3)
    print(el3)
    print(postmen_edges_list_to_genome(el3))