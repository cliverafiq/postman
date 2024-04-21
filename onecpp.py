import networkx as nx
from adjacency import AdjMatrix, adjmatrix_to_edges, pretty_edge_list
import itertools

def adj_mat_to_netx_graph(am: AdjMatrix):
    edges = adjmatrix_to_edges(am)
    g = nx.Graph()
    for edge in edges:
        g.add_edge(*edge, weight=am[edge[0]][edge[1]])
    return g


def get_shortest_paths_distances(graph, pairs, edge_weight_name):
    """Compute shortest distance between each pair of nodes in a graph.  Return a dictionary keyed on node pairs (tuples)."""
    distances = {}
    for pair in pairs:
        distances[pair] = nx.dijkstra_path_length(graph, pair[0], pair[1], weight=edge_weight_name)
    return distances

def eulerize(G):
    if G.order() == 0:
        raise nx.NetworkXPointlessConcept("Cannot Eulerize null graph")
    if not nx.is_connected(G):
        raise nx.NetworkXError("G is not connected")
    odd_degree_nodes = [n for n, d in G.degree() if d % 2 == 1]
    G = nx.MultiGraph(G)
    if len(odd_degree_nodes) == 0:
        return G

    # get all shortest paths between vertices of odd degree
    odd_deg_pairs_paths = [
        (m, {n: nx.shortest_path(G, source=m, target=n, weight="weight")})
        for m, n in itertools.combinations(odd_degree_nodes, 2)
    ]

    # use the number of vertices in a graph + 1 as an upper bound on
    # the maximum length of a path in G
    upper_bound_on_max_path_length = len(G) + 1

    # use "len(G) + 1 - len(P)",
    # where P is a shortest path between vertices n and m,
    # as edge-weights in a new graph
    # store the paths in the graph for easy indexing later
    Gp = nx.Graph()
    for n, Ps in odd_deg_pairs_paths:
        for m, P in Ps.items():
            if n != m:
                Gp.add_edge(
                    m, n, weight=upper_bound_on_max_path_length - len(P), path=P
                )

    # find the minimum weight matching of edges in the weighted graph
    best_matching = nx.Graph(list(nx.max_weight_matching(Gp)))

    # duplicate each edge along each path in the set of paths in Gp
    for m, n in best_matching.edges():
        path = Gp[m][n]["path"]
        G.add_edges_from(nx.utils.pairwise(path))
    return G


def chinese_postman(graph):
    odd_degree_nodes = [node for node, degree in graph.degree() if degree % 2 != 0]

    if not odd_degree_nodes:
        # If all nodes have even degree, return the sum of all edge weights
        return sum(graph[u][v]['weight'] for u, v in graph.edges())

    mg = eulerize(graph)
    eulerian_circuit = list(nx.eulerian_circuit(mg))
    print(pretty_edge_list(eulerian_circuit))
    # Calculate the total weight of the Eulerian circuit
    total_weight = sum(mg[u][v][0]['weight'] for u, v in eulerian_circuit)

    return total_weight 
