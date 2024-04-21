from typing import List


total_weight = 0
for i in range(len(graph)):
    for j in range(i+1, len(graph[i])):
        total_weight += graph[i][j]


def findoddvertices():
    oddvertices = []
    for x in range(len(graph)):
        zerocount = 0

        for i in range(len(graph[x])):

            if graph[x][i] != 0:
                zerocount = zerocount + 1

        if zerocount % 2 != 0:
            oddvertices.append(x)

    return (oddvertices)


def oddpairings(oddvertices):
    pairs = []

    for x in range(len(oddvertices)):
        for c in range(x + 1, len(oddvertices)):
            pair = [oddvertices[x], oddvertices[c]]
            pairs.append(pair)

    return pairs

    return pairs


class Graphy(WeightedGraph):
    def __init__(self, graph: List[List[float]]):
        super().__init__()
        self.graph = graph

    def cost(self, from_node: Location, to_node: Location) -> float:
        return self.graph[from_node][to_node]

    def neighbors(self, id: Location) -> list[Location]:
        lst = []
        for x in range(len(graph[id])):
            if graph[id][x] != 0:
                lst.append(x)
        return lst


mygraph = Graphy(graph)

odd = findoddvertices()
print("The amount of odd vertices are: ", odd)
pairs = oddpairings(odd)
print("The possible pairings for all odd vertices are: ", pairs)

unique_pairings = find_unique_pairings(odd)
print(f"Total unique pairings: {len(unique_pairings)}")

minimum = 99999999999

for pairing in unique_pairings:
    total = 0
    for i in range(len(pairing)):
        swap = pairing[i][0] <= pairing[i][1]
        start, end = sorted(pairing[i][:2])
        path = []
        ret = dijkstra_search(mygraph, start, end)
        current = end
        while current != start:
            path.append((ret[0][current], current))
            current = ret[0][current]
        path.reverse()
        total = total + ret[1][end]

    pairing_str = ' '.join([str(p) for p in pairing])
    print("The shortest way of joining " + pairing_str + " has a total length of " + str(total))

    if ret[1][end] < minimum:
        shortest = pairing
        shortest_total = total

print(str(shortest) +" Is the shortest pairing with the total length of "+ str(shortest_total))
print("So the shortest possible way to solve this graph is "+ str(shortest_total + total_weight))
