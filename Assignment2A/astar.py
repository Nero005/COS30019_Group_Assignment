import heapq
from core import heuristic, parse_problem

def search(nodes, graph, origin, destinations):
    order = 0             # creation order, used to break ties
    nodes_created = 1     # the origin counts as created
    g = 0
    f = g + heuristic(origin, destinations, nodes)

    # each entry: (f, node, order, g, path)
    frontier = [(f, origin, order, g, [origin])]

    while frontier:
        f, node, _, g, path = heapq.heappop(frontier)

        #goal test when a node is popped, not when it is added
        if node in destinations:
            return node, nodes_created, path

        for neighbour, cost in graph.get(node, []):
            if neighbour in path:    #don't loop back along this path
                continue
            order += 1
            nodes_created += 1
            new_g = g + cost
            new_f = new_g + heuristic(neighbour, destinations, nodes)
            heapq.heappush(frontier,(new_f, neighbour, order, new_g, path + [neighbour]))
    
    return None, nodes_created, [] #no goal reachable

if __name__ == "__main__":
    nodes, graph, origin, destinations = parse_problem("tests/test_01.txt")
    goal, nodes_created, path = search(nodes, graph, origin, destinations)
    print(goal, nodes_created)
    print(" ".join(str(n) for n in path))