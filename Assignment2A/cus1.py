import heapq

from core import parse_problem

def ucs(nodes, graph, origin, destinations):
    created = 1 
    order = 0 
    frontier = [(0, origin, order, [origin])]

    while frontier:
        cost, node, _, path = heapq.heappop(frontier)

        if node in destinations:
            return node, created, path

        for neighbour, edge_cost in graph.get(node, []):
            if neighbour in path: 
                continue
            order += 1
            created += 1
            heapq.heappush(frontier, (cost + edge_cost, neighbour, order, path + [neighbour]))

    return None, created, []

def search(nodes, graph, origin, destinations):
    return ucs(nodes, graph, origin, destinations)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python cus1.py <filename>")
    else:
        n, g, o, d = parse_problem(sys.argv[1])
        goal, created, path = search(n, g, o, d)
        print(goal, created)
        print(" ".join(str(x) for x in path))
