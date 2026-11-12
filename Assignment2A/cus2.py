from core import heuristic, parse_problem

# CUS2: IDA* (Iterative Deepening A*), an informed search.
# Depth-first search with a limit on f = g + h. If no goal is found within the
# limit, the limit is raised to the smallest f that went over it, and the
# search starts again from the origin. Only the current path is stored.

def search(nodes, graph, origin, destinations):
    nodes_created = 1       # the origin counts as created
    EPS = 1e-9              # float tolerance when comparing f with the limit
    
    limit = heuristic(origin, destinations, nodes)
    
    while True:
        # result of one round: goal found, or the next limit to try
        found, next_limit, nodes_created = dfs_limited(
            nodes, graph, origin, destinations, [origin], 0, limit, nodes_created, EPS)
        
        if found is not None:
            goal, path = found
            return goal, nodes_created, path

        if next_limit == float("inf"):    # nothing was cut off, so no goal is reachable
            return None, nodes_created, []

        limit = next_limit

def dfs_limited(nodes, graph, node, destinations, path, g, limit, nodes_created, EPS):
    # goal test when the node is visited
    if node in destinations:
        return (node, path), limit, nodes_created

    smallest_over = float("inf")    # smallest f seen that went over the limit

    # graph[node] is sorted by node number, so smaller numbers are tried first
    for neighbour, cost in graph.get(node, []):
        if neighbour in path:       # don't loop back along this path
            continue
        nodes_created += 1
        new_g = g + cost
        new_f = new_g + heuristic(neighbour, destinations, nodes)

        if new_f > limit + EPS:     # too expensive for this round
            smallest_over = min(smallest_over, new_f)
            continue

        found, over, nodes_created = dfs_limited(
            nodes, graph, neighbour, destinations, path + [neighbour],
            new_g, limit, nodes_created, EPS)
        if found is not None:
            return found, limit, nodes_created
        smallest_over = min(smallest_over, over)

    return None, smallest_over, nodes_created

if __name__ == "__main__":
    nodes, graph, origin, destinations = parse_problem("tests/test_01.txt")
    goal, nodes_created, path = search(nodes, graph, origin, destinations)
    print(goal, nodes_created)
    print(" ".join(str(n) for n in path))
    
    