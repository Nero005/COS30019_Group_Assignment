import heapq
import sys

from core import heuristic, parse_problem

def gbfs(nodes, graph, origin, destinations):
    counter = 0          # insertion order, for chronological tie-breaking
    created = 1          # the root counts as a created node
    frontier = [(heuristic(origin, destinations, nodes), origin, counter, [origin])]

    while frontier:
        h, node, _, path = heapq.heappop(frontier)

        if node in destinations:
            return node, created, path

        # neighbours are already sorted by node number (done in core.py)
        for nxt, _cost in graph.get(node, []):
            if nxt in path:          # skip loops along this branch
                continue
            counter += 1
            created += 1
            heapq.heappush(
                frontier,
                (heuristic(nxt, destinations, nodes), nxt, counter, path + [nxt]),
            )
    return None, created, []

def search(nodes, graph, origin, destinations):
    return gbfs(nodes, graph, origin, destinations)

def main():
    if len(sys.argv) != 3:
        print("Usage: python gbfs.py <filename> <method>")
        return
    filename, method = sys.argv[1], sys.argv[2].upper()

    nodes, graph, origin, destinations = parse_problem(filename)

    if method != "GBFS":
        print(f"Method {method} not implemented in this file.")
        return

    result = gbfs(nodes, graph, origin, destinations)
    print(f"{filename} {method}")
    if result[0] is None:
        print(f"No goal is reachable; {result[1]}")
    else:
        goal, created, path = result
        print(f"{goal} {created}")
        print(" ".join(map(str, path)))

if __name__ == "__main__":
    main()