import sys
from core import parse_problem
from collections import deque


def bfs(nodes, edges, origin, destinations):
    created = 1 # root counts as a created node
    frontier = deque([[origin]]) # each entry is a full path

    while frontier:
        path = frontier.popleft()
        node = path[-1]

        if node in destinations:
            return node, created, path

        # children (neighbours) in ascending order
        for nxt, _cost in sorted(edges.get(node, [])):
            if nxt in path: # avoid cycles along this branch
                continue
            created += 1
            frontier.append(path + [nxt])

    return None, created, []


def search(nodes, graph, origin, destinations):
    return bfs(nodes, graph, origin, destinations)


def main():
    if len(sys.argv) != 2:
        print("Usage: python bfs.py <filename>")
        return
    filename = sys.argv[1]

    nodes, edges, origin, destinations = parse_problem(filename)
    goal, created, path = bfs(nodes, edges, origin, destinations)

    print(f"{filename} BFS")
    if goal is None:
        print(f"No goal is reachable; {created}")
    else:
        print(f"{goal} {created}")
        print(" ".join(map(str, path)))


if __name__ == "__main__":
    main()
