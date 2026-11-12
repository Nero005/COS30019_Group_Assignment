import sys
from core import parse_problem

def dfs(nodes, edges, origin, destinations):

    created = 1 
    frontier = [[origin]] 

    while frontier:
        path = frontier.pop()  
        node = path[-1]

        if node in destinations:
            return node, created, path

        for nxt, _cost in reversed(sorted(edges.get(node, []))):
            if nxt in path: 
                continue
            created += 1
            frontier.append(path + [nxt])

    return None, created, []

def search(nodes, graph, origin, destinations):
    return dfs(nodes, graph, origin, destinations)

def main():
    if len(sys.argv) != 2:
        print("Usage: python dfs.py <filename>")
        return
    filename = sys.argv[1]

    nodes, edges, origin, destinations = parse_problem(filename)
    goal, created, path = dfs(nodes, edges, origin, destinations)

    print(f"{filename} DFS")
    if goal is None:
        print(f"No goal is reachable; {created}")
    else:
        print(f"{goal} {created}")
        print(" ".join(map(str, path)))

if __name__ == "__main__":
    main()
