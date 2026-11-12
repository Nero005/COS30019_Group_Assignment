import sys
import importlib
from core import parse_problem

# method name on the command line -> the file that holds that algorithm
METHODS = {
    "DFS": "dfs",
    "BFS": "bfs",
    "GBFS": "gbfs",
    "AS": "astar",
    "CUS1": "cus1",
    "CUS2": "cus2",
}

def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        print("Methods: " + ", ".join(METHODS))
        return

    filename = sys.argv[1]
    method = sys.argv[2].upper()

    if method not in METHODS:
        print("Unknown method: " + method)
        print("Methods: " + ", ".join(METHODS))
        return

    try:
        nodes, graph, origin, destinations = parse_problem(filename)
    except FileNotFoundError:
        print("File not found: " + filename)
        return

    # load only the file for the chosen method
    algorithm = importlib.import_module(METHODS[method])
    goal, nodes_created, path = algorithm.search(nodes, graph, origin, destinations)

    # line 1: filename method
    print(filename + " " + method)
    if goal is None:
        print("No goal is reachable; " + str(nodes_created))
    else:
        # line 2: goal number_of_nodes
        print(str(goal) + " " + str(nodes_created))
        # line 3: the path
        print(" ".join(str(n) for n in path))

if __name__ == "__main__":
    main()