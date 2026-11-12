import math

def heuristic(node, destinations, nodes):
    x1, y1 = nodes[node]
    return min(math.dist((x1, y1), nodes[d]) for d in destinations)

def parse_problem(filename):
    nodes = {}
    graph = {}
    origin = None
    destinations = []
    section = None

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # a heading line switches which section we are reading
            if line == "Nodes:":
                section = "nodes"
            elif line == "Edges:":
                section = "edges"
            elif line == "Origin:":
                section = "origin"
            elif line == "Destinations:":
                section = "destinations"

            # a data line, handled by whichever section we are in
            elif section == "nodes":
                # looks like "1: (4,1)"
                label, coords = line.split(":")
                x, y = coords.strip().strip("()").split(",")
                nodes[int(label)] = (int(x), int(y))

            elif section == "edges":
                # look like "(2,1): 4" (one-way, from 2 to 1, cost 4)
                edge, cost = line.split(":")
                a, b = edge.strip().strip("()").split(",")
                graph.setdefault(int(a), []).append((int(b), int(cost)))

            elif section == "origin":
                origin = int(line)

            elif section == "destinations":
                #looks like "5; 4"
                destinations = [int(d) for d in line.split(";")]

    #sort each node's neighbours by node number (tie-break rule)
    for key in graph:
        graph[key].sort()

    return nodes, graph, origin, destinations