import os
import sys
import heapq
import importlib
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation, PillowWriter
from core import parse_problem, heuristic
from search import METHODS

# animate.py replays a search one step at a time and saves it as a GIF.
# Each tracer below follows the same rules as the real algorithm (same
# tie-breaking, same node counting) but also records every step. At the end
# the replay is checked against the real search() result.

RAD = 0.1  # how much two-way edges curve apart

# each step: (node, path, frontier_nodes, nodes_created, round, note)

def trace_bfs(nodes, graph, origin, destinations):
    steps, created = [], 1
    queue = deque([(origin, [origin])])
    while queue:
        node, path = queue.popleft()
        steps.append((node, path, [n for n, _ in queue], created, 0, ""))
        if node in destinations:
            return steps, node, created, path
        for nb, cost in sorted(graph.get(node, [])):
            if nb in path:
                continue
            created += 1
            queue.append((nb, path + [nb]))
    return steps, None, created, []

def trace_dfs(nodes, graph, origin, destinations):
    steps, created = [], 1
    stack = [(origin, [origin])]
    while stack:
        node, path = stack.pop()
        steps.append((node, path, [n for n, _ in stack], created, 0, ""))
        if node in destinations:
            return steps, node, created, path
        for nb, cost in sorted(graph.get(node, []), reverse=True):  # smaller node is popped first
            if nb in path:
                continue
            created += 1
            stack.append((nb, path + [nb]))
    return steps, None, created, []

def trace_gbfs(nodes, graph, origin, destinations):
    steps, created, counter = [], 1, 0
    frontier = [(heuristic(origin, destinations, nodes), origin, counter, [origin])]
    while frontier:
        h, node, _, path = heapq.heappop(frontier)
        steps.append((node, path, [e[1] for e in frontier], created, 0, "h = %.2f" % h))
        if node in destinations:
            return steps, node, created, path
        for nb, cost in graph.get(node, []):
            if nb in path:
                continue
            counter += 1
            created += 1
            heapq.heappush(frontier, (heuristic(nb, destinations, nodes), nb, counter, path + [nb]))
    return steps, None, created, []

def trace_astar(nodes, graph, origin, destinations):
    steps, created, order = [], 1, 0
    frontier = [(heuristic(origin, destinations, nodes), origin, order, 0, [origin])]
    while frontier:
        f, node, _, g, path = heapq.heappop(frontier)
        steps.append((node, path, [e[1] for e in frontier], created, 0,
                      "g = %g, f = %.2f" % (g, f)))
        if node in destinations:
            return steps, node, created, path
        for nb, cost in graph.get(node, []):
            if nb in path:
                continue
            order += 1
            created += 1
            new_g = g + cost
            new_f = new_g + heuristic(nb, destinations, nodes)
            heapq.heappush(frontier, (new_f, nb, order, new_g, path + [nb]))
    return steps, None, created, []

def trace_ucs(nodes, graph, origin, destinations):
    steps, created, order = [], 1, 0
    frontier = [(0, origin, order, [origin])]
    while frontier:
        g, node, _, path = heapq.heappop(frontier)
        steps.append((node, path, [e[1] for e in frontier], created, 0, "g = %g" % g))
        if node in destinations:
            return steps, node, created, path
        for nb, cost in graph.get(node, []):
            if nb in path:
                continue
            order += 1
            created += 1
            heapq.heappush(frontier, (g + cost, nb, order, path + [nb]))
    return steps, None, created, []

def trace_idastar(nodes, graph, origin, destinations):
    steps = []
    state = {"created": 1}
    EPS = 1e-9

    def visit(node, path, g, limit, rnd):
        steps.append((node, path, [], state["created"], rnd, "limit = %.2f, g = %g" % (limit, g)))
        if node in destinations:
            return (node, path), limit
        smallest = float("inf")
        for nb, cost in graph.get(node, []):
            if nb in path:
                continue
            state["created"] += 1
            new_g = g + cost
            new_f = new_g + heuristic(nb, destinations, nodes)
            if new_f > limit + EPS:
                smallest = min(smallest, new_f)
                continue
            found, over = visit(nb, path + [nb], new_g, limit, rnd)
            if found is not None:
                return found, limit
            smallest = min(smallest, over)
        return None, smallest

    limit = heuristic(origin, destinations, nodes)
    rnd = 0
    while True:
        found, nxt = visit(origin, [origin], 0, limit, rnd)
        if found is not None:
            return steps, found[0], state["created"], found[1]
        if nxt == float("inf"):
            return steps, None, state["created"], []
        limit = nxt
        rnd += 1

TRACERS = {"BFS": trace_bfs, "DFS": trace_dfs, "GBFS": trace_gbfs,
           "AS": trace_astar, "CUS1": trace_ucs, "CUS2": trace_idastar}

def draw_frame(ax, nodes, graph, origin, destinations, title,
               expanded, frontier, current, path, final):
    ax.clear()
    for a in graph:
        for b, cost in graph[a]:
            x1, y1 = nodes[a]
            x2, y2 = nodes[b]
            on_path = (a in path and b in path
                       and path.index(b) == path.index(a) + 1)
            colour = "gray"
            width = 1
            if on_path:
                colour = "red" if final else "royalblue"
                width = 3
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="->", color=colour, lw=width,
                                        shrinkA=14, shrinkB=14,
                                        connectionstyle="arc3,rad=" + str(RAD)))
            lx = (x1 + x2) / 2 + 0.5 * RAD * (y2 - y1)
            ly = (y1 + y2) / 2 - 0.5 * RAD * (x2 - x1)
            ax.text(lx, ly, str(cost), fontsize=8, ha="center", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))

    for n, (x, y) in nodes.items():
        if final and n in path:
            fill = "lightpink"
        elif n == current:
            fill = "tomato"
        elif n in frontier:
            fill = "gold"
        elif n in expanded:
            fill = "lightskyblue"
        else:
            fill = "lightgray"
        edge, lw = "black", 1
        if n == origin:
            edge, lw = "green", 3
        elif n in destinations:
            edge, lw = "orange", 3
        ax.scatter(x, y, s=700, c=fill, edgecolors=edge, linewidths=lw, zorder=3)
        ax.text(x, y, str(n), ha="center", va="center", fontweight="bold", zorder=4)

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10)
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)

    # colour key, placed under the graph
    def dot(fill, edge="black", lw=1):
        return Line2D([0], [0], marker="o", linestyle="", markersize=11,
                      markerfacecolor=fill, markeredgecolor=edge, markeredgewidth=lw)
    key = [
        (dot("white", "green", 3), "Origin"),
        (dot("white", "orange", 3), "Destination"),
        (dot("tomato"), "Expanding now"),
        (dot("gold"), "Frontier (waiting)"),
        (dot("lightskyblue"), "Already expanded"),
        (dot("lightgray"), "Not reached yet"),
        (Line2D([0], [0], color="royalblue", lw=3), "Path to current node"),
        (Line2D([0], [0], color="red", lw=3), "Final path"),
    ]
    ax.legend([k[0] for k in key], [k[1] for k in key], loc="upper center",
              bbox_to_anchor=(0.5, -0.06), ncol=4, fontsize=8, frameon=False,
              columnspacing=1.2, handletextpad=0.4)

def main():
    if len(sys.argv) not in (3, 4):
        print("Usage: python animate.py <filename> <method> [noshow]")
        return
    filename = sys.argv[1]
    method = sys.argv[2].upper()
    show = not (len(sys.argv) == 4 and sys.argv[3].lower() == "noshow")   # window pops up unless "noshow"
    if method not in TRACERS:
        print("Animation is not available for: " + method)
        return

    nodes, graph, origin, destinations = parse_problem(filename)
    steps, goal, created, path = TRACERS[method](nodes, graph, origin, destinations)

    # check the replay against the real algorithm
    real = importlib.import_module(METHODS[method]).search(nodes, graph, origin, destinations)
    if real != (goal, created, path):
        print("WARNING: the replay does not match the real %s result: %s vs %s"
              % (method, (goal, created, path), real))

    # frames: one per step, then a few holding the final path
    frames = list(range(len(steps))) + [len(steps) - 1] * 4
    fig, ax = plt.subplots(figsize=(7, 7.6))
    fig.subplots_adjust(top=0.9, bottom=0.2)   # room for the colour key
    expanded_by_frame = []
    seen, last_round = set(), 0
    for node, p, fr, c, rnd, note in steps:
        if rnd != last_round:       # IDA*: a new round starts from scratch
            seen, last_round = set(), rnd
        seen = seen | {node}
        expanded_by_frame.append(seen)

    def update(i):
        idx = frames[i]
        final = i >= len(steps)
        node, p, fr, c, rnd, note = steps[idx]
        if final:
            title = "%s: goal %s, nodes created %d\npath: %s" % (
                method, goal, created, " ".join(map(str, path)))
        else:
            title = "%s  step %d/%d  expanding node %d  (nodes created %d)" % (
                method, idx + 1, len(steps), node, c)
            if note:
                title += "\n" + note
            if method == "CUS2":
                title += "  [round %d]" % (rnd + 1)
        draw_frame(ax, nodes, graph, origin, destinations, title,
                   expanded_by_frame[idx], set(fr), node,
                   path if final else p, final)

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000, repeat=False)

    os.makedirs("visPhoto", exist_ok=True)
    test_name = os.path.splitext(os.path.basename(filename))[0]
    out = os.path.join("visPhoto", test_name + "_" + method + ".gif")
    anim.save(out, writer=PillowWriter(fps=1))
    print("Saved " + out)
    if show:
        # the saved run used up the first animation, so make a fresh one to play in the window
        window = FuncAnimation(fig, update, frames=len(frames), interval=1000, repeat=False)
        plt.show()

main()
