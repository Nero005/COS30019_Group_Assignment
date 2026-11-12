import os
import csv
import glob
import time
import importlib
from core import parse_problem
from search import METHODS

# run_test.py runs every method on every test file in tests/ and prints one
# results table. It also saves results.csv (all data) and results_table.csv
# (one compact row per test) for the report.
# Each table cell is:  goal / nodes created / path cost

TEST_FOLDER = "tests"
ORDER = ["DFS", "BFS", "GBFS", "AS", "CUS1", "CUS2"]

def path_cost(graph, path):
    # add up the edge costs along the path
    total = 0
    for a, b in zip(path, path[1:]):
        total += dict(graph[a])[b]
    return total

def run_one(module_name, nodes, graph, origin, destinations):
    # returns (goal, nodes_created, cost, seconds), or None if the method is not available
    try:
        algorithm = importlib.import_module(module_name)
        start = time.perf_counter()
        goal, created, path = algorithm.search(nodes, graph, origin, destinations)
        seconds = time.perf_counter() - start
    except (ImportError, AttributeError):
        return None            # file is empty or has no search() yet
    cost = path_cost(graph, path) if goal is not None else None
    return goal, created, cost, seconds

def main():
    files = sorted(glob.glob(os.path.join(TEST_FOLDER, "test_*.txt")))
    rows = []        # for results.csv
    table = []       # for printing

    for filename in files:
        name = os.path.splitext(os.path.basename(filename))[0]
        if os.path.getsize(filename) == 0:
            continue           # empty test file, nothing to run
        try:
            nodes, graph, origin, destinations = parse_problem(filename)
        except Exception as error:
            print("Skipped " + name + ": could not read the file (" + str(error) + ")")
            continue

        cells = []
        for method in ORDER:
            result = run_one(METHODS[method], nodes, graph, origin, destinations)
            if result is None:
                cells.append("n/a")
                rows.append([name, method, "n/a", "", "", ""])
                continue
            goal, created, cost, seconds = result
            if goal is None:
                cells.append("none/" + str(created))
            else:
                cells.append(str(goal) + "/" + str(created) + "/" + str(cost))
            rows.append([name, method, goal if goal is not None else "none",
                         created, cost if cost is not None else "", round(seconds * 1000, 3)])
        table.append((name, cells))

    # print the table
    print("Each cell: goal / nodes created / path cost   (none/N = no goal reachable, N nodes created)")
    print()
    print("%-9s" % "test" + "".join("%-12s" % m for m in ORDER))
    print("-" * (9 + 12 * len(ORDER)))
    for name, cells in table:
        print("%-9s" % name + "".join("%-12s" % c for c in cells))

    # compact table for the report: one row per test, one column per method
    with open("results_table.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Test"] + ORDER)
        for name, cells in table:
            # spaces around "/" stop Excel turning a cell like 4/8/10 into a date
            writer.writerow([name] + [c.replace("/", " / ") for c in cells])

    # save the same data for the report
    with open("results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "method", "goal", "nodes_created", "path_cost", "time_ms"])
        writer.writerows(rows)
    print()
    print("Saved results.csv (full data) and results_table.csv (compact table), " + str(len(table)) + " tests")

if __name__ == "__main__":
    main()
