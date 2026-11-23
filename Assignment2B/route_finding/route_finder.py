"""
route_finder.py - Task 4: integrate Part A (search) with Part B (traffic prediction)

Given an origin and a destination SCATS site, a date and time, a prediction model and one of the
six Part A search methods, it returns up to five routes with their estimated travel times.

How it works
  1. travel_time.py (Task 3) predicts the traffic at every site with the chosen model
     (LSTM, GRU or XGBoost regression) and turns it into a travel time for every road.
     The nodes are the 40 SCATS sites; the cost of each edge is its travel time in seconds.
  2. The chosen Part A method (DFS, BFS, GBFS, AS, CUS1, CUS2), unchanged, searches that graph.
     Its route is always route 1.
  3. More routes (up to max_routes) come from Yen's k-shortest-paths method: it takes each
     route found so far, blocks one of its roads at a time, and asks the same search method
     for a detour. The cheapest new routes are kept. No route visits a site twice.

Heuristic (GBFS, AS, CUS2)
  Part A's heuristic is the straight-line distance between node coordinates. Here the site
  coordinates are converted to "seconds of driving at 60 km/h" before the search, so the same
  Part A heuristic becomes: straight-line km / 60 km/h. No road is faster than 60 km/h and every
  road also adds a 30 s intersection delay, so this never overestimates the real travel time
  (A* stays optimal). The 0.99 factor keeps it safely below the real straight-line distance.

Run (from the Assignment2B folder):
    python route_finding/route_finder.py --origin 2000 --destination 3002 --algorithm AS ^
                                          --model xgboost --time "2006-10-30 08:00"
    (any option left out comes from config.json)

Use in the GUI:
    from route_finder import RouteFinder
    finder = RouteFinder()
    result = finder.find_routes("2000", "3002", "2006-10-30 08:00", model="xgboost", algorithm="AS")
"""
import argparse
import importlib
import json
import math
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))        # the Assignment2B folder
PART_A = os.path.join(os.path.dirname(ROOT), "Assignment2A")
sys.path.insert(0, os.path.join(ROOT, "travel_time_estimation"))
sys.path.append(PART_A)                  # appended, so Part B modules are found first

from travel_time import MODELS, TravelTimeEstimator                      # noqa: E402

CONFIG_FILE = os.path.join(ROOT, "config.json")
ALGORITHMS = {"DFS": "dfs", "BFS": "bfs", "GBFS": "gbfs", "AS": "astar", "CUS1": "cus1", "CUS2": "cus2"}
ALGORITHM_NAMES = {"DFS": "Depth-First Search", "BFS": "Breadth-First Search",
                   "GBFS": "Greedy Best-First Search", "AS": "A*",
                   "CUS1": "Uniform Cost Search (CUS1)", "CUS2": "IDA* (CUS2)"}
SECONDS_PER_KM = 3600 / 60.0             # driving time for 1 km at the 60 km/h speed limit
HEURISTIC_SAFETY = 0.99
SLOT = pd.Timedelta(minutes=15)


def load_config(path=CONFIG_FILE):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def seconds_coordinates(sites):
    """Site coordinates in 'seconds at 60 km/h' (flat projection, fine for a 10 km area),
    so math.dist between two sites = straight-line driving time at the speed limit."""
    lat0 = math.radians(sites["lat"].mean())
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * math.cos(lat0)
    scale = SECONDS_PER_KM * HEURISTIC_SAFETY
    return {s: ((row["long"] - sites["long"].mean()) * km_per_deg_lon * scale,
                (row["lat"] - sites["lat"].mean()) * km_per_deg_lat * scale)
            for s, row in sites.iterrows()}


def path_cost(graph, path):
    costs = {}
    for a, b in zip(path, path[1:]):
        costs.setdefault(a, dict(graph[a]))
        if b not in costs[a]:
            return None                  # not a valid path in this graph
    return sum(costs[a][b] for a, b in zip(path, path[1:]))


class RouteFinder:
    """Keeps one TravelTimeEstimator per model, so each model is loaded only once."""

    def __init__(self, config=None):
        self.config = config or load_config()
        self._estimators = {}

    def estimator(self, model):
        if model not in MODELS:
            raise ValueError(f"Unknown model '{model}'. Choose from: {', '.join(MODELS)}")
        if model not in self._estimators:
            self._estimators[model] = TravelTimeEstimator(model=model,
                                                          flow_at=self.config.get("flow_at", "destination"))
        return self._estimators[model]

    def sites(self):
        """{site: description} for drop-down lists, e.g. '2000': 'WARRIGAL_RD / TOORAK_RD'."""
        est = self.estimator(self.config.get("default_model", "xgboost"))
        return {s: site_label(locs) for s, locs in est.sites["locations"].items()}

    def time_range(self, model=None):
        """Earliest and latest departure times the models can predict (they need 3 hours of history)."""
        est = self.estimator(model or self.config.get("default_model", "xgboost"))
        index = est.predictor.flows.index
        lookback = getattr(est.predictor, "lookback", 12)
        return index[lookback], index[-1]

    # ------------------------------------------------------------ main entry point
    def find_routes(self, origin, destination, when, model=None, algorithm=None, max_routes=None):
        model = (model or self.config["default_model"]).lower()
        algorithm = (algorithm or self.config["default_algorithm"]).upper()
        max_routes = int(max_routes or self.config.get("max_routes", 5))
        origin, destination = str(origin).zfill(4), str(destination).zfill(4)

        est = self.estimator(model)
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {', '.join(ALGORITHMS)}")
        for name, site in (("Origin", origin), ("Destination", destination)):
            if site not in est.sites.index:
                raise ValueError(f"{name} {site} is not one of the 40 SCATS sites with traffic data.")
        when = self._check_time(when, model)

        # Part B: predicted travel time on every road at this time
        nodes, graph = est.build_graph(str(when))
        search_nodes = seconds_coordinates(est.sites)
        legs = est.edge_times(str(when)).set_index(["from", "to"])
        roads = {}
        for _, e in est.edges.iterrows():
            roads[(e["from_site"], e["to_site"])] = roads[(e["to_site"], e["from_site"])] = e["road"]

        # Part A: the chosen search method finds route 1
        module = importlib.import_module(ALGORITHMS[algorithm])
        t0 = time.perf_counter()
        goal, nodes_created, first = module.search(search_nodes, graph, origin, [destination])
        search_seconds = time.perf_counter() - t0

        paths = []
        if goal is not None:
            paths = [first]
            if max_routes > 1:
                paths += yen_more_routes(module, search_nodes, graph, first, destination, max_routes - 1)

        routes = [self._describe(p, graph, legs, roads, est) for p in paths]
        fastest = min(range(len(routes)), key=lambda i: routes[i]["travel_time_s"]) if routes else None
        return {"origin": origin, "destination": destination, "time": str(when),
                "model": model, "algorithm": algorithm, "algorithm_name": ALGORITHM_NAMES[algorithm],
                "nodes_created": nodes_created, "search_ms": round(search_seconds * 1000, 2),
                "routes": routes, "fastest_index": fastest}

    # ------------------------------------------------------------ helpers
    def _check_time(self, when, model):
        try:
            when = pd.Timestamp(when)
        except (ValueError, TypeError):
            raise ValueError(f"'{when}' is not a valid date and time (use YYYY-MM-DD HH:MM).")
        when = when.floor("15min")                      # the data is in 15-minute slots
        first, last = self.time_range(model)
        if not first <= when <= last:
            raise ValueError(f"Time must be between {first:%Y-%m-%d %H:%M} and {last:%Y-%m-%d %H:%M} "
                             f"(the models need the 3 hours of data before the departure time).")
        return when

    @staticmethod
    def _describe(path, graph, legs, roads, est):
        rows = []
        for a, b in zip(path, path[1:]):
            leg = legs.loc[(a, b)]
            rows.append({"from": a, "to": b, "road": roads.get((a, b), ""),
                         "distance_km": float(leg["distance_km"]), "speed_kmh": float(leg["speed_kmh"]),
                         "travel_time_s": float(leg["travel_time_s"])})
        total = sum(r["travel_time_s"] for r in rows)
        return {"path": path, "travel_time_s": round(total, 1), "travel_time_min": round(total / 60, 1),
                "distance_km": round(sum(r["distance_km"] for r in rows), 3),
                "intersections": max(len(path) - 1, 0), "legs": rows}


def yen_more_routes(module, nodes, graph, first_path, destination, extra):
    """Yen's k-shortest-paths method, using the chosen Part A search for every detour.
    Returns up to `extra` more loop-free routes, cheapest first."""
    found = [first_path]
    candidates = []                                        # (cost, path)
    for _ in range(extra):
        previous = found[-1]
        for i in range(len(previous) - 1):
            spur, root = previous[i], previous[:i + 1]
            # block the next road of every found route that shares this root,
            # and the sites already in the root (so the detour cannot loop back)
            blocked_edges = {(p[i], p[i + 1]) for p in found if len(p) > i + 1 and p[:i + 1] == root}
            blocked_nodes = set(root[:-1])
            trimmed = {a: [(b, c) for b, c in nbrs if (a, b) not in blocked_edges and b not in blocked_nodes]
                       for a, nbrs in graph.items() if a not in blocked_nodes}
            goal, _, spur_path = module.search(nodes, trimmed, spur, [destination])
            if goal is None:
                continue
            candidate = root[:-1] + spur_path
            if candidate in found or any(candidate == p for _, p in candidates):
                continue
            candidates.append((path_cost(graph, candidate), candidate))
        if not candidates:
            break
        candidates.sort(key=lambda c: (c[0], c[1]))
        found.append(candidates.pop(0)[1])
    return found[1:]


def site_label(locations):
    """'WARRIGAL_RD N of TOORAK_RD; ...' -> 'WARRIGAL_RD / TOORAK_RD' (the main roads at the site)."""
    roads = []
    for loc in locations.split(";"):
        parts = loc.strip().split()
        for i, word in enumerate(parts):
            if word.lower() == "of":
                for name in (" ".join(parts[:i - 1]), " ".join(parts[i + 1:])):
                    if name and name not in roads:
                        roads.append(name)
                break
    return " / ".join(roads[:2])


def print_result(result, sites):
    print(f"{result['algorithm_name']} | model {result['model']} | {result['time']}")
    print(f"From {result['origin']} ({sites[result['origin']]}) to "
          f"{result['destination']} ({sites[result['destination']]})")
    if not result["routes"]:
        print("No route found.")
        return
    print(f"Nodes created by {result['algorithm']}: {result['nodes_created']} "
          f"({result['search_ms']} ms)\n")
    for i, r in enumerate(result["routes"]):
        tags = []
        if i == 0:
            tags.append(f"found by {result['algorithm']}")
        if i == result["fastest_index"]:
            tags.append("fastest")
        print(f"Route {i + 1}: {r['travel_time_min']} min, {r['distance_km']:.2f} km "
              f"({', '.join(tags) if tags else 'alternative'})")
        print("   " + " -> ".join(r["path"]))
    best = result["routes"][result["fastest_index"]]
    print(f"\nFastest route, road by road:")
    for leg in best["legs"]:
        print(f"   {leg['from']} -> {leg['to']}  {leg['road']:<16} {leg['distance_km']:.2f} km  "
              f"{leg['speed_kmh']:.1f} km/h  {leg['travel_time_s']:.0f} s")


def main():
    config = load_config()
    p = argparse.ArgumentParser(description="Find up to 5 routes between two SCATS sites")
    p.add_argument("--origin", default=config["default_origin"])
    p.add_argument("--destination", default=config["default_destination"])
    p.add_argument("--time", default=config["default_datetime"], help="e.g. 2006-10-30 08:00")
    p.add_argument("--model", default=config["default_model"], choices=list(MODELS))
    p.add_argument("--algorithm", default=config["default_algorithm"], type=str.upper,
                   choices=list(ALGORITHMS))
    p.add_argument("--routes", type=int, default=config.get("max_routes", 5))
    args = p.parse_args()

    finder = RouteFinder(config)
    try:
        result = finder.find_routes(args.origin, args.destination, args.time,
                                    model=args.model, algorithm=args.algorithm, max_routes=args.routes)
    except ValueError as error:
        print("Error:", error)
        sys.exit(1)
    print_result(result, finder.sites())


if __name__ == "__main__":
    main()
