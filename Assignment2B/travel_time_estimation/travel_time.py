"""
travel_time.py - travel time estimation for each edge (COS30019 Assignment 2B, Task 3)

For a chosen date and time, a trained model (LSTM, GRU or XGBoost regression) predicts the
traffic flow at every SCATS site, and each edge from build_graph.py gets a travel time:

  1. Predicted flow at a site (cars per 15 min, all approaches added together)
     -> flow per approach per hour = site flow / number of approaches x 4
     The conversion formula only works up to 1500 cars/hour (one approach), so the
     whole-intersection total cannot be used directly.
  2. Flow -> speed, using the formula in "Traffic Flow to Travel Time Conversion v1.0":
         flow = -1.4648375 * speed^2 + 93.75 * speed
     solved for speed on the "under capacity" (green) side of the curve:
         speed = (93.75 + sqrt(93.75^2 - 4 * 1.4648375 * flow)) / (2 * 1.4648375)
     Speed is capped at the 60 km/h speed limit (this happens when flow <= 351),
     and flow above capacity (1500) is treated as capacity (speed 32 km/h).
  3. Edge travel time (seconds) = distance / speed * 3600 + 30 s delay at the intersection.

Which site's flow is used for an edge A -> B is set by FLOW_AT:
    "destination" (default) - flow at B, as in the assignment brief
    "origin"                - flow at A, as in the conversion document
(Ask the tutor which one to use; changing it is one setting.)

Use from the route finder (Task 4):
    from travel_time import TravelTimeEstimator
    est = TravelTimeEstimator(model="xgboost")
    nodes, graph = est.build_graph("2006-10-30 08:00")
    # nodes: {site: (longitude, latitude)}
    # graph: {site: [(neighbour, seconds), ...]}  sorted by neighbour, same style as Part A

Run a quick check (from the Assignment2B folder):
    python travel_time_estimation/travel_time.py --model xgboost --time "2006-10-30 08:00"
It prints every edge and saves a map coloured by speed to results/travel_time/.
"""
import argparse
import math
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Assignment2B folder
sys.path.insert(0, os.path.join(ROOT, "Machine_Learning_Algorithm"))
sys.path.insert(0, os.path.join(ROOT, "data_cleaning_processing"))

# constants from the brief and the conversion document
A = -1.4648375
B = 93.75
SPEED_LIMIT = 60.0          # km/h
CAPACITY_FLOW = 1500.0      # cars/hour at the top of the curve
CAPACITY_SPEED = 32.0       # km/h at capacity
INTERSECTION_DELAY = 30.0   # seconds per controlled intersection
FLOW_AT = "destination"     # "destination" (brief) or "origin" (conversion document)

# where each trained model is saved (see the Machine_Learning_Algorithm scripts)
MODELS = {
    "lstm": ("train_lstm", os.path.join("results", "lstm", "train_lstm.pt")),
    "gru": ("train_gru", os.path.join("results", "gru", "train_gru.pt")),
    "xgboost": ("xgb_regression", os.path.join("results", "xgboost", "xgb_regression.json")),
}


# ---------------------------------------------------------------- formula
def flow_to_speed(flow_per_hour):
    """Speed (km/h) for a flow (cars/hour on one approach), under-capacity side of the curve."""
    flow = min(max(flow_per_hour, 0.0), CAPACITY_FLOW)
    disc = B * B + 4 * A * flow                       # A is negative
    speed = (-B - math.sqrt(max(disc, 0.0))) / (2 * A)
    return max(min(speed, SPEED_LIMIT), CAPACITY_SPEED)


def edge_travel_time(distance_km, flow_per_hour):
    """Seconds to drive one edge, including the 30 s intersection delay."""
    speed = flow_to_speed(flow_per_hour)
    return distance_km / speed * 3600 + INTERSECTION_DELAY


# ---------------------------------------------------------------- estimator
def load_predictor(model):
    """Load the trained LSTM, GRU or XGBoost regression model (all share the same interface)."""
    if model not in MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {', '.join(MODELS)}")
    module_name, checkpoint = MODELS[model]
    module = __import__(module_name)
    flows_csv = os.path.join(ROOT, "processed_data", "processed_data.csv")
    return module.TrafficPredictor(os.path.join(ROOT, checkpoint), flows_csv)


class TravelTimeEstimator:
    def __init__(self, model="xgboost", flow_at=FLOW_AT,
                 sites_csv=None, edges_csv=None, predictor=None):
        data_dir = os.path.join(ROOT, "processed_data")
        sites = pd.read_csv(sites_csv or os.path.join(data_dir, "sites.csv"), dtype={"site": str})
        edges = pd.read_csv(edges_csv or os.path.join(data_dir, "edges.csv"),
                            dtype={"from_site": str, "to_site": str})
        sites["site"] = sites["site"].str.zfill(4)
        self.sites = sites.set_index("site")
        # number of approaches (detectors) added together for each site's flow
        if "detectors" in self.sites.columns:
            self.approaches = self.sites["detectors"].astype(int).to_dict()
        else:                                    # older sites.csv: count the location names instead
            self.approaches = {s: len(locs.split(";")) for s, locs in self.sites["locations"].items()}
        self.edges = edges
        self.flow_at = flow_at
        self.model = model
        self.predictor = predictor or load_predictor(model)

    def site_flows(self, when):
        """Predicted flow per approach in cars/hour, for every site, at time `when`."""
        predicted = self.predictor.forecast(when, steps=1)          # {site: [cars/15min]}
        return {s: values[0] / self.approaches[s] * 4 for s, values in predicted.items()}

    def edge_times(self, when):
        """Table of every directed edge with distance, flow, speed and travel time."""
        flows = self.site_flows(when)
        rows = []
        for _, e in self.edges.iterrows():
            for a, b in ((e["from_site"], e["to_site"]), (e["to_site"], e["from_site"])):   # two-way
                flow = flows[b] if self.flow_at == "destination" else flows[a]
                rows.append({"from": a, "to": b, "distance_km": e["distance_km"],
                             "flow_per_hour": round(flow, 1),
                             "speed_kmh": round(flow_to_speed(flow), 2),
                             "travel_time_s": round(edge_travel_time(e["distance_km"], flow), 1)})
        return pd.DataFrame(rows)

    def build_graph(self, when):
        """Nodes and weighted edges in the same style as Part A (core.parse_problem):
        nodes {site: (longitude, latitude)}, graph {site: [(neighbour, seconds), ...]}."""
        times = self.edge_times(when)
        nodes = {s: (row["long"], row["lat"]) for s, row in self.sites.iterrows()}
        graph = {s: [] for s in nodes}
        for _, r in times.iterrows():
            graph[r["from"]].append((r["to"], r["travel_time_s"]))
        for s in graph:
            graph[s].sort()                     # smaller site number first (Part A tie-break)
        return nodes, graph


def save_speed_map(est, table, when, path):
    """Map of the sites with each road coloured by its predicted speed (red = slow, green = 60 km/h).
    A two-way road is drawn once, using the slower of its two directions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        print("matplotlib not installed - skipping the map")
        return None
    sites = est.sites
    slowest = {}
    for _, r in table.iterrows():
        key = tuple(sorted((r["from"], r["to"])))
        slowest[key] = min(slowest.get(key, SPEED_LIMIT), r["speed_kmh"])

    cmap = plt.get_cmap("RdYlGn")
    norm = Normalize(vmin=CAPACITY_SPEED, vmax=SPEED_LIMIT)
    fig, ax = plt.subplots(figsize=(10, 9))
    for (a, b), speed in slowest.items():
        ax.plot([sites.at[a, "long"], sites.at[b, "long"]], [sites.at[a, "lat"], sites.at[b, "lat"]],
                color=cmap(norm(speed)), lw=4, solid_capstyle="round", zorder=1)
    ax.scatter(sites["long"], sites["lat"], s=50, color="white", edgecolors="black", zorder=2)
    for s in sites.index:
        ax.annotate(s, (sites.at[s, "long"], sites.at[s, "lat"]), xytext=(4, 4),
                    textcoords="offset points", fontsize=8)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.7)
    bar.set_label("Predicted speed (km/h)")
    ax.set_title(f"Predicted road speeds - {est.model} - {when}")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_aspect(1 / math.cos(math.radians(sites["lat"].mean())))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def main():
    p = argparse.ArgumentParser(description="Estimate the travel time on every edge")
    p.add_argument("--model", default="xgboost", choices=list(MODELS))
    p.add_argument("--time", default="2006-10-30 08:00", help="date and time, e.g. 2006-10-30 08:00")
    p.add_argument("--flow-at", default=FLOW_AT, choices=["destination", "origin"])
    args = p.parse_args()

    est = TravelTimeEstimator(model=args.model, flow_at=args.flow_at)
    table = est.edge_times(args.time)
    print(f"Model: {args.model} | time: {args.time} | flow taken at the edge's {args.flow_at} site")
    print(table.to_string(index=False))
    print(f"\n{len(table)} directed edges | travel time {table['travel_time_s'].min():.0f}-"
          f"{table['travel_time_s'].max():.0f} s | speed {table['speed_kmh'].min():.1f}-"
          f"{table['speed_kmh'].max():.1f} km/h")

    stamp = pd.Timestamp(args.time).strftime("%Y%m%d_%H%M")
    path = save_speed_map(est, table, args.time,
                          os.path.join(ROOT, "results", "travel_time", f"speed_map_{args.model}_{stamp}.png"))
    if path:
        print("Saved", path)


if __name__ == "__main__":
    main()
