"""
build_graph.py - build the road network between the 40 SCATS sites (COS30019 Assignment 2B, Task 3)

The dataset has no list of which sites are connected, so the edges are built from the
location names in processed_data/sites.csv (for example "WARRIGAL_RD N of TOORAK_RD"):

  1. Each site gets the set of roads it sits on (both the approach road and the cross road).
  2. Sites on the same road are sorted along the road, and each site is linked to the next
     one. Links are two-way, because the dataset counts traffic in both directions.
  3. A link longer than MAX_LINK_KM is skipped: it usually means two different roads with
     the same name (e.g. HIGH_ST in Kew and HIGH_ST in Glen Iris).
  4. A road that changes name is joined with ROAD_ALIASES (e.g. Whitehorse Rd becomes
     Maroondah Hwy at Union Rd).
  5. Any site still not connected to the main network is linked to its nearest connected
     site, so every site can be used as an origin or destination. These links are marked
     "nearest" in the output so they can be reported as an assumption.

Distances are straight-line (haversine) distances between site coordinates, in km.
The edge list was checked by eye against the VicRoads Boroondara map (IBM data_Bor.pdf).
Edit edges.csv by hand if you find a wrong or missing link.

Run (from the Assignment2B folder):
    python travel_time_estimation/build_graph.py      # -> processed_data/edges.csv, graph_map.png
"""
import math
import os
import re
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Assignment2B folder
MAX_LINK_KM = 4.0

# roads that continue under another name, or freeway ramps that belong to the same road
ROAD_ALIASES = {
    "MAROONDAH_HWY": "WHITEHORSE_RD",          # Whitehorse Rd becomes Maroondah Hwy near Union Rd
    "BURWOOD_HWY": "TOORAK_RD",                # Toorak Rd becomes Burwood Hwy at Warrigal Rd
    "OFFRAMP_EASTERN_FWY": "EASTERN_FWY",
    "EASTERN_FWY_W_BD_RAMPS": "EASTERN_FWY",
    "HARP_ST": "HARP_RD",
}

# "WARRIGAL_RD N of TOORAK_RD" -> approach road "WARRIGAL_RD", cross road "TOORAK_RD"
LOCATION_PATTERN = re.compile(r"^(.*?)\s+(?:N|S|E|W|NE|NW|SE|SW)\s+of\s+(.*)$", re.IGNORECASE)


def haversine_km(lat1, lon1, lat2, lon2):
    """Straight-line distance on the Earth's surface, in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def roads_of(locations):
    """All road names a site sits on, with aliases applied."""
    roads = set()
    for loc in locations.split(";"):
        m = LOCATION_PATTERN.match(loc.strip())
        names = m.groups() if m else (loc.strip(),)
        for name in names:
            name = name.strip().upper()
            roads.add(ROAD_ALIASES.get(name, name))
    return roads


def load_sites(path):
    sites = pd.read_csv(path, dtype={"site": str})
    sites["site"] = sites["site"].str.zfill(4)
    sites["roads"] = sites["locations"].apply(roads_of)
    return sites.set_index("site")


def order_along_road(members, sites):
    """Sort the sites on one road along the road's main direction
    (east-west if the road spreads more east-west, otherwise north-south)."""
    lats = sites.loc[members, "lat"]
    lons = sites.loc[members, "long"]
    east_west = (lons.max() - lons.min()) * math.cos(math.radians(lats.mean())) >= (lats.max() - lats.min())
    key = lons if east_west else lats
    return list(key.sort_values().index)


def build_edges(sites):
    edges = {}                                        # (a, b) with a < b -> (road, how)

    def dist(a, b):
        return haversine_km(sites.at[a, "lat"], sites.at[a, "long"], sites.at[b, "lat"], sites.at[b, "long"])

    # 1-4: link consecutive sites on the same road
    all_roads = sorted(set().union(*sites["roads"]))
    for road in all_roads:
        members = [s for s in sites.index if road in sites.at[s, "roads"]]
        if len(members) < 2:
            continue
        ordered = order_along_road(members, sites)
        for a, b in zip(ordered, ordered[1:]):
            if dist(a, b) <= MAX_LINK_KM:
                edges.setdefault(tuple(sorted((a, b))), (road, "road"))

    # 5: connect any separate piece of the network to its nearest site in the main network
    while True:
        groups = connected_groups(sites.index, edges)
        if len(groups) == 1:
            break
        main = max(groups, key=len)
        others = [g for g in groups if g is not main]
        best = min(((dist(a, b), a, b) for g in others for a in g for b in main))
        _, a, b = best
        edges[tuple(sorted((a, b)))] = ("(nearest site)", "nearest")

    rows = []
    for (a, b), (road, how) in sorted(edges.items()):
        rows.append({"from_site": a, "to_site": b, "road": road,
                     "distance_km": round(dist(a, b), 3), "link_type": how})
    return pd.DataFrame(rows)


def connected_groups(nodes, edges):
    neighbours = {n: set() for n in nodes}
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    seen, groups = set(), []
    for n in nodes:
        if n in seen:
            continue
        group, stack = set(), [n]
        while stack:
            x = stack.pop()
            if x in group:
                continue
            group.add(x)
            stack.extend(neighbours[x] - group)
        seen |= group
        groups.append(group)
    return groups


def save_map(sites, edges, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping the map")
        return
    plt.figure(figsize=(10, 9))
    for _, e in edges.iterrows():
        a, b = e["from_site"], e["to_site"]
        style = "-" if e["link_type"] == "road" else "--"
        plt.plot([sites.at[a, "long"], sites.at[b, "long"]], [sites.at[a, "lat"], sites.at[b, "lat"]],
                 style, color="gray", lw=1.2, zorder=1)
    plt.scatter(sites["long"], sites["lat"], s=60, color="tab:pink", edgecolors="black", zorder=2)
    for s in sites.index:
        plt.annotate(s, (sites.at[s, "long"], sites.at[s, "lat"]), xytext=(4, 4),
                     textcoords="offset points", fontsize=8)
    plt.title("SCATS sites and road links (dashed = linked to nearest site)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.gca().set_aspect(1 / math.cos(math.radians(sites["lat"].mean())))
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main():
    sites_csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "processed_data", "sites.csv")
    out_dir = os.path.dirname(sites_csv)
    sites = load_sites(sites_csv)
    edges = build_edges(sites)
    edges.to_csv(os.path.join(out_dir, "edges.csv"), index=False)
    save_map(sites, edges, os.path.join(out_dir, "graph_map.png"))

    degree = pd.concat([edges["from_site"], edges["to_site"]]).value_counts()
    print(f"{len(sites)} sites, {len(edges)} two-way links "
          f"({(edges['link_type'] == 'nearest').sum()} linked to the nearest site)")
    print(f"Links per site: min {degree.min()}, max {degree.max()}")
    print(f"Saved edges.csv and graph_map.png to {out_dir}")


if __name__ == "__main__":
    main()
