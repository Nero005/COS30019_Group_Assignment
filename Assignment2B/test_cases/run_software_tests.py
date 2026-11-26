"""
run_software_tests.py - software test cases for the TBRGS (COS30019 Assignment 2B)
(The machine learning test cases are in run_ml_tests.py.)

Runs every test case on the real Boroondara network (40 SCATS sites, October 2006 data),
checks the result against what is expected, and saves a results table for the report.
ST01-ST15 are the main cases for the report; ST16-ST24 are extra cases (appendix).

Every successful route search is also checked automatically:
  - each route starts at the origin and ends at the destination
  - each step uses a real road (an edge in processed_data/edges.csv)
  - no route visits a site twice, and no two routes are the same
  - a route's travel time equals the sum of its road times
  - no more routes than asked for

Run (from the Assignment2B folder, after training the three models):
    python test_cases/run_software_tests.py
Outputs: results/tests/software_test_results.csv  and  results/tests/software_test_summary.txt
"""
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # the Assignment2B folder
sys.path.insert(0, os.path.join(ROOT, "route_finding"))
from route_finder import ALGORITHMS, RouteFinder                        # noqa: E402
from travel_time import edge_travel_time, flow_to_speed                  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "tests")
PEAK = "2006-10-30 08:00"      # Monday morning peak
NIGHT = "2006-10-31 03:00"     # Tuesday 3am
EPS = 0.11                     # seconds; times are rounded to 0.1 s

# Each case: id, scenario, inputs for find_routes (or a special "kind"), what is expected, and a check.
# A check returns (passed, actual result as text).
CASES = [
    # ------------------------------------------------------------ normal trips
    dict(id="ST01", scenario="Assignment example trip, weekday morning peak",
         inputs=dict(origin="2000", destination="3002", when=PEAK, model="xgboost", algorithm="AS"),
         expected="5 routes found; A* route 1 is the fastest of them"),
    dict(id="ST02", scenario="Same trip at night (3am, almost no traffic)",
         inputs=dict(origin="2000", destination="3002", when=NIGHT, model="xgboost", algorithm="AS"),
         expected="Every road at the 60 km/h limit; trip faster than ST01 at peak"),
    dict(id="ST03", scenario="Neighbouring sites joined by one road",
         inputs=dict(origin="2000", destination="4043", when=PEAK, model="xgboost", algorithm="AS"),
         expected="Fastest route is the direct road 2000 -> 4043"),
    dict(id="ST04", scenario="Origin is the same as the destination",
         inputs=dict(origin="2000", destination="2000", when=PEAK, model="xgboost", algorithm="AS"),
         expected="One route [2000] with travel time 0"),
    dict(id="ST05", scenario="Far-apart sites (south-east to north of the map)",
         inputs=dict(origin="0970", destination="2827", when=PEAK, model="xgboost", algorithm="AS"),
         expected="Route found across the whole network"),
    dict(id="ST06", scenario="Origin only reachable through a 'nearest site' link (Richmond)",
         inputs=dict(origin="4812", destination="3180", when=PEAK, model="xgboost", algorithm="AS"),
         expected="Route found; it leaves 4812 through its only link (to 4262)"),
    dict(id="ST16", scenario="Reverse direction of ST01 (3002 -> 2000)",
         inputs=dict(origin="3002", destination="2000", when=PEAK, model="xgboost", algorithm="AS"),
         expected="Route found; uses the same roads in reverse order"),
    dict(id="ST17", scenario="Weekend morning (Saturday 28 Oct 08:00)",
         inputs=dict(origin="2000", destination="3002", when="2006-10-28 08:00", model="xgboost", algorithm="AS"),
         expected="Route found; trip not slower than the weekday peak (ST01)"),
    dict(id="ST18", scenario="Time not on a 15-minute step (08:07)",
         inputs=dict(origin="2000", destination="3002", when="2006-10-30 08:07", model="xgboost", algorithm="AS"),
         expected="Rounded down to 08:00; same result as ST01"),
    dict(id="ST19", scenario="Only one route asked for",
         inputs=dict(origin="2000", destination="3002", when=PEAK, model="xgboost", algorithm="AS", max_routes=1),
         expected="Exactly 1 route, the same as ST01 route 1"),
    # ------------------------------------------------------------ the six search methods
    dict(id="ST07", scenario="Depth-First Search on a long trip",
         inputs=dict(origin="0970", destination="2827", when=PEAK, model="xgboost", algorithm="DFS"),
         expected="Route found; not faster than the optimal A* route (ST05)"),
    dict(id="ST08", scenario="Breadth-First Search on a long trip",
         inputs=dict(origin="0970", destination="2827", when=PEAK, model="xgboost", algorithm="BFS"),
         expected="Route found; not faster than the optimal A* route (ST05)"),
    dict(id="ST09", scenario="Greedy Best-First Search on a long trip",
         inputs=dict(origin="0970", destination="2827", when=PEAK, model="xgboost", algorithm="GBFS"),
         expected="Route found; not faster than the optimal A* route (ST05)"),
    dict(id="ST10", scenario="Uniform Cost Search (CUS1) on a long trip",
         inputs=dict(origin="0970", destination="2827", when=PEAK, model="xgboost", algorithm="CUS1"),
         expected="Same travel time as A* (ST05): both are optimal"),
    dict(id="ST11", scenario="IDA* (CUS2) on a long trip",
         inputs=dict(origin="0970", destination="2827", when=PEAK, model="xgboost", algorithm="CUS2"),
         expected="Same travel time as A* (ST05): both are optimal"),
    # ------------------------------------------------------------ the three models
    dict(id="ST12", scenario="LSTM model instead of XGBoost (ST01 trip)",
         inputs=dict(origin="2000", destination="3002", when=PEAK, model="lstm", algorithm="AS"),
         expected="Route found; travel time within 20% of the XGBoost result"),
    dict(id="ST13", scenario="GRU model instead of XGBoost (ST01 trip)",
         inputs=dict(origin="2000", destination="3002", when=PEAK, model="gru", algorithm="AS"),
         expected="Route found; travel time within 20% of the XGBoost result"),
    # ------------------------------------------------------------ travel time formula
    dict(id="ST20", scenario="Flow-to-speed conversion at known points", kind="formula",
         expected="Flow 0 and 351 -> 60 km/h (speed limit); 1500 -> 32 km/h; 1 km at 60 km/h = 90 s with delay"),
    # ------------------------------------------------------------ invalid input
    dict(id="ST14", scenario="Unknown origin site 9999",
         inputs=dict(origin="9999", destination="3002", when=PEAK, model="xgboost", algorithm="AS"),
         expected="Error: site is not one of the 40 SCATS sites"),
    dict(id="ST15", scenario="Date outside the data (2 November 2006)",
         inputs=dict(origin="2000", destination="3002", when="2006-11-02 08:00", model="xgboost", algorithm="AS"),
         expected="Error: time must be within the data range"),
    dict(id="ST21", scenario="Time with less than 3 hours of history (1 Oct 01:00)",
         inputs=dict(origin="2000", destination="3002", when="2006-10-01 01:00", model="xgboost", algorithm="AS"),
         expected="Error: time must be within the data range"),
    dict(id="ST22", scenario="Text that is not a date",
         inputs=dict(origin="2000", destination="3002", when="hello", model="xgboost", algorithm="AS"),
         expected="Error: not a valid date and time"),
    dict(id="ST23", scenario="Unknown search method",
         inputs=dict(origin="2000", destination="3002", when=PEAK, model="xgboost", algorithm="XYZ"),
         expected="Error: unknown algorithm"),
    dict(id="ST24", scenario="Unknown model",
         inputs=dict(origin="2000", destination="3002", when=PEAK, model="svm", algorithm="AS"),
         expected="Error: unknown model"),
]


# ---------------------------------------------------------------- checks
def route_problems(result, edges, max_routes):
    """General checks for every successful search. Returns a list of problems (empty = OK)."""
    problems, seen = [], set()
    o, d = result["origin"], result["destination"]
    if len(result["routes"]) > max_routes:
        problems.append(f"{len(result['routes'])} routes > {max_routes}")
    for i, r in enumerate(result["routes"], 1):
        p = r["path"]
        if p[0] != o or p[-1] != d:
            problems.append(f"route {i} does not go from {o} to {d}")
        if len(set(p)) != len(p):
            problems.append(f"route {i} visits a site twice")
        bad = [(a, b) for a, b in zip(p, p[1:]) if (a, b) not in edges]
        if bad:
            problems.append(f"route {i} uses a road that does not exist: {bad[0]}")
        if abs(sum(leg["travel_time_s"] for leg in r["legs"]) - r["travel_time_s"]) > EPS * len(p):
            problems.append(f"route {i} time is not the sum of its roads")
        if tuple(p) in seen:
            problems.append(f"route {i} is a duplicate")
        seen.add(tuple(p))
    return problems


def describe(result):
    if not result["routes"]:
        return "no route"
    r = result["routes"][0]
    others = ", ".join(f"{x['travel_time_min']}" for x in result["routes"][1:])
    return (f"{len(result['routes'])} route(s); route 1 {r['travel_time_min']} min "
            f"[{' > '.join(r['path'])}]" + (f"; others {others} min" if others else "") +
            f"; {result['nodes_created']} nodes")


def specific_check(case, result, done):
    """The check written for each case. `done` holds earlier results, for comparisons."""
    cid, routes = case["id"], result["routes"]
    fastest = min(r["travel_time_s"] for r in routes) if routes else None
    first = routes[0]["travel_time_s"] if routes else None
    if cid == "ST01":
        return len(routes) == 5 and abs(first - fastest) <= EPS
    if cid == "ST02":
        speeds = [leg["speed_kmh"] for r in routes for leg in r["legs"]]
        return all(abs(s - 60) < 0.01 for s in speeds) and first < done["ST01"]["routes"][0]["travel_time_s"]
    if cid == "ST03":
        best = routes[result["fastest_index"]]["path"]
        return best == ["2000", "4043"]
    if cid == "ST04":
        return len(routes) == 1 and routes[0]["path"] == ["2000"] and routes[0]["travel_time_s"] == 0
    if cid in ("ST05", "ST12", "ST13"):
        if cid == "ST05":
            return bool(routes)
        ref = done["ST01"]["routes"][0]["travel_time_s"]
        return bool(routes) and abs(first - ref) / ref <= 0.20
    if cid == "ST06":
        return all(r["path"][1] == "4262" for r in routes)
    if cid == "ST16":
        return bool(routes) and routes[0]["path"] == list(reversed(done["ST01"]["routes"][0]["path"]))
    if cid == "ST17":
        return bool(routes) and first <= done["ST01"]["routes"][0]["travel_time_s"] + EPS
    if cid == "ST18":
        ref = done["ST01"]
        return result["time"] == ref["time"] and routes[0]["path"] == ref["routes"][0]["path"]
    if cid == "ST19":
        return len(routes) == 1 and routes[0]["path"] == done["ST01"]["routes"][0]["path"]
    if cid in ("ST07", "ST08", "ST09"):
        return bool(routes) and first >= done["ST05"]["routes"][0]["travel_time_s"] - EPS
    if cid in ("ST10", "ST11"):
        return bool(routes) and abs(first - done["ST05"]["routes"][0]["travel_time_s"]) <= EPS
    return False


def run_formula_case():
    speeds = {f: flow_to_speed(f) for f in (0, 351, 1500)}
    seconds = edge_travel_time(1.0, 0)
    passed = (abs(speeds[0] - 60) < 1e-9 and abs(speeds[351] - 60) < 1e-9
              and abs(speeds[1500] - 32) < 0.1 and abs(seconds - 90) < 1e-6)
    actual = (f"0 -> {speeds[0]:.1f} km/h, 351 -> {speeds[351]:.1f} km/h, "
              f"1500 -> {speeds[1500]:.1f} km/h; 1 km at 60 km/h = {seconds:.0f} s")
    return passed, actual


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    finder = RouteFinder()
    est = finder.estimator("xgboost")
    edges = set()
    for _, e in est.edges.iterrows():
        edges.add((e["from_site"], e["to_site"]))
        edges.add((e["to_site"], e["from_site"]))

    rows, done = [], {}
    for case in sorted(CASES, key=lambda c: c["id"]):
        t0 = time.perf_counter()
        expects_error = case["expected"].startswith("Error")
        if case.get("kind") == "formula":
            passed, actual = run_formula_case()
        else:
            args = dict(case["inputs"])
            max_routes = args.pop("max_routes", finder.config.get("max_routes", 5))
            try:
                result = finder.find_routes(args["origin"], args["destination"], args["when"],
                                            model=args["model"], algorithm=args["algorithm"],
                                            max_routes=max_routes)
                done[case["id"]] = result
                problems = route_problems(result, edges, max_routes)
                actual = describe(result) + (f" | PROBLEMS: {'; '.join(problems)}" if problems else "")
                passed = (not expects_error) and not problems and specific_check(case, result, done)
            except ValueError as error:
                actual = f"Error: {error}"
                passed = expects_error
        seconds = time.perf_counter() - t0
        inputs = case.get("inputs", {})
        rows.append({"id": case["id"], "report": "main" if int(case["id"][2:]) <= 15 else "extra",
                     "scenario": case["scenario"],
                     "origin": inputs.get("origin", ""), "destination": inputs.get("destination", ""),
                     "time": inputs.get("when", ""), "model": inputs.get("model", ""),
                     "algorithm": inputs.get("algorithm", ""), "expected": case["expected"],
                     "actual": actual, "result": "PASS" if passed else "FAIL",
                     "seconds": round(seconds, 3)})
        print(f"{case['id']}  {'PASS' if passed else 'FAIL'}  {case['scenario']}")
        print(f"        expected: {case['expected']}")
        print(f"        actual:   {actual}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "software_test_results.csv"), index=False)
    passed = (table["result"] == "PASS").sum()
    summary = (f"TBRGS software test cases: {passed} of {len(table)} passed "
               f"(ST01-ST15 main, ST16-ST24 extra)\n\n" +
               table[["id", "report", "scenario", "origin", "destination", "time", "model", "algorithm", "result"]]
               .to_string(index=False))
    with open(os.path.join(OUT_DIR, "software_test_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\n{passed} of {len(table)} software test cases passed. Saved "
          f"results/tests/software_test_results.csv and software_test_summary.txt")
    return 0 if passed == len(table) else 1


if __name__ == "__main__":
    sys.exit(main())
