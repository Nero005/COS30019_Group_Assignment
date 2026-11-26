"""
run_ml_tests.py - machine learning test cases for the TBRGS (COS30019 Assignment 2B)
(The software / route-finding test cases are in run_software_tests.py.)

Tests the three saved models (LSTM, GRU, XGBoost regression) on the test days they never saw
in training (27-31 October 2006): single predictions at peak, night and weekend times, whole
test-set accuracy against the baselines, behaviour further ahead, output validity, and that
the saved results are permanent (same input -> same output, and the same metrics as reported).

ML01-ML15 are the main cases for the report; ML16-ML18 are extra cases (appendix).

Tolerances (set before running, not tuned to pass):
    busy times     prediction within 15% of the actual count
    night (3am)    within 20 cars of the actual count (percentages mean little for tiny counts)

Run (from the Assignment2B folder, after training the three models):
    python test_cases/run_ml_tests.py
Outputs: results/tests/ml_test_results.csv  and  results/tests/ml_test_summary.txt
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # the Assignment2B folder
sys.path.insert(0, os.path.join(ROOT, "travel_time_estimation"))
sys.path.insert(0, os.path.join(ROOT, "Machine_Learning_Algorithm"))
from travel_time import load_predictor                                    # noqa: E402
from evaluate_models import baseline_predictions, metrics, model_predictions   # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "tests")
FLOWS_CSV = os.path.join(ROOT, "processed_data", "processed_data.csv")
MODELS = {"lstm": "LSTM", "gru": "GRU", "xgboost": "XGBoost regression"}
TEST_START = pd.Timestamp("2006-10-27 00:00")
BUSY_TOL = 0.15          # 15%
NIGHT_TOL = 20           # cars per 15 minutes


def load_flows():
    flows = pd.read_csv(FLOWS_CSV, index_col=0, parse_dates=True)
    flows.columns = flows.columns.astype(str).str.zfill(4)
    return flows


class Lab:
    """Loads the models and the test-set predictions once, shared by all cases."""

    def __init__(self):
        self.flows = load_flows()
        self.sites = list(self.flows.columns)
        self.test_times = self.flows.index[self.flows.index >= TEST_START]
        self.true = self.flows.loc[self.test_times].values.astype(float)
        self.predictors = {m: load_predictor(m) for m in MODELS}
        self._horizon = {}

    def predict(self, model, when, site):
        return self.predictors[model].forecast(when, steps=1)[site][0]

    def actual(self, when, site):
        return float(self.flows.loc[pd.Timestamp(when), site])

    def horizon(self, model):
        """Predictions for every test time, 1-4 steps ahead (cached)."""
        if model not in self._horizon:
            self._horizon[model] = model_predictions(self.predictors[model], self.test_times, self.sites, 4)
        return self._horizon[model]


# ---------------------------------------------------------------- case helpers
def single_point(lab, models, when, site, kind):
    """Predict one site at one time with each model and compare with the actual count."""
    actual = lab.actual(when, site)
    parts, ok = [], True
    for m in models:
        pred = lab.predict(m, when, site)
        err = abs(pred - actual)
        if kind == "busy":
            ok &= err <= BUSY_TOL * actual
            parts.append(f"{MODELS[m]} {pred:.0f} ({err / actual * 100:.1f}% off)")
        else:
            ok &= err <= NIGHT_TOL
            parts.append(f"{MODELS[m]} {pred:.0f} ({err:.0f} cars off)")
    return ok, f"actual {actual:.0f}; " + ", ".join(parts)


def case_hard_site(lab):
    """Site 3001 at 8am is expected to be harder than site 2000 at 8am for every model."""
    when = "2006-10-30 08:00"
    ok, parts = True, []
    for m in MODELS:
        e2000 = abs(lab.predict(m, when, "2000") - lab.actual(when, "2000")) / lab.actual(when, "2000")
        e3001 = abs(lab.predict(m, when, "3001") - lab.actual(when, "3001")) / lab.actual(when, "3001")
        ok &= e3001 > e2000
        parts.append(f"{MODELS[m]}: 3001 {e3001 * 100:.1f}% vs 2000 {e2000 * 100:.1f}%")
    return ok, "; ".join(parts)


def case_beats_baseline(lab, which):
    persistence, hist_avg = baseline_predictions(lab.flows, lab.test_times)
    base = persistence if which == "persistence" else hist_avg
    base_mae = metrics(lab.true, base)["MAE"]
    ok, parts = True, []
    for m in MODELS:
        mae = metrics(lab.true, lab.horizon(m)[1])["MAE"]
        ok &= mae < base_mae
        parts.append(f"{MODELS[m]} {mae:.2f}")
    return ok, f"baseline MAE {base_mae:.2f}; " + ", ".join(parts)


def case_horizon(lab):
    ok, parts = True, []
    for m in MODELS:
        h = lab.horizon(m)
        valid = ~np.isnan(h[4]).any(axis=1)
        mae15 = metrics(lab.true[valid], h[1][valid])["MAE"]
        mae60 = metrics(lab.true[valid], h[4][valid])["MAE"]
        ok &= mae60 >= mae15
        parts.append(f"{MODELS[m]} {mae15:.1f} -> {mae60:.1f}")
    return ok, "MAE 15 min -> 60 min ahead: " + ", ".join(parts)


def case_valid_output(lab):
    ok, parts = True, []
    for m in MODELS:
        p = lab.horizon(m)[1]
        good = p.shape == lab.true.shape and np.isfinite(p).all() and (p >= 0).all()
        ok &= bool(good)
        parts.append(f"{MODELS[m]} {p.shape[0]}x{p.shape[1]} {'OK' if good else 'BAD'}")
    return ok, ", ".join(parts)


def case_repeatable(lab):
    ok, parts = True, []
    for m in MODELS:
        a = lab.predictors[m].forecast("2006-10-30 08:00", steps=2)
        b = lab.predictors[m].forecast("2006-10-30 08:00", steps=2)
        same = all(a[s] == b[s] for s in lab.sites)
        ok &= same
        parts.append(f"{MODELS[m]} {'identical' if same else 'DIFFERENT'}")
    return ok, ", ".join(parts)


def case_saved_metrics(lab):
    """The MAE saved by each training script must equal the MAE recomputed now from the saved model."""
    folders = {"lstm": "lstm", "gru": "gru", "xgboost": "xgboost"}
    ok, parts = True, []
    for m, name in MODELS.items():
        with open(os.path.join(ROOT, "results", folders[m], "metrics.json")) as f:
            saved = json.load(f)[name]["MAE"]
        now = metrics(lab.true, lab.horizon(m)[1])["MAE"]
        ok &= abs(saved - now) < 0.01
        parts.append(f"{MODELS[m]} saved {saved:.3f} / now {now:.3f}")
    return ok, ", ".join(parts)


def case_peak_vs_night(lab):
    ok, parts = True, []
    for m in MODELS:
        peak = lab.predict(m, "2006-10-30 08:00", "2000")
        night = lab.predict(m, "2006-10-31 03:00", "2000")
        ok &= peak > 10 * night
        parts.append(f"{MODELS[m]} {peak:.0f} vs {night:.0f}")
    return ok, "8am vs 3am at site 2000: " + ", ".join(parts)


def case_too_early(lab):
    parts, ok = [], True
    for m in MODELS:
        try:
            lab.predictors[m].forecast("2006-10-01 01:00", steps=1)
            ok = False
            parts.append(f"{MODELS[m]} predicted (no error)")
        except (ValueError, KeyError, IndexError) as error:
            parts.append(f"{MODELS[m]} refused ({type(error).__name__})")
    return ok, ", ".join(parts)


def case_no_leakage(lab):
    d = np.load(os.path.join(ROOT, "processed_data", "ml_data.npz"), allow_pickle=True)
    train_end, val_start = pd.Timestamp(d["t_train"][-1]), pd.Timestamp(d["t_val"][0])
    val_end, test_start = pd.Timestamp(d["t_val"][-1]), pd.Timestamp(d["t_test"][0])
    ok = train_end < val_start and val_end < test_start and test_start == TEST_START
    return ok, (f"train ends {train_end:%d %b %H:%M}, validation {val_start:%d %b}-{val_end:%d %b %H:%M}, "
                f"test starts {test_start:%d %b %H:%M}")


ALL = list(MODELS)
CASES = [
    ("ML01", "XGBoost: site 2000, Monday 30 Oct 08:00 (morning peak)", "Within 15% of actual",
     lambda lab: single_point(lab, ["xgboost"], "2006-10-30 08:00", "2000", "busy")),
    ("ML02", "LSTM: site 2000, Monday 30 Oct 08:00 (morning peak)", "Within 15% of actual",
     lambda lab: single_point(lab, ["lstm"], "2006-10-30 08:00", "2000", "busy")),
    ("ML03", "GRU: site 2000, Monday 30 Oct 08:00 (morning peak)", "Within 15% of actual",
     lambda lab: single_point(lab, ["gru"], "2006-10-30 08:00", "2000", "busy")),
    ("ML04", "XGBoost: site 2000, Tuesday 31 Oct 03:00 (night)", "Within 20 cars of actual",
     lambda lab: single_point(lab, ["xgboost"], "2006-10-31 03:00", "2000", "night")),
    ("ML05", "LSTM: site 2000, Tuesday 31 Oct 03:00 (night)", "Within 20 cars of actual",
     lambda lab: single_point(lab, ["lstm"], "2006-10-31 03:00", "2000", "night")),
    ("ML06", "GRU: site 2000, Tuesday 31 Oct 03:00 (night)", "Within 20 cars of actual",
     lambda lab: single_point(lab, ["gru"], "2006-10-31 03:00", "2000", "night")),
    ("ML07", "All models: site 2000, Sunday 29 Oct 12:00 (weekend)", "Each within 15% of actual",
     lambda lab: single_point(lab, ALL, "2006-10-29 12:00", "2000", "busy")),
    ("ML08", "All models: site 3002, Monday 30 Oct 17:30 (evening peak)", "Each within 15% of actual",
     lambda lab: single_point(lab, ALL, "2006-10-30 17:30", "3002", "busy")),
    ("ML09", "All models: hardest site 3001 vs site 2000 at 08:00", "Error at 3001 larger than at 2000 "
     "for every model (known weak site)", case_hard_site),
    ("ML10", "All models vs persistence baseline (whole test set)", "Every model has lower MAE",
     lambda lab: case_beats_baseline(lab, "persistence")),
    ("ML11", "All models vs historical-average baseline (whole test set)", "Every model has lower MAE",
     lambda lab: case_beats_baseline(lab, "historical")),
    ("ML12", "Predicting further ahead (15 vs 60 minutes)", "Error at 60 min >= error at 15 min "
     "for every model", case_horizon),
    ("ML13", "Output validity over the whole test set", "480 x 40 predictions, all finite and >= 0",
     case_valid_output),
    ("ML14", "Repeatability (same input twice)", "Identical predictions every time", case_repeatable),
    ("ML15", "Saved results are permanent", "MAE in each metrics.json equals the MAE recomputed "
     "from the saved model", case_saved_metrics),
    ("ML16", "Peak busier than night (site 2000)", "08:00 prediction > 10 x 03:00 prediction "
     "for every model", case_peak_vs_night),
    ("ML17", "Too little history (1 Oct 01:00)", "Every model refuses to predict", case_too_early),
    ("ML18", "No data leakage between train / validation / test", "Train ends before validation, "
     "validation ends before the test days (27 Oct)", case_no_leakage),
]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading the three models...")
    lab = Lab()
    rows = []
    for cid, scenario, expected, run in CASES:
        t0 = time.perf_counter()
        try:
            passed, actual = run(lab)
        except Exception as error:                    # a crash is a failed test, not a stopped run
            passed, actual = False, f"Error: {error}"
        rows.append({"id": cid, "report": "main" if int(cid[2:]) <= 15 else "extra",
                     "scenario": scenario, "expected": expected, "actual": actual,
                     "result": "PASS" if passed else "FAIL",
                     "seconds": round(time.perf_counter() - t0, 2)})
        print(f"{cid}  {'PASS' if passed else 'FAIL'}  {scenario}")
        print(f"        expected: {expected}")
        print(f"        actual:   {actual}")

    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, "ml_test_results.csv"), index=False)
    passed = (table["result"] == "PASS").sum()
    summary = (f"TBRGS machine learning test cases: {passed} of {len(table)} passed "
               f"(ML01-ML15 main, ML16-ML18 extra)\n\n" +
               table[["id", "report", "scenario", "result"]].to_string(index=False))
    with open(os.path.join(OUT_DIR, "ml_test_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"\n{passed} of {len(table)} machine learning test cases passed. Saved "
          f"results/tests/ml_test_results.csv and ml_test_summary.txt")
    return 0 if passed == len(table) else 1


if __name__ == "__main__":
    sys.exit(main())
