"""
xgb_regression.py - XGBoost regression model for traffic flow (COS30019 Assignment 2B)

Uses XGBRegressor with the squared-error regression objective to predict the traffic
flow (cars per 15 minutes) at every SCATS site for the next 15 minutes.

It uses the same data, the same split and the same metrics as train_lstm.py and
train_gru.py, so the three models can be compared fairly:
    data   processed_data/processed_data.csv  (from data_processing.py, one column per site)
    split  train days 1-21, validation days 22-26, test days 27-31 (by time, never shuffled)
    input  the last LOOKBACK readings (12 = 3 hours), the same as the LSTM and GRU windows
    output the reading 15 minutes later
XGBoost cannot see the order of a sequence the way an LSTM/GRU does, so each reading in the
window becomes its own input column, plus time features (time of day, day of week,
weekend), the reading at the same time the day before, and the site as a category.
One model is trained for all 40 sites.

Steps (from the Assignment2B folder)
    python data_cleaning_processing/data_processing.py          # once -> processed_data/
    python Machine_Learning_Algorithm/xgb_regression.py         # -> results/xgboost/

Outputs (in --out): xgb_regression.json (the model), metrics.json, feature_importance.png,
                    prediction_<site>.png

Use inside the route guidance system (same interface as the LSTM and GRU predictors)
    from xgb_regression import TrafficPredictor
    p = TrafficPredictor("results/xgboost/xgb_regression.json", "processed_data/processed_data.csv")
    flows = p.forecast("2006-10-30 08:00", steps=4)   # {site: [cars/15min, ...]}
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Assignment2B folder

# the split and window must match data_processing.py
SLOTS_PER_DAY = 96
LOOKBACK = 12            # last 12 readings (3 hours)
TRAIN_DAYS = 21          # 1-21 Oct  -> train
VAL_DAYS = 5             # 22-26 Oct -> validation (early stopping); 27-31 Oct -> test

DEFAULTS = {
    "n_estimators": 2000,        # upper limit; early stopping picks the real number of trees
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,            # each tree sees 80% of the rows
    "colsample_bytree": 0.8,     # and 80% of the input columns
    "early_stopping_rounds": 50, # stop when the validation error has not improved for 50 trees
    "seed": 42,
}


# ---------------------------------------------------------------- data
def load_flows(csv_path):
    """processed_data.csv -> DataFrame (timestamps x sites), site ids as 4-digit text."""
    flows = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    flows.columns = flows.columns.astype(str).str.zfill(4)
    return flows


def feature_columns():
    return ([f"lag_{k}" for k in range(LOOKBACK, 0, -1)]
            + ["lag_day", "time_of_day", "day_of_week", "is_weekend", "site"])


def build_rows(values, timestamps, sites, targets):
    """One row per (target time, site).
    lag_1 is the reading just before the target, lag_12 the reading 3 hours before."""
    n_sites = values.shape[1]
    rows = []
    for t in targets:
        window = values[t - LOOKBACK:t]                     # (LOOKBACK, n_sites)
        day_before = values[t - SLOTS_PER_DAY] if t >= SLOTS_PER_DAY else np.full(n_sites, np.nan)
        ts = pd.Timestamp(timestamps[t])
        for i in range(n_sites):
            rows.append(list(window[:, i]) + [day_before[i],
                        ts.hour * 4 + ts.minute // 15, ts.day_of_week, int(ts.day_of_week >= 5), sites[i]])
    X = pd.DataFrame(rows, columns=feature_columns())
    X["site"] = pd.Categorical(X["site"], categories=list(sites))   # same categories every time
    return X


def split_data(flows):
    """Same boundaries as data_processing.prepare_ml_data, so the test targets are identical."""
    values = flows.values.astype(float)
    train_end = TRAIN_DAYS * SLOTS_PER_DAY
    val_end = train_end + VAL_DAYS * SLOTS_PER_DAY
    parts = {"train": range(LOOKBACK, train_end),        # a target needs 12 readings before it
             "val": range(train_end, val_end),
             "test": range(val_end, len(values))}
    sites = list(flows.columns)
    out = {}
    for name, targets in parts.items():
        out[f"X_{name}"] = build_rows(values, flows.index, sites, targets)
        out[f"y_{name}"] = values[list(targets)].reshape(-1)                 # same row order
        out[f"t_{name}"] = flows.index[list(targets)]
    out["sites"] = sites
    return out


# ---------------------------------------------------------------- model
def train_model(d, cfg):
    """XGBoost regression (reg:squarederror), early stopping on the validation days."""
    model = XGBRegressor(
        objective="reg:squarederror",   # regression: minimise the squared prediction error
        n_estimators=cfg["n_estimators"],
        learning_rate=cfg["learning_rate"],
        max_depth=cfg["max_depth"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        early_stopping_rounds=cfg["early_stopping_rounds"],
        tree_method="hist",
        enable_categorical=True,        # lets 'site' be used as a category
        random_state=cfg["seed"],
    )
    t0 = time.time()
    model.fit(d["X_train"], d["y_train"], eval_set=[(d["X_val"], d["y_val"])], verbose=False)
    return model, time.time() - t0


def regression_metrics(y_true, y_pred):
    """Metrics in cars / 15 min over all sites and time steps (same as train_lstm.py / train_gru.py)."""
    err = y_pred - y_true
    mask = y_true > 5                            # avoid MAPE blow-up on near-zero flow
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE_%": float(np.mean(np.abs(err[mask] / y_true[mask])) * 100),
            "R2": float(1 - ss_res / ss_tot)}


def evaluate(model, d):
    """Test-set metrics, plus the same persistence baseline as the LSTM/GRU scripts
    (next reading = last reading)."""
    n_sites = len(d["sites"])
    pred = np.clip(model.predict(d["X_test"]), 0, None).reshape(-1, n_sites)   # (time, sites)
    true = d["y_test"].reshape(-1, n_sites)
    naive = d["X_test"]["lag_1"].to_numpy().reshape(-1, n_sites)
    result = {"XGBoost regression": regression_metrics(true, pred),
              "Persistence baseline": regression_metrics(true, naive)}
    mae = np.mean(np.abs(pred - true), axis=0)
    result["per_site_MAE"] = {s: float(m) for s, m in zip(d["sites"], mae)}
    return result, true, pred


def save_plots(model, d, true, pred, out_dir, n_sites=3):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots")
        return
    imp = pd.Series(model.feature_importances_, index=feature_columns()).sort_values()
    plt.figure(figsize=(6, 5))
    plt.barh(imp.index, imp.values)
    plt.xlabel("Importance (gain share)"); plt.title("XGBoost regression - feature importance")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "feature_importance.png"), dpi=150); plt.close()

    for i, site in enumerate(d["sites"][:n_sites]):
        plt.figure(figsize=(9, 3.5))
        plt.plot(d["t_test"][:288], true[:288, i], label="actual")      # first 3 test days
        plt.plot(d["t_test"][:288], pred[:288, i], label="XGBoost regression")
        plt.ylabel("Cars / 15 min"); plt.title(f"SCATS {site} - test period"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"prediction_{site}.png"), dpi=150)
        plt.close()


# ---------------------------------------------------------------- use in the route finder
class TrafficPredictor:
    """Load a trained model and forecast flow for every site (same interface as the
    LSTM/GRU TrafficPredictor). History comes from processed_data.csv."""

    def __init__(self, model_path, flows_csv):
        self.model = XGBRegressor()
        self.model.load_model(model_path)
        self.flows = load_flows(flows_csv)
        self.sites = list(self.flows.columns)

    def forecast(self, timestamp, steps=1):
        """Cars/15 min for `steps` consecutive slots starting at `timestamp`.
        Readings before `timestamp` are real; later steps feed on predictions."""
        t0 = self.flows.index.get_loc(pd.Timestamp(timestamp))
        if t0 < LOOKBACK:
            raise ValueError("Not enough history before the requested time.")
        values = self.flows.values[:t0].astype(float)          # history only
        times = list(self.flows.index[:t0])
        out = []
        for k in range(steps):
            ts = pd.Timestamp(timestamp) + pd.Timedelta(minutes=15 * k)
            values = np.vstack([values, np.zeros(len(self.sites))])   # slot to predict
            times.append(ts)
            X = build_rows(values, times, self.sites, [len(values) - 1])
            p = np.clip(self.model.predict(X), 0, None)
            values[-1] = p                                      # the next step uses this prediction
            out.append(p)
        out = np.array(out)                                     # (steps, n_sites)
        return {s: out[:, i].tolist() for i, s in enumerate(self.sites)}


def main():
    p = argparse.ArgumentParser(description="Train an XGBoost regression traffic flow model")
    p.add_argument("--data", default=os.path.join(ROOT, "processed_data"),
                   help="folder containing processed_data.csv")
    p.add_argument("--config", help="optional JSON file overriding the defaults")
    p.add_argument("--out", default=os.path.join(ROOT, "results", "xgboost"))
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=None)
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    cfg.update({k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None})

    flows = load_flows(os.path.join(args.data, "processed_data.csv"))
    d = split_data(flows)
    os.makedirs(args.out, exist_ok=True)
    n = len(d["sites"])
    print(f"{n} sites | train {len(d['y_train']) // n} val {len(d['y_val']) // n} "
          f"test {len(d['y_test']) // n} time steps")

    model, seconds = train_model(d, cfg)
    metrics, true, pred = evaluate(model, d)
    metrics["train_seconds"] = seconds
    metrics["trees_used"] = int(model.best_iteration + 1)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_site_MAE"}, indent=2))

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    model.save_model(os.path.join(args.out, "xgb_regression.json"))
    save_plots(model, d, true, pred, args.out)
    print(f"Saved model, metrics and plots to {args.out}/")


if __name__ == "__main__":
    main()
