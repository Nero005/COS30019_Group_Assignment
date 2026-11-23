"""
evaluate_models.py - compare the trained traffic flow models (COS30019 Assignment 2B)

Loads the three saved models (LSTM, GRU, XGBoost regression) and tests them on exactly the
same test days (27-31 October 2006, all 40 SCATS sites), next to two simple baselines:
    Persistence         the next 15 minutes = the last 15 minutes
    Historical average  the average flow at that site, time of day and day type
                        (weekday or weekend) over the training days (1-21 October)

Every model is called through its TrafficPredictor.forecast(), the same function the
route guidance system uses, so the numbers describe the models as they are used.

Metrics (in cars per 15 minutes, over all sites and test times):
    MAE    mean absolute error
    RMSE   root mean squared error (punishes big misses more)
    NRMSE  RMSE / mean actual flow (the measure used in the lecturer's example)
    MAPE   mean absolute percentage error (only where actual flow > 5, to avoid dividing by ~0)
    R2     share of the variation in traffic explained by the model (1 = perfect)

Outputs (in results/evaluation/):
    comparison_table.csv     one row per model, all metrics, for 15 minutes ahead
    horizon_table.csv        accuracy when predicting 15, 30, 45 and 60 minutes ahead
    peak_offpeak_table.csv   accuracy in weekday peak hours vs the rest of the time
    error_by_hour.csv        MAE for each hour of the day
    per_site_mae.csv         MAE for each SCATS site
    evaluation_summary.txt   the tables above as text
    metrics_comparison.png, error_by_horizon.png, error_by_hour.png, per_site_mae.png,
    actual_vs_predicted.png, scatter_actual_vs_predicted.png

Run (from the Assignment2B folder, after training the three models):
    python Machine_Learning_Algorithm/evaluate_models.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # the Assignment2B folder
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "data_cleaning_processing"))

SLOTS_PER_DAY = 96
TRAIN_DAYS, VAL_DAYS = 21, 5          # same split as data_processing.py
HORIZONS = [1, 2, 3, 4]               # steps of 15 minutes ahead (15, 30, 45, 60 min)
PEAK_HOURS = [(7.0, 9.5), (16.5, 19.0)]   # weekday morning and evening peaks

# name shown in the tables -> (module, saved model file, results folder of the training script)
MODELS = {
    "LSTM": ("train_lstm", os.path.join("results", "lstm", "train_lstm.pt"), "lstm"),
    "GRU": ("train_gru", os.path.join("results", "gru", "train_gru.pt"), "gru"),
    "XGBoost regression": ("xgb_regression", os.path.join("results", "xgboost", "xgb_regression.json"), "xgboost"),
}
COLOURS = {"LSTM": "tab:blue", "GRU": "tab:orange", "XGBoost regression": "tab:green",
           "Persistence": "tab:gray", "Historical average": "tab:brown"}


# ---------------------------------------------------------------- metrics
def metrics(true, pred):
    """All metrics over every site and time in the arrays (cars per 15 minutes)."""
    true, pred = np.asarray(true, float), np.asarray(pred, float)
    err = pred - true
    mask = true > 5
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": rmse,
            "NRMSE": rmse / float(np.mean(true)),
            "MAPE_%": float(np.mean(np.abs(err[mask] / true[mask])) * 100),
            "R2": float(1 - np.sum(err ** 2) / np.sum((true - true.mean()) ** 2))}


# ---------------------------------------------------------------- predictions
def load_flows(path):
    flows = pd.read_csv(path, index_col=0, parse_dates=True)
    flows.columns = flows.columns.astype(str).str.zfill(4)
    return flows


def model_predictions(predictor, test_times, sites, max_steps):
    """Forecast from every test time, up to max_steps ahead.
    Returns {h: array (n_test, n_sites)}, where row i is the prediction for test_times[i]
    made h steps (h x 15 minutes) earlier. Missing entries (near the start) are NaN."""
    n = len(test_times)
    preds = {h: np.full((n, len(sites)), np.nan) for h in range(1, max_steps + 1)}
    for i, t in enumerate(test_times):
        steps = min(max_steps, n - i)
        out = predictor.forecast(t, steps=steps)                 # {site: [step1, step2, ...]}
        block = np.array([out[s] for s in sites]).T              # (steps, n_sites)
        for k in range(steps):                                   # forecast made at t, k+1 steps ahead,
            preds[k + 1][i + k] = block[k]                       # is a prediction for test_times[i+k]
    return preds


def baseline_predictions(flows, test_times):
    """Persistence and historical-average predictions for the test times."""
    values = flows.values.astype(float)
    pos = flows.index.get_indexer(test_times)
    persistence = values[pos - 1]                                # the reading 15 minutes earlier

    def slot_and_daytype(idx):
        idx = pd.DatetimeIndex(idx)
        return np.asarray(idx.hour * 4 + idx.minute // 15), np.asarray(idx.day_of_week >= 5)

    train = flows.iloc[:TRAIN_DAYS * SLOTS_PER_DAY]
    slot, weekend = slot_and_daytype(train.index)
    avg = train.groupby([slot, weekend]).mean()                  # (slot, weekend) -> mean flow per site
    test_slot, test_weekend = slot_and_daytype(test_times)
    hist_avg = avg.loc[list(zip(test_slot, test_weekend))].values
    return persistence, hist_avg


# ---------------------------------------------------------------- tables
def peak_mask(times):
    times = pd.DatetimeIndex(times)
    hour = times.hour + times.minute / 60
    weekday = times.dayofweek < 5
    in_peak = np.zeros(len(times), bool)
    for start, end in PEAK_HOURS:
        in_peak |= (hour >= start) & (hour < end)
    return in_peak & weekday


def build_tables(true, one_step, horizon_preds, test_times, sites, extra):
    comparison = pd.DataFrame({name: metrics(true, p) for name, p in one_step.items()}).T
    for name, info in extra.items():
        for k, v in info.items():
            comparison.loc[name, k] = v

    rows = []
    for name, preds in horizon_preds.items():
        for h, p in preds.items():
            ok = ~np.isnan(p).any(axis=1)
            m = metrics(true[ok], p[ok])
            rows.append({"model": name, "minutes_ahead": h * 15, **m})
    horizon = pd.DataFrame(rows)

    peak = peak_mask(test_times)
    peak_rows = []
    for name, p in one_step.items():
        for label, mask in (("weekday peak", peak), ("other times", ~peak)):
            peak_rows.append({"model": name, "period": label, **metrics(true[mask], p[mask])})
    peak_table = pd.DataFrame(peak_rows)

    hours = pd.DatetimeIndex(test_times).hour
    by_hour = pd.DataFrame({name: [np.mean(np.abs(p[hours == h] - true[hours == h])) for h in range(24)]
                            for name, p in one_step.items()}, index=pd.Index(range(24), name="hour"))
    per_site = pd.DataFrame({name: np.mean(np.abs(p - true), axis=0) for name, p in one_step.items()},
                            index=pd.Index(sites, name="site"))
    return comparison, horizon, peak_table, by_hour, per_site


# ---------------------------------------------------------------- charts
def save_charts(out, comparison, horizon, by_hour, per_site, true, one_step, test_times, sites):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping charts")
        return
    names = list(comparison.index)

    # 1. main metrics side by side
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, col in zip(axes, ["MAE", "RMSE", "NRMSE", "R2"]):
        vals = comparison[col]
        ax.bar(range(len(names)), vals, color=[COLOURS[n] for n in names])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" regression", "\nregression").replace(" average", "\naverage")
                            for n in names], fontsize=8)
        ax.set_title(col + (" (higher is better)" if col == "R2" else " (lower is better)"), fontsize=10)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3f}" if col in ("NRMSE", "R2") else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
        if col == "R2":
            ax.set_ylim(min(0.9, vals.min() - 0.01), 1.0)
    fig.suptitle("Test days 27-31 October 2006, 15 minutes ahead, all 40 sites")
    fig.tight_layout(); fig.savefig(os.path.join(out, "metrics_comparison.png"), dpi=150); plt.close(fig)

    # 2. accuracy further ahead
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, g in horizon.groupby("model", sort=False):
        ax.plot(g["minutes_ahead"], g["MAE"], marker="o", label=name, color=COLOURS[name])
    ax.set_xlabel("Minutes ahead"); ax.set_ylabel("MAE (cars / 15 min)")
    ax.set_xticks([h * 15 for h in HORIZONS]); ax.set_title("Error when predicting further ahead")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "error_by_horizon.png"), dpi=150); plt.close(fig)

    # 3. error through the day
    fig, ax = plt.subplots(figsize=(9, 4))
    for name in by_hour.columns:
        ax.plot(by_hour.index, by_hour[name], marker=".", label=name, color=COLOURS[name])
    ax.set_xlabel("Hour of day"); ax.set_ylabel("MAE (cars / 15 min)"); ax.set_xticks(range(0, 24, 2))
    ax.set_title("Error by hour of day (test days)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "error_by_hour.png"), dpi=150); plt.close(fig)

    # 4. error per site (models only, sorted by the best model's error)
    model_cols = [n for n in per_site.columns if n in MODELS]
    order = per_site[model_cols].mean(axis=1).sort_values().index
    fig, ax = plt.subplots(figsize=(12, 4))
    width = 0.8 / len(model_cols)
    for j, name in enumerate(model_cols):
        ax.bar(np.arange(len(order)) + j * width, per_site.loc[order, name], width,
               label=name, color=COLOURS[name])
    ax.set_xticks(np.arange(len(order)) + 0.4 - width / 2)
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_ylabel("MAE (cars / 15 min)"); ax.set_title("Error per SCATS site"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "per_site_mae.png"), dpi=150); plt.close(fig)

    # 5. one busy site on one weekday: actual vs every model
    site_i = int(np.argmax(true.mean(axis=0)))
    day = pd.DatetimeIndex(test_times).normalize() == pd.Timestamp("2006-10-30")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(np.array(test_times)[day], true[day, site_i], color="black", lw=2, label="Actual")
    for name in model_cols:
        ax.plot(np.array(test_times)[day], one_step[name][day, site_i], lw=1.2, label=name, color=COLOURS[name])
    ax.set_ylabel("Cars / 15 min"); ax.set_title(f"SCATS {sites[site_i]} on Monday 30 October 2006")
    ax.legend(); ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(os.path.join(out, "actual_vs_predicted.png"), dpi=150); plt.close(fig)

    # 6. predicted vs actual for every site and time
    fig, axes = plt.subplots(1, len(model_cols), figsize=(5 * len(model_cols), 4.5), sharey=True)
    top = max(true.max(), max(one_step[n].max() for n in model_cols))
    for ax, name in zip(np.atleast_1d(axes), model_cols):
        ax.scatter(true.ravel(), one_step[name].ravel(), s=2, alpha=0.25, color=COLOURS[name])
        ax.plot([0, top], [0, top], "k--", lw=1)
        ax.set_title(f"{name} (R2 = {comparison.loc[name, 'R2']:.3f})")
        ax.set_xlabel("Actual (cars / 15 min)")
    np.atleast_1d(axes)[0].set_ylabel("Predicted (cars / 15 min)")
    fig.tight_layout(); fig.savefig(os.path.join(out, "scatter_actual_vs_predicted.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="Compare the trained traffic flow models")
    p.add_argument("--data", default=os.path.join(ROOT, "processed_data", "processed_data.csv"))
    p.add_argument("--out", default=os.path.join(ROOT, "results", "evaluation"))
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    flows = load_flows(args.data)
    sites = list(flows.columns)
    test_start = (TRAIN_DAYS + VAL_DAYS) * SLOTS_PER_DAY
    test_times = flows.index[test_start:]
    true = flows.values[test_start:].astype(float)
    print(f"Test: {len(test_times)} time steps ({test_times[0]} to {test_times[-1]}) x {len(sites)} sites")

    one_step, horizon_preds, extra = {}, {}, {}
    for name, (module_name, model_file, folder) in MODELS.items():
        path = os.path.join(ROOT, model_file)
        if not os.path.exists(path):
            print(f"Skipping {name}: {model_file} not found (run its training script first)")
            continue
        module = __import__(module_name)
        predictor = module.TrafficPredictor(path, args.data)
        t0 = time.time()
        preds = model_predictions(predictor, test_times, sites, max(HORIZONS))
        seconds = time.time() - t0
        one_step[name] = preds[1]
        horizon_preds[name] = preds
        info = {"predict_ms_per_call": seconds / len(test_times) * 1000}
        metrics_file = os.path.join(ROOT, "results", folder, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                info["train_seconds"] = json.load(f).get("train_seconds", np.nan)
        extra[name] = info
        print(f"Evaluated {name} ({seconds:.1f} s)")

    if not one_step:
        sys.exit("No trained models found. Train them first (see the Machine_Learning_Algorithm scripts).")

    persistence, hist_avg = baseline_predictions(flows, test_times)
    one_step["Persistence"] = persistence
    one_step["Historical average"] = hist_avg
    # persistence h steps ahead: the last reading known when predicting, i.e. h readings earlier
    values = flows.values.astype(float)
    horizon_preds["Persistence"] = {h: values[test_start - h:test_start - h + len(test_times)]
                                    for h in HORIZONS}

    comparison, horizon, peak_table, by_hour, per_site = build_tables(
        true, one_step, horizon_preds, test_times, sites, extra)

    comparison.to_csv(os.path.join(args.out, "comparison_table.csv"), float_format="%.4f")
    horizon.to_csv(os.path.join(args.out, "horizon_table.csv"), index=False, float_format="%.4f")
    peak_table.to_csv(os.path.join(args.out, "peak_offpeak_table.csv"), index=False, float_format="%.4f")
    by_hour.to_csv(os.path.join(args.out, "error_by_hour.csv"), float_format="%.3f")
    per_site.to_csv(os.path.join(args.out, "per_site_mae.csv"), float_format="%.3f")
    save_charts(args.out, comparison, horizon, by_hour, per_site, true, one_step, test_times, sites)

    pd.set_option("display.width", 140)
    text = ["MODEL COMPARISON - test days 27-31 October 2006, all 40 sites, cars per 15 minutes",
            "", "1. Predicting 15 minutes ahead", comparison.round(3).to_string(),
            "", "2. Predicting further ahead (MAE / RMSE / NRMSE)",
            horizon.pivot(index="model", columns="minutes_ahead", values="MAE").round(2)
                   .add_prefix("MAE ").to_string(),
            "", "3. Weekday peak hours (7:00-9:30, 16:30-19:00) vs other times",
            peak_table.pivot(index="model", columns="period", values="MAE").round(2)
                      .add_prefix("MAE ").to_string(),
            "", "4. Hardest sites (highest average MAE across the models)",
            per_site[[c for c in per_site.columns if c in MODELS]].mean(axis=1)
                    .sort_values(ascending=False).head(5).round(2).to_string()]
    summary = "\n".join(text)
    with open(os.path.join(args.out, "evaluation_summary.txt"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print("\n" + summary)
    print(f"\nSaved tables and charts to {args.out}")


if __name__ == "__main__":
    main()
