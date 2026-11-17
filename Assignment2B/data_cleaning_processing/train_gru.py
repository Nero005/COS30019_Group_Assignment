"""
train_gru.py

Steps
    python data_processing.py Scats_Data_October_2006.xls data      # once
    python train_gru.py --data data --out results/gru

Outputs (in --out): train_gru.pt, metrics.json, loss_curve.png, prediction_<site>.png

Use inside the route guidance system
    from train_gru import TrafficPredictor
    p = TrafficPredictor("results/gru/train_gru.pt", "data/processed_data.csv")
    flows = p.forecast("2006-10-30 08:00", steps=4)   # {site: [cars/15min, ...]}
    per_hour = {s: [f * 4 for f in v] for s, v in flows.items()}
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data_processing import inverse_scale, load_ml_data

DEFAULTS = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "batch_size": 64,
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 10,          # early stopping on validation loss
    "seed": 42,
}


class GRUNet(nn.Module):
    """Stacked GRU -> linear layer giving the next reading of every site."""

    def __init__(self, n_sites, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(n_sites, hidden_size, num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, n_sites)

    def forward(self, x):                       # x: (batch, lookback, n_sites)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


def _loader(X, y, batch_size, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _run_epoch(model, loader, loss_fn, device, opt=None):
    model.train(opt is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total += loss.item() * len(xb)
            n += len(xb)
    return total / n


def train_model(d, cfg, device):
    """Adam + MSE; keeps the weights with the best validation loss."""
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    n_sites = d["X_train"].shape[2]
    model = GRUNet(n_sites, cfg["hidden_size"], cfg["num_layers"], cfg["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = nn.MSELoss()
    train_dl = _loader(d["X_train"], d["y_train"], cfg["batch_size"], True)
    val_dl = _loader(d["X_val"], d["y_val"], cfg["batch_size"], False)

    hist = {"train": [], "val": []}
    best, best_state, bad = float("inf"), None, 0
    t0 = time.time()
    for ep in range(1, cfg["epochs"] + 1):
        tr = _run_epoch(model, train_dl, loss_fn, device, opt)
        va = _run_epoch(model, val_dl, loss_fn, device)
        hist["train"].append(tr)
        hist["val"].append(va)
        print(f"epoch {ep:3d}  train {tr:.5f}  val {va:.5f}")
        if va < best - 1e-6:
            best, bad = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print("early stopping")
                break
    model.load_state_dict(best_state)
    hist["train_seconds"] = time.time() - t0
    return model, hist


def regression_metrics(y_true, y_pred):
    """Metrics in cars / 15 min over all sites and time steps."""
    err = y_pred - y_true
    mask = y_true > 5                            # avoid MAPE blow-up on near-zero flow
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {"MAE": float(np.mean(np.abs(err))),
            "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAPE_%": float(np.mean(np.abs(err[mask] / y_true[mask])) * 100),
            "R2": float(1 - ss_res / ss_tot)}


@torch.no_grad()
def predict(model, X, device, batch_size=512):
    model.eval()
    out = [model(torch.from_numpy(X[i:i + batch_size]).to(device)).cpu().numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out)


def evaluate(model, d, device):
    """Test-set metrics in original units, plus a persistence baseline
    (next reading = last reading) that any model should beat."""
    pred = np.clip(inverse_scale(predict(model, d["X_test"], device), d["data_min"], d["data_max"]), 0, None)
    true = inverse_scale(d["y_test"], d["data_min"], d["data_max"])
    naive = inverse_scale(d["X_test"][:, -1, :], d["data_min"], d["data_max"])
    result = {"GRU": regression_metrics(true, pred),
              "Persistence baseline": regression_metrics(true, naive)}
    # per-site MAE (useful for the insights section)
    mae = np.mean(np.abs(pred - true), axis=0)
    result["per_site_MAE"] = {str(s): float(m) for s, m in zip(d["sites"], mae)}
    return result, true, pred


def save_plots(hist, d, true, pred, out_dir, n_sites=3):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - skipping plots")
        return
    plt.figure(figsize=(6, 4))
    plt.plot(hist["train"], label="train")
    plt.plot(hist["val"], label="validation")
    plt.xlabel("Epoch"); plt.ylabel("MSE (scaled)"); plt.title("GRU loss"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150); plt.close()

    for i, site in enumerate(d["sites"][:n_sites]):
        plt.figure(figsize=(9, 3.5))
        plt.plot(d["t_test"][:288], true[:288, i], label="actual")     # first 3 test days
        plt.plot(d["t_test"][:288], pred[:288, i], label="GRU")
        plt.ylabel("Cars / 15 min"); plt.title(f"SCATS {site} - test period"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"prediction_{site}.png"), dpi=150)
        plt.close()


class TrafficPredictor:
    """Load a trained checkpoint and forecast flow for every site.
    History comes from processed_data.csv (unscaled flows from data_processing.py)."""

    def __init__(self, checkpoint, flows_csv, device="cpu"):
        ck = torch.load(checkpoint, map_location=device)
        cfg = ck["config"]
        self.lookback = int(ck["lookback"])
        if int(ck["horizon"]) != 1:
            raise ValueError("TrafficPredictor assumes HORIZON = 1")
        self.sites = list(ck["sites"])
        self.lo, self.hi = ck["data_min"], ck["data_max"]
        self.span = np.where(self.hi - self.lo == 0, 1, self.hi - self.lo)
        self.flows = pd.read_csv(flows_csv, index_col=0, parse_dates=True)
        self.flows.columns = self.flows.columns.astype(str).str.zfill(4)
        self.flows = self.flows[self.sites]
        self.device = device
        self.model = GRUNet(len(self.sites), cfg["hidden_size"], cfg["num_layers"], cfg["dropout"])
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(device).eval()

    @torch.no_grad()
    def forecast(self, timestamp, steps=1):
        """Cars/15 min for `steps` consecutive slots starting at `timestamp`.
        Readings before `timestamp` are real; later steps feed on predictions."""
        t0 = self.flows.index.get_loc(pd.Timestamp(timestamp))
        if t0 < self.lookback:
            raise ValueError("Not enough history before the requested time.")
        hist = (self.flows.values[t0 - self.lookback:t0] - self.lo) / self.span
        out = []
        for _ in range(steps):
            x = torch.tensor(hist[None], dtype=torch.float32, device=self.device)
            p = self.model(x).cpu().numpy()[0]
            out.append(np.clip(p * self.span + self.lo, 0, None))
            hist = np.vstack([hist[1:], p])
        out = np.array(out)                                  # (steps, n_sites)
        return {s: out[:, i].tolist() for i, s in enumerate(self.sites)}


def main():
    p = argparse.ArgumentParser(description="Train a GRU traffic flow model")
    p.add_argument("--data", default=".", help="folder containing ml_data.npz")
    p.add_argument("--config", help="optional JSON file overriding the defaults")
    p.add_argument("--out", default="results/gru")
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", type=type(v), default=None)
    args = p.parse_args()

    cfg = dict(DEFAULTS)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    cfg.update({k: getattr(args, k) for k in DEFAULTS if getattr(args, k) is not None})

    d = load_ml_data(os.path.join(args.data, "ml_data.npz"))
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(d['sites'])} sites | train {len(d['y_train'])} val {len(d['y_val'])} "
          f"test {len(d['y_test'])} windows | device {device}")

    model, hist = train_model(d, cfg, device)
    metrics, true, pred = evaluate(model, d, device)
    metrics["train_seconds"] = hist["train_seconds"]
    metrics["epochs_run"] = len(hist["train"])
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_site_MAE"}, indent=2))

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    torch.save({"state_dict": model.state_dict(), "config": cfg,
                "sites": [str(s) for s in d["sites"]],
                "data_min": d["data_min"], "data_max": d["data_max"],
                "lookback": int(d["lookback"]), "horizon": int(d["horizon"])},
               os.path.join(args.out, "train_gru.pt"))
    save_plots(hist, d, true, pred, args.out)
    print(f"Saved model, metrics and plots to {args.out}/")


if __name__ == "__main__":
    main()