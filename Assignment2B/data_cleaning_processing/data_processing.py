"""
data_processing.py

  clean:    load the Excel file, fix site ids and dates, drop sparse detectors, fill gaps per
            detector, then sum the approaches -> one flow per SCATS site per 15 minutes.
  prep:      scale to 0-1 (fitted on the training days only), build sliding windows
            (past LOOKBACK readings -> next reading) and split by time into train/val/test.

Outputs (written to output_dir)
  processed_data.csv   cleaned flows (unscaled): one row per 15-min slot, one column per site
  sites.csv            site id, lat, long, location names
  ml_data.npz          windows + scaling info for the ML models (read it with load_ml_data below)
  cleaning_report.txt  counts of everything found/fixed (for the report)

Usage : python data_processing.py [path_to_xls] [output_dir]
Needs : pandas, numpy, and xlrd to read the old .xls format (pip install xlrd)

    from data_processing import load_ml_data, inverse_scale
    d = load_ml_data("ml_data.npz")
    d["X_train"]  shape (samples, LOOKBACK, n_sites), scaled to ~0-1
    d["y_train"]  shape (samples, n_sites)           (the next 15-min reading of every site)
    One site only: X = d["X_train"][:, :, [i]], y = d["y_train"][:, i]   (i = column of that site)
    inverse_scale(prediction, d["data_min"], d["data_max"]) turns scaled values back into cars/15 min.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#settings
SLOTS = [f"V{i:02d}" for i in range(96)]   # V00..V95 = 15-min counts (NOT 'VR Internal ...')
SLOTS_PER_DAY = 96
MIN_DAYS = 15        # drop a detector with fewer days than this
MAX_INTERP = 4       # linearly interpolate gaps up to 4 slots (1 hour)
LOOKBACK = 12        # input window: the last 12 readings (3 hours)
HORIZON = 1          # predict this many steps ahead (1 = the next 15 minutes)
TRAIN_DAYS = 21      # 1-21 Oct   -> train
VAL_DAYS = 5         # 22-26 Oct  -> validation; the remaining days (27-31 Oct) -> test

report = []


def log(msg):
    print(msg)
    report.append(msg)


#cleaning
def load_raw(src):
    """Read the data sheet. The header is on the 2nd row site ids stay text to keep leading 0"""
    try:
        raw = pd.read_excel(src, sheet_name="Data", header=1, dtype={"SCATS Number": str})
    except ImportError as err:
        sys.exit(f"{err}\nReading the old .xls format needs xlrd:  pip install xlrd")
    raw = raw.rename(columns={"SCATS Number": "site"}).copy()   # copy() defragments the wide frame
    raw["site"] = raw["site"].str.strip().str.zfill(4)
    raw["date"] = pd.to_datetime(raw["Date"]).dt.normalize()    # Date column carries a stray 00:15
    log(f"Loaded {len(raw)} rows, {raw['site'].nunique()} sites, {raw['Location'].nunique()} location names")
    log(f"Missing cells in V00-V95: {int(raw[SLOTS].isna().sum().sum())}; "
        f"negatives: {int((raw[SLOTS] < 0).sum().sum())}")
    return raw


def identify_detectors(raw):
    """Give every detector an id. Some location names repeat inside a site (e.g. 4335 HIGH_ST NE of
    CHARLES_ST has two detectors) so VR Internal Loc is part of the key otherwise they would
    look like duplicates"""
    raw["det"] = raw["site"] + "|" + raw["Location"].str.strip() + "|" + raw["VR Internal Loc"].astype(str)
    dups = raw.duplicated(["det", "date"]).sum()
    log(f"True duplicate (detector, date) rows after using detector id: {dups}")
    return raw.drop_duplicates(["det", "date"])


def drop_sparse_detectors(raw):
    """remove detectors that have fewer than MIN_DAYS days of data"""
    days = raw.groupby("det")["date"].nunique()
    too_few = days[days < MIN_DAYS]
    for det, n in too_few.items():
        log(f"Dropped detector {det}: only {n} day(s) of data")
    raw = raw[~raw["det"].isin(too_few.index)]
    log(f"Detectors with fewer than 31 days (kept, gaps filled): "
        f"{int((days.loc[raw['det'].unique()] < 31).sum())}")
    return raw


def fill_detector(series):
    """Fill the gaps of one detector's full timeline
    - gaps of <= MAX_INTERP slots: straight line between the values either side
    - longer gaps (e.g. whole missing day): median of the same weekday and time slot
    - last resort: median of that time slot over all days"""
    gap = series.isna()
    gap_len = gap.groupby((gap != gap.shift()).cumsum()).transform("sum")   # length of the gap each NaN is in
    short_gap = gap & (gap_len <= MAX_INTERP)
    # interpolate only the short gaps (interpolate(limit=...) alone would also half-fill long gaps)
    s = series.where(~short_gap, series.interpolate(limit_area="inside"))
    s = s.fillna(s.groupby([s.index.dayofweek, s.index.time]).transform("median"))
    return s.fillna(s.groupby(s.index.time).transform("median"))


def build_site_matrix(raw):
    """Wide -> long -> full October timeline per detector -> fill gaps -> sum detectors per site.
    Gaps are filled PER DETECTOR before summing, otherwise a missing approach looks like a traffic drop."""
    long = raw.melt(id_vars=["site", "det", "date"], value_vars=SLOTS, var_name="slot", value_name="flow")
    long["ts"] = long["date"] + pd.to_timedelta(long["slot"].str[1:].astype(int) * 15, unit="m")

    full_index = pd.date_range("2006-10-01 00:00", "2006-10-31 23:45", freq="15min")   # 31*96 = 2976
    det_site = raw.drop_duplicates("det").set_index("det")["site"]

    filled, n_missing = {}, 0
    for det, g in long.groupby("det"):
        s = g.set_index("ts")["flow"].reindex(full_index).astype(float)
        n_missing += int(s.isna().sum())
        filled[det] = fill_detector(s)
    log(f"Cells missing after reindexing to a full October timeline (missing days etc.): {n_missing}")

    det_df = pd.DataFrame(filled)
    site_df = det_df.T.groupby(det_site.loc[det_df.columns]).sum().T.round().astype(int)
    site_df.index.name = "timestamp"
    site_df = site_df[sorted(site_df.columns)]
    log(f"Final matrix: {site_df.shape[0]} time steps x {site_df.shape[1]} sites")
    return site_df


def flag_zero_runs(raw):
    """Report (do not remove) sites with >= 2 hours of zeros. These are mostly overnight on minor
    approaches, so they are probably genuine low volume rather than detector faults."""
    longest = (raw[SLOTS] == 0).astype(int).T.rolling(8).sum().max()   # longest 2h window of zeros per row
    runs8 = (longest >= 8).groupby(raw["site"]).sum()
    log("Sites with long (>=2h) zero runs (mostly overnight on minor approaches, so probably genuine low volume): "
        + ", ".join(f"{s} ({n} rows)" for s, n in runs8[runs8 > 0].items()))
    log(f"Overall share of zero readings: {(raw[SLOTS] == 0).values.mean():.2%}")


def build_sites_table(raw, site_df):
    """One row per site: median lat/long over its detectors (0/0 rows ignored) and its location names."""
    loc = raw.copy()
    loc.loc[(loc["NB_LATITUDE"] == 0) | (loc["NB_LONGITUDE"] == 0), ["NB_LATITUDE", "NB_LONGITUDE"]] = np.nan
    n_bad = int(loc["NB_LATITUDE"].isna().sum())
    sites = loc.groupby("site").agg(lat=("NB_LATITUDE", "median"), long=("NB_LONGITUDE", "median"),
                                    locations=("Location", lambda x: "; ".join(sorted(set(x))))).reset_index()
    log(f"Rows with 0/0 coordinates: {n_bad}; sites with no valid coordinates: "
        f"{sites.loc[sites['lat'].isna(), 'site'].tolist()}  -> fill from Traffic_Count_Locations_with_LONG_LAT.csv")
    return sites[sites["site"].isin(site_df.columns)]


#prepare for ml
def make_windows(values, first_target, last_target):
    """Sliding windows. For every target index t in [first_target, last_target):
         input  = values[t-HORIZON+1-LOOKBACK : t-HORIZON+1]   (only readings BEFORE the target)
         target = values[t]
    Inputs may reach back into the previous split (history is allowed); targets never do."""
    X, y, idx = [], [], []
    for t in range(first_target, last_target):
        end = t - HORIZON + 1
        start = end - LOOKBACK
        if start < 0:
            continue
        X.append(values[start:end])
        y.append(values[t])
        idx.append(t)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), np.array(idx)


def prepare_ml_data(site_df):
    """Split by time (never shuffled), scale with train-only statistics, then window each split."""
    values = site_df.values.astype(np.float32)               # (time steps, sites)
    n_days = len(values) // SLOTS_PER_DAY
    train_end = TRAIN_DAYS * SLOTS_PER_DAY
    val_end = train_end + VAL_DAYS * SLOTS_PER_DAY
    log(f"Split by time: train days 1-{TRAIN_DAYS}, validation days {TRAIN_DAYS + 1}-{TRAIN_DAYS + VAL_DAYS}, "
        f"test days {TRAIN_DAYS + VAL_DAYS + 1}-{n_days}")

    # min-max scaling per site, fitted on the TRAINING rows only (so no test information leaks in)
    data_min = values[:train_end].min(axis=0)
    data_max = values[:train_end].max(axis=0)
    span = np.where(data_max - data_min == 0, 1, data_max - data_min) 
    scaled = (values - data_min) / span

    out = {}
    for name, lo, hi in [("train", 0, train_end), ("val", train_end, val_end), ("test", val_end, len(values))]:
        X, y, idx = make_windows(scaled, lo, hi)
        out[f"X_{name}"], out[f"y_{name}"] = X, y
        out[f"t_{name}"] = site_df.index.values[idx]         # timestamp of every target
        log(f"  {name}: X {X.shape}, y {y.shape}")
    out["sites"] = np.array(site_df.columns, dtype=str)      # plain text, so np.load needs no pickle
    out["data_min"], out["data_max"] = data_min, data_max
    out["lookback"], out["horizon"] = LOOKBACK, HORIZON
    log(f"Window: last {LOOKBACK} readings ({LOOKBACK * 15} min) -> reading {HORIZON * 15} min ahead; "
        f"scaled with train-only min/max per site")
    return out


def load_ml_data(path="ml_data.npz"):
    """Load ml_data.npz into a plain dict of arrays (see the module docstring for the keys)."""
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def inverse_scale(scaled, data_min, data_max):
    """Turn scaled values back into cars per 15 minutes. `scaled` can be (..., n_sites) or one site
    (pass that site's data_min / data_max, e.g. d['data_min'][i])."""
    return scaled * (data_max - data_min) + data_min


#main
def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "Scats_Data_October_2006.xls"
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(src)
    raw = identify_detectors(raw)
    raw = drop_sparse_detectors(raw)
    site_df = build_site_matrix(raw)
    flag_zero_runs(raw)
    sites = build_sites_table(raw, site_df)
    ml = prepare_ml_data(site_df)

    site_df.to_csv(out_dir / "processed_data.csv")
    sites.to_csv(out_dir / "sites.csv", index=False)
    np.savez(out_dir / "ml_data.npz", **ml)
    log(f"Saved processed_data.csv, sites.csv, ml_data.npz, cleaning_report.txt to {out_dir.resolve()}")
    (out_dir / "cleaning_report.txt").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
