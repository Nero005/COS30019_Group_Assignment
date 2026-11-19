"""
XGBoost regression model for traffic flow prediction (COS30019 Assignment 2B).

Predicts the traffic flow (number of cars in the next 15 minutes) at a SCATS
location, using XGBRegressor with the squared-error regression objective.

Steps:
  1. load_scats()      read the SCATS October 2006 data and turn it into long format
  2. make_features()   build the inputs (recent traffic, time of day, day of week, location)
  3. split_by_date()   train on the earlier days, test on the last week (no random split)
  4. train_model()     fit XGBRegressor, using early stopping on a validation set
  5. evaluate()        MAE, RMSE, NRMSE and R2, compared with a simple baseline
  6. predict_flow()    predict one location at one date and time (for the route finder)

Run:  python xgb_regression.py "Scats Data October 2006.xls"
"""

import re
import sys
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# ---- settings (change these here, or move them to the config file later) ----
DATA_FILE = "Scats Data October 2006.xls"
MODEL_FILE = "xgb_regression_model.json"
TEST_START = "2006-10-25"          # test = 25 to 31 October, train = 1 to 24 October
VALID_START = "2006-10-22"         # last 3 training days are used for early stopping
LAGS = [1, 2, 3, 4]                # previous 15-minute intervals (up to 1 hour back)
DAY_LAG = 96                       # same time on the previous day (96 intervals per day)


def load_scats(path=DATA_FILE):
    """Read the 'Data' sheet and return one row per location per 15 minutes:
    columns site, location, datetime, flow."""
    df = pd.read_excel(path, sheet_name="Data", header=1)   # real header is on the 2nd row
    volume_cols = [c for c in df.columns if re.fullmatch(r"V\d\d", str(c))]   # V00 to V95

    df = df[["SCATS Number", "Location", "Date"] + volume_cols]
    df = df.rename(columns={"SCATS Number": "site", "Location": "location"})
    df["day"] = pd.to_datetime(df["Date"]).dt.normalize()   # keep only the calendar day

    # some locations have the same day twice; keep the first copy
    df = df.drop_duplicates(subset=["location", "day"], keep="first")

    # wide (96 columns per day) -> long (one row per 15-minute interval)
    long = df.melt(id_vars=["site", "location", "day"], value_vars=volume_cols,
                   var_name="interval", value_name="flow")
    interval = long["interval"].str[1:].astype(int)             # "V05" -> 5
    long["datetime"] = long["day"] + pd.to_timedelta(interval * 15, unit="min")
    long = long[["site", "location", "datetime", "flow"]]
    return long.sort_values(["location", "datetime"]).reset_index(drop=True)


def make_features(long):
    """Add the model inputs. Each location is put on a full 15-minute grid first,
    so a lag always means 'exactly N intervals earlier', even when days are missing."""
    parts = []
    start, end = long["datetime"].min(), long["datetime"].max()
    full_grid = pd.date_range(start, end, freq="15min")

    for location, group in long.groupby("location"):
        g = group.set_index("datetime").reindex(full_grid)      # missing days become NaN
        g.index.name = "datetime"
        g["location"] = location
        g["site"] = group["site"].iloc[0]
        for lag in LAGS:
            g[f"lag_{lag}"] = g["flow"].shift(lag)
        g["lag_day"] = g["flow"].shift(DAY_LAG)
        parts.append(g.reset_index())

    data = pd.concat(parts, ignore_index=True)
    data["time_of_day"] = data["datetime"].dt.hour * 4 + data["datetime"].dt.minute // 15  # 0 to 95
    data["day_of_week"] = data["datetime"].dt.dayofweek                                   # Monday = 0
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["location"] = data["location"].astype("category")   # XGBoost uses it as a category

    # keep only rows where the target and all inputs are known
    needed = ["flow"] + feature_columns()
    return data.dropna(subset=needed).reset_index(drop=True)


def feature_columns():
    return [f"lag_{lag}" for lag in LAGS] + ["lag_day", "time_of_day",
                                             "day_of_week", "is_weekend", "location"]


def split_by_date(data):
    """Time-based split: earlier days for training, the last week for testing."""
    train = data[data["datetime"] < TEST_START]
    test = data[data["datetime"] >= TEST_START]
    fit = train[train["datetime"] < VALID_START]        # used to build the trees
    valid = train[train["datetime"] >= VALID_START]     # used only to decide when to stop
    return fit, valid, test


def train_model(fit, valid):
    """Train XGBRegressor (regression with squared error)."""
    model = XGBRegressor(
        objective="reg:squarederror",   # regression: minimise the squared prediction error
        n_estimators=2000,              # upper limit; early stopping picks the real number
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,                  # each tree sees 80% of the rows
        colsample_bytree=0.8,           # and 80% of the inputs
        tree_method="hist",
        enable_categorical=True,        # lets 'location' be used as a category
        early_stopping_rounds=50,       # stop when validation error stops improving
        random_state=42,
    )
    cols = feature_columns()
    model.fit(fit[cols], fit["flow"],
              eval_set=[(valid[cols], valid["flow"])], verbose=False)
    return model


def scores(actual, predicted):
    rmse = root_mean_squared_error(actual, predicted)
    return {"MAE": mean_absolute_error(actual, predicted),
            "RMSE": rmse,
            "NRMSE": rmse / np.mean(actual),   # same measure as the lecturer's example
            "R2": r2_score(actual, predicted)}


def evaluate(model, test):
    """Score the model on the test week, next to a simple baseline:
    'the next 15 minutes will be the same as the last 15 minutes'."""
    predicted = model.predict(test[feature_columns()])
    predicted = np.clip(predicted, 0, None)               # a flow cannot be negative
    return {"XGBoost regression": scores(test["flow"], predicted),
            "Baseline (last value)": scores(test["flow"], test["lag_1"])}


def predict_flow(model, data, location, when):
    """Predict the flow (cars per 15 minutes) at one location for the 15-minute
    interval starting at 'when', using the real readings before it. Returns None
    if the inputs for that time are missing from the data."""
    when = pd.Timestamp(when)
    row = data[(data["location"] == location) & (data["datetime"] == when)]
    if row.empty:
        return None
    value = model.predict(row[feature_columns()])[0]
    return max(0.0, float(value))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_FILE
    print("Loading", path)
    long = load_scats(path)
    data = make_features(long)
    fit, valid, test = split_by_date(data)
    print(f"Rows: train {len(fit)}, validation {len(valid)}, test {len(test)}")

    model = train_model(fit, valid)
    print("Trees used (after early stopping):", model.best_iteration + 1)

    print("\nTest week results (25-31 October 2006):")
    print(pd.DataFrame(evaluate(model, test)).T.round(3).to_string())

    importance = pd.Series(model.feature_importances_, index=feature_columns())
    print("\nFeature importance:")
    print(importance.sort_values(ascending=False).round(3).to_string())

    model.save_model(MODEL_FILE)
    print("\nModel saved to", MODEL_FILE)

    # example prediction for one location and time in the test week
    location = "WARRIGAL_RD N of TOORAK_RD"
    when = "2006-10-26 08:00"
    actual = test[(test["location"] == location) & (test["datetime"] == when)]["flow"]
    print(f"\nExample: {location} at {when}")
    print("  predicted:", round(predict_flow(model, data, location, when), 1),
          " actual:", actual.iloc[0] if not actual.empty else "n/a")


if __name__ == "__main__":
    main()
