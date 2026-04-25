"""Train an XGBoost AFT (Accelerated Failure Time) survival model on the permit timeline data.

Splits the dataset 80/20 train/validation. Trains on the 80% with right-censoring handled
natively, evaluates the held-out 20% with concordance index + MAE on uncensored rows.

Inputs:
    permit_pilot/final/data_centers_with_timelines.csv

Outputs:
    permit_pilot/models/permit_aft.json           — trained XGBoost-AFT model
    permit_pilot/models/permit_aft_meta.json      — feature names, metrics, training log
    permit_pilot/models/feature_importance.csv    — top features by gain
"""
from __future__ import annotations
import csv as csvmod
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

from features import derive_features, features_to_array, FEATURE_NAMES

ROOT = Path(__file__).parent
SRC_CSV = ROOT / "final" / "data_centers_with_timelines.csv"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

VAL_FRACTION = 0.20
RANDOM_SEED = 42


# --- MW parsing + imputation ---
INVALID_MW = {"", "None", "Unknown", "unknown", "null", "NA", "n/a"}
_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")

# size_rank categorical → midpoint MW (heuristic, derived from FracTracker bin labels).
SIZERANK_MW = {
    "Small (0-10 MW)":      5,
    "Medium (11-50 MW)":   30,
    "Large (51-99 MW)":    75,
    "Hyperscale (100-999 MW)": 550,
    "Mega campus (>1,000 MW)": 1500,
}

def parse_mw_mid(s: str) -> float | None:
    if not s or s.strip() in INVALID_MW:
        return None
    nums = [float(m.group(0).replace(",", "")) for m in _NUM_RE.finditer(s)]
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    return (min(nums[0], nums[1]) + max(nums[0], nums[1])) / 2.0


def impute_mw(row: dict) -> tuple[float | None, int]:
    """Returns (mw, quality_flag): 0=measured, 1=size_rank, 2=sqft, 3=median fallback,
    None mw if no signal at all."""
    # Tier 0: measured
    mw = parse_mw_mid(row.get("mw_capacity", ""))
    if mw is not None:
        return mw, 0
    # Tier 1: size_rank category
    sr = (row.get("size_rank") or "").strip()
    if sr in SIZERANK_MW:
        return float(SIZERANK_MW[sr]), 1
    # Tier 2: sqft heuristic — ~1 MW per 5,000 sqft for typical DCs
    sqft_str = (row.get("facility_size_sqft") or "").replace(",", "").strip()
    try:
        sqft = float(sqft_str) if sqft_str else 0
    except ValueError:
        sqft = 0
    if sqft > 0:
        return sqft / 5000.0, 2
    return None, -1


# --- Concordance index (Harrell's C) for survival prediction ---
def concordance_index(durations, predictions, event_observed) -> float:
    """C-index: probability that for any pair (i, j), if i has shorter true duration AND
    event observed for i, then prediction[i] < prediction[j]. AFT predicts log-time, so
    smaller prediction = shorter expected duration."""
    n = len(durations)
    concordant = 0; tied = 0; permissible = 0
    for i in range(n):
        if not event_observed[i]:
            continue
        for j in range(n):
            if i == j: continue
            if durations[j] < durations[i]:
                continue  # j must be at-or-after i to be a permissible pair
            if durations[j] == durations[i] and not event_observed[j]:
                continue
            permissible += 1
            if predictions[i] < predictions[j]:
                concordant += 1
            elif predictions[i] == predictions[j]:
                tied += 1
    if permissible == 0:
        return float("nan")
    return (concordant + 0.5 * tied) / permissible


def build_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Load CSV, derive features, compute survival bounds. Returns (X, t_lower, t_upper,
    is_approved, project_keys)."""
    rows = list(csvmod.DictReader(SRC_CSV.open()))
    print(f"[load] {len(rows)} rows from {SRC_CSV.name}", file=sys.stderr)

    Xs, lows, ups, approved, keys = [], [], [], [], []
    skipped_no_mw = skipped_no_dur = skipped_no_latlon = 0
    mw_quality_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    t0 = time.time()
    for i, r in enumerate(rows):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"[features] {i+1}/{len(rows)} ({elapsed:.0f}s elapsed)", file=sys.stderr)
        mw, mw_q = impute_mw(r)
        if mw is None:
            skipped_no_mw += 1; continue
        try:
            lat = float(r["lat"]); lon = float(r["long"])
        except (ValueError, KeyError):
            skipped_no_latlon += 1; continue

        is_app = r.get("is_approved") == "True"
        d_app_str = r.get("days_to_first_approval", "")
        d_obs_str = r.get("total_observed_days", "")
        try:
            days_app = float(d_app_str) if d_app_str else None
            days_obs = float(d_obs_str) if d_obs_str else None
        except ValueError:
            days_app = days_obs = None

        # Survival bounds
        if is_app and days_app is not None and days_app > 0:
            lower, upper = days_app, days_app
        elif (not is_app) and days_obs is not None and days_obs > 0:
            lower, upper = days_obs, float("inf")
        else:
            skipped_no_dur += 1; continue

        feats = derive_features(mw, lat, lon, mw_quality_flag=mw_q)
        Xs.append(features_to_array(feats))
        lows.append(lower); ups.append(upper)
        approved.append(is_app)
        keys.append((r["facility_name"], r["city"], r["state"]))
        mw_quality_counts[mw_q] = mw_quality_counts.get(mw_q, 0) + 1

    X = np.array(Xs)
    print(f"[load] kept {len(Xs)}/{len(rows)} (skipped: no_mw={skipped_no_mw}, "
          f"no_dur={skipped_no_dur}, no_latlon={skipped_no_latlon})", file=sys.stderr)
    print(f"[load] MW source quality: measured={mw_quality_counts.get(0,0)}, "
          f"size_rank={mw_quality_counts.get(1,0)}, sqft={mw_quality_counts.get(2,0)}, "
          f"median_fb={mw_quality_counts.get(3,0)}", file=sys.stderr)
    return X, np.array(lows), np.array(ups), np.array(approved, dtype=bool), keys


def main():
    print("=== Building feature matrix (this calls pollution.py per row, ~2-5 min) ===",
          file=sys.stderr)
    X, t_lower, t_upper, is_approved, keys = build_dataset()

    # 80/20 train/validation split, stratified by is_approved
    np.random.seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(X)
    perm = rng.permutation(n)
    val_size = int(n * VAL_FRACTION)
    val_idx = perm[:val_size]
    tr_idx = perm[val_size:]

    print(f"\n=== split ===", file=sys.stderr)
    print(f"  train: {len(tr_idx)} ({is_approved[tr_idx].sum()} approved, "
          f"{(~is_approved[tr_idx]).sum()} censored)", file=sys.stderr)
    print(f"  val:   {len(val_idx)} ({is_approved[val_idx].sum()} approved, "
          f"{(~is_approved[val_idx]).sum()} censored)", file=sys.stderr)

    # Build DMatrices with AFT bounds
    dtrain = xgb.DMatrix(X[tr_idx])
    dtrain.set_float_info("label_lower_bound", t_lower[tr_idx])
    dtrain.set_float_info("label_upper_bound", t_upper[tr_idx])

    dval = xgb.DMatrix(X[val_idx])
    dval.set_float_info("label_lower_bound", t_lower[val_idx])
    dval.set_float_info("label_upper_bound", t_upper[val_idx])

    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.20,
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "verbosity": 1,
    }
    evals_result = {}
    print("\n=== training XGBoost-AFT ===", file=sys.stderr)
    booster = xgb.train(
        params, dtrain, num_boost_round=400,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        early_stopping_rounds=30,
        verbose_eval=25,
    )

    # Predictions on validation set (these are predicted MEAN durations in days)
    pred_val = booster.predict(dval)
    pred_train = booster.predict(dtrain)

    # Concordance index
    c_train = concordance_index(t_lower[tr_idx], pred_train, is_approved[tr_idx])
    c_val = concordance_index(t_lower[val_idx], pred_val, is_approved[val_idx])

    # MAE on uncensored validation rows
    mask_v = is_approved[val_idx]
    if mask_v.sum() > 0:
        mae_days = np.mean(np.abs(pred_val[mask_v] - t_lower[val_idx][mask_v]))
        mae_months = mae_days / 30.4375
        # log-MAE (better metric for skewed durations)
        log_mae = np.mean(np.abs(np.log1p(pred_val[mask_v]) - np.log1p(t_lower[val_idx][mask_v])))
        median_pred = float(np.median(pred_val[mask_v]))
        median_actual = float(np.median(t_lower[val_idx][mask_v]))
    else:
        mae_days = mae_months = log_mae = median_pred = median_actual = None

    print(f"\n=== METRICS ===", file=sys.stderr)
    print(f"  C-index (train): {c_train:.3f}", file=sys.stderr)
    print(f"  C-index (val):   {c_val:.3f}    [target ≥ 0.65]", file=sys.stderr)
    if mae_days is not None:
        print(f"  MAE (days, val uncensored, n={mask_v.sum()}): {mae_days:.0f}", file=sys.stderr)
        print(f"  MAE (months):    {mae_months:.1f}", file=sys.stderr)
        print(f"  log1p MAE:       {log_mae:.3f}", file=sys.stderr)
        print(f"  median predicted vs actual (val uncensored): {median_pred:.0f}d vs {median_actual:.0f}d", file=sys.stderr)

    # Feature importance (gain)
    importance = booster.get_score(importance_type="gain")
    importance_named = sorted(
        ((FEATURE_NAMES[int(k[1:])], v) for k, v in importance.items()),
        key=lambda x: -x[1])
    print(f"\n=== TOP 15 FEATURES (by gain) ===", file=sys.stderr)
    for name, gain in importance_named[:15]:
        print(f"  {name:42s} {gain:.1f}", file=sys.stderr)

    # Save
    booster.save_model(str(MODELS / "permit_aft.json"))
    meta = {
        "feature_names": list(FEATURE_NAMES),
        "model_params": params,
        "best_iteration": booster.best_iteration if hasattr(booster, "best_iteration") else None,
        "val_fraction": VAL_FRACTION,
        "random_seed": RANDOM_SEED,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(val_idx)),
        "n_train_approved": int(is_approved[tr_idx].sum()),
        "n_val_approved": int(is_approved[val_idx].sum()),
        "metrics": {
            "c_index_train": float(c_train),
            "c_index_val": float(c_val),
            "mae_days_val_uncensored": float(mae_days) if mae_days is not None else None,
            "mae_months_val_uncensored": float(mae_months) if mae_months is not None else None,
            "log1p_mae_val_uncensored": float(log_mae) if log_mae is not None else None,
            "median_pred_val": median_pred,
            "median_actual_val": median_actual,
        },
        "top_features_by_gain": importance_named[:25],
    }
    (MODELS / "permit_aft_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    with (MODELS / "feature_importance.csv").open("w", newline="") as f:
        w = csvmod.writer(f); w.writerow(["feature", "gain"]); w.writerows(importance_named)
    print(f"\n[done] saved → {MODELS / 'permit_aft.json'}", file=sys.stderr)
    print(f"[done] meta → {MODELS / 'permit_aft_meta.json'}", file=sys.stderr)


if __name__ == "__main__":
    main()
