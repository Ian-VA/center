"""Stage 5: LLM-based permit-outcome prediction.

Uses Claude Sonnet 4.6 with prompt-cached full historical dataset (1,330 projects from
data_centers_with_timelines.csv) as in-context reference. Same 4-input signature, structured
JSON output.

API:
    predict(mw_capacity, lat, lon, pollution_cost_usd_per_year=None) -> dict

CLI:
    python predict.py --mw 200 --lat 38.95 --lon -77.45 --pollution-cost 1500000
    python predict.py --validate --n 50      # held-out evaluation, AUC + MAE vs. train.py split
"""
from __future__ import annotations
import argparse
import csv as csvmod
import json
import math
import os
import random
import re
import sys
import time
from functools import lru_cache
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent
TIMELINES_CSV = ROOT / "final" / "data_centers_with_timelines.csv"
MODEL = "claude-sonnet-4-6"
RANDOM_SEED = 42        # match train.py for fair held-out comparison
VAL_FRACTION = 0.20


def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    keyfile = os.path.expanduser("~/.anthropic_key")
    if os.path.exists(keyfile):
        return open(keyfile).read().strip()
    raise RuntimeError("Set ANTHROPIC_API_KEY or place key in ~/.anthropic_key")


@lru_cache(maxsize=1)
def _reference_block() -> str:
    rows = list(csvmod.DictReader(TIMELINES_CSV.open()))
    lines = ["HISTORICAL US DATA CENTER PERMIT TIMELINES (1,330 projects, FracTracker + LLM-extracted events).",
             "Each line: # name (city, state) | status | mw | approved | terminal | days_to_approval | timeline"]
    for i, r in enumerate(rows, 1):
        lines.append(
            f"#{i} {r['facility_name']} ({r['city']}, {r['state']}) | "
            f"status={r.get('fractracker_status','')} | mw={r.get('mw_capacity','') or '?'} | "
            f"approved={r.get('is_approved','')} | terminal={r.get('terminal_outcome','')} | "
            f"days_to_approval={r.get('days_to_first_approval') or 'censored'} | "
            f"timeline: {(r.get('timeline_summary','') or '')[:280]}"
        )
    return "\n".join(lines)


@lru_cache(maxsize=1)
def _fr_index():
    out = []
    for r in csvmod.DictReader(TIMELINES_CSV.open()):
        try:
            la, lo = float(r["lat"]), float(r["long"])
        except (ValueError, KeyError):
            continue
        out.append((la, lo, r["state"], r["facility_name"], r["fractracker_status"]))
    return out


def _haversine(lat1, lon1, lat2, lon2):
    R = 3959.0
    la1r, la2r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(la1r)*math.cos(la2r)*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _derive_geo(lat, lon):
    fr = _fr_index()
    by_d = sorted(fr, key=lambda x: _haversine(lat, lon, x[0], x[1]))
    near5 = by_d[:5]
    states = [n[2] for n in near5]
    state = max(set(states), key=states.count) if states else None
    return {
        "inferred_state": state,
        "nearest_neighbors": [(n[3], n[4]) for n in near5],
        "nearest_distance_miles": round(_haversine(lat, lon, by_d[0][0], by_d[0][1]), 2),
    }


SYSTEM_PROMPT = """You are a US data center permit-outcome analyst. Given a candidate project's numeric features (MW capacity, lat/lon, optional pollution-cost-per-year), reason against the historical dataset of 1,330 US data centers below and predict:
  1. p_approved (0–1)
  2. expected_days_to_first_approval (integer, days)
  3. CIs on both
  4. The 5 most similar historical projects (with reasoning)
  5. Key factors driving the prediction
  6. Plain-language reasoning (4–8 sentences)

Calibration anchors (from the reference data):
- Median days_to_first_approval (uncensored) ≈ 243d (8mo); p25=96, p75=583, p90=1461
- Outcomes: 460 approved, 96 denied, 40 withdrawn, 649 still pending

Match similar cases on MW band, state/ISO, project_status, pollution_cost tier.

Output ONLY via the report_prediction tool."""

PREDICTION_TOOL = {
    "name": "report_prediction",
    "description": "Return the structured permit-outcome prediction.",
    "input_schema": {
        "type": "object",
        "properties": {
            "p_approved": {"type": "number", "minimum": 0, "maximum": 1},
            "p_approved_ci_low": {"type": "number", "minimum": 0, "maximum": 1},
            "p_approved_ci_high": {"type": "number", "minimum": 0, "maximum": 1},
            "expected_days_to_first_approval": {"type": "integer"},
            "days_ci_low": {"type": "integer"},
            "days_ci_high": {"type": "integer"},
            "most_similar_cases": {
                "type": "array", "minItems": 3, "maxItems": 5,
                "items": {"type": "object",
                          "properties": {"name": {"type": "string"},
                                         "why_similar": {"type": "string"},
                                         "outcome": {"type": "string"}},
                          "required": ["name", "why_similar", "outcome"]},
            },
            "key_factors": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 7},
            "reasoning": {"type": "string"},
        },
        "required": ["p_approved", "p_approved_ci_low", "p_approved_ci_high",
                     "expected_days_to_first_approval", "days_ci_low", "days_ci_high",
                     "most_similar_cases", "key_factors", "reasoning"],
    },
}


def predict(mw_capacity: float, lat: float, lon: float,
            pollution_cost_usd_per_year: float | None = None) -> dict:
    """LLM-driven permit-outcome prediction using full historical dataset as cached context."""
    if pollution_cost_usd_per_year is None:
        try:
            from pollution import datacenter_pollution
            pollution_cost_usd_per_year = datacenter_pollution(lat, lon, mw_capacity)["social_cost_usd_per_year"]
        except Exception:
            pass

    geo = _derive_geo(lat, lon)
    user_msg = (
        f"CANDIDATE PROJECT FEATURES:\n"
        f"  mw_capacity = {mw_capacity}\n"
        f"  lat = {lat}, lon = {lon}\n"
        f"  pollution_cost_usd_per_year = {pollution_cost_usd_per_year}\n\n"
        f"DERIVED GEOGRAPHIC CONTEXT:\n"
        f"  inferred_state = {geo['inferred_state']}\n"
        f"  nearest historical neighbors: {geo['nearest_neighbors']}\n"
        f"  nearest_distance_miles = {geo['nearest_distance_miles']}\n\n"
        f"Compare against the 1,330 historical projects above and emit the prediction."
    )
    client = anthropic.Anthropic(api_key=_api_key())
    response = client.messages.create(
        model=MODEL, max_tokens=2048,
        system=[
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": _reference_block(), "cache_control": {"type": "ephemeral"}},
        ],
        tools=[PREDICTION_TOOL],
        tool_choice={"type": "tool", "name": "report_prediction"},
        messages=[{"role": "user", "content": user_msg}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "report_prediction":
            out = dict(block.input)
            out["derived_context"] = geo
            return out
    raise RuntimeError("No prediction returned")


# ----- Held-out validation -----
def validate(n_max=100):
    rows = list(csvmod.DictReader(TIMELINES_CSV.open()))
    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    val_size = int(len(rows) * VAL_FRACTION)
    val_indices = indices[:val_size]

    _NUM = re.compile(r"\d[\d,]*\.?\d*")
    def parse_mw(s):
        if not s or s.strip() in ("", "None", "Unknown"):
            return None
        nums = [float(m.group(0).replace(",", "")) for m in _NUM.finditer(s)]
        if not nums:
            return None
        return (min(nums[0], nums[1]) + max(nums[0], nums[1])) / 2.0 if len(nums) >= 2 else nums[0]

    val_eval = []
    for i in val_indices:
        r = rows[i]
        mw = parse_mw(r.get("mw_capacity", ""))
        if mw is None:
            continue
        try:
            lat = float(r["lat"]); lon = float(r["long"])
        except (ValueError, KeyError):
            continue
        is_app = r.get("is_approved") == "True"
        try:
            days = float(r.get("days_to_first_approval") or "0")
        except ValueError:
            days = 0
        val_eval.append((r, mw, lat, lon, is_app, days))
        if len(val_eval) >= n_max:
            break

    print(f"Running LLM predictions on {len(val_eval)} held-out projects...", file=sys.stderr)
    results = []
    t0 = time.time()
    for i, (r, mw, lat, lon, is_app, days) in enumerate(val_eval, 1):
        try:
            pred = predict(mw, lat, lon)
        except Exception as e:
            print(f"[{i}] ERROR {type(e).__name__}: {e}", file=sys.stderr); continue
        results.append({
            "name": r["facility_name"], "city": r["city"], "state": r["state"],
            "actual_is_approved": is_app, "actual_days_to_approval": days,
            "pred_p_approved": pred["p_approved"],
            "pred_days": pred["expected_days_to_first_approval"],
        })
        if i % 5 == 0:
            print(f"  {i}/{len(val_eval)} ({time.time()-t0:.0f}s, ~${0.08*i:.2f})",
                  file=sys.stderr)

    pos = [r["pred_p_approved"] for r in results if r["actual_is_approved"]]
    neg = [r["pred_p_approved"] for r in results if not r["actual_is_approved"]]
    auc = (sum(1 for p in pos for n in neg if p > n) +
           0.5 * sum(1 for p in pos for n in neg if p == n)) / max(1, len(pos) * len(neg))
    unc = [r for r in results if r["actual_is_approved"] and r["actual_days_to_approval"] > 0]
    mae = sum(abs(r["pred_days"] - r["actual_days_to_approval"]) for r in unc) / max(1, len(unc))

    print(f"\n=== LLM VALIDATION (n={len(results)}) ===", file=sys.stderr)
    print(f"  AUC (is_approved):                {auc:.3f}    [chance = 0.5]", file=sys.stderr)
    print(f"  MAE days (uncensored, n={len(unc)}):  {mae:.0f} ({mae/30.4:.1f} months)", file=sys.stderr)

    out = ROOT / "models" / "llm_validation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"n_validated": len(results), "auc_is_approved": auc,
                               "mae_days_uncensored": mae, "n_uncensored": len(unc),
                               "predictions": results}, indent=2))
    print(f"\nFull log → {out}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--mw", type=float)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--pollution-cost", type=float)
    args = ap.parse_args()

    if args.validate:
        validate(n_max=args.n)
    elif args.mw is not None and args.lat is not None and args.lon is not None:
        print(json.dumps(predict(args.mw, args.lat, args.lon, args.pollution_cost), indent=2))
    else:
        ap.error("Provide --validate, or --mw/--lat/--lon for a single prediction.")
