"""Permit-timeline prediction from the raw event log.

Reads `permit_timeline_events.csv` (4,445 events across ~1,230 projects) directly,
matches application→decision permit cycles per project, and produces a kNN+KM
forecast for a candidate site.

For each historical project we look for the earliest matched cycle of
  rezoning_applied            → rezoning_council_decision
  site_plan_applied           → site_plan_approved
  conditional_use_permit_applied  → conditional_use_permit_decision
  special_use_permit_applied      → special_use_permit_decision
  air_permit_applied          → air_permit_issued
  state_environmental_review_initiated → state_environmental_review_decision

Projects with a matched application→decision pair contribute an *event* of
length (decision_date − application_date). Projects with an application but no
decision yet contribute a *right-censored* observation of (last_event − application).
Projects with no permit application observed at all are skipped (we have no
permit clock to fit). Kaplan-Meier on the K-nearest-by-(lat, lon, mw, state)
gives a censoring-aware median + CI.

Sonnet 4.6 is used only for narrative `key_factors`; numerics and similar cases
come from the kNN/KM directly.

API:
    predict(mw_capacity, lat, lon, pollution_cost_usd_per_year=None) -> dict

CLI:
    python predict.py --mw 200 --lat 38.95 --lon -77.45 --pollution-cost 1500000
"""
from __future__ import annotations

import argparse
import csv as csvmod
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime, date
from functools import lru_cache
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent
EVENTS_CSV = ROOT / "final" / "permit_timeline_events.csv"
MODEL = "claude-sonnet-4-6"

# Application → decision pairings the dataset uses. Each pair represents one regulatory
# review cycle from filing to a binding government decision.
PERMIT_PAIRS: dict[str, str] = {
    "rezoning_applied": "rezoning_council_decision",
    "site_plan_applied": "site_plan_approved",
    "conditional_use_permit_applied": "conditional_use_permit_decision",
    "special_use_permit_applied": "special_use_permit_decision",
    "air_permit_applied": "air_permit_issued",
    "state_environmental_review_initiated": "state_environmental_review_decision",
}
_APPLICATION_EVENTS = set(PERMIT_PAIRS.keys())

# kNN tuning. K=20 is large enough to give 8-15 same-state neighbors in populous states
# (TX, GA, VA) and falls back to nearby-state neighbors when the home-state pool is small.
KNN_K = 20
DIST_WEIGHT_MILES = 100.0
DIST_WEIGHT_MW = 50.0
DIST_PENALTY_DIFFERENT_STATE = 1.5  # ≈ 150 miles equivalent — keeps cross-state out unless local pool is sparse

# Dataset-wide fallback anchors (computed at startup from all matched cycles).
_GLOBAL_ANCHORS: dict | None = None


# ---------------- Anthropic key ----------------

def _api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    keyfile = os.path.expanduser("~/.anthropic_key")
    if os.path.exists(keyfile):
        return open(keyfile).read().strip()
    raise RuntimeError("Set ANTHROPIC_API_KEY or place key in ~/.anthropic_key")


# ---------------- Parsing ----------------

_NUM_RE = re.compile(r"\d[\d,]*\.?\d*")


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _parse_mw(s: str | None) -> float | None:
    if not s:
        return None
    s = s.strip()
    if s in ("", "None", "Unknown"):
        return None
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


# ---------------- Project assembly ----------------

@lru_cache(maxsize=1)
def _projects() -> list[dict]:
    """Aggregate the raw event log into one record per project.

    Each record carries: lat, lon, state, mw, name, city, status,
    `events` (sorted list of (date, event_type, decision)), and a `cycle` describing
    the project's first matched permit cycle:
      - ('event', days, permit_type)   — application + decision both observed
      - ('censored', days, permit_type) — application but no decision yet
      - None                            — no permit application observed at all
    """
    by_key: dict[tuple[str, str], dict] = defaultdict(lambda: {"evs": [], "meta": {}})
    for r in csvmod.DictReader(EVENTS_CSV.open()):
        key = (
            (r.get("project_facility_name") or "").strip(),
            (r.get("project_city") or "").strip(),
        )
        d = _parse_date(r.get("date"))
        if not d:
            continue
        by_key[key]["evs"].append((d, r.get("event_type") or "", r.get("decision") or ""))
        m = by_key[key]["meta"]
        if "lat" not in m:
            try:
                lat = float(r.get("project_lat") or "")
                lon = float(r.get("project_long") or "")
            except ValueError:
                continue
            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            m.update({
                "lat": lat,
                "lon": lon,
                "state": (r.get("project_state") or "").strip(),
                "mw": _parse_mw(r.get("project_mw")),
                "name": (r.get("project_facility_name") or "").strip(),
                "city": (r.get("project_city") or "").strip(),
                "status": (r.get("project_status") or "").strip(),
                "operator": (r.get("project_operator") or "").strip(),
            })

    projects: list[dict] = []
    for key, data in by_key.items():
        meta = data["meta"]
        if "lat" not in meta:
            continue
        events = sorted(data["evs"])

        # Index events by type
        by_type: dict[str, list[date]] = defaultdict(list)
        for d, t, _ in events:
            by_type[t].append(d)

        # Find earliest matched application → decision pair (first cycle to complete)
        cycle: tuple[str, int, str] | None = None
        best_event_days: int | None = None
        best_event_perm: str | None = None
        for app_t, dec_t in PERMIT_PAIRS.items():
            apps = by_type.get(app_t, [])
            decs = by_type.get(dec_t, [])
            if not apps or not decs:
                continue
            first_app = min(apps)
            later_decs = [d for d in decs if d >= first_app]
            if not later_decs:
                continue
            days = (min(later_decs) - first_app).days
            if days <= 0:
                continue
            if best_event_days is None or days < best_event_days:
                best_event_days = days
                best_event_perm = app_t
        if best_event_days is not None and best_event_perm is not None:
            cycle = ("event", best_event_days, best_event_perm)
        else:
            # No matched cycle — but if there's an unfulfilled application, censor at last-event
            apps_present = [(t, by_type[t]) for t in _APPLICATION_EVENTS if by_type.get(t)]
            if apps_present and events:
                first_app_d = min(d for _, ds in apps_present for d in ds)
                last_d = max(d for d, _, _ in events)
                days_since = (last_d - first_app_d).days
                if days_since > 0:
                    earliest_app_type = min(apps_present, key=lambda x: min(x[1]))[0]
                    cycle = ("censored", days_since, earliest_app_type)

        projects.append({
            **meta,
            "events": events,
            "cycle": cycle,
        })
    return projects


def _global_anchors() -> dict:
    """Median + percentiles of all matched permit cycles, computed once."""
    global _GLOBAL_ANCHORS
    if _GLOBAL_ANCHORS is not None:
        return _GLOBAL_ANCHORS
    days = sorted(
        p["cycle"][1] for p in _projects()
        if p["cycle"] and p["cycle"][0] == "event"
    )
    if not days:
        _GLOBAL_ANCHORS = {"n": 0, "median": 100, "p25": 30, "p75": 250, "p90": 400}
        return _GLOBAL_ANCHORS

    def pct(p: float) -> int:
        idx = max(0, min(len(days) - 1, int(round((len(days) - 1) * p / 100))))
        return int(days[idx])

    _GLOBAL_ANCHORS = {
        "n": len(days),
        "median": pct(50),
        "p25": pct(25),
        "p75": pct(75),
        "p90": pct(90),
    }
    return _GLOBAL_ANCHORS


# ---------------- Geometry ----------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3959.0
    la1r, la2r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(la1r) * math.cos(la2r) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _infer_state(lat: float, lon: float) -> str | None:
    projs = _projects()
    if not projs:
        return None
    near = sorted(projs, key=lambda p: _haversine(lat, lon, p["lat"], p["lon"]))[:5]
    states = [p["state"] for p in near if p.get("state")]
    return max(set(states), key=states.count) if states else None


# ---------------- Kaplan-Meier ----------------

def _kaplan_meier(events: list[tuple[float, bool]]) -> dict:
    """KM survival fit. events: list of (time, is_event=True if approval observed).

    Returns dict with p25 / median / p75 (times where S(t) crosses 0.75 / 0.5 / 0.25),
    plus n_events, n_total, max_observed_time, final_survival.
    A percentile is None if S(t) never crosses that threshold within the observed window.
    """
    if not events:
        return {"p25": None, "median": None, "p75": None, "n_events": 0,
                "n_total": 0, "max_observed_time": 0, "final_survival": 1.0}
    ordered = sorted(events, key=lambda e: (e[0], not e[1]))
    s = 1.0
    n_at_risk = len(ordered)
    targets: dict[float, float | None] = {0.75: None, 0.5: None, 0.25: None}
    n_events = 0
    last_t = ordered[-1][0]

    i = 0
    while i < len(ordered) and n_at_risk > 0:
        t = ordered[i][0]
        d = c = 0
        while i < len(ordered) and ordered[i][0] == t:
            if ordered[i][1]:
                d += 1
            else:
                c += 1
            i += 1
        if d > 0:
            old_s = s
            s *= (n_at_risk - d) / n_at_risk
            n_events += d
            for thresh in (0.75, 0.5, 0.25):
                if targets[thresh] is None and old_s > thresh >= s:
                    targets[thresh] = t
        n_at_risk -= d + c

    return {
        "p25": targets[0.75],
        "median": targets[0.5],
        "p75": targets[0.25],
        "final_survival": s,
        "n_events": n_events,
        "n_total": len(ordered),
        "max_observed_time": last_t,
    }


# ---------------- kNN ----------------

def _knn_predict(mw_capacity: float, lat: float, lon: float, k: int = KNN_K) -> dict:
    projs = _projects()
    inferred_state = _infer_state(lat, lon)

    # Score every project (regardless of whether it has a cycle); we'll filter cycle
    # availability inside KM, not here, so the K nearest can include censored neighbors.
    scored = []
    for p in projs:
        mw = p["mw"]
        if mw is None or mw <= 0:
            continue
        miles = _haversine(lat, lon, p["lat"], p["lon"])
        mw_gap = abs(mw - mw_capacity)
        same_state = p["state"] and p["state"] == inferred_state
        state_pen = 0.0 if same_state else DIST_PENALTY_DIFFERENT_STATE
        score = miles / DIST_WEIGHT_MILES + mw_gap / DIST_WEIGHT_MW + state_pen
        scored.append((score, miles, mw_gap, p))
    scored.sort(key=lambda x: x[0])
    neighbors = scored[:k]

    # Build KM input from neighbors that have permit-clock data
    km_events: list[tuple[float, bool]] = []
    for _, _, _, p in neighbors:
        cyc = p.get("cycle")
        if cyc is None:
            continue
        kind, days, _ = cyc
        if days <= 0:
            continue
        km_events.append((float(days), kind == "event"))

    km = _kaplan_meier(km_events)
    anchors = _global_anchors()

    # Decide source: if local KM has enough events for a stable median, use it. With n<3
    # events the local median is dominated by individual observations (degenerate CI), so
    # we blend with the global anchor: keep the local point but widen the CI to global.
    MIN_LOCAL_EVENTS = 3
    if km["median"] is not None and km["n_events"] >= MIN_LOCAL_EVENTS:
        days_source = "km_local"
        median_days = int(round(km["median"]))
        p25 = int(round(km["p25"])) if km["p25"] is not None else max(int(median_days * 0.4), 1)
        p75 = (
            int(round(km["p75"])) if km["p75"] is not None
            else int(round(max(km["max_observed_time"], median_days * 2)))
        )
    elif km["median"] is not None:
        # Local KM has 1-2 events. Use the local median but widen the CI to global anchors
        # so we don't claim fake precision off n=1.
        days_source = "km_local_widened"
        median_days = int(round(km["median"]))
        p25 = anchors["p25"]
        p75 = anchors["p90"]
    elif km["p25"] is not None:
        days_source = "km_partial"
        median_days = anchors["median"]
        p25 = int(round(km["p25"]))
        p75 = anchors["p90"]
    else:
        days_source = "global_fallback"
        median_days = anchors["median"]
        p25 = anchors["p25"]
        p75 = anchors["p90"]

    # p_approved: KM-style "ever approved within observation window"
    if km["n_total"] > 0:
        p_approved = km["n_events"] / km["n_total"]
    else:
        p_approved = 0.0
    n = max(1, km["n_total"])
    z = 1.28
    denom = 1 + z * z / n
    centre = (p_approved + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p_approved * (1 - p_approved) / n + z * z / (4 * n * n)) / denom
    p_low = max(0.0, centre - spread)
    p_high = min(1.0, centre + spread)

    neighbor_summary = [
        {
            "name": p["name"],
            "city": p["city"],
            "state": p["state"],
            "status": p["status"],
            "operator": p.get("operator", ""),
            "mw": mw_val,
            "miles": round(miles_val, 1),
            "mw_gap": round(mw_gap_val, 1),
            "cycle": p["cycle"],  # tuple or None
        }
        for _, miles_val, mw_gap_val, p in neighbors
        for mw_val in [p["mw"]]
    ]

    return {
        "median_days": median_days,
        "days_p25": p25,
        "days_p75": p75,
        "days_source": days_source,
        "p_approved": round(p_approved, 3),
        "p_approved_ci_low": round(p_low, 3),
        "p_approved_ci_high": round(p_high, 3),
        "k": len(neighbors),
        "n_km_observations": km["n_total"],
        "n_events": km["n_events"],
        "inferred_state": inferred_state,
        "nearest_distance_miles": round(neighbors[0][1], 2) if neighbors else None,
        "neighbors": neighbor_summary,
        "global_anchors": anchors,
    }


# ---------------- LLM key-factors ----------------

NARRATIVE_TOOL = {
    "name": "report_key_factors",
    "description": "List 3-5 drivers of the kNN+KM permit-timeline prediction.",
    "input_schema": {
        "type": "object",
        "properties": {
            "key_factors": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 5,
                "description": "Each factor ≤ 12 words, no padding.",
            },
        },
        "required": ["key_factors"],
    },
}

NARRATIVE_SYSTEM = """You explain pre-computed kNN+Kaplan-Meier permit-timeline forecasts.
The number of days and probability of approval are already calculated by a survival model
over historical data center projects with matched application→decision permit cycles. Your
only job is to identify 3-5 key factors that drive the prediction (each ≤ 12 words).

Be terse. No prose. No preamble. Output ONLY via the report_key_factors tool."""


def _build_similar_cases(knn: dict, n: int = 5) -> list[dict]:
    """Top-n nearest by score, deduped by (name, city). Real cycle info when available."""
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for c in knn["neighbors"]:
        key = (c["name"].lower(), c["city"].lower())
        if key in seen:
            continue
        seen.add(key)
        outcome = c["status"] or "unknown"
        bits = [f"{c['miles']:.0f} mi", f"{c['mw']:.0f} MW"]
        if c["state"]:
            bits.append(c["state"])
        if c["cycle"]:
            kind, days, perm = c["cycle"]
            short_perm = perm.split("_applied")[0].split("_initiated")[0]
            if kind == "event":
                bits.append(f"{short_perm} {days}d")
            else:
                bits.append(f"{short_perm} pending {days}d+")
        out.append({"name": c["name"], "why_similar": " · ".join(bits), "outcome": outcome})
        if len(out) >= n:
            break
    return out


def _llm_key_factors(
    mw_capacity: float, lat: float, lon: float,
    pollution_cost: float | None, knn: dict, similar_cases: list[dict],
) -> list[str]:
    similar_lines = "\n".join(
        f"- {c['name']} ({c['outcome']}): {c['why_similar']}" for c in similar_cases
    )
    user_msg = (
        f"CANDIDATE PROJECT:\n"
        f"  mw_capacity = {mw_capacity}\n"
        f"  lat = {lat}, lon = {lon}\n"
        f"  pollution_cost_usd_per_year = {pollution_cost}\n"
        f"  inferred_state = {knn['inferred_state']}\n\n"
        f"PREDICTION (pre-computed — explain, do NOT recompute):\n"
        f"  days = {knn['median_days']} (range {knn['days_p25']}-{knn['days_p75']})\n"
        f"  p_approved = {knn['p_approved']:.2f}\n\n"
        f"SIMILAR HISTORICAL CASES:\n{similar_lines}\n\n"
        f"Return 3-5 key_factors via the tool. Each ≤ 12 words. Focus on substantive drivers"
        f" (location, MW, pollution, similar-case outcomes). Do NOT mention sample size,"
        f" data sparsity, model fallbacks, or 'insufficient data' — those are meta and useless."
    )
    client = anthropic.Anthropic(api_key=_api_key())
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=[
            {"type": "text", "text": NARRATIVE_SYSTEM, "cache_control": {"type": "ephemeral"}},
        ],
        tools=[NARRATIVE_TOOL],
        tool_choice={"type": "tool", "name": "report_key_factors"},
        messages=[{"role": "user", "content": user_msg}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "report_key_factors":
            return list(block.input.get("key_factors", []))
    raise RuntimeError("No key_factors returned")


# ---------------- Public API ----------------

def predict(
    mw_capacity: float, lat: float, lon: float,
    pollution_cost_usd_per_year: float | None = None,
) -> dict:
    """Hybrid kNN+KM permit-timeline prediction.

    Numerics from KM over the K-nearest historical projects (by lat/lon/MW/state).
    Similar cases ranked directly from kNN. Sonnet 4.6 generates only key_factors.
    """
    # Don't fall back to pollution.py's social_cost_usd_per_year here — that includes
    # CO2/CH4/N2O climate damage costs ($190/ton CO2 etc.), while the frontend's
    # "health-impact social cost" is COBRA's PM2.5+O3-only number. Mixing them feeds
    # Sonnet a different (much larger) figure than what the user sees on screen.
    # If the caller doesn't pass it, leave it None so Sonnet treats it as unknown.
    knn = _knn_predict(mw_capacity, lat, lon)
    similar_cases = _build_similar_cases(knn)
    key_factors = _llm_key_factors(
        mw_capacity, lat, lon, pollution_cost_usd_per_year, knn, similar_cases
    )

    return {
        "p_approved": knn["p_approved"],
        "p_approved_ci_low": knn["p_approved_ci_low"],
        "p_approved_ci_high": knn["p_approved_ci_high"],
        "expected_days_to_first_approval": knn["median_days"],
        "days_ci_low": knn["days_p25"],
        "days_ci_high": knn["days_p75"],
        "key_factors": key_factors,
        "most_similar_cases": similar_cases,
        "derived_context": {
            "inferred_state": knn["inferred_state"],
            "nearest_distance_miles": knn["nearest_distance_miles"],
            "k": knn["k"],
            "n_km_events": knn["n_events"],
            "n_km_observations": knn["n_km_observations"],
            "days_source": knn["days_source"],
            "global_anchors": knn["global_anchors"],
        },
    }


# ---------------- CLI ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mw", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--pollution-cost", type=float)
    args = ap.parse_args()
    print(json.dumps(predict(args.mw, args.lat, args.lon, args.pollution_cost), indent=2))
