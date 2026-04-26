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
WIDE_CSV = ROOT / "final" / "data_centers_wide.csv"
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


def _ood_cap(
    mw_capacity: float, square_footage: float | None, dist: dict,
) -> tuple[float | None, str | None]:
    """Hard p_approved cap when the candidate is out-of-distribution.

    LLM prompt rules don't bind reliably for tail-end inputs, so we enforce a
    deterministic ceiling here instead of trusting Sonnet to apply the cap itself.
    Returns (cap, reason). Cap is None when the candidate is in-distribution.
    """
    sqft_d = dist["sqft"]
    mw_d = dist["mw"]
    ratio_d = dist["sqft_per_mw"]
    candidates: list[tuple[float, str]] = []

    if square_footage:
        if square_footage > 5 * sqft_d["max"]:
            candidates.append((0.05,
                f"sqft {square_footage:,.0f} > 5× historical max ({sqft_d['max']:,})"))
        elif square_footage > 2 * sqft_d["max"]:
            candidates.append((0.10,
                f"sqft {square_footage:,.0f} > 2× historical max ({sqft_d['max']:,})"))
        elif square_footage > sqft_d["p99"]:
            candidates.append((0.30,
                f"sqft {square_footage:,.0f} > p99 ({sqft_d['p99']:,})"))

    if mw_capacity > 2 * mw_d["max"]:
        candidates.append((0.05,
            f"MW {mw_capacity:,.0f} > 2× historical max ({mw_d['max']:,})"))
    elif mw_capacity > mw_d["p99"]:
        candidates.append((0.40,
            f"MW {mw_capacity:,.0f} > p99 ({mw_d['p99']:,})"))

    if square_footage and mw_capacity > 0:
        ratio = square_footage / mw_capacity
        if ratio > 3 * ratio_d["p90"]:
            candidates.append((0.20,
                f"sqft/MW {ratio:,.0f} > 3× p90 ({3 * ratio_d['p90']:,})"
                f" — implausibly under-powered"))
        elif ratio < ratio_d["p10"] / 3 and ratio_d["p10"] > 0:
            candidates.append((0.20,
                f"sqft/MW {ratio:,.0f} < p10/3 ({ratio_d['p10'] // 3:,})"
                f" — implausibly over-powered"))

    if not candidates:
        return (None, None)
    cap, reason = min(candidates, key=lambda x: x[0])
    return (cap, reason)


@lru_cache(maxsize=1)
def _scale_distribution() -> dict:
    """Empirical sqft + MW + sqft/MW distributions from the historical dataset.

    Used to ground the LLM's plausibility check on real numbers rather than vibes.
    Falls back to hardcoded estimates if the wide CSV is unavailable.
    """
    fallback = {
        "sqft": {"n": 894, "p50": 200_000, "p90": 1_000_000, "p99": 2_100_000, "max": 2_100_000},
        "mw": {"n": 504, "p50": 100, "p90": 750, "p99": 1_500, "max": 2_000},
        "sqft_per_mw": {"p10": 1_500, "p50": 5_000, "p90": 12_000},
    }
    if not WIDE_CSV.exists():
        return fallback
    sqfts: list[float] = []
    mws: list[float] = []
    ratios: list[float] = []
    try:
        for row in csvmod.DictReader(WIDE_CSV.open()):
            sqft_raw = (row.get("facility_size_sqft") or "").strip().replace(",", "")
            mw_raw = (row.get("project_mw") or row.get("mw") or "").strip()
            sqft = None
            mw = None
            try:
                sqft = float(sqft_raw)
                if sqft <= 0:
                    sqft = None
            except ValueError:
                pass
            mw = _parse_mw(mw_raw)
            if sqft is not None:
                sqfts.append(sqft)
            if mw is not None:
                mws.append(mw)
            if sqft is not None and mw is not None and mw > 0:
                ratios.append(sqft / mw)
    except (OSError, csvmod.Error):
        return fallback

    def pct(arr: list[float], p: float) -> int:
        if not arr:
            return 0
        s = sorted(arr)
        return int(s[min(len(s) - 1, int(round((len(s) - 1) * p / 100)))])

    if not sqfts or not mws:
        return fallback
    return {
        "sqft": {
            "n": len(sqfts), "p50": pct(sqfts, 50), "p90": pct(sqfts, 90),
            "p99": pct(sqfts, 99), "max": int(max(sqfts)),
        },
        "mw": {
            "n": len(mws), "p50": pct(mws, 50), "p90": pct(mws, 90),
            "p99": pct(mws, 99), "max": int(max(mws)),
        },
        "sqft_per_mw": {
            "p10": pct(ratios, 10), "p50": pct(ratios, 50), "p90": pct(ratios, 90),
        },
    }


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


# ---------------- LLM probability assessment ----------------

ASSESSMENT_TOOL = {
    "name": "report_permit_assessment",
    "description": (
        "Produce a permit-approval probability with a confidence band, plus 3-5 key drivers."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "p_approved": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Final probability the project is ever approved (0-1).",
            },
            "p_approved_ci_low": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Lower bound of an ~80% confidence band for p_approved.",
            },
            "p_approved_ci_high": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Upper bound of an ~80% confidence band for p_approved.",
            },
            "key_factors": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "maxItems": 5,
                "description": "Each factor ≤ 12 words, no padding.",
            },
        },
        "required": [
            "p_approved", "p_approved_ci_low", "p_approved_ci_high", "key_factors",
        ],
    },
}

ASSESSMENT_SYSTEM = """You assess US data-center permit approval probability.

The kNN+KM baseline you are given was computed using ONLY the candidate's lat/lon and
MW. It does not see square footage, generators, or pollution cost — so red flags from
those fields can and should override the baseline. Treat the baseline as the prior
when the candidate sits inside the historical distribution; depart from it materially
when out-of-distribution evidence is present.

Hard rules:
  - If candidate sqft > p99 of historical sqft, cap p_approved ≤ 0.35 (this is a
    rarely-seen mega-project; opposition + review burden is acute). If sqft > 2× the
    historical max, cap p_approved ≤ 0.20.
  - If candidate MW > p99 of historical MW, cap p_approved ≤ 0.40.
  - sqft / MW typically falls in [p10, p90] of the historical ratio. If the candidate
    ratio is OUTSIDE that band by ≥ 3×, the project is implausibly proportioned
    (under- or over-powered for its size). Cap p_approved ≤ 0.25 and flag this in
    key_factors. Hyperscale norms: ~5,000 sqft/MW; under 1,500 means under-powered,
    over 15,000 means dramatically under-utilized space.
  - Big diesel backup (>100 MW total) triggers Title V air-permit scrutiny — mark
    down by 0.05–0.15 depending on state.
  - Pollution social cost > $5M/yr signals dirty grid + scrutiny risk — mark down
    proportionally.

When in-distribution and analogs are mostly approved/operating, stay near the kNN
baseline. When out-of-distribution, depart sharply — do NOT politely nudge.

Substantive factors to reason about:
  - kNN baseline (anchor for in-distribution candidates).
  - Scale realism vs the historical sqft/MW/ratio percentiles you'll be given.
  - Location and state regime: VA/TX/GA approve readily; CA/NY/coastal opposition is
    higher; rural counties with utility partnerships approve more.
  - Generator footprint and fuel mix.
  - Pollution social cost magnitude.
  - The 5 nearest neighbors' actual outcomes.

CI band: tight (±0.05–0.10) when many in-distribution analogs agree; wide (±0.15–0.25)
when out-of-distribution or analogs disagree.

key_factors should be grounded PRIMARILY in the 5 nearest historical data centers and
the candidate's location/state regime. Reference specific neighbors by short name when
useful (e.g. "Microsoft Project Fulton (approved, 12mi) supports approval"). Only fall
back to scale/generator/pollution drivers when location+analogs alone don't resolve
the call, or when an out-of-distribution scale signal forces a cap.

Be terse. No prose. Output ONLY via the report_permit_assessment tool."""


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


def _format_generators(generators: list[dict] | None) -> str:
    if not generators:
        return "  generators = none specified\n"
    lines = ["  generators ="]
    for g in generators:
        fuel = g.get("fuel", "?")
        mw = g.get("powerMW", 0) or 0
        mode = g.get("mode", "?")
        rh = g.get("runHours", 0) or 0
        lines.append(f"    - {fuel} {mw} MW, mode={mode}, run_hours={rh}")
    return "\n".join(lines) + "\n"


def _llm_assessment(
    mw_capacity: float, lat: float, lon: float,
    pollution_cost: float | None, square_footage: float | None,
    generators: list[dict] | None,
    knn: dict, similar_cases: list[dict],
) -> dict:
    """Sonnet returns p_approved (+ CI) and key_factors via a forced tool call."""
    similar_lines = "\n".join(
        f"- {c['name']} ({c['outcome']}): {c['why_similar']}" for c in similar_cases
    )
    sqft_line = (
        f"  square_footage = {square_footage:,.0f}\n" if square_footage else ""
    )
    pollution_line = (
        f"  pollution_cost_usd_per_year = {pollution_cost:,.0f}\n"
        if pollution_cost is not None else "  pollution_cost_usd_per_year = unknown\n"
    )
    dist = _scale_distribution()
    sqft_d = dist["sqft"]; mw_d = dist["mw"]; ratio_d = dist["sqft_per_mw"]
    candidate_ratio_line = ""
    if square_footage and mw_capacity > 0:
        ratio = square_footage / mw_capacity
        candidate_ratio_line = f"  candidate_sqft_per_mw = {ratio:,.0f}\n"

    user_msg = (
        f"CANDIDATE PROJECT:\n"
        f"  mw_capacity = {mw_capacity}\n"
        f"{sqft_line}"
        f"{candidate_ratio_line}"
        f"  lat = {lat}, lon = {lon}\n"
        f"{pollution_line}"
        f"  inferred_state = {knn['inferred_state']}\n"
        f"{_format_generators(generators)}\n"
        f"HISTORICAL DISTRIBUTION (for plausibility check):\n"
        f"  sqft (n={sqft_d['n']}): p50={sqft_d['p50']:,}, p90={sqft_d['p90']:,},"
        f" p99={sqft_d['p99']:,}, max={sqft_d['max']:,}\n"
        f"  MW (n={mw_d['n']}): p50={mw_d['p50']:,}, p90={mw_d['p90']:,},"
        f" p99={mw_d['p99']:,}, max={mw_d['max']:,}\n"
        f"  sqft/MW ratio: p10={ratio_d['p10']:,}, p50={ratio_d['p50']:,},"
        f" p90={ratio_d['p90']:,}\n\n"
        f"kNN+KM BASELINE (lat/lon/MW only — does NOT see sqft/generators/pollution):\n"
        f"  baseline_p_approved = {knn['p_approved']:.3f}"
        f" (CI {knn['p_approved_ci_low']:.3f} – {knn['p_approved_ci_high']:.3f},"
        f" n={knn['n_km_observations']}, events={knn['n_events']})\n\n"
        f"5 NEAREST HISTORICAL ANALOGS:\n{similar_lines}\n\n"
        f"Return p_approved + CI band + 3-5 key_factors via the tool."
    )
    client = anthropic.Anthropic(api_key=_api_key())
    response = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=[
            {"type": "text", "text": ASSESSMENT_SYSTEM, "cache_control": {"type": "ephemeral"}},
        ],
        tools=[ASSESSMENT_TOOL],
        tool_choice={"type": "tool", "name": "report_permit_assessment"},
        messages=[{"role": "user", "content": user_msg}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == "report_permit_assessment":
            payload = block.input
            lo = float(payload["p_approved_ci_low"])
            hi = float(payload["p_approved_ci_high"])
            if lo > hi:
                lo, hi = hi, lo
            return {
                "p_approved": float(payload["p_approved"]),
                "p_approved_ci_low": lo,
                "p_approved_ci_high": hi,
                "key_factors": list(payload.get("key_factors", [])),
            }
    raise RuntimeError("No assessment returned")


# ---------------- Public API ----------------

def predict(
    mw_capacity: float, lat: float, lon: float,
    pollution_cost_usd_per_year: float | None = None,
    square_footage: float | None = None,
    generators: list[dict] | None = None,
) -> dict:
    """Hybrid permit-timeline prediction.

    Days come from Kaplan-Meier over the K-nearest historical projects (by lat/lon/MW/state).
    Probability of approval comes from Sonnet 4.6, which sees the kNN baseline as a reference
    anchor plus the candidate's MW, square footage, generators, location, and pollution cost,
    and adjusts for out-of-distribution signals (e.g. a 6M-sqft outlier).
    """
    # Don't fall back to pollution.py's social_cost_usd_per_year here — that includes
    # CO2/CH4/N2O climate damage costs ($190/ton CO2 etc.), while the frontend's
    # "health-impact social cost" is COBRA's PM2.5+O3-only number. Mixing them feeds
    # Sonnet a different (much larger) figure than what the user sees on screen.
    # If the caller doesn't pass it, leave it None so Sonnet treats it as unknown.
    knn = _knn_predict(mw_capacity, lat, lon)
    similar_cases = _build_similar_cases(knn)
    assessment = _llm_assessment(
        mw_capacity, lat, lon,
        pollution_cost_usd_per_year, square_footage, generators,
        knn, similar_cases,
    )

    p = assessment["p_approved"]
    lo = assessment["p_approved_ci_low"]
    hi = assessment["p_approved_ci_high"]
    cap, cap_reason = _ood_cap(mw_capacity, square_footage, _scale_distribution())
    cap_applied = None
    if cap is not None and p > cap:
        # Clamp the point + CI band to respect the OOD ceiling.
        p_capped = cap
        hi_capped = min(hi, cap + 0.05)
        lo_capped = min(lo, p_capped)
        cap_applied = {
            "cap": cap, "reason": cap_reason,
            "llm_p_approved_before_cap": p,
            "llm_p_approved_ci_before_cap": [lo, hi],
        }
        p, lo, hi = p_capped, lo_capped, hi_capped

    return {
        "p_approved": p,
        "p_approved_ci_low": lo,
        "p_approved_ci_high": hi,
        "expected_days_to_first_approval": knn["median_days"],
        "days_ci_low": knn["days_p25"],
        "days_ci_high": knn["days_p75"],
        "key_factors": assessment["key_factors"],
        "most_similar_cases": similar_cases,
        "derived_context": {
            "inferred_state": knn["inferred_state"],
            "nearest_distance_miles": knn["nearest_distance_miles"],
            "k": knn["k"],
            "n_km_events": knn["n_events"],
            "n_km_observations": knn["n_km_observations"],
            "days_source": knn["days_source"],
            "global_anchors": knn["global_anchors"],
            "knn_baseline_p_approved": knn["p_approved"],
            "knn_baseline_p_approved_ci_low": knn["p_approved_ci_low"],
            "knn_baseline_p_approved_ci_high": knn["p_approved_ci_high"],
            "out_of_distribution_cap": cap_applied,
        },
    }


# ---------------- CLI ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mw", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--pollution-cost", type=float)
    ap.add_argument("--sqft", type=float, help="Proposed facility square footage")
    ap.add_argument(
        "--generators", type=str,
        help='JSON list, e.g. \'[{"fuel":"Diesel","powerMW":50,"mode":"backup","runHours":50}]\'',
    )
    args = ap.parse_args()
    gens = json.loads(args.generators) if args.generators else None
    print(json.dumps(
        predict(args.mw, args.lat, args.lon, args.pollution_cost, args.sqft, gens),
        indent=2,
    ))
