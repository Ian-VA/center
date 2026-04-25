"""Feature engineering for the permit-AFT model.

Single source of truth used by BOTH training (apply to all 1,178 historical projects) and
inference (apply to a candidate project's 4 user inputs). Same function, same feature order.

Public:
    derive_features(mw_capacity, lat, lon, pollution_cost_usd_per_year=None) -> dict[str, float]
    FEATURE_NAMES — ordered tuple of feature names, used to construct numpy rows
"""
from __future__ import annotations
import math
import os
from functools import lru_cache
from pathlib import Path

import duckdb

from pollution import datacenter_pollution

ROOT = Path(__file__).parent
ECHO_DB = ROOT / "echo_bulk" / "permit_pilot.duckdb"

# --- ISO / RTO mapping (matches predict.py) ---
STATE_TO_ISO = {
    "VA":"PJM","WV":"PJM","MD":"PJM","DE":"PJM","DC":"PJM","PA":"PJM","NJ":"PJM","OH":"PJM",
    "KY":"PJM","IN":"PJM","MI":"PJM","IL":"MISO_PJM",
    "NC":"NonISO_SE","SC":"NonISO_SE","TN":"NonISO_SE","AL":"NonISO_SE","MS":"NonISO_SE",
    "GA":"NonISO_SE","FL":"NonISO_SE",
    "TX":"ERCOT",
    "OK":"SPP","KS":"SPP","NE":"SPP","AR":"SPP_MISO","LA":"SPP_MISO","MO":"SPP_MISO",
    "ND":"MROW","SD":"MROW","MN":"MROW","IA":"MROW","WI":"MROW",
    "NY":"NYISO","MA":"ISONE","CT":"ISONE","ME":"ISONE","VT":"ISONE","NH":"ISONE","RI":"ISONE",
    "CA":"CAISO",
    "WA":"NWPP","OR":"NWPP","ID":"NWPP","MT":"NWPP","WY":"NWPP","UT":"NWPP",
    "CO":"NonISO_W","AZ":"NonISO_W","NV":"NonISO_W","NM":"NonISO_W",
    "AK":"NonISO_W","HI":"NonISO_W",
}
# One-hot dimensions for ISO. Kept compact (top markets + Other) to keep features/rows ratio sane.
ISO_DIMS = ["PJM","ERCOT","CAISO","MROW","NonISO_SE","NYISO","Other"]

# eGRID subregions: top-9 by US DC presence, rest lumped into "Other".
EGRID_DIMS = ["RFCE","RFCW","ERCT","MROW","CAMX","SRSO","SRVC","SRMV","NWPP","Other"]

DC_NAICS_PREFIXES = ("518210","518111","518112","541513","541512","541519",
                     "517311","517312","517410","517919")


# --- Cached spatial indices ---

@lru_cache(maxsize=1)
def _echo_lookup_table():
    """Returns numpy arrays of (lats, lons, naics_strings) for ALL ECHO facilities."""
    import numpy as np
    con = duckdb.connect(str(ECHO_DB), read_only=True)
    rows = con.execute("""
      SELECT FAC_LAT, FAC_LONG, COALESCE(FAC_NAICS_CODES, '') AS naics
      FROM echo
      WHERE FAC_LAT IS NOT NULL AND FAC_LONG IS NOT NULL
    """).fetchall()
    con.close()
    lats = np.array([r[0] for r in rows], dtype=np.float64)
    lons = np.array([r[1] for r in rows], dtype=np.float64)
    naics = [r[2] for r in rows]
    has_dc_naics = np.array([any(p in n for p in DC_NAICS_PREFIXES) for n in naics], dtype=np.bool_)
    return lats, lons, has_dc_naics


@lru_cache(maxsize=1)
def _fractracker_neighbors():
    """Returns lat/lon/state/status arrays for FracTracker projects (for local-DC density features)."""
    import numpy as np
    con = duckdb.connect()
    rows = con.execute(f"""
      SELECT TRY_CAST(lat AS DOUBLE) AS plat,
             TRY_CAST("long" AS DOUBLE) AS plong,
             state, status
      FROM read_csv_auto('{ROOT}/echo_bulk/fractracker.csv', sample_size=-1)
      WHERE TRY_CAST(lat AS DOUBLE) IS NOT NULL
        AND TRY_CAST("long" AS DOUBLE) IS NOT NULL
    """).fetchall()
    con.close()
    lats = np.array([r[0] for r in rows])
    lons = np.array([r[1] for r in rows])
    states = [r[2] for r in rows]
    statuses = [r[3] for r in rows]
    return lats, lons, states, statuses


# --- Distance helpers ---

def _haversine_vec_miles(lat0, lon0, lats, lons):
    """Vectorized haversine: scalar (lat0, lon0) to numpy arrays (lats, lons), in miles."""
    import numpy as np
    R = 3959.0
    la0r = math.radians(lat0); lo0r = math.radians(lon0)
    larr = np.radians(lats); loarr = np.radians(lons)
    dlat = larr - la0r; dlon = loarr - lo0r
    a = np.sin(dlat / 2) ** 2 + np.cos(la0r) * np.cos(larr) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# --- Feature derivation ---

FEATURE_NAMES = (
    "mw_numeric_mid", "log_mw", "mw_quality_flag",
    "lat", "lon",
    *(f"iso_{x}" for x in ISO_DIMS),
    *(f"egrid_{x}" for x in EGRID_DIMS),
    "co2_intensity_lb_per_mwh",
    "grid_coal_pct", "grid_gas_pct", "grid_nuclear_pct",
    "nearest_echo_distance_miles", "nearest_echo_dc_naics_within_1mi",
    "local_dc_count_25mi", "local_proposed_count_25mi",
    "local_operating_count_25mi", "local_cancelled_count_25mi",
    "local_approval_rate_25mi",
    "log_pollution_cost", "pollution_cost_per_mw",
    "mw_x_is_PJM", "pollution_x_local_cancelled",
)


def derive_features(
    mw_capacity: float,
    lat: float,
    lon: float,
    pollution_cost_usd_per_year: float | None = None,
    mw_quality_flag: int = 0,
) -> dict[str, float]:
    """Expand the user inputs into the full feature vector the AFT model trains on.

    mw_quality_flag (training-only signal; defaults to 0 at inference):
        0 = measured (parsed from FracTracker mw_capacity)
        1 = imputed from size_rank category
        2 = imputed from sqft heuristic
        3 = median fallback
    """
    import numpy as np

    f: dict[str, float] = {}

    # MW
    f["mw_numeric_mid"] = float(mw_capacity)
    f["log_mw"] = math.log1p(mw_capacity)
    f["mw_quality_flag"] = float(mw_quality_flag)

    # Raw lat/lon
    f["lat"] = float(lat); f["lon"] = float(lon)

    # ISO / state — derive state from nearest FracTracker neighbor
    lats, lons, states, statuses = _fractracker_neighbors()
    dists = _haversine_vec_miles(lat, lon, lats, lons)
    nearest_idx = int(np.argmin(dists))
    inferred_state = states[nearest_idx] if nearest_idx < len(states) else None
    iso = STATE_TO_ISO.get(inferred_state, "Unknown")
    for d in ISO_DIMS:
        f[f"iso_{d}"] = 1.0 if iso == d else 0.0

    # Pollution + grid context (eGRID)
    pol = datacenter_pollution(lat, lon, mw_capacity)
    sub = pol["geo"]["subregion_code"] or "Other"
    if sub not in EGRID_DIMS:
        sub = "Other"
    for d in EGRID_DIMS:
        f[f"egrid_{d}"] = 1.0 if sub == d else 0.0
    f["co2_intensity_lb_per_mwh"] = pol["emissions"]["CO2"]["factor_lb_per_MWh"] or 0.0
    mix = pol["resource_mix_pct"]
    f["grid_coal_pct"] = mix.get("coal") or 0.0
    f["grid_gas_pct"] = mix.get("gas") or 0.0
    f["grid_nuclear_pct"] = mix.get("nuclear") or 0.0

    # Pollution cost (auto-derive if not provided)
    if pollution_cost_usd_per_year is None:
        pollution_cost_usd_per_year = pol["social_cost_usd_per_year"] or 0.0
    f["log_pollution_cost"] = math.log1p(max(0.0, pollution_cost_usd_per_year))
    f["pollution_cost_per_mw"] = pollution_cost_usd_per_year / max(1.0, mw_capacity)

    # ECHO industrial-neighborhood signal
    elats, elons, ehas_dc = _echo_lookup_table()
    edists = _haversine_vec_miles(lat, lon, elats, elons)
    nearest_e = int(np.argmin(edists))
    f["nearest_echo_distance_miles"] = float(edists[nearest_e])
    within_1mi = edists < 1.0
    f["nearest_echo_dc_naics_within_1mi"] = float(np.any(ehas_dc[within_1mi]))

    # Local DC ecosystem (25mi radius FracTracker neighbors)
    nearby = dists < 25.0
    nearby_statuses = [s for s, m in zip(statuses, nearby) if m]
    f["local_dc_count_25mi"] = float(len(nearby_statuses))
    f["local_proposed_count_25mi"] = float(sum(1 for s in nearby_statuses if s == "Proposed"))
    f["local_operating_count_25mi"] = float(sum(1 for s in nearby_statuses if s == "Operating"))
    f["local_cancelled_count_25mi"] = float(sum(1 for s in nearby_statuses if s == "Cancelled"))
    n_terminal = sum(1 for s in nearby_statuses
                     if s in ("Operating", "Approved/Permitted/Under construction",
                              "Expanding", "Cancelled", "Suspended"))
    n_approved = sum(1 for s in nearby_statuses
                     if s in ("Operating", "Approved/Permitted/Under construction", "Expanding"))
    f["local_approval_rate_25mi"] = n_approved / max(1, n_terminal)

    # Interactions
    f["mw_x_is_PJM"] = f["log_mw"] * f["iso_PJM"]
    f["pollution_x_local_cancelled"] = f["log_pollution_cost"] * f["local_cancelled_count_25mi"]

    return f


def features_to_array(features_dict: dict[str, float]):
    """Convert dict to numpy row in the canonical FEATURE_NAMES order."""
    import numpy as np
    return np.array([features_dict.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float64)


if __name__ == "__main__":
    # Quick smoke test
    import json
    f = derive_features(mw_capacity=200, lat=38.95, lon=-77.45)
    print(json.dumps(f, indent=2))
    print(f"\n{len(FEATURE_NAMES)} features total")
