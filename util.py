import json
import math
from functools import lru_cache
from pathlib import Path
import openpyxl
import pandas as pd
from data import CobraData
from center.pollution import datacenter_pollution

from center.pollution import DEFAULT_UTILIZATION, DEFAULT_PUE

ROOT = Path(__file__).parent

@lru_cache(maxsize=1)
def _load_counties() -> list[dict]:
    """Load counties data from JSON file."""
    counties_path = ROOT / "data" / "counties.json"
    with open(counties_path, "r") as f:
        return json.load(f)

def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points on Earth in miles."""
    R = 3959.0  # Earth's radius in miles
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def estimate_county(lat: float, lon: float) -> dict:
    """Estimate the county for given lat/lon by finding the closest county centroid."""
    counties = _load_counties()
    best_distance = float("inf")
    best_county = None
    for county in counties:
        clat = county.get("LAT")
        clon = county.get("LON")
        if clat is None or clon is None:
            continue
        distance = _haversine_miles(lat, lon, clat, clon)
        if distance < best_distance:
            best_distance = distance
            best_county = county
    if best_county is None:
        raise ValueError(f"No county found for lat={lat}, lon={lon}")
    return {
        "state": best_county["STNAME"],
        "county": best_county["CYNAME"],
        "fips": best_county["FIPS"],
        "distance_miles": round(best_distance, 2)
    }


EGRID_PATH = ROOT / "center" / "echo_bulk" / "egrid2023.xlsx"
EGRID_URL = "https://www.epa.gov/system/files/documents/2025-01/egrid2023_data_rev1.xlsx"

_RESOURCE_MIX_CODES = {
    "coal": "SRCLPR", "oil": "SROLPR", "gas": "SRGSPR", "nuclear": "SRNCPR",
    "hydro": "SRHYPR", "biomass": "SRBMPR", "wind": "SRWIPR", "solar": "SRSOPR",
    "geothermal": "SRGTPR", "other_fossil": "SROFPR",
    "total_nonrenewable": "SRTNPR", "total_renewable": "SRTRPR",
}

@lru_cache(maxsize=1)
def _open_egrid() -> openpyxl.Workbook:
    if not EGRID_PATH.exists():
        import sys
        import urllib.request
        EGRID_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[eGRID] file missing — downloading {EGRID_URL} → {EGRID_PATH} (~21MB)",
              file=sys.stderr)
        urllib.request.urlretrieve(EGRID_URL, EGRID_PATH)
        print(f"[eGRID] saved {EGRID_PATH.stat().st_size/1e6:.0f} MB", file=sys.stderr)
    return openpyxl.load_workbook(str(EGRID_PATH), read_only=True, data_only=True)

@lru_cache(maxsize=1)
def _subregion_table() -> dict:
    """{SUBRGN: {col_code: value, ...}} from SRL23 sheet."""
    wb = _open_egrid()
    ws = wb["SRL23"]
    rows = list(ws.iter_rows(values_only=True))
    codes = rows[1]
    out = {}
    for r in rows[2:]:
        if not r or not r[1]:
            continue
        d = {c: v for c, v in zip(codes, r) if c}
        sub = d.get("SUBRGN")
        if isinstance(sub, str):
            out[sub] = d
    return out

@lru_cache(maxsize=1)
def _plant_locations() -> list[tuple]:
    """List of (lat, lon, subrgn) for every eGRID plant — used for nearest-plant lookup."""
    wb = _open_egrid()
    ws = wb["PLNT23"]
    rows = list(ws.iter_rows(values_only=True))
    codes = rows[1]
    idx_lat = codes.index("LAT")
    idx_lon = codes.index("LON")
    idx_sub = codes.index("SUBRGN")
    out = []
    for r in rows[2:]:
        if not r:
            continue
        lat = r[idx_lat]; lon = r[idx_lon]; sub = r[idx_sub]
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)) and isinstance(sub, str):
            if math.isfinite(lat) and math.isfinite(lon):
                out.append((lat, lon, sub))
    return out

def lookup_subregion(lat: float, lon: float) -> str:
    """Find eGRID subregion code via the nearest power plant in the eGRID plant list."""
    plants = _plant_locations()
    best_d = float("inf"); best = None
    for plat, plon, sub in plants:
        d = _haversine_miles(lat, lon, plat, plon)
        if d < best_d:
            best_d = d; best = sub
    if best is None:
        raise ValueError(f"No eGRID plant found near lat={lat}, lon={lon}")
    return best

def most_common_fuel(lat: float, lon: float) -> str:
    """Compute the most common fuel used for electricity generation in the EPA subregion for given coordinates."""
    sub_code = lookup_subregion(lat, lon)
    sub = _subregion_table().get(sub_code)
    if not sub:
        raise ValueError(f"No data for subregion {sub_code}")
    
    mix = {}
    for fuel, code in _RESOURCE_MIX_CODES.items():
        if "total_" not in fuel:  # Exclude totals
            v = sub.get(code)
            mix[fuel] = v if isinstance(v, (int, float)) else 0.0
    
    if not mix:
        raise ValueError(f"No resource mix data for subregion {sub_code}")
    
    most_common = max(mix, key=mix.get)
    return most_common

DIESEL_EMISSION_FACTOR = {
    "NOx": 4.41,
    "SO2": 0.29
} #lb/MMBtu
GAS_EMISSION_FACTOR = {
    "NOx": 1.94,
    "SO2": 5.88e-4,
    "VOC": 1.2e-1,
    "PM25": 3.84e-2
} #lb/MMBtu
MWH_TO_MMBTU = 3.41
LB_TO_KG = 0.453592

def generator_pollution(
    mw_capacity: float,
    is_diesel: bool,
    utilization: float = DEFAULT_UTILIZATION,
    pue: float = DEFAULT_PUE
):
    annual_mWh = mw_capacity * 8760.0 * utilization * pue
    annual_MMBtu = annual_mWh * MWH_TO_MMBTU

    if is_diesel:
        emission_factor = DIESEL_EMISSION_FACTOR
    else:
        emission_factor = GAS_EMISSION_FACTOR

    return {
        k: v * annual_MMBtu * LB_TO_KG for k, v in emission_factor.items()
    }


def load_sectors() -> pd.DataFrame:
    return pd.read_json(ROOT / "data" / "sectors.json")


def get_grid_fuel_id(common_fuel: str) -> int:
    if common_fuel == "coal":
        return 544  # Bit coal
    elif common_fuel == "gas":
        return 545  # Natural gas
    else:
        return 548  # Distillate oil


def get_generator_fuel_id(is_diesel: bool) -> int:
    if is_diesel:
        return 548
    else:
        return 545  # Natural Gas


def get_tier_ids(sectors: pd.DataFrame, fuel_id: int) -> list:
    return list(
        sectors[sectors["ID"] == fuel_id][
            ["TIER1", "TIER2", "TIER3"]
        ].iloc[0]
    )


def load_counties() -> pd.DataFrame:
    return pd.read_json(ROOT / "data" / "counties.json")


def get_source_index(counties: pd.DataFrame, fips: str) -> int:
    return counties[counties["FIPS"] == int(fips)]["SOURCEINDX"].iloc[0]


def prepare_emissions(
    lat: float,
    lon: float,
    total_power: float,
    generator_power: float,
    generator_is_diesel: bool
) -> tuple:
    sectors = load_sectors()
    county = estimate_county(lat, lon)
    common_fuel = most_common_fuel(lat, lon)
    grid_fuel_id = get_grid_fuel_id(common_fuel)
    grid_tier_ids = get_tier_ids(sectors, grid_fuel_id)
    grid_emissions = datacenter_pollution(lat, lon, total_power - generator_power)["emissions"]
    grid_emissions = {k: v["tons_per_year"] for k, v in grid_emissions.items()}
    counties = load_counties()
    sourceindx = get_source_index(counties, county["fips"])
    data = CobraData()
    raw_base = data.load_emissions_base()
    base_emissions = data.summarize_emissions(raw_base)
    # Note: EPA data doesn't have PM25, VOC, SOA
    grid_raw = data.modify_emissions(raw_base, grid_emissions, grid_tier_ids, [sourceindx])
    generator_fuel_id = get_generator_fuel_id(generator_is_diesel)
    generator_tier_ids = get_tier_ids(sectors, generator_fuel_id)
    generator_emissions = generator_pollution(generator_power, generator_is_diesel)
    generator_raw = data.modify_emissions(grid_raw, generator_emissions, generator_tier_ids, [sourceindx])
    modified_emissions = data.summarize_emissions(generator_raw)
    return data, base_emissions, modified_emissions
