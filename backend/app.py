from flask import Flask, request, jsonify
import pandas as pd
import jax.numpy as jnp

from data import CobraData
from compute import Compute
from util import prepare_emissions
from predict import predict as predict_permit_timeline

app = Flask(__name__)

# SOURCEINDX(1..N) → FIPS, ordered by SOURCEINDX so it aligns with the per-county result vector.
_counties = pd.read_json("data/counties.json").sort_values("SOURCEINDX")
COUNTY_FIPS = [f"{int(f):05d}" for f in _counties["FIPS"].tolist()]
COUNTY_NAMES = [f"{r.CYNAME}, {r.STNAME}" for r in _counties.itertuples()]


@app.post("/compute")
def compute_effects():
    payload = request.json or {}
    lat = payload["lat"]
    lon = payload["lon"]
    total_power = payload["total_power"]
    generators = payload.get("generators")

    if generators:
        data, base_emissions, modified_emissions = prepare_emissions(
            lat, lon, total_power, generators=generators,
        )
    else:
        # Legacy single-generator fallback for older clients.
        generator_power = payload.get("generator_power", 0)
        generator_is_diesel = payload.get("fuel") == "Diesel"
        data, base_emissions, modified_emissions = prepare_emissions(
            lat, lon, total_power,
            generator_power=generator_power,
            generator_is_diesel=generator_is_diesel,
        )

    # Calculate the difference in pollutants
    compute = Compute(data)

    base_aq_vectors = compute.vectorize(base_emissions)
    control_aq_vectors = compute.vectorize(modified_emissions)
    delta_pm = compute.compute_pm(control_aq_vectors) - compute.compute_pm(
        base_aq_vectors
    )
    delta_o3 = compute.compute_o3(control_aq_vectors) - compute.compute_o3(
        base_aq_vectors
    )

    # Compute health impacts using the deltas
    impact = compute.compute_valuation(delta_pm, delta_o3)

    total_costs_by_county = (jnp.vstack(impact.values()).sum(axis=0)).tolist()

    naaqs_idx = (jnp.nonzero(compute.compute_pm(control_aq_vectors) > 35)[0]).tolist()
    naaqs_fips = [COUNTY_FIPS[i] for i in naaqs_idx if 0 <= i < len(COUNTY_FIPS)]

    return jsonify({
        "health_cost_by_county": total_costs_by_county,
        "county_fips": COUNTY_FIPS[:len(total_costs_by_county)],
        "county_names": COUNTY_NAMES[:len(total_costs_by_county)],
        "naaqs_violations": [i + 1 for i in naaqs_idx],  # 1-indexed sourceindx
        "naaqs_violation_fips": naaqs_fips,
    })


def _first_present(payload: dict, *keys):
    for k in keys:
        if k in payload and payload[k] is not None:
            return payload[k]
    return None


def _to_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@app.post("/predict-permit")
def predict_permit_endpoint():
    payload = request.json or {}
    try:
        lat = float(_first_present(payload, "lat", "latitude"))
        lon = float(_first_present(payload, "lon", "lng", "longitude"))
        mw = float(_first_present(
            payload, "mw_capacity", "mwCapacity", "mw", "total_power", "totalPower",
        ))
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"missing or invalid lat/lon/mw_capacity: {e}"}), 400

    pollution_cost = _to_float(_first_present(
        payload, "pollution_cost_usd_per_year", "pollutionCostUsdPerYear",
        "pollution_cost", "pollutionCost",
    ))
    square_footage = _to_float(_first_present(
        payload, "square_footage", "squareFootage", "sqft",
        "square_feet", "squareFeet", "facility_size_sqft", "facilitySizeSqft",
    ))

    generators = _first_present(payload, "generators", "generatorList")
    if generators is not None and not isinstance(generators, list):
        generators = None

    try:
        result = predict_permit_timeline(
            mw, lat, lon, pollution_cost, square_footage, generators,
        )
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    # Echo what the server actually parsed, so frontend mismatches are debuggable.
    result["_request_received"] = {
        "lat": lat, "lon": lon, "mw_capacity": mw,
        "square_footage": square_footage,
        "pollution_cost_usd_per_year": pollution_cost,
        "generators": generators,
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run()
