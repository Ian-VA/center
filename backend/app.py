from flask import Flask, request, jsonify
import pandas as pd
import jax.numpy as jnp
from dataclasses import asdict, fields
from pathlib import Path

from data import CobraData
from compute import Compute
from util import prepare_emissions
from predict import predict as predict_permit_timeline
from resource_usage import DCCalcInputs, calculate_data_center
from price_change import (
    current_demand,
    elasticity,
    price_change as calculate_price_change,
)

ROOT = Path(__file__).parent
app = Flask(__name__)

# SOURCEINDX(1..N) → FIPS, ordered by SOURCEINDX so it aligns with the per-county result vector.
_counties = pd.read_json(ROOT / "data" / "counties.json").sort_values("SOURCEINDX")
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


def _require_float(payload: dict, *keys):
    value = _to_float(_first_present(payload, *keys))
    if value is None:
        joined = "/".join(keys)
        raise ValueError(f"missing or invalid {joined}")
    return value


def _resource_usage_payload(payload: dict) -> dict:
    allowed = {f.name for f in fields(DCCalcInputs)}
    return {k: v for k, v in payload.items() if k in allowed}


@app.post("/resource-usage")
def resource_usage_endpoint():
    payload = request.json or {}
    clean_payload = _resource_usage_payload(payload)

    try:
        inputs = DCCalcInputs(**clean_payload)
        results = calculate_data_center(inputs)
    except (TypeError, ValueError, ZeroDivisionError) as e:
        return jsonify({"error": f"invalid resource usage payload: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    response = asdict(results)
    response["_request_received"] = clean_payload
    return jsonify(response)


@app.post("/price-change")
def price_change_endpoint():
    payload = request.json or {}
    try:
        current_price = _require_float(payload, "current_price", "currentPrice")
        new_demand = _require_float(payload, "new_demand", "newDemand")

        old_demand = _to_float(_first_present(payload, "old_demand", "oldDemand"))
        supply_elasticity = _to_float(_first_present(
            payload, "supply_elasticity", "supplyElasticity",
        ))
        demand_elasticity = _to_float(_first_present(
            payload, "demand_elasticity", "demandElasticity",
        ))

        if old_demand is None or supply_elasticity is None or demand_elasticity is None:
            resource = _first_present(payload, "resource", "utility")
            location = _first_present(payload, "location", "region")
            if not resource or not location:
                raise ValueError(
                    "missing explicit elasticity/demand fields or resource/location defaults"
                )
            if resource not in elasticity:
                raise ValueError(f"unsupported resource: {resource}")
            if location not in current_demand:
                raise ValueError(f"unsupported location: {location}")
            if resource not in current_demand[location]:
                raise ValueError(f"unsupported resource for location: {location}/{resource}")

            old_demand = float(current_demand[location][resource])
            supply_elasticity = float(elasticity[resource]["supply"])
            demand_elasticity = float(elasticity[resource]["demand"])
        else:
            resource = _first_present(payload, "resource", "utility")
            location = _first_present(payload, "location", "region")

        denominator = supply_elasticity - demand_elasticity
        if denominator == 0:
            raise ValueError("supply_elasticity - demand_elasticity cannot be zero")
        if old_demand <= 0:
            raise ValueError("old_demand must be greater than zero")

        changed_price = calculate_price_change(
            current_price,
            old_demand,
            new_demand,
            supply_elasticity,
            demand_elasticity,
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500

    return jsonify({
        "price_change": changed_price,
        "current_price": current_price,
        "old_demand": old_demand,
        "new_demand": new_demand,
        "supply_elasticity": supply_elasticity,
        "demand_elasticity": demand_elasticity,
        "resource": resource,
        "location": location,
        "_request_received": payload,
    })


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
