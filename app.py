from flask import Flask, request, jsonify
import pandas as pd
import jax.numpy as jnp

from center.pollution import datacenter_pollution
from data import CobraData
from compute import Compute
from util import prepare_emissions

app = Flask(__name__)


@app.post("/compute")
def compute_effects():
    lat = request.json["lat"]
    lon = request.json["lon"]
    total_power = request.json["total_power"]
    generator_power = request.json["generator_power"]
    generator_is_diesel = request.json["fuel"] == "Diesel"

    data, base_emissions, modified_emissions = prepare_emissions(
        lat, lon, total_power, generator_power, generator_is_diesel
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



    return jsonify({
        "health_cost_by_county": total_costs_by_county,
        "naaqs_violations": (jnp.nonzero(compute.compute_pm(control_aq_vectors) > 35)[0] + 1).tolist() #Ambient PM2.5 must not exceed 35μg/m3
    })


if __name__ == '__main__':
    app.run()
