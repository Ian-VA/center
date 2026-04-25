from flask import Flask, request, jsonify
import pandas as pd

from center.pollution import datacenter_pollution
from data import CobraData
from compute import Compute
from util import estimate_county, most_common_fuel, generator_pollution

app = Flask(__name__)


@app.post("/compute")
def compute_effects():
    lat = request.json["lat"]
    lon = request.json["lon"]
    total_power = request.json["total_power"]
    generator_power = request.json["generator_power"]
    generator_is_diesel = request.json["fuel"] == "Diesel"

    sectors = pd.read_json("data/sectors.json")

    county = estimate_county(lat, lon)
    #The most common fuel used by the grid, needed to calculate stack height
    common_fuel = most_common_fuel(lat, lon)
    if common_fuel == "coal":
        grid_fuel_id = 544 #Bit coal
    elif common_fuel == "gas":
        grid_fuel_id = 545 #Natural gas
    else:
        grid_fuel_id = 548 #Distillate oil

    grid_tier_ids = list(
        sectors[sectors["ID"] == grid_fuel_id][
            ["TIER1", "TIER2", "TIER3"]
        ].iloc[0]
    )

    grid_emissions = datacenter_pollution(lat, lon,total_power - generator_power)

    counties = pd.read_json("data/counties.json")
    # Map the county FIPS codes to source indexes
    # Safe assumption is to assume all grid emissions are in the same county
    sourceindx = counties[counties["FIPS"] == county["fips"]]["SOURCEINDX"][0]
    # Map the sector ID to the tier IDs

    data = CobraData()
    raw_base = data.load_emissions_base()
    base_emissions = data.summarize_emissions(raw_base)

    #Note: EPA data doesn't have PM25, VOC, SOA
    grid_emissions = grid_emissions["emissions"]
    grid_raw = data.modify_emissions(
        raw_base, grid_emissions, grid_tier_ids, [sourceindx]
    )

    if generator_is_diesel:
        generator_fuel_id = 548
    else:
        generator_fuel_id = 545 #Natural Gas

    generator_tier_ids = list(
        sectors[sectors["ID"] == generator_fuel_id][
            ["TIER1", "TIER2", "TIER3"]
        ].iloc[0]
    )

    generator_emissions = generator_pollution(generator_power, generator_is_diesel)

    generator_raw = data.modify_emissions(
        grid_raw, generator_emissions, generator_tier_ids, [sourceindx]
    )

    modified_emissions = data.summarize_emissions(generator_raw)

    # Calculate the difference in pollutants
    compute = Compute(data)

    base_aq_vectors = compute.vectorize(base_emissions)
    control_aq_vectors = compute.vectorize(modified_emissions)
    delta_pm = compute.compute_pm(base_aq_vectors) - compute.compute_pm(
        control_aq_vectors
    )
    delta_o3 = compute.compute_o3(base_aq_vectors) - compute.compute_o3(
        control_aq_vectors
    )

    # Compute health impacts using the deltas
    impact = compute.compute_valuation(delta_pm, delta_o3)

    return jsonify(impact)


if __name__ == '__main__':
    app.run()
