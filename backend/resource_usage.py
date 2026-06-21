"""
Data Center Energy Usage Calculator
===================================

Planning-level data center power, cooling, water, cost, and carbon model.

The model favors explicit component equations over flat percentages:
IT equipment, UPS losses, PDU/distribution losses, cooling COP, economizer
operation, water usage effectiveness, and emissions are calculated as
separate terms so assumptions remain visible.

All calculations are planning-level estimates only. Verify with manufacturer
data, metered facility data, current code, AHJ requirements, and qualified
MEP/electrical engineering review before design use.

References
----------
- Server idle-to-peak model: A Review of Power Consumption Models of Servers
  in Data Centers, Applied Energy, 2020, DOI: 10.1016/j.apenergy.2020.114806.
- Aggregate server modeling: Generalizable Machine Learning Models for
  Predicting Data Center Server Power, Efficiency, and Throughput,
  arXiv:2503.06439.
- Modular facility/load chain modeling: A Complete Model for Modular
  Simulation of Data Centre Power Load, arXiv:1804.00703.
- Full-facility PUE/CUE modeling: Prediction of Overall Energy Consumption of
  Data Centers in Different Locations, Sensors, 2022,
  DOI: 10.3390/s22103704.
- Chiller part-load COP: Energy performance analysis of multi-chiller cooling
  systems for data centers concerning progressive loading throughout the
  lifecycle under typical climates, Building Simulation, 2024,
  DOI: 10.1007/s12273-024-1167-9.
- WUE definition: Water Usage Effectiveness (WUE): A Green Grid Data Center
  Sustainability Metric, The Green Grid White Paper #35, 2011.
- Climate/technology WUE modeling: Climate- and technology-specific PUE and
  WUE estimations for U.S. data centers using a hybrid statistical and
  thermodynamics-based approach, Resources, Conservation and Recycling, 2022,
  DOI: 10.1016/j.resconrec.2022.106323.
- Carbon framing: Environmental Burden of United States Data Centers in the
  Artificial Intelligence Era, arXiv:2411.09786.
- EPA eGRID 2023 summary data: https://www.epa.gov/egrid/summary-data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# Reference data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CoolingType:
    id: str
    label: str
    cop_min: float
    cop_max: float
    cop_default: float
    uses_cooling_tower_water: bool


@dataclass(frozen=True)
class ClimateZone:
    id: str
    label: str
    free_cooling_hours: int


@dataclass(frozen=True)
class EGridRegion:
    id: str
    label: str
    factor: float          # lb CO2e / kWh (EPA eGRID 2023 total output rate)


@dataclass(frozen=True)
class RackPreset:
    id: str
    label: str
    peak_kw: float         # peak kW per rack


COOLING_TYPES: list[CoolingType] = [
    CoolingType("dx", "Direct Expansion (DX)", 2.0, 3.8, 3.0, False),
    CoolingType("chilled-water", "Chilled Water", 2.5, 6.0, 4.2, True),
    CoolingType("evaporative", "Evaporative / Indirect Free Cooling", 3.0, 15.0, 8.0, True),
    CoolingType("rear-door", "Rear-Door Heat Exchangers", 3.0, 7.0, 5.0, False),
    CoolingType("immersion", "Immersion Cooling", 4.0, 14.0, 9.0, False),
]

# These retain the existing 8-zone interface. The free-cooling hours remain
# planning defaults; replace with ASHRAE/climate-specific hourly data when
# available.
CLIMATE_ZONES: list[ClimateZone] = [
    ClimateZone("hot-humid", "Hot-Humid (Miami, Houston)", 500),
    ClimateZone("hot-dry", "Hot-Dry (Phoenix, Las Vegas)", 1_200),
    ClimateZone("mixed-humid", "Mixed-Humid (Atlanta, DC)", 2_500),
    ClimateZone("mixed-dry", "Mixed-Dry (Denver, Boise)", 3_200),
    ClimateZone("cool-humid", "Cool-Humid (Chicago, Boston)", 4_500),
    ClimateZone("cool-dry", "Cool-Dry (Salt Lake, Reno)", 5_000),
    ClimateZone("cold", "Cold (Minneapolis, Fargo)", 6_000),
    ClimateZone("subarctic", "Subarctic (Fairbanks, Anchorage)", 7_500),
]

EGRID_REGIONS: list[EGridRegion] = [
    EGridRegion("CAMX", "CA/Mexico (CAMX)", 0.429983),
    EGridRegion("ERCT", "Texas (ERCT)", 0.736629),
    EGridRegion("MROE", "Midwest Reliability (MROE)", 1.404963),
    EGridRegion("NEWE", "New England (NEWE)", 0.543178),
    EGridRegion("NWPP", "Northwest (NWPP)", 0.635267),
    EGridRegion("NYUP", "New York (NYUP)", 0.242776),
    EGridRegion("RFCM", "RFC Michigan (RFCM)", 0.975978),
    EGridRegion("RFCW", "RFC West (RFCW)", 0.916054),
    EGridRegion("RMPA", "Rocky Mountain (RMPA)", 1.042539),
    EGridRegion("SRMW", "SERC Midwest (SRMW)", 1.248582),
    EGridRegion("SRSO", "SERC South (SRSO)", 0.846007),
    EGridRegion("SRVC", "SERC Virginia/Carolina (SRVC)", 0.596326),
    EGridRegion("SPNO", "SPP North (SPNO)", 0.867740),
    EGridRegion("SPSO", "SPP South (SPSO)", 0.875567),
]

RACK_PRESETS: list[RackPreset] = [
    RackPreset("low", "Low Density (4 kW peak)", 4.0),
    RackPreset("medium", "Medium Density (8 kW peak)", 8.0),
    RackPreset("high", "High Density (15 kW peak)", 15.0),
    RackPreset("ultra", "Ultra-High / AI (30 kW peak)", 30.0),
    RackPreset("custom", "Custom", 0.0),
]

# Average US passenger vehicle annual CO2e (metric tons)
VEHICLE_CO2_METRIC_TONS_PER_YEAR = 4.29

# Conversion constants
BTU_PER_KW = 3_412.14
BTU_PER_TON = 12_000
HOURS_PER_YEAR = 8_760
LBS_PER_METRIC_TON = 2_204.62
LITERS_PER_GALLON = 3.785411784
LATENT_HEAT_VAPORIZATION_KJ_PER_KG = 2_450.0


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------

def _get_cooling_type(id_: str) -> CoolingType:
    return next((c for c in COOLING_TYPES if c.id == id_), COOLING_TYPES[1])


def _get_climate_zone(id_: str) -> ClimateZone:
    return next((c for c in CLIMATE_ZONES if c.id == id_), CLIMATE_ZONES[2])


def _get_egrid_region(id_: str) -> EGridRegion:
    return next((r for r in EGRID_REGIONS if r.id == id_), EGRID_REGIONS[7])


def _get_rack_preset(id_: str) -> RackPreset:
    return next((p for p in RACK_PRESETS if p.id == id_), RACK_PRESETS[1])


# ---------------------------------------------------------------------------
# Input / output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DCCalcInputs:
    """Research-oriented user inputs for the data center calculator."""

    # IT equipment
    num_racks: int = 50
    rack_preset: str = "medium"
    kw_peak_per_rack: float = 10.0       # used when rack_preset == "custom"
    server_utilization: float = 0.60     # average utilization, 0..1
    server_idle_power_fraction: float = 0.60

    # Non-IT loads
    lighting_w_per_sqft: float = 1.2
    data_hall_sqft: float = 5_000
    security_kw: float = 2.0
    other_kw: float = 0.0

    # Electrical chain
    redundancy: str = "N+1"              # "N" | "N+1" | "2N" | "2N+1"
    ups_no_load_loss_kw: float = 2.0
    ups_load_loss_fraction: float = 0.03
    pdu_idle_kw: float = 0.5
    pdu_loss_coefficient: float = 0.00002

    # Cooling
    cooling_type: str = "chilled-water"
    rated_cop: Optional[float] = None
    cooling_plr: Optional[float] = None
    hot_aisle_containment: bool = True
    containment_savings_fraction: float = 0.20
    climate_zone: str = "mixed-humid"
    economizer_mechanical_reduction_fraction: float = 0.85
    altitude_ft: float = 500

    # Water
    site_wue_l_per_kwh: Optional[float] = None  # measured override, if known
    cooling_tower_cycles_of_concentration: float = 4.0
    cooling_tower_drift_fraction: float = 0.002
    humidification_liters_per_kwh_it: Optional[float] = None

    # Financial & emissions
    electric_rate: float = 0.08
    egrid_region: str = "RFCW"
    marginal_emission_factor_lb_per_kwh: Optional[float] = None
    hourly_marginal_emission_factors_lb_per_kwh: Optional[Sequence[float]] = None


@dataclass
class DCCalcResults:
    """All outputs from the data center power, cooling, water, and carbon model."""

    # Resolved inputs
    kw_peak_per_rack: float = 0.0
    server_utilization: float = 0.0
    server_idle_power_fraction: float = 0.0

    # IT load
    it_peak_kw: float = 0.0
    it_average_kw: float = 0.0
    annual_it_energy_kwh: float = 0.0

    # Non-IT loads
    lighting_load_kw: float = 0.0
    total_non_it_kw: float = 0.0

    # Electrical chain
    ups_fixed_loss_kw: float = 0.0
    ups_variable_loss_kw: float = 0.0
    ups_loss_kw: float = 0.0
    ups_input_power_kw: float = 0.0
    ups_implied_efficiency_pct: float = 0.0
    pdu_loss_kw: float = 0.0
    power_distribution_loss_kw: float = 0.0

    # Heat rejection
    heat_rejection_kw: float = 0.0
    heat_rejection_btu_hr: float = 0.0
    heat_rejection_tons: float = 0.0

    # Cooling
    rated_cop: float = 0.0
    actual_cop: float = 0.0
    cooling_plr: float = 0.0
    containment_factor: float = 1.0
    effective_cooling_load_kw: float = 0.0
    effective_cooling_load_tons: float = 0.0
    cooling_energy_kw: float = 0.0
    altitude_derating: float = 1.0
    cooling_capacity_tons: float = 0.0
    free_cooling_hours: int = 0
    free_cooling_fraction: float = 0.0
    blended_cooling_energy_kwh: float = 0.0
    crah_count: int = 0

    # Water
    site_wue_l_per_kwh: float = 0.0
    calculated_wue_l_per_kwh: float = 0.0
    evaporation_water_liters_per_year: float = 0.0
    blowdown_water_liters_per_year: float = 0.0
    drift_water_liters_per_year: float = 0.0
    humidification_water_liters_per_year: float = 0.0
    water_usage_liters_per_year: float = 0.0
    water_usage_gallons_per_year: float = 0.0
    water_model: str = ""

    # Electrical infrastructure
    redundancy_multiplier: float = 1.0
    ups_capacity_kva: float = 0.0
    generator_capacity_kw: float = 0.0
    transformer_kva: float = 0.0
    utility_amps: float = 0.0
    pdu_count: int = 0

    # Facility totals
    total_facility_power_kw: float = 0.0
    annual_facility_energy_kwh: float = 0.0

    # PUE
    pue: float = 0.0
    pue_rating: str = ""
    pue_color: str = ""

    # Annual cost
    annual_cost: float = 0.0

    # Carbon footprint
    carbon_model: str = ""
    emission_factor_lb_per_kwh: float = 0.0
    annual_co2_lbs: float = 0.0
    annual_co2_metric_tons: float = 0.0

    # Warnings
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _get_kw_peak_per_rack(inputs: DCCalcInputs) -> float:
    if inputs.rack_preset == "custom":
        return _clamp(inputs.kw_peak_per_rack, 0.5, 150.0)
    preset = _get_rack_preset(inputs.rack_preset)
    return preset.peak_kw if preset.peak_kw > 0 else 8.0


def _pue_rating(pue: float) -> tuple[str, str]:
    if pue < 1.2:
        return "Excellent", "#22c55e"
    if pue < 1.4:
        return "Good", "#84cc16"
    if pue < 1.6:
        return "Average", "#f59e0b"
    if pue < 2.0:
        return "Poor", "#ef4444"
    return "Very Poor", "#dc2626"


def _altitude_derating(altitude_ft: float) -> float:
    """Return cooling-capacity sizing multiplier for altitude."""
    if altitude_ft <= 5_000:
        return 1.0
    thousands_above = math.ceil((altitude_ft - 5_000) / 1_000)
    return 1.0 + 0.03 * thousands_above


def _redundancy_multiplier(redundancy: str, crah_count: int) -> float:
    if redundancy == "N":
        return 1.0
    if redundancy == "N+1":
        return (crah_count + 1) / crah_count
    if redundancy == "2N":
        return 2.0
    if redundancy == "2N+1":
        return 2.0 + 1.0 / crah_count
    return 1.0


def _part_load_adjusted_cop(rated_cop: float, plr: float, cooling: CoolingType) -> float:
    """
    Return COP adjusted for part-load operation.

    Zhang, Li, and Wang (Building Simulation, 2024,
    DOI: 10.1007/s12273-024-1167-9) show large COP variation across lifecycle
    part-load ratios. This compact curve is a planning approximation, not a
    manufacturer chiller curve.
    """
    bounded_plr = _clamp(plr, 0.05, 1.25)
    if bounded_plr <= 1.0:
        plr_factor = 0.35 + 0.65 * math.sqrt(bounded_plr)
    else:
        plr_factor = 1.0 - 0.10 * (bounded_plr - 1.0)
    return _clamp(rated_cop * plr_factor, cooling.cop_min, cooling.cop_max)


def _average_hourly_factor(values: Sequence[float]) -> Optional[float]:
    clean = [v for v in values if v >= 0]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _default_humidification_l_per_kwh_it(cooling_type: str, climate_zone: str) -> float:
    """Return a small reduced-order humidification proxy by climate."""
    if cooling_type in {"rear-door", "immersion", "evaporative"}:
        return 0.0

    climate_factors = {
        "hot-humid": 0.005,
        "hot-dry": 0.030,
        "mixed-humid": 0.010,
        "mixed-dry": 0.025,
        "cool-humid": 0.015,
        "cool-dry": 0.030,
        "cold": 0.040,
        "subarctic": 0.050,
    }
    return climate_factors.get(climate_zone, 0.010)


# ---------------------------------------------------------------------------
# Main calculation function
# ---------------------------------------------------------------------------

def calculate_data_center(inputs: DCCalcInputs) -> DCCalcResults:
    """
    Run the full data center energy calculation.

    Returns a component-level model suitable for planning comparisons.
    """
    warnings: list[str] = []

    cooling_system = _get_cooling_type(inputs.cooling_type)
    climate = _get_climate_zone(inputs.climate_zone)
    egrid = _get_egrid_region(inputs.egrid_region)

    kw_peak_per_rack = _get_kw_peak_per_rack(inputs)
    utilization = _clamp(inputs.server_utilization, 0.0, 1.0)
    idle_fraction = _clamp(inputs.server_idle_power_fraction, 0.30, 0.90)

    rated_cop = inputs.rated_cop if inputs.rated_cop is not None else cooling_system.cop_default
    rated_cop = _clamp(rated_cop, cooling_system.cop_min, cooling_system.cop_max)

    # Server power uses the common idle-to-peak model:
    # P(u) = P_idle + (P_peak - P_idle) * u.
    # See Applied Energy 2020 DOI: 10.1016/j.apenergy.2020.114806 and
    # the SPECpower-based aggregate modeling discussion in arXiv:2503.06439.
    it_peak_kw = inputs.num_racks * kw_peak_per_rack
    it_average_kw = it_peak_kw * (idle_fraction + (1.0 - idle_fraction) * utilization)
    annual_it_energy_kwh = it_average_kw * HOURS_PER_YEAR

    lighting_load_kw = (inputs.data_hall_sqft * inputs.lighting_w_per_sqft) / 1_000
    total_non_it_kw = lighting_load_kw + inputs.security_kw + inputs.other_kw

    # UPS loss is modeled as fixed no-load loss plus load-proportional loss.
    # This follows the engineering point emphasized in Schneider Electric
    # White Paper SPD_NRAN-66CK3D_EN: UPS efficiency is load-dependent.
    ups_fixed_loss_kw = max(inputs.ups_no_load_loss_kw, 0.0)
    ups_variable_loss_kw = it_average_kw * max(inputs.ups_load_loss_fraction, 0.0)
    ups_loss_kw = ups_fixed_loss_kw + ups_variable_loss_kw
    ups_input_kw = it_average_kw + ups_loss_kw
    ups_implied_efficiency_pct = (
        it_average_kw / ups_input_kw * 100.0 if ups_input_kw > 0 else 0.0
    )

    # PDU/distribution loss is explicit instead of a flat percentage:
    # P_PDU = P_idle + lambda * load^2, adapted from the modular load-chain
    # structure in Rahmani et al., arXiv:1804.00703.
    pdu_loss_kw = max(inputs.pdu_idle_kw, 0.0) + max(inputs.pdu_loss_coefficient, 0.0) * ups_input_kw**2
    power_distribution_loss_kw = pdu_loss_kw

    heat_rejection_kw = it_average_kw + ups_loss_kw + pdu_loss_kw + total_non_it_kw
    heat_rejection_btu_hr = heat_rejection_kw * BTU_PER_KW
    heat_rejection_tons = heat_rejection_btu_hr / BTU_PER_TON

    containment_savings = (
        _clamp(inputs.containment_savings_fraction, 0.0, 0.50)
        if inputs.hot_aisle_containment
        else 0.0
    )
    containment_factor = 1.0 - containment_savings
    effective_cooling_load_kw = heat_rejection_kw * containment_factor
    effective_cooling_load_tons = effective_cooling_load_kw * BTU_PER_KW / BTU_PER_TON

    cooling_plr = inputs.cooling_plr
    if cooling_plr is None:
        cooling_plr = effective_cooling_load_kw / it_peak_kw if it_peak_kw > 0 else 0.0
    cooling_plr = _clamp(cooling_plr, 0.05, 1.25)

    actual_cop = _part_load_adjusted_cop(rated_cop, cooling_plr, cooling_system)
    cooling_energy_kw = effective_cooling_load_kw / actual_cop if actual_cop > 0 else 0.0

    alt_derating = _altitude_derating(inputs.altitude_ft)
    cooling_capacity_tons = effective_cooling_load_tons * alt_derating

    free_cooling_hrs = climate.free_cooling_hours
    free_cooling_fraction = free_cooling_hrs / HOURS_PER_YEAR
    economizer_reduction = _clamp(inputs.economizer_mechanical_reduction_fraction, 0.0, 1.0)
    blended_cooling_energy_kwh = (
        cooling_energy_kw * HOURS_PER_YEAR * (1.0 - free_cooling_fraction * economizer_reduction)
    )

    crah_count = max(1, math.ceil(cooling_capacity_tons / 30.0))
    redundancy_mult = _redundancy_multiplier(inputs.redundancy, crah_count)

    ups_capacity_kva = (ups_input_kw / 0.9) * redundancy_mult
    instantaneous_facility_kw = (
        it_average_kw + ups_loss_kw + pdu_loss_kw + total_non_it_kw + cooling_energy_kw
    )
    generator_capacity_kw = instantaneous_facility_kw * 1.25 * redundancy_mult
    transformer_kva = ups_capacity_kva * 1.15
    utility_amps = (transformer_kva * 1_000) / (480.0 * math.sqrt(3))
    pdu_count = math.ceil(inputs.num_racks / 20) * 2

    non_cooling_energy_kwh = (
        it_average_kw + ups_loss_kw + pdu_loss_kw + total_non_it_kw
    ) * HOURS_PER_YEAR
    annual_facility_energy_kwh = non_cooling_energy_kwh + blended_cooling_energy_kwh

    # PUE = total facility energy / IT equipment energy. See The Green Grid
    # PUE definition and Zhang & Liu, Sensors 2022, DOI: 10.3390/s22103704.
    pue = (
        annual_facility_energy_kwh / annual_it_energy_kwh
        if annual_it_energy_kwh > 0
        else 0.0
    )
    pue_rating, pue_color = _pue_rating(pue)

    annual_cost = annual_facility_energy_kwh * inputs.electric_rate

    # The Green Grid defines WUE as annual onsite water use divided by annual
    # IT energy. Lei & Masanet (2022, DOI: 10.1016/j.resconrec.2022.106323)
    # show WUE should vary with climate, cooling technology, free cooling, and
    # operating parameters. This reduced-order model calculates water first and
    # derives WUE, instead of assuming a constant WUE by cooling type.
    evaporation_water_liters = 0.0
    blowdown_water_liters = 0.0
    drift_water_liters = 0.0
    if cooling_system.uses_cooling_tower_water:
        non_free_cooling_hours = HOURS_PER_YEAR - free_cooling_hrs
        tower_heat_rejection_kw = effective_cooling_load_kw + cooling_energy_kw
        evaporation_water_liters = (
            tower_heat_rejection_kw
            * non_free_cooling_hours
            * 3_600.0
            / LATENT_HEAT_VAPORIZATION_KJ_PER_KG
        )
        cycles = max(inputs.cooling_tower_cycles_of_concentration, 1.1)
        blowdown_water_liters = evaporation_water_liters / (cycles - 1.0)
        drift_fraction = _clamp(inputs.cooling_tower_drift_fraction, 0.0, 0.02)
        drift_water_liters = evaporation_water_liters * drift_fraction

    humidification_l_per_kwh = (
        inputs.humidification_liters_per_kwh_it
        if inputs.humidification_liters_per_kwh_it is not None
        else _default_humidification_l_per_kwh_it(inputs.cooling_type, inputs.climate_zone)
    )
    humidification_l_per_kwh = max(humidification_l_per_kwh, 0.0)
    humidification_water_liters = annual_it_energy_kwh * humidification_l_per_kwh

    calculated_water_liters = (
        evaporation_water_liters
        + blowdown_water_liters
        + drift_water_liters
        + humidification_water_liters
    )
    calculated_wue = (
        calculated_water_liters / annual_it_energy_kwh
        if annual_it_energy_kwh > 0
        else 0.0
    )

    if inputs.site_wue_l_per_kwh is not None:
        site_wue = max(inputs.site_wue_l_per_kwh, 0.0)
        water_usage_liters = annual_it_energy_kwh * site_wue
        water_model = "measured WUE override"
    else:
        site_wue = calculated_wue
        water_usage_liters = calculated_water_liters
        water_model = "calculated thermodynamic approximation"
    water_usage_gallons = water_usage_liters / LITERS_PER_GALLON

    carbon_model = "eGRID annual average"
    emission_factor = egrid.factor
    if inputs.hourly_marginal_emission_factors_lb_per_kwh is not None:
        hourly_factor = _average_hourly_factor(inputs.hourly_marginal_emission_factors_lb_per_kwh)
        if hourly_factor is not None:
            emission_factor = hourly_factor
            carbon_model = "hourly marginal average"
    elif inputs.marginal_emission_factor_lb_per_kwh is not None:
        emission_factor = max(inputs.marginal_emission_factor_lb_per_kwh, 0.0)
        carbon_model = "marginal supplied"
    else:
        # Guidi et al., arXiv:2411.09786, frame data center emissions around
        # attributable/marginal grid impacts. eGRID remains a fallback here
        # because no hourly marginal data ships with this standalone module.
        warnings.append(
            "Carbon uses annual-average eGRID because no marginal emission factor "
            "or hourly marginal series was supplied."
        )

    annual_co2_lbs = annual_facility_energy_kwh * emission_factor
    annual_co2_metric_tons = annual_co2_lbs / LBS_PER_METRIC_TON

    if inputs.altitude_ft > 5_000:
        pct = round((alt_derating - 1.0) * 100)
        warnings.append(
            f"Altitude above 5,000 ft. Cooling equipment capacity increased by {pct}% "
            "for planning; verify against manufacturer altitude curves."
        )

    nominal_capacity_tons = crah_count * 30.0
    if cooling_capacity_tons > nominal_capacity_tons:
        unit_str = "unit" if crah_count == 1 else "units"
        warnings.append(
            f"Altitude derating requires {cooling_capacity_tons:.1f} tons but "
            f"{crah_count} CRAH/CRAC {unit_str} at 30 tons each provide only "
            f"{round(nominal_capacity_tons)} tons nominal."
        )

    if cooling_plr < 0.35:
        warnings.append(
            "Cooling system is at low part-load ratio; COP and PUE are sensitive "
            "to actual chiller staging and controls."
        )

    if pue > 2.0:
        warnings.append(
            "PUE exceeds 2.0. Verify airflow, cooling controls, containment, UPS "
            "loading, economizer operation, and measured metering."
        )

    return DCCalcResults(
        kw_peak_per_rack=kw_peak_per_rack,
        server_utilization=utilization,
        server_idle_power_fraction=idle_fraction,
        it_peak_kw=it_peak_kw,
        it_average_kw=it_average_kw,
        annual_it_energy_kwh=annual_it_energy_kwh,
        lighting_load_kw=lighting_load_kw,
        total_non_it_kw=total_non_it_kw,
        ups_fixed_loss_kw=ups_fixed_loss_kw,
        ups_variable_loss_kw=ups_variable_loss_kw,
        ups_loss_kw=ups_loss_kw,
        ups_input_power_kw=ups_input_kw,
        ups_implied_efficiency_pct=ups_implied_efficiency_pct,
        pdu_loss_kw=pdu_loss_kw,
        power_distribution_loss_kw=power_distribution_loss_kw,
        heat_rejection_kw=heat_rejection_kw,
        heat_rejection_btu_hr=heat_rejection_btu_hr,
        heat_rejection_tons=heat_rejection_tons,
        rated_cop=rated_cop,
        actual_cop=actual_cop,
        cooling_plr=cooling_plr,
        containment_factor=containment_factor,
        effective_cooling_load_kw=effective_cooling_load_kw,
        effective_cooling_load_tons=effective_cooling_load_tons,
        cooling_energy_kw=cooling_energy_kw,
        altitude_derating=alt_derating,
        cooling_capacity_tons=cooling_capacity_tons,
        free_cooling_hours=free_cooling_hrs,
        free_cooling_fraction=free_cooling_fraction,
        blended_cooling_energy_kwh=blended_cooling_energy_kwh,
        crah_count=crah_count,
        site_wue_l_per_kwh=site_wue,
        calculated_wue_l_per_kwh=calculated_wue,
        evaporation_water_liters_per_year=evaporation_water_liters,
        blowdown_water_liters_per_year=blowdown_water_liters,
        drift_water_liters_per_year=drift_water_liters,
        humidification_water_liters_per_year=humidification_water_liters,
        water_usage_liters_per_year=water_usage_liters,
        water_usage_gallons_per_year=water_usage_gallons,
        water_model=water_model,
        redundancy_multiplier=redundancy_mult,
        ups_capacity_kva=ups_capacity_kva,
        generator_capacity_kw=generator_capacity_kw,
        transformer_kva=transformer_kva,
        utility_amps=utility_amps,
        pdu_count=pdu_count,
        total_facility_power_kw=instantaneous_facility_kw,
        annual_facility_energy_kwh=annual_facility_energy_kwh,
        pue=pue,
        pue_rating=pue_rating,
        pue_color=pue_color,
        annual_cost=annual_cost,
        carbon_model=carbon_model,
        emission_factor_lb_per_kwh=emission_factor,
        annual_co2_lbs=annual_co2_lbs,
        annual_co2_metric_tons=annual_co2_metric_tons,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def print_report(inputs: DCCalcInputs, r: DCCalcResults) -> None:
    """Print a formatted console report of inputs and calculated results."""

    def fmt(value: float, decimals: int = 1) -> str:
        return f"{value:,.{decimals}f}"

    def fmt_int(value: float) -> str:
        return f"{int(round(value)):,}"

    sep = "-" * 60

    print(f"\n{'=' * 60}")
    print("  DATA CENTER ENERGY USAGE CALCULATOR")
    print(f"{'=' * 60}")

    cooling_label = _get_cooling_type(inputs.cooling_type).label
    climate_label = _get_climate_zone(inputs.climate_zone).label
    egrid_label = _get_egrid_region(inputs.egrid_region).label

    print("\nINPUTS")
    print(sep)
    print(f"  IT Racks                   : {inputs.num_racks}")
    print(f"  Peak Power Per Rack        : {fmt(r.kw_peak_per_rack, 1)} kW")
    print(f"  Server Utilization         : {fmt(r.server_utilization * 100, 1)} %")
    print(f"  Idle Power Fraction        : {fmt(r.server_idle_power_fraction * 100, 1)} %")
    print(f"  Data Hall Area             : {fmt_int(inputs.data_hall_sqft)} sq ft")
    print(f"  Lighting                   : {fmt(inputs.lighting_w_per_sqft, 1)} W/sq ft")
    print(f"  Security/Monitoring        : {fmt(inputs.security_kw, 1)} kW")
    print(f"  Other Ancillary            : {fmt(inputs.other_kw, 1)} kW")
    print(f"  Redundancy                 : {inputs.redundancy}")
    print(f"  Cooling System             : {cooling_label}")
    print(f"  Hot Aisle Containment      : {'Yes' if inputs.hot_aisle_containment else 'No'}")
    print(f"  Climate Zone               : {climate_label}")
    print(f"  Altitude                   : {fmt_int(inputs.altitude_ft)} ft")
    print(f"  Electric Rate              : ${fmt(inputs.electric_rate, 4)}/kWh")
    print(f"  Emissions Region           : {egrid_label}")

    if r.warnings:
        print("\nWARNINGS")
        print(sep)
        for warning in r.warnings:
            print(f"  - {warning}")

    print("\nPOWER SUMMARY")
    print(sep)
    print(f"  IT Peak Load               : {fmt(r.it_peak_kw, 1)} kW")
    print(f"  IT Average Load            : {fmt(r.it_average_kw, 1)} kW")
    print(f"  Lighting Load              : {fmt(r.lighting_load_kw, 2)} kW")
    print(f"  Total Non-IT Loads         : {fmt(r.total_non_it_kw, 2)} kW")
    print(f"  UPS Fixed Loss             : {fmt(r.ups_fixed_loss_kw, 2)} kW")
    print(f"  UPS Variable Loss          : {fmt(r.ups_variable_loss_kw, 2)} kW")
    print(f"  UPS Losses                 : {fmt(r.ups_loss_kw, 2)} kW")
    print(f"  UPS Implied Efficiency     : {fmt(r.ups_implied_efficiency_pct, 2)} %")
    print(f"  PDU/Distribution Loss      : {fmt(r.pdu_loss_kw, 2)} kW")
    print(f"  Total Facility Power       : {fmt(r.total_facility_power_kw, 1)} kW")

    print("\nELECTRICAL INFRASTRUCTURE")
    print(sep)
    print(f"  Redundancy Multiplier      : {fmt(r.redundancy_multiplier, 3)}x")
    print(f"  UPS Capacity               : {fmt_int(r.ups_capacity_kva)} kVA")
    print(f"  Generator Capacity         : {fmt_int(r.generator_capacity_kw)} kW")
    print(f"  Transformer                : {fmt_int(r.transformer_kva)} kVA")
    print(f"  Utility Feed @ 480V 3ph    : {fmt_int(r.utility_amps)} A")
    print(f"  PDU Count (dual-feed)      : {r.pdu_count} units")

    print("\nCOOLING SUMMARY")
    print(sep)
    print(f"  Heat Rejection             : {fmt(r.heat_rejection_kw, 1)} kW")
    print(f"  Heat Rejection             : {fmt_int(r.heat_rejection_btu_hr)} BTU/hr")
    print(f"  Heat Rejection             : {fmt(r.heat_rejection_tons, 1)} tons")
    if inputs.hot_aisle_containment:
        savings = (1.0 - r.containment_factor) * 100.0
        print(f"  Containment Savings        : {fmt(savings, 1)} %")
    print(f"  Effective Cooling Load     : {fmt(r.effective_cooling_load_kw, 1)} kW")
    print(f"  Effective Cooling Load     : {fmt(r.effective_cooling_load_tons, 1)} tons")
    print(f"  Cooling Capacity (derated) : {fmt(r.cooling_capacity_tons, 1)} tons")
    print(f"  Cooling PLR                : {fmt(r.cooling_plr * 100, 1)} %")
    print(f"  Rated COP                  : {fmt(r.rated_cop, 2)}")
    print(f"  Actual COP                 : {fmt(r.actual_cop, 2)}")
    print(f"  Mechanical Cooling         : {fmt(r.cooling_energy_kw, 1)} kW")
    print(f"  CRAH/CRAC Units            : {r.crah_count} units (@ 30 tons each)")
    print(f"  Free Cooling Hours         : {fmt_int(r.free_cooling_hours)} hrs/yr")
    print(f"  Free Cooling Fraction      : {fmt(r.free_cooling_fraction * 100, 1)} %")
    print(f"  Annual Cooling (blended)   : {fmt_int(r.blended_cooling_energy_kwh)} kWh")
    if inputs.altitude_ft > 5_000:
        print(f"  Altitude Derating          : {fmt(r.altitude_derating, 2)}x")

    print("\nPUE, WATER, AND COST")
    print(sep)
    print(f"  PUE                        : {fmt(r.pue, 2)} ({r.pue_rating})")
    print(f"  Annual IT Energy           : {fmt_int(r.annual_it_energy_kwh)} kWh")
    print(f"  Annual Facility Energy     : {fmt_int(r.annual_facility_energy_kwh)} kWh")
    print(f"  Annual Energy Cost         : ${r.annual_cost:,.2f}")
    print(f"  Water Model                : {r.water_model}")
    print(f"  Calculated WUE             : {fmt(r.calculated_wue_l_per_kwh, 3)} L/kWh IT")
    print(f"  Effective WUE              : {fmt(r.site_wue_l_per_kwh, 3)} L/kWh IT")
    print(f"  Evaporation Water          : {fmt_int(r.evaporation_water_liters_per_year)} L/yr")
    print(f"  Blowdown Water             : {fmt_int(r.blowdown_water_liters_per_year)} L/yr")
    print(f"  Drift Water                : {fmt_int(r.drift_water_liters_per_year)} L/yr")
    print(f"  Humidification Water       : {fmt_int(r.humidification_water_liters_per_year)} L/yr")
    print(f"  Estimated Water Usage      : {fmt_int(r.water_usage_liters_per_year)} L/yr")
    print(f"  Estimated Water Usage      : {fmt_int(r.water_usage_gallons_per_year)} gal/yr")

    vehicles = r.annual_co2_metric_tons / VEHICLE_CO2_METRIC_TONS_PER_YEAR
    print("\nCARBON FOOTPRINT")
    print(sep)
    print(f"  Carbon Model               : {r.carbon_model}")
    print(f"  Emission Factor            : {fmt(r.emission_factor_lb_per_kwh, 3)} lb CO2e/kWh")
    print(f"  Annual CO2e                : {fmt_int(r.annual_co2_lbs)} lbs")
    print(f"  Annual CO2e                : {fmt(r.annual_co2_metric_tons, 1)} metric tons")
    print(f"  Vehicle Equivalent         : ~{fmt(vehicles, 0)} avg US vehicles/yr")

    print(f"\n{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    inputs = DCCalcInputs(
        num_racks=50,
        rack_preset="medium",
        kw_peak_per_rack=10.0,
        server_utilization=0.60,
        server_idle_power_fraction=0.60,
        lighting_w_per_sqft=1.2,
        data_hall_sqft=5_000,
        security_kw=2.0,
        other_kw=0.0,
        redundancy="N+1",
        ups_no_load_loss_kw=2.0,
        ups_load_loss_fraction=0.03,
        pdu_idle_kw=0.5,
        pdu_loss_coefficient=0.00002,
        cooling_type="chilled-water",
        rated_cop=None,
        cooling_plr=None,
        hot_aisle_containment=True,
        containment_savings_fraction=0.20,
        climate_zone="mixed-humid",
        altitude_ft=500,
        site_wue_l_per_kwh=None,
        electric_rate=0.08,
        egrid_region="RFCW",
    )

    results = calculate_data_center(inputs)
    print_report(inputs, results)

    print("-" * 60)
    print("Custom scenario: 120 GPU racks @ 30 kW peak, Denver, 2N, DX cooling")
    print("-" * 60)

    gpu_inputs = DCCalcInputs(
        num_racks=120,
        rack_preset="ultra",
        server_utilization=0.75,
        server_idle_power_fraction=0.55,
        lighting_w_per_sqft=1.5,
        data_hall_sqft=12_000,
        security_kw=5.0,
        other_kw=10.0,
        redundancy="2N",
        ups_no_load_loss_kw=8.0,
        ups_load_loss_fraction=0.025,
        pdu_idle_kw=2.0,
        pdu_loss_coefficient=0.000005,
        cooling_type="dx",
        rated_cop=3.2,
        cooling_plr=None,
        hot_aisle_containment=False,
        climate_zone="mixed-dry",
        altitude_ft=5_280,
        site_wue_l_per_kwh=None,
        electric_rate=0.065,
        egrid_region="RMPA",
        marginal_emission_factor_lb_per_kwh=None,
    )

    gpu_results = calculate_data_center(gpu_inputs)
    print_report(gpu_inputs, gpu_results)
