import type { GeneratorFuel, OnsiteGenerator } from './generators';

export interface ComputeRequest {
  lat: number;
  lon: number;
  total_power: number;
  generator_power: number;
  fuel: GeneratorFuel;
  generators: OnsiteGenerator[];
}

export interface ComputeResult {
  health_cost_by_county: number[];
  county_fips?: string[];
  county_names?: string[];
  naaqs_violations: number[];
  naaqs_violation_fips?: string[];
}

export interface CountyCost {
  fips: string;
  name?: string;
  usdPerYear: number;
}

export interface ComputeError {
  error: string;
}

export type ComputeResponse = ComputeResult | ComputeError;

export function isComputeError(value: ComputeResponse): value is ComputeError {
  return typeof (value as ComputeError).error === 'string';
}

async function parseJsonResponse<T>(res: Response, fallback: T): Promise<T | ComputeError> {
  const text = await res.text();
  let parsed: unknown;
  try {
    parsed = text ? JSON.parse(text) : null;
  } catch {
    if (!res.ok) {
      const snippet = text.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 500);
      return { error: `Backend ${res.status}: ${snippet || '(empty body)'}` };
    }
    return { error: `Invalid response from backend (status ${res.status})` };
  }
  if (!res.ok) {
    const message =
      parsed && typeof parsed === 'object' && parsed !== null && 'error' in parsed
        ? String((parsed as { error: unknown }).error)
        : `Backend returned status ${res.status}`;
    return { error: message };
  }
  return (parsed ?? fallback) as T;
}

export interface ImpactSummary {
  totalUsdPerYear: number;
  countyCount: number;
  topCountyUsd: number;
  topCountyName?: string;
  topCountyFips?: string;
  /** Total minus the top county — what's spread across "all the other counties on the map." */
  remainderUsdPerYear: number;
  naaqsViolationCount: number;
  countyCosts: CountyCost[];
  naaqsFips: string[];
}

export function summarizeResult(res: ComputeResult): ImpactSummary {
  const costs = Array.isArray(res.health_cost_by_county) ? res.health_cost_by_county : [];
  const fips = Array.isArray(res.county_fips) ? res.county_fips : [];
  const names = Array.isArray(res.county_names) ? res.county_names : [];
  const countyCosts: CountyCost[] = [];
  let total = 0;
  let topAbs = 0;
  let topVal = 0;
  let topName: string | undefined;
  let topFips: string | undefined;
  for (let i = 0; i < costs.length; i++) {
    const v = costs[i];
    if (typeof v === 'number' && Number.isFinite(v)) {
      total += v;
      if (Math.abs(v) > topAbs) {
        topAbs = Math.abs(v);
        topVal = v;
        topName = names[i];
        topFips = fips[i];
      }
      if (fips[i]) {
        countyCosts.push({ fips: fips[i], name: names[i], usdPerYear: v });
      }
    }
  }
  return {
    totalUsdPerYear: total,
    countyCount: costs.length,
    topCountyUsd: topVal,
    topCountyName: topName,
    topCountyFips: topFips,
    remainderUsdPerYear: total - topVal,
    naaqsViolationCount: Array.isArray(res.naaqs_violations) ? res.naaqs_violations.length : 0,
    countyCosts,
    naaqsFips: Array.isArray(res.naaqs_violation_fips) ? res.naaqs_violation_fips : [],
  };
}

export type ApprovalLikelihood = 'Low' | 'Medium' | 'High';

/** Bins the kNN/KM p_approved into a low/medium/high category. Thresholds chosen
 *  around the dataset's overall approval rate so the categories are interpretable
 *  relative to a typical historical project. */
export function approvalLikelihood(pApproved: number): ApprovalLikelihood {
  if (pApproved >= 0.6) return 'High';
  if (pApproved >= 0.3) return 'Medium';
  return 'Low';
}

export const APPROVAL_LIKELIHOOD_TONE: Record<ApprovalLikelihood, string> = {
  High: 'text-emerald-700 dark:text-emerald-300',
  Medium: 'text-amber-700 dark:text-amber-300',
  Low: 'text-rose-700 dark:text-rose-300',
};

export interface PermitPredictionRequest {
  lat: number;
  lon: number;
  mw_capacity: number;
  pollution_cost_usd_per_year?: number | null;
}

export interface SimilarCase {
  name: string;
  why_similar: string;
  outcome: string;
}

export interface PermitPrediction {
  p_approved: number;
  p_approved_ci_low: number;
  p_approved_ci_high: number;
  expected_days_to_first_approval: number;
  days_ci_low: number;
  days_ci_high: number;
  most_similar_cases: SimilarCase[];
  key_factors: string[];
  derived_context?: {
    inferred_state?: string | null;
    nearest_distance_miles?: number;
    k?: number;
    n_km_events?: number;
    n_km_observations?: number;
    days_source?: 'km_local' | 'km_local_widened' | 'km_partial' | 'global_fallback';
    global_anchors?: { n: number; median: number; p25: number; p75: number; p90: number };
  };
}

export type PermitPredictionResponse = PermitPrediction | ComputeError;

export function isPermitError(value: PermitPredictionResponse): value is ComputeError {
  return typeof (value as ComputeError).error === 'string';
}

export interface ResourceUsageRequest {
  num_racks: number;
  rack_preset: string;
  kw_peak_per_rack: number;
  server_utilization: number;
  server_idle_power_fraction: number;
  data_hall_sqft: number;
  redundancy: string;
  cooling_type: string;
  rated_cop?: number | null;
  hot_aisle_containment: boolean;
  climate_zone: string;
  altitude_ft: number;
  electric_rate: number;
  egrid_region: string;
  cooling_tower_cycles_of_concentration: number;
  cooling_tower_drift_fraction: number;
  site_wue_l_per_kwh?: number | null;
}

export interface ResourceUsageResult {
  it_peak_kw: number;
  it_average_kw: number;
  total_facility_power_kw: number;
  annual_it_energy_kwh: number;
  annual_facility_energy_kwh: number;
  pue: number;
  pue_rating: string;
  calculated_wue_l_per_kwh: number;
  site_wue_l_per_kwh: number;
  water_usage_liters_per_year: number;
  water_usage_gallons_per_year: number;
  evaporation_water_liters_per_year: number;
  blowdown_water_liters_per_year: number;
  drift_water_liters_per_year: number;
  humidification_water_liters_per_year: number;
  water_model: string;
  annual_cost: number;
  annual_co2_metric_tons: number;
  warnings?: string[];
}

export type ResourceUsageResponse = ResourceUsageResult | ComputeError;

export function isResourceUsageError(value: ResourceUsageResponse): value is ComputeError {
  return typeof (value as ComputeError).error === 'string';
}

export interface PriceChangeRequest {
  current_price: number;
  new_demand: number;
  location?: string;
  resource?: 'water' | 'electricity';
  old_demand?: number;
  supply_elasticity?: number;
  demand_elasticity?: number;
}

export interface PriceChangeResult {
  price_change: number;
  current_price: number;
  old_demand: number;
  new_demand: number;
  supply_elasticity: number;
  demand_elasticity: number;
  location?: string;
  resource?: string;
}

export type PriceChangeResponse = PriceChangeResult | ComputeError;

export function isPriceChangeError(value: PriceChangeResponse): value is ComputeError {
  return typeof (value as ComputeError).error === 'string';
}

export const LA_BASELINE_DEMAND = {
  electricityKwhPerYear: 64_896e9,
  waterLitersPerYear: 1_489_564 * 1.233e6,
};

export async function predictPermit(
  req: PermitPredictionRequest,
): Promise<PermitPredictionResponse> {
  const res = await fetch('/api/predict-permit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return parseJsonResponse<PermitPrediction>(res, {} as PermitPrediction);
}

export async function computeImpact(req: ComputeRequest): Promise<ComputeResponse> {
  const res = await fetch('/api/compute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return parseJsonResponse<ComputeResult>(res, { health_cost_by_county: [], naaqs_violations: [] });
}

export async function calculateResourceUsage(
  req: ResourceUsageRequest,
): Promise<ResourceUsageResponse> {
  const res = await fetch('/api/resource-usage', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return parseJsonResponse<ResourceUsageResult>(res, {} as ResourceUsageResult);
}

export async function calculatePriceChange(
  req: PriceChangeRequest,
): Promise<PriceChangeResponse> {
  const res = await fetch('/api/price-change', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return parseJsonResponse<PriceChangeResult>(res, {} as PriceChangeResult);
}
