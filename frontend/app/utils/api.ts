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

export async function predictPermit(
  req: PermitPredictionRequest,
): Promise<PermitPredictionResponse> {
  const res = await fetch('/api/predict-permit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
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
  return parsed as PermitPrediction;
}

export async function computeImpact(req: ComputeRequest): Promise<ComputeResponse> {
  const res = await fetch('/api/compute', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
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
  return (parsed ?? { health_cost_by_county: [], naaqs_violations: [] }) as ComputeResult;
}
