import type { LatLng } from './area';
import type { OnsiteGenerator } from './generators';
import type { ComputeResult, PermitPrediction } from './api';

export interface SavedResult {
  computedAt: number;
  totalUsdPerYear: number;
  topCountyUsd: number;
  naaqsViolationCount: number;
  naaqsFips: string[];
  costByFips: Record<string, number>;
  permitDays?: number;
  permitDaysCiLow?: number;
  permitDaysCiHigh?: number;
  permitProbability?: number;
  permitProbabilityCiLow?: number;
  permitProbabilityCiHigh?: number;
  /** Full backend payloads — when present, the analysis page loads them directly
   *  instead of re-running compute + predict. Skipping a re-run avoids both the
   *  COBRA call and the Sonnet call (~30s + ~$0.05). */
  fullResult?: ComputeResult;
  permitPrediction?: PermitPrediction;
}

export interface SavedAnalysis {
  id: string;
  name: string;
  createdAt: number;
  updatedAt: number;
  polygon: LatLng[];
  mwUsage: number;
  gridUsage: number;
  onsiteUsage: number;
  generators: OnsiteGenerator[];
  result?: SavedResult;
}

const STORAGE_KEY = 'center.saved-analyses.v1';

export function loadSavedAnalyses(): SavedAnalysis[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.map((item: SavedAnalysis) => ({
      ...item,
      generators: Array.isArray(item.generators) ? item.generators : [],
    }));
  } catch {
    return [];
  }
}

export function persistSavedAnalyses(items: SavedAnalysis[]): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  } catch {
    // Quota exceeded or storage unavailable.
  }
}

export function generateAnalysisId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID();
  }
  return `a-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}
