'use client';

import { useSearchParams } from 'next/navigation';
import dynamic from 'next/dynamic';

const MapComponent = dynamic(() => import('../components/MapComponent'), { ssr: false });
const CountyChoroplethMap = dynamic(
  () => import('../components/CountyChoroplethMap'),
  { ssr: false },
);
import NameInputModal from '../components/NameInputModal';
import { useRouter } from 'next/navigation';
import { Suspense, useEffect, useMemo, useState } from 'react';
import {
  decodePolygon,
  formatSquareFeet,
  polygonAreaSquareFeet,
  polygonCentroid,
} from '../utils/area';
import { decodeGenerators, type OnsiteGenerator } from '../utils/generators';
import {
  APPROVAL_LIKELIHOOD_TONE,
  approvalLikelihood,
  calculatePriceChange,
  calculateResourceUsage,
  computeImpact,
  isComputeError,
  isPermitError,
  isPriceChangeError,
  isResourceUsageError,
  LA_BASELINE_DEMAND,
  predictPermit,
  summarizeResult,
  type ComputeResult,
  type PermitPrediction,
  type PriceChangeResult,
  type ResourceUsageResult,
} from '../utils/api';
import { decodeResourceAssumptions } from '../utils/resourceAssumptions';
import { useLiveLayers } from '../hooks/useLiveLayers';
import {
  generateAnalysisId,
  loadSavedAnalyses,
  persistSavedAnalyses,
  type SavedAnalysis,
  type SavedResult,
} from '../utils/savedAnalyses';

const usdFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
});

const compactNumber = new Intl.NumberFormat('en-US', {
  maximumFractionDigits: 1,
  notation: 'compact',
});

function AnalysisContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const { liveData, liveStatus } = useLiveLayers();

  const polygonParam = searchParams.get('polygon');
  const polygon = useMemo(() => decodePolygon(polygonParam), [polygonParam]);
  const centroid = useMemo(() => polygonCentroid(polygon), [polygon]);
  const areaSqFt = useMemo(
    () => (polygon.length >= 3 ? polygonAreaSquareFeet(polygon) : 0),
    [polygon],
  );

  const racksParam = searchParams.get('racks');
  const legacyMw = searchParams.get('mw');
  const rackCount = useMemo(() => {
    const explicit = Number(racksParam);
    if (Number.isFinite(explicit) && explicit > 0) return explicit;
    const legacy = Number(legacyMw);
    if (Number.isFinite(legacy) && legacy > 0) return Math.max(1, Math.round((legacy * 1000) / 8));
    return 0;
  }, [racksParam, legacyMw]);
  const resourceAssumptions = useMemo(
    () => decodeResourceAssumptions(searchParams.get('resource')),
    [searchParams],
  );
  const grid = searchParams.get('grid');
  const onsite = searchParams.get('onsite');
  const generators: OnsiteGenerator[] = useMemo(
    () => decodeGenerators(searchParams.get('generators')),
    [searchParams],
  );

  // If ?saved=ID is in the URL and that saved analysis carries cached payloads,
  // load them synchronously *during render* so the API-call effects can bail on
  // their very first run instead of racing with a separate hydration effect.
  const savedId = searchParams.get('saved');
  const savedHydration = useMemo(() => {
    if (!savedId || typeof window === 'undefined') return null;
    try {
      const entry = loadSavedAnalyses().find((a) => a.id === savedId);
      if (entry?.result?.fullResult && entry.result.permitPrediction) {
        return {
          result: entry.result.fullResult,
          permit: entry.result.permitPrediction,
          resourceUsage: entry.result.resourceUsage,
          waterPrice: entry.result.waterPriceChange,
          electricityPrice: entry.result.electricityPriceChange,
        };
      }
    } catch {}
    return null;
  }, [savedId]);

  const [result, setResult] = useState<ComputeResult | null>(
    savedHydration?.result ?? null,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [permit, setPermit] = useState<PermitPrediction | null>(
    savedHydration?.permit ?? null,
  );
  const [permitLoading, setPermitLoading] = useState(false);
  const [permitError, setPermitError] = useState<string | null>(null);
  const [resourceUsage, setResourceUsage] = useState<ResourceUsageResult | null>(
    savedHydration?.resourceUsage ?? null,
  );
  const [resourceLoading, setResourceLoading] = useState(false);
  const [resourceError, setResourceError] = useState<string | null>(null);
  const [waterPrice, setWaterPrice] = useState<PriceChangeResult | null>(
    savedHydration?.waterPrice ?? null,
  );
  const [electricityPrice, setElectricityPrice] = useState<PriceChangeResult | null>(
    savedHydration?.electricityPrice ?? null,
  );
  const [priceLoading, setPriceLoading] = useState(false);
  const [priceError, setPriceError] = useState<string | null>(null);
  const [savedToast, setSavedToast] = useState<string | null>(null);
  const [saveModalOpen, setSaveModalOpen] = useState(false);

  useEffect(() => {
    if (savedHydration?.resourceUsage) return;
    if (rackCount <= 0) return;

    const ac = new AbortController();
    setResourceLoading(true);
    setResourceError(null);
    setResourceUsage(null);
    setResult(null);
    setPermit(null);
    setWaterPrice(null);
    setElectricityPrice(null);

    calculateResourceUsage({
      num_racks: rackCount,
      rack_preset: resourceAssumptions.rackPreset,
      kw_peak_per_rack: resourceAssumptions.kwPeakPerRack,
      server_utilization: resourceAssumptions.serverUtilization,
      server_idle_power_fraction: resourceAssumptions.serverIdlePowerFraction,
      data_hall_sqft: areaSqFt,
      redundancy: resourceAssumptions.redundancy,
      cooling_type: resourceAssumptions.coolingType,
      rated_cop: resourceAssumptions.ratedCop,
      hot_aisle_containment: resourceAssumptions.hotAisleContainment,
      climate_zone: resourceAssumptions.climateZone,
      altitude_ft: resourceAssumptions.altitudeFt,
      electric_rate: resourceAssumptions.electricRate,
      egrid_region: resourceAssumptions.egridRegion,
      cooling_tower_cycles_of_concentration:
        resourceAssumptions.coolingTowerCyclesOfConcentration,
      cooling_tower_drift_fraction: resourceAssumptions.coolingTowerDriftFraction,
      site_wue_l_per_kwh: resourceAssumptions.siteWueLPerKwh,
    })
      .then((res) => {
        if (ac.signal.aborted) return;
        if (isResourceUsageError(res)) {
          setResourceError(res.error);
        } else {
          setResourceUsage(res);
        }
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        setResourceError(err instanceof Error ? err.message : 'Unknown error');
      })
      .finally(() => {
        if (!ac.signal.aborted) setResourceLoading(false);
      });

    return () => ac.abort();
  }, [areaSqFt, rackCount, resourceAssumptions, savedHydration]);

  useEffect(() => {
    if (savedHydration) return;
    if (!centroid || !onsite || !resourceUsage) return;
    const totalPower = resourceUsage.total_facility_power_kw / 1000;
    const onsitePct = Number(onsite);
    if (!Number.isFinite(totalPower) || totalPower <= 0) return;

    const generatorPower = (totalPower * onsitePct) / 100;
    const fuel = generators[0]?.fuel ?? 'Diesel';

    const ac = new AbortController();
    setLoading(true);
    setError(null);
    setResult(null);

    computeImpact({
      lat: centroid[0],
      lon: centroid[1],
      total_power: totalPower,
      generator_power: generatorPower,
      fuel,
      generators,
    })
      .then((res) => {
        if (ac.signal.aborted) return;
        if (isComputeError(res)) {
          setError(res.error);
        } else {
          setResult(res);
        }
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        setError(err instanceof Error ? err.message : 'Unknown error');
      })
      .finally(() => {
        if (!ac.signal.aborted) setLoading(false);
      });

    return () => ac.abort();
  }, [centroid, onsite, generators, savedHydration, resourceUsage]);

  const summary = useMemo(() => (result ? summarizeResult(result) : null), [result]);

  // Permit-timeline prediction. Waits for the COBRA result so Sonnet sees the same
  // pollution-cost number the user sees in the summary card; firing earlier lets the
  // backend fall back to pollution.py's CO2-inclusive figure, which disagrees with
  // the COBRA health-cost total (different metric).
  useEffect(() => {
    if (savedHydration) return;
    if (!centroid || !resourceUsage) return;
    const totalPower = resourceUsage.total_facility_power_kw / 1000;
    if (!Number.isFinite(totalPower) || totalPower <= 0) return;
    if (!result) return; // wait for the health-cost computation

    const ac = new AbortController();
    setPermitLoading(true);
    setPermitError(null);
    setPermit(null);

    predictPermit({
      lat: centroid[0],
      lon: centroid[1],
      mw_capacity: totalPower,
      pollution_cost_usd_per_year: summarizeResult(result).totalUsdPerYear,
    })
      .then((res) => {
        if (ac.signal.aborted) return;
        if (isPermitError(res)) {
          setPermitError(res.error);
        } else {
          setPermit(res);
        }
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        setPermitError(err instanceof Error ? err.message : 'Unknown error');
      })
      .finally(() => {
        if (!ac.signal.aborted) setPermitLoading(false);
      });

    return () => ac.abort();
  }, [centroid, result, savedHydration, resourceUsage]);

  useEffect(() => {
    if (savedHydration?.waterPrice && savedHydration?.electricityPrice) return;
    if (!resourceUsage) return;

    const ac = new AbortController();
    setPriceLoading(true);
    setPriceError(null);
    setWaterPrice(null);
    setElectricityPrice(null);

    Promise.all([
      calculatePriceChange({
        current_price: 0.01,
        location: 'LA',
        resource: 'water',
        new_demand:
          LA_BASELINE_DEMAND.waterLitersPerYear + resourceUsage.water_usage_liters_per_year,
      }),
      calculatePriceChange({
        current_price: 0.17,
        location: 'LA',
        resource: 'electricity',
        new_demand:
          LA_BASELINE_DEMAND.electricityKwhPerYear +
          resourceUsage.annual_facility_energy_kwh,
      }),
    ])
      .then(([water, electricity]) => {
        if (ac.signal.aborted) return;
        if (isPriceChangeError(water)) {
          setPriceError(water.error);
        } else {
          setWaterPrice(water);
        }
        if (isPriceChangeError(electricity)) {
          setPriceError(electricity.error);
        } else {
          setElectricityPrice(electricity);
        }
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        setPriceError(err instanceof Error ? err.message : 'Unknown error');
      })
      .finally(() => {
        if (!ac.signal.aborted) setPriceLoading(false);
      });

    return () => ac.abort();
  }, [resourceUsage, savedHydration]);

  const costByFips = useMemo(() => {
    const map: Record<string, number> = {};
    if (summary) {
      for (const c of summary.countyCosts) map[c.fips] = c.usdPerYear;
    }
    return map;
  }, [summary]);

  const openSaveModal = () => {
    if (!summary) return;
    if (polygon.length < 3) {
      alert('Site polygon is missing — cannot save.');
      return;
    }
    setSaveModalOpen(true);
  };

  const commitSave = (name: string) => {
    if (!summary) return;
    const trimmed = name.trim() || 'Untitled site';

    const savedResult: SavedResult = {
      computedAt: Date.now(),
      totalUsdPerYear: summary.totalUsdPerYear,
      topCountyUsd: summary.topCountyUsd,
      naaqsViolationCount: summary.naaqsViolationCount,
      naaqsFips: summary.naaqsFips,
      costByFips,
      fullResult: result ?? undefined,
      permitPrediction: permit ?? undefined,
      resourceUsage: resourceUsage ?? undefined,
      waterPriceChange: waterPrice ?? undefined,
      electricityPriceChange: electricityPrice ?? undefined,
      ...(permit && {
        permitDays: permit.expected_days_to_first_approval,
        permitDaysCiLow: permit.days_ci_low,
        permitDaysCiHigh: permit.days_ci_high,
        permitProbability: permit.p_approved,
        permitProbabilityCiLow: permit.p_approved_ci_low,
        permitProbabilityCiHigh: permit.p_approved_ci_high,
      }),
    };

    const id = generateAnalysisId();
    const now = Date.now();
    const item: SavedAnalysis = {
      id,
      name: trimmed,
      createdAt: now,
      updatedAt: now,
      polygon,
      rackCount,
      resourceAssumptions,
      mwUsage: resourceUsage ? resourceUsage.total_facility_power_kw / 1000 : 0,
      gridUsage: Number(grid) || 0,
      onsiteUsage: Number(onsite) || 0,
      generators,
      result: savedResult,
    };
    const all = loadSavedAnalyses();
    persistSavedAnalyses([item, ...all]);
    setSaveModalOpen(false);
    setSavedToast(`Saved "${trimmed}"`);
    setTimeout(() => setSavedToast(null), 2500);
  };

  return (
    <div className="flex flex-1 min-h-0">
      <aside className="w-1/3 min-w-[360px] border-r border-border bg-surface-muted/60 p-8 overflow-y-auto">
        <div className="space-y-6">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-accent">
              Permit Analysis
            </p>
            <h1 className="mt-1 text-2xl font-semibold tracking-tight text-foreground">
              Site Results
            </h1>
          </div>

          <div className="rounded-2xl border border-border bg-surface p-5 shadow-sm">
            <dl className="grid grid-cols-2 gap-4 text-sm">
              <div className="col-span-2 min-w-0">
                <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-foreground/50">
                  Centroid
                </dt>
                <dd className="mt-1 break-all font-mono text-sm text-foreground [overflow-wrap:anywhere]">
                  {centroid
                    ? `${centroid[0].toFixed(5)}, ${centroid[1].toFixed(5)}`
                    : '—'}
                </dd>
              </div>
              <div className="col-span-2 min-w-0">
                <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-foreground/50">
                  Site footprint
                </dt>
                <dd className="mt-1 break-all text-foreground tabular-nums [overflow-wrap:anywhere]">
                  {polygon.length >= 3
                    ? formatSquareFeet(areaSqFt)
                    : '—'}
                </dd>
              </div>
              <div className="min-w-0">
                <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-foreground/50">
                  Rack count
                </dt>
                <dd className="mt-1 break-all text-foreground tabular-nums [overflow-wrap:anywhere]">
                  {rackCount.toLocaleString()}
                </dd>
              </div>
              <div className="min-w-0">
                <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-foreground/50">
                  Derived facility MW
                </dt>
                <dd className="mt-1 break-all text-foreground tabular-nums [overflow-wrap:anywhere]">
                  {resourceUsage
                    ? `${(resourceUsage.total_facility_power_kw / 1000).toFixed(2)} MW`
                    : '—'}
                </dd>
              </div>
              <div className="min-w-0">
                <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-foreground/50">
                  Grid / Onsite
                </dt>
                <dd className="mt-1 break-all text-foreground tabular-nums [overflow-wrap:anywhere]">{grid}% / {onsite}%</dd>
              </div>
              <div className="min-w-0">
                <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-foreground/50">
                  IT peak / avg
                </dt>
                <dd className="mt-1 break-all text-foreground tabular-nums [overflow-wrap:anywhere]">
                  {resourceUsage
                    ? `${(resourceUsage.it_peak_kw / 1000).toFixed(2)} / ${(resourceUsage.it_average_kw / 1000).toFixed(2)} MW`
                    : '—'}
                </dd>
              </div>
              <div className="col-span-2 min-w-0">
                <dt className="text-xs font-semibold uppercase tracking-[0.14em] text-foreground/50">
                  Generators
                </dt>
                <dd className="mt-1 text-sm text-foreground">
                  {generators.length === 0
                    ? '—'
                    : generators
                        .map(
                          (g) =>
                            `${g.fuel}${g.powerMW ? ` · ${g.powerMW} MW` : ''}`,
                        )
                        .join(', ')}
                </dd>
              </div>
            </dl>
          </div>

          <div className="rounded-2xl border border-border bg-surface p-6 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-foreground/50">
              Health-impact social cost
            </p>
            {loading && (
              <p className="mt-3 text-sm text-foreground/60">Computing pollution impact…</p>
            )}
            {error && (
              <p className="mt-3 text-sm text-rose-700 dark:text-rose-300">{error}</p>
            )}
            {summary && !loading && !error && (
              <>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-3xl font-semibold tracking-tight text-foreground">
                    {usdFormatter.format(summary.totalUsdPerYear)}
                  </span>
                  <span className="text-sm font-medium text-foreground/60">/ year</span>
                </div>
              </>
            )}
            {!loading && !error && !summary && (
              <p className="mt-3 text-sm text-foreground/60">
                Provide MW, onsite split, and a polygon to compute impact.
              </p>
            )}
          </div>

          <div className="rounded-2xl border border-border bg-surface p-6 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-foreground/50">
              Resource usage
            </p>
            {resourceLoading && (
              <p className="mt-3 text-sm text-foreground/60">Calculating energy and water use…</p>
            )}
            {resourceError && (
              <p className="mt-3 text-sm text-rose-700 dark:text-rose-300">
                {resourceError}
              </p>
            )}
            {resourceUsage && !resourceLoading && !resourceError && (
              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-foreground/50">
                    Annual energy
                  </p>
                  <p className="mt-1 font-mono text-base tabular-nums text-foreground">
                    {compactNumber.format(resourceUsage.annual_facility_energy_kwh)} kWh
                  </p>
                </div>
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-foreground/50">
                    PUE
                  </p>
                  <p className="mt-1 font-mono text-base tabular-nums text-foreground">
                    {resourceUsage.pue.toFixed(2)}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-foreground/50">
                    Annual water
                  </p>
                  <p className="mt-1 font-mono text-base tabular-nums text-foreground">
                    {compactNumber.format(resourceUsage.water_usage_gallons_per_year)} gal
                  </p>
                </div>
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-foreground/50">
                    WUE
                  </p>
                  <p className="mt-1 font-mono text-base tabular-nums text-foreground">
                    {resourceUsage.site_wue_l_per_kwh.toFixed(3)} L/kWh
                  </p>
                </div>
                <div className="col-span-2 rounded-lg border border-border bg-surface-muted/50 px-3 py-2 text-xs text-foreground/70">
                  <div className="flex justify-between gap-3">
                    <span>Evaporation</span>
                    <span className="font-mono tabular-nums">
                      {compactNumber.format(resourceUsage.evaporation_water_liters_per_year)} L/yr
                    </span>
                  </div>
                  <div className="mt-1 flex justify-between gap-3">
                    <span>Blowdown / drift</span>
                    <span className="font-mono tabular-nums">
                      {compactNumber.format(
                        resourceUsage.blowdown_water_liters_per_year +
                          resourceUsage.drift_water_liters_per_year,
                      )}{' '}
                      L/yr
                    </span>
                  </div>
                  <div className="mt-1 flex justify-between gap-3">
                    <span>Humidification</span>
                    <span className="font-mono tabular-nums">
                      {compactNumber.format(resourceUsage.humidification_water_liters_per_year)} L/yr
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          <div className="rounded-2xl border border-border bg-surface p-6 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-foreground/50">
              Utility price impact
            </p>
            {priceLoading && (
              <p className="mt-3 text-sm text-foreground/60">Estimating utility price response…</p>
            )}
            {priceError && (
              <p className="mt-3 text-sm text-rose-700 dark:text-rose-300">
                {priceError}
              </p>
            )}
            {(waterPrice || electricityPrice) && !priceLoading && !priceError && (
              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-foreground/50">
                    Water
                  </p>
                  <p className="mt-1 font-mono text-base tabular-nums text-foreground">
                    {waterPrice ? `$${waterPrice.price_change.toFixed(4)}` : '—'}
                  </p>
                </div>
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-foreground/50">
                    Electricity
                  </p>
                  <p className="mt-1 font-mono text-base tabular-nums text-foreground">
                    {electricityPrice ? `$${electricityPrice.price_change.toFixed(4)}` : '—'}
                  </p>
                </div>
                <p className="col-span-2 text-xs text-foreground/55">
                  Baseline estimate uses the backend LA water and electricity demand tables.
                </p>
              </div>
            )}
          </div>

          {summary && summary.naaqsViolationCount > 0 && (
            <div className="rounded-2xl border border-rose-300 bg-rose-50 p-5 shadow-sm dark:border-rose-900 dark:bg-rose-950/40">
              <div className="flex items-center gap-2">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-4 w-4 text-rose-700 dark:text-rose-300"
                  aria-hidden="true"
                >
                  <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
                  <path d="M12 9v4" />
                  <path d="M12 17h.01" />
                </svg>
                <span className="text-xs font-semibold uppercase tracking-[0.14em] text-rose-800 dark:text-rose-200">
                  NAAQS PM2.5 violations
                </span>
              </div>
              <p className="mt-2 text-2xl font-semibold tracking-tight text-rose-900 dark:text-rose-100">
                {summary.naaqsViolationCount.toLocaleString()}
                <span className="ml-2 text-sm font-medium text-rose-800/80 dark:text-rose-300/70">
                  {summary.naaqsViolationCount === 1 ? 'county' : 'counties'} above 35 μg/m³
                </span>
              </p>
              <p className="mt-2 text-xs text-rose-800/80 dark:text-rose-300/70">
                Modeled ambient PM2.5 exceeds the EPA 24-hour standard. Permit risk is elevated in these counties.
              </p>
            </div>
          )}

          {summary && summary.naaqsViolationCount === 0 && !loading && !error && (
            <div className="rounded-2xl border border-emerald-300 bg-emerald-50 p-5 shadow-sm dark:border-emerald-900 dark:bg-emerald-950/40">
              <div className="flex items-center gap-2">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-4 w-4 text-emerald-700 dark:text-emerald-300"
                  aria-hidden="true"
                >
                  <path d="M20 6 9 17l-5-5" />
                </svg>
                <span className="text-xs font-semibold uppercase tracking-[0.14em] text-emerald-800 dark:text-emerald-200">
                  NAAQS clear
                </span>
              </div>
              <p className="mt-2 text-sm text-emerald-900/80 dark:text-emerald-100/80">
                Modeled PM2.5 stays below the 35 μg/m³ standard in all counties.
              </p>
            </div>
          )}

          <div className="rounded-2xl border border-border bg-surface p-6 shadow-sm">
            <p className="text-xs font-semibold uppercase tracking-[0.18em] text-foreground/50">
              Permit-timeline forecast
            </p>
            {permitLoading && (
              <p className="mt-3 text-sm text-foreground/60">
                Reasoning over historical permit dataset…
              </p>
            )}
            {permitError && (
              <p className="mt-3 text-sm text-rose-700 dark:text-rose-300">
                {permitError}
              </p>
            )}
            {permit && !permitLoading && !permitError && (
              <>
                <div className="mt-2 flex items-baseline gap-2">
                  <span className="text-3xl font-semibold tracking-tight text-foreground tabular-nums">
                    {permit.expected_days_to_first_approval}
                  </span>
                  <span className="text-sm font-medium text-foreground/60">days to first approval</span>
                </div>
                <p className="mt-1 text-xs text-foreground/55 tabular-nums">
                  80% range: {permit.days_ci_low}–{permit.days_ci_high} days ·{' '}
                  {(permit.expected_days_to_first_approval / 30.4).toFixed(1)} mo midpoint
                </p>
                <div className="mt-3 flex items-baseline gap-2">
                  <span
                    className={`text-2xl font-semibold tracking-tight ${
                      APPROVAL_LIKELIHOOD_TONE[approvalLikelihood(permit.p_approved)]
                    }`}
                  >
                    {approvalLikelihood(permit.p_approved)}
                  </span>
                  <span className="text-xs font-medium text-foreground/60">
                    likelihood of approval
                  </span>
                </div>
                {permit.key_factors && permit.key_factors.length > 0 && (
                  <div className="mt-4">
                    <h3 className="text-[10px] font-semibold uppercase tracking-[0.14em] text-foreground/55">
                      Key factors
                    </h3>
                    <ul className="mt-1.5 space-y-1 text-xs text-foreground/80">
                      {permit.key_factors.slice(0, 5).map((f, i) => (
                        <li key={i} className="flex gap-2">
                          <span className="text-foreground/40">·</span>
                          <span className="flex-1">{f}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {permit.most_similar_cases && permit.most_similar_cases.length > 0 && (
                  <details className="mt-3 group">
                    <summary className="cursor-pointer text-[11px] font-semibold uppercase tracking-[0.14em] text-foreground/55 hover:text-foreground/80">
                      Similar historical cases ({permit.most_similar_cases.length})
                    </summary>
                    <ul className="mt-2 space-y-2 text-xs">
                      {permit.most_similar_cases.map((c, i) => (
                        <li key={i} className="rounded-lg border border-border bg-surface-muted/40 px-3 py-2">
                          <div className="flex items-center justify-between gap-2">
                            <span className="font-semibold text-foreground">{c.name}</span>
                            <span className="rounded bg-surface px-1.5 py-0.5 text-[10px] font-semibold text-foreground/65">
                              {c.outcome}
                            </span>
                          </div>
                          <p className="mt-1 text-foreground/65">{c.why_similar}</p>
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </>
            )}
            {!permit && !permitLoading && !permitError && (
              <p className="mt-3 text-sm text-foreground/60">
                Provide MW and a polygon to forecast permit timeline.
              </p>
            )}
          </div>

          {summary && !loading && !error && (
            <button
              onClick={openSaveModal}
              className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-accent-foreground shadow-sm transition hover:bg-accent-hover"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="h-4 w-4"
                aria-hidden="true"
              >
                <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z" />
                <path d="M17 21v-8H7v8" />
                <path d="M7 3v5h8" />
              </svg>
              Save site & results
            </button>
          )}
          {savedToast && (
            <p className="text-center text-xs font-medium text-emerald-700 dark:text-emerald-300">
              {savedToast}
            </p>
          )}

          <button
            onClick={() => router.back()}
            className="inline-flex w-full items-center justify-center gap-2 rounded-lg border border-border bg-surface px-4 py-2.5 text-sm font-medium text-foreground transition hover:border-border-strong hover:bg-surface-muted"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-4 w-4"
              aria-hidden="true"
            >
              <path d="m15 18-6-6 6-6" />
            </svg>
            Back to dashboard
          </button>
        </div>
      </aside>
      <div className="flex w-2/3 flex-col">
        <div className="flex-1 min-h-0 border-b border-border">
          <MapComponent
            polygon={polygon}
            isInteractive={false}
            liveData={liveData}
            liveStatus={liveStatus}
            generators={generators}
          />
        </div>
        <div className="flex-1 min-h-0 relative">
          <NameInputModal
            open={saveModalOpen}
            title="Save site & results"
            description="Give this site a name so you can compare it with other saved analyses."
            initialValue="Untitled site"
            placeholder="Untitled site"
            submitLabel="Save"
            onClose={() => setSaveModalOpen(false)}
            onSubmit={commitSave}
            extra={
              summary ? (
                <div className="rounded-lg border border-border bg-surface-muted/60 px-3 py-2 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-semibold text-foreground/65">
                      Total social cost
                    </span>
                    <span className="font-mono tabular-nums text-foreground">
                      {usdFormatter.format(summary.totalUsdPerYear)}/yr
                    </span>
                  </div>
                  <div className="mt-1 flex items-center justify-between">
                    <span className="font-semibold text-foreground/65">
                      NAAQS Compliant
                    </span>
                    <span
                      className={`font-semibold ${
                        summary.naaqsViolationCount === 0
                          ? 'text-emerald-700 dark:text-emerald-300'
                          : 'text-rose-700 dark:text-rose-300'
                      }`}
                    >
                      {summary.naaqsViolationCount === 0 ? 'yes' : 'no'}
                    </span>
                  </div>
                  {permit && (
                    <>
                      <div className="mt-1 flex items-center justify-between">
                        <span className="font-semibold text-foreground/65">
                          Days to first approval
                        </span>
                        <span className="font-mono tabular-nums text-foreground">
                          {permit.expected_days_to_first_approval}
                        </span>
                      </div>
                      <div className="mt-1 flex items-center justify-between">
                        <span className="font-semibold text-foreground/65">
                          Likelihood of approval
                        </span>
                        <span
                          className={`font-semibold ${
                            APPROVAL_LIKELIHOOD_TONE[approvalLikelihood(permit.p_approved)]
                          }`}
                        >
                          {approvalLikelihood(permit.p_approved)}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              ) : null
            }
          />

          <div className="absolute left-3 top-3 z-[400] rounded-md border border-border bg-surface/95 px-3 py-1.5 text-[10px] font-semibold uppercase tracking-[0.16em] text-foreground/70 shadow backdrop-blur">
            County health-cost choropleth
          </div>
          {summary ? (
            <CountyChoroplethMap
              costByFips={costByFips}
              highlightFips={summary.naaqsFips}
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center bg-surface-muted/40 text-sm text-foreground/55">
              {loading
                ? 'Computing county-level impact…'
                : error
                  ? 'County map unavailable — analysis failed.'
                  : 'County map will appear once analysis runs.'}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function Analysis() {
  return (
    <Suspense
      fallback={
        <div className="flex flex-1 items-center justify-center text-sm text-foreground/60">
          Loading analysis…
        </div>
      }
    >
      <AnalysisContent />
    </Suspense>
  );
}
