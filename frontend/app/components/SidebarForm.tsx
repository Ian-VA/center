'use client';

import { useRouter } from 'next/navigation';
import { FormEvent, useEffect, useMemo, useState } from 'react';
import {
  encodePolygon,
  polygonAreaSquareFeet,
  type LatLng,
} from '../utils/area';
import { polygonsOverlap } from '../utils/spatial';
import {
  SENSITIVE_LAYERS,
  type SensitiveCategory,
  type SensitiveFeature,
} from '../data/sensitive-layers';
import type { SavedAnalysis } from '../utils/savedAnalyses';
import { encodeGenerators, type OnsiteGenerator } from '../utils/generators';
import { approvalLikelihood } from '../utils/api';
import NameInputModal from './NameInputModal';

interface SidebarFormProps {
  polygon: LatLng[];
  setPolygon: (polygon: LatLng[]) => void;
  setCenterOn: (point: LatLng | undefined) => void;
  mwUsage: number;
  gridUsage: number;
  onsiteUsage: number;
  setMwUsage: (value: number) => void;
  setGridUsage: (value: number) => void;
  setOnsiteUsage: (value: number) => void;
  generators: OnsiteGenerator[];
  liveData?: Partial<Record<SensitiveCategory, SensitiveFeature[]>>;
  validationError?: string | null;
  savedAnalyses: SavedAnalysis[];
  currentAnalysis: SavedAnalysis | null;
  onSaveAnalysis: () => void;
  onLoadAnalysis: (id: string) => void;
  onDeleteAnalysis: (id: string) => void;
  onNewAnalysis: () => void;
  onRenameAnalysis: (id: string, name: string) => void;
}

const inputClass =
  'mt-1.5 block w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-foreground shadow-sm transition placeholder:text-foreground/40 focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30';

const labelClass =
  'block text-xs font-semibold uppercase tracking-[0.12em] text-foreground/60 whitespace-nowrap';

const ghostButtonClass =
  'inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-border bg-surface px-3 py-1.5 text-xs font-semibold text-foreground/80 transition hover:border-border-strong hover:bg-surface-muted disabled:cursor-not-allowed disabled:opacity-50';

function formatShortUsd(v: number): string {
  const abs = Math.abs(v);
  const sign = v < 0 ? '-' : '';
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

const numberInputProps = (
  raw: string,
  onRawChange: (value: string) => void,
  onNumberChange: (value: number) => void,
) => ({
  value: raw,
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => {
    const v = e.target.value;
    onRawChange(v);
    if (v === '' || v === '-' || v === '.') {
      onNumberChange(0);
    } else {
      const parsed = Number(v);
      if (!Number.isNaN(parsed)) onNumberChange(parsed);
    }
  },
});

const SidebarForm: React.FC<SidebarFormProps> = ({
  polygon,
  setPolygon,
  setCenterOn,
  mwUsage,
  gridUsage,
  onsiteUsage,
  setMwUsage,
  setGridUsage,
  setOnsiteUsage,
  generators,
  liveData,
  validationError,
  savedAnalyses,
  currentAnalysis,
  onSaveAnalysis,
  onLoadAnalysis,
  onDeleteAnalysis,
  onNewAnalysis,
  onRenameAnalysis,
}) => {
  const router = useRouter();

  const [address, setAddress] = useState('');
  const [geocoding, setGeocoding] = useState(false);
  const [savedMenuOpen, setSavedMenuOpen] = useState(false);
  const [renameTarget, setRenameTarget] = useState<SavedAnalysis | null>(null);

  const [mwInput, setMwInput] = useState<string>(mwUsage ? String(mwUsage) : '');
  const [gridInput, setGridInput] = useState<string>(String(gridUsage));
  const [onsiteInput, setOnsiteInput] = useState<string>(String(onsiteUsage));

  const parseInput = (raw: string): number => {
    if (raw === '' || raw === '-' || raw === '.') return 0;
    const n = Number(raw);
    return Number.isNaN(n) ? 0 : n;
  };

  useEffect(() => {
    if (parseInput(mwInput) !== mwUsage) {
      setMwInput(mwUsage ? String(mwUsage) : '');
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mwUsage]);

  useEffect(() => {
    if (parseInput(gridInput) !== gridUsage) {
      setGridInput(String(gridUsage));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gridUsage]);

  useEffect(() => {
    if (parseInput(onsiteInput) !== onsiteUsage) {
      setOnsiteInput(String(onsiteUsage));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [onsiteUsage]);

  const areaSqFt = useMemo(() => polygonAreaSquareFeet(polygon), [polygon]);
  const polygonComplete = polygon.length >= 3;

  const requiredOnsiteMW = useMemo(
    () => Math.max(0, (mwUsage * onsiteUsage) / 100),
    [mwUsage, onsiteUsage],
  );
  // Only PRIME generators provide continuous onsite power. Backup gens run a few hours
  // a year — they don't satisfy the onsite-share requirement, but they still count
  // toward emissions in the backend (run_hours-weighted).
  const placedPrimeMW = useMemo(
    () =>
      generators.reduce((sum, g) => {
        const placed = g.lat !== null && g.lon !== null;
        if (!placed || g.mode !== 'prime') return sum;
        return sum + (Number.isFinite(g.powerMW) ? g.powerMW : 0);
      }, 0),
    [generators],
  );
  const placedPrimeCount = useMemo(
    () =>
      generators.filter(
        (g) => g.lat !== null && g.lon !== null && g.mode === 'prime',
      ).length,
    [generators],
  );
  const onsiteShortfall = Math.max(0, requiredOnsiteMW - placedPrimeMW);
  const onsiteCovered =
    onsiteUsage === 0 ||
    (placedPrimeCount > 0 && onsiteShortfall <= 1e-6);
  const canRunAnalysis = polygonComplete && onsiteCovered;

  const overlaps = useMemo(() => {
    if (!polygonComplete) return [] as { category: SensitiveCategory; label: string; feature: SensitiveFeature }[];
    const hits: { category: SensitiveCategory; label: string; feature: SensitiveFeature }[] = [];
    for (const layer of SENSITIVE_LAYERS) {
      const features = liveData?.[layer.id] ?? layer.features;
      for (const feature of features) {
        if (polygonsOverlap(polygon, feature.polygon)) {
          hits.push({ category: layer.id, label: layer.label, feature });
        }
      }
    }
    return hits;
  }, [polygon, polygonComplete, liveData]);

  const banHit = overlaps.some((h) => h.category === 'ban');

  const handleGeocode = async () => {
    if (!address) return;
    setGeocoding(true);
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(address)}`,
      );
      const data = await response.json();
      if (data && data[0]) {
        setCenterOn([Number(data[0].lat), Number(data[0].lon)]);
      } else {
        alert('Address not found.');
      }
    } catch {
      alert('Error geocoding address.');
    } finally {
      setGeocoding(false);
    }
  };

  const undoVertex = () => {
    if (polygon.length === 0) return;
    setPolygon(polygon.slice(0, -1));
  };

  const clearPolygon = () => setPolygon([]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!polygonComplete) {
      alert('Draw a polygon with at least 3 points on the map first.');
      return;
    }
    if (!onsiteCovered) {
      const msg = placedPrimeCount === 0
        ? `Onsite usage > 0%, so at least one prime-power generator must be placed inside the site (${requiredOnsiteMW.toFixed(1)} MW total).`
        : `Prime generators must cover ${requiredOnsiteMW.toFixed(1)} MW. Place ${onsiteShortfall.toFixed(1)} MW more of prime power.`;
      alert(msg);
      return;
    }
    const params = new URLSearchParams({
      polygon: encodePolygon(polygon),
      area: String(Math.round(areaSqFt)),
      mw: String(mwUsage),
      grid: String(gridUsage),
      onsite: String(onsiteUsage),
      generators: encodeGenerators(generators),
    });
    // If we're viewing a saved site that already has cached results, signal the
    // analysis page to load them instead of re-running compute + predict.
    if (currentAnalysis?.result?.fullResult && currentAnalysis?.result?.permitPrediction) {
      params.set('saved', currentAnalysis.id);
    }
    router.push(`/analysis?${params.toString()}`);
  };


  return (
    <div className="space-y-3">
      <div className="rounded-2xl border border-border bg-surface px-3 py-2 shadow-sm">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setSavedMenuOpen((v) => !v)}
            className="flex flex-1 items-center gap-2 rounded-lg px-2 py-1.5 text-left transition hover:bg-surface-muted"
            aria-expanded={savedMenuOpen}
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-4 w-4 text-foreground/55"
              aria-hidden="true"
            >
              <path d="M4 4h6l2 2h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2Z" />
            </svg>
            <span className="flex-1 truncate text-xs font-semibold text-foreground">
              {currentAnalysis ? currentAnalysis.name : 'Saved analyses'}
            </span>
            <span className="text-[10px] font-semibold text-foreground/45">
              {savedAnalyses.length}
            </span>
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className={`h-3.5 w-3.5 text-foreground/55 transition-transform ${savedMenuOpen ? 'rotate-180' : ''}`}
              aria-hidden="true"
            >
              <path d="m6 9 6 6 6-6" />
            </svg>
          </button>
          <button
            type="button"
            onClick={() => {
              onNewAnalysis();
              setSavedMenuOpen(false);
            }}
            className="inline-flex items-center gap-1 rounded-lg border border-border bg-surface-muted px-2.5 py-1.5 text-[11px] font-semibold text-foreground/80 transition hover:border-border-strong hover:bg-surface"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-3.5 w-3.5"
              aria-hidden="true"
            >
              <path d="M12 5v14" />
              <path d="M5 12h14" />
            </svg>
            New
          </button>
        </div>
        {savedMenuOpen && (
          <ul className="mt-1 max-h-60 overflow-y-auto border-t border-border pt-1">
            {savedAnalyses.length === 0 && (
              <li className="px-2 py-3 text-center text-[11px] text-foreground/50">
                No saved analyses yet. Use Save below to keep one.
              </li>
            )}
            {savedAnalyses.map((analysis) => {
              const isCurrent = currentAnalysis?.id === analysis.id;
              return (
                <li
                  key={analysis.id}
                  className={`group flex items-center gap-1 rounded-lg px-1 ${
                    isCurrent ? 'bg-accent-soft' : 'hover:bg-surface-muted'
                  }`}
                >
                  <button
                    type="button"
                    onClick={() => {
                      onLoadAnalysis(analysis.id);
                      setSavedMenuOpen(false);
                    }}
                    className="flex-1 truncate px-2 py-1.5 text-left text-xs"
                  >
                    <span
                      className={`block truncate font-medium ${isCurrent ? 'text-accent' : 'text-foreground'}`}
                    >
                      {analysis.name}
                    </span>
                    <span className="block text-[10px] text-foreground/45">
                      {new Date(analysis.updatedAt).toLocaleDateString()} ·{' '}
                      {analysis.polygon.length} pts
                    </span>
                    {analysis.result && (
                      <>
                        <span className="mt-0.5 block text-[10px] font-mono tabular-nums text-foreground/65">
                          {formatShortUsd(analysis.result.totalUsdPerYear)}/yr
                          {analysis.result.naaqsViolationCount > 0 && (
                            <span className="ml-1 rounded bg-rose-100 px-1 text-[9px] font-semibold uppercase text-rose-700 dark:bg-rose-950/60 dark:text-rose-300">
                              {analysis.result.naaqsViolationCount} NAAQS
                            </span>
                          )}
                        </span>
                        {(analysis.result.permitDays != null ||
                          analysis.result.permitProbability != null) && (
                          <span className="block truncate text-[10px] font-mono tabular-nums text-foreground/55">
                            {analysis.result.permitDays != null && (
                              <>{analysis.result.permitDays}d</>
                            )}
                            {analysis.result.permitDays != null &&
                              analysis.result.permitProbability != null && (
                                <span> · </span>
                              )}
                            {analysis.result.permitProbability != null && (
                              <>{approvalLikelihood(analysis.result.permitProbability).toLowerCase()}</>
                            )}
                          </span>
                        )}
                      </>
                    )}
                  </button>
                  <button
                    type="button"
                    onClick={() => setRenameTarget(analysis)}
                    className="rounded p-1 text-foreground/40 opacity-0 transition hover:text-foreground/70 group-hover:opacity-100"
                    aria-label="Rename"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="h-3.5 w-3.5"
                      aria-hidden="true"
                    >
                      <path d="M12 20h9" />
                      <path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L7 19l-4 1 1-4Z" />
                    </svg>
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      if (window.confirm(`Delete "${analysis.name}"?`)) {
                        onDeleteAnalysis(analysis.id);
                      }
                    }}
                    className="rounded p-1 text-foreground/40 opacity-0 transition hover:text-rose-600 group-hover:opacity-100"
                    aria-label="Delete"
                  >
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="h-3.5 w-3.5"
                      aria-hidden="true"
                    >
                      <path d="M3 6h18" />
                      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                    </svg>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    <form
      onSubmit={handleSubmit}
      className="rounded-2xl border border-border bg-surface p-6 shadow-sm"
      suppressHydrationWarning
    >
      <div className="mb-5">
        {!currentAnalysis && (
          <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-accent">
            New Analysis
          </p>
        )}
        <h2 className={`${currentAnalysis ? '' : 'mt-1'} text-lg font-semibold tracking-tight text-foreground`}>
          Proposed Data Center Details
        </h2>
        <p className="mt-1 text-sm text-foreground/60">
          Sketch a site boundary to model permit acceptance and pollution impact.
        </p>
      </div>

      <div className="space-y-5">
        {validationError && (
          <div className="rounded-lg border border-rose-300 bg-rose-50 px-3 py-2 text-xs font-medium text-rose-800 shadow-sm dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-200">
            {validationError}
          </div>
        )}
        <div>
          <label className={labelClass} htmlFor="address">
            Address <span className="text-foreground/40 normal-case tracking-normal">(optional)</span>
          </label>
          <input
            id="address"
            type="text"
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="123 Server Lane, Ashburn, VA"
            className={inputClass}
          />
          <button
            type="button"
            onClick={handleGeocode}
            disabled={geocoding || !address}
            suppressHydrationWarning
            className="mt-2 inline-flex w-full items-center justify-center gap-2 rounded-lg bg-secondary-soft px-4 py-2 text-sm font-semibold text-secondary-hover ring-1 ring-secondary/30 transition hover:bg-secondary hover:text-white hover:ring-secondary disabled:cursor-not-allowed disabled:opacity-50"
          >
            {geocoding ? 'Geocoding…' : 'Find on map'}
          </button>
        </div>

        <div className="rounded-xl border border-border bg-surface-muted/60 p-4">
          <div className="flex items-center justify-between">
            <span className={labelClass}>Site Boundary</span>
            <span className="text-[11px] font-semibold text-foreground/50">
              {polygon.length} {polygon.length === 1 ? 'point' : 'points'}
            </span>
          </div>

          <div className="mt-3 flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="min-w-0 break-all text-2xl font-semibold leading-tight tracking-tight text-foreground tabular-nums [overflow-wrap:anywhere]">
              {polygonComplete
                ? Math.round(areaSqFt).toLocaleString()
                : polygon.length > 0
                  ? '—'
                  : '0'}
            </span>
            <span className="text-sm font-medium text-foreground/60">ft²</span>
          </div>

          {polygon.length > 0 && polygon.length < 3 && (
            <p className="mt-3 text-xs text-foreground/50">
              Add {3 - polygon.length} more {3 - polygon.length === 1 ? 'point' : 'points'} to close the polygon.
            </p>
          )}

          <div className="mt-3 flex gap-2">
            <button
              type="button"
              onClick={undoVertex}
              disabled={polygon.length === 0}
              suppressHydrationWarning
              className={ghostButtonClass}
            >
              Undo
            </button>
            <button
              type="button"
              onClick={clearPolygon}
              disabled={polygon.length === 0}
              suppressHydrationWarning
              className={ghostButtonClass}
            >
              Clear
            </button>
          </div>
        </div>

        {overlaps.length > 0 && (
          <div
            className={`rounded-xl border px-4 py-3 shadow-sm ${
              banHit
                ? 'border-rose-300 bg-rose-50 dark:border-rose-900 dark:bg-rose-950/40'
                : 'border-amber-300 bg-amber-50 dark:border-amber-900 dark:bg-amber-950/40'
            }`}
          >
            <div className="flex items-center gap-2">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                className={`h-4 w-4 ${banHit ? 'text-rose-700 dark:text-rose-300' : 'text-amber-700 dark:text-amber-300'}`}
                aria-hidden="true"
              >
                <path d="M12 9v4" />
                <path d="M12 17h.01" />
                <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0Z" />
              </svg>
              <span
                className={`text-xs font-semibold uppercase tracking-[0.14em] ${
                  banHit
                    ? 'text-rose-800 dark:text-rose-200'
                    : 'text-amber-800 dark:text-amber-200'
                }`}
              >
                {banHit ? 'Build prohibited' : 'Sensitive area conflict'}
              </span>
            </div>
            <ul className="mt-2 space-y-1.5 text-xs">
              {overlaps.map(({ category, label, feature }) => (
                <li
                  key={feature.id}
                  className={`flex items-start gap-2 ${
                    banHit
                      ? 'text-rose-900 dark:text-rose-100'
                      : 'text-amber-900 dark:text-amber-100'
                  }`}
                >
                  <span
                    className={`mt-1 inline-block h-2 w-2 flex-shrink-0 rounded-full ${
                      category === 'ban'
                        ? 'bg-rose-600'
                        : category === 'pushback'
                          ? 'bg-blue-600'
                          : category === 'park' || category === 'tribal'
                            ? 'bg-red-600'
                            : 'bg-yellow-500'
                    }`}
                  />
                  <span className="flex-1 break-words">
                    <span className="font-semibold">{feature.name}</span>
                    <span className="opacity-70"> · {label}</span>
                  </span>
                </li>
              ))}
            </ul>
            {banHit && (
              <p className="mt-2 text-[11px] font-medium text-rose-800/80 dark:text-rose-300/70">
                This jurisdiction has prohibited new data center construction.
              </p>
            )}
          </div>
        )}

        <div>
          <label className={labelClass} htmlFor="mw">
            MW Usage
          </label>
          <input
            id="mw"
            type="number"
            inputMode="decimal"
            placeholder="e.g. 25"
            className={inputClass}
            required
            {...numberInputProps(mwInput, setMwInput, setMwUsage)}
          />
        </div>

        <div className="grid grid-cols-2 items-end gap-3">
          <div>
            <label className={labelClass} htmlFor="grid">
              % Grid Usage
            </label>
            <input
              id="grid"
              type="number"
              inputMode="numeric"
              min={0}
              max={100}
              placeholder="0"
              className={inputClass}
              required
              {...numberInputProps(gridInput, setGridInput, setGridUsage)}
            />
          </div>
          <div>
            <label className={labelClass} htmlFor="onsite">
              % Onsite Usage
            </label>
            <input
              id="onsite"
              type="number"
              inputMode="numeric"
              min={0}
              max={100}
              placeholder="0"
              className={inputClass}
              required
              {...numberInputProps(onsiteInput, setOnsiteInput, setOnsiteUsage)}
            />
          </div>
        </div>

        {polygonComplete && onsiteUsage > 0 && !onsiteCovered && (
          <div className="rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs dark:border-amber-900 dark:bg-amber-950/40">
            <div className="font-semibold text-amber-800 dark:text-amber-200">
              {placedPrimeCount === 0
                ? 'Prime generator required'
                : 'Prime power underspecified'}
            </div>
            <p className="mt-1 text-amber-900/80 dark:text-amber-100/80">
              {placedPrimeCount === 0 ? (
                <>
                  Onsite usage is {onsiteUsage}% — at least one{' '}
                  <strong>prime-power</strong> generator must be placed inside the site
                  to provide continuous {requiredOnsiteMW.toFixed(1)} MW. Backup
                  generators can be added too, but only prime gens count toward the
                  onsite share.
                </>
              ) : (
                <>
                  Prime generators must cover{' '}
                  <span className="font-mono tabular-nums">{requiredOnsiteMW.toFixed(1)} MW</span>{' '}
                  ({onsiteUsage}% of {mwUsage} MW).
                  <span className="block mt-0.5">
                    Placed prime power:{' '}
                    <span className="font-mono tabular-nums">{placedPrimeMW.toFixed(1)} MW</span>{' '}
                    <span className="opacity-70">
                      (need {onsiteShortfall.toFixed(1)} MW more)
                    </span>
                  </span>
                </>
              )}
            </p>
          </div>
        )}

        <div className="flex gap-2">
          <button
            type="button"
            onClick={onSaveAnalysis}
            disabled={!polygonComplete}
            suppressHydrationWarning
            className="inline-flex items-center justify-center gap-1.5 rounded-lg border border-border bg-surface-muted px-4 py-2.5 text-sm font-semibold text-foreground/80 transition hover:border-border-strong hover:bg-surface disabled:cursor-not-allowed disabled:opacity-50"
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
            {currentAnalysis ? 'Update' : 'Save'}
          </button>
          <button
            type="submit"
            className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2.5 text-sm font-semibold text-accent-foreground shadow-sm transition hover:bg-accent-hover focus:outline-none focus:ring-2 focus:ring-accent/40 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={!canRunAnalysis}
            suppressHydrationWarning
          >
            Run analysis
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
              <path d="M5 12h14" />
              <path d="m13 5 7 7-7 7" />
            </svg>
          </button>
        </div>
      </div>
    </form>
      <NameInputModal
        open={renameTarget !== null}
        title="Rename analysis"
        initialValue={renameTarget?.name ?? ''}
        placeholder="Untitled site"
        submitLabel="Rename"
        onClose={() => setRenameTarget(null)}
        onSubmit={(name) => {
          if (renameTarget) onRenameAnalysis(renameTarget.id, name);
          setRenameTarget(null);
        }}
      />
    </div>
  );
};

export default SidebarForm;
