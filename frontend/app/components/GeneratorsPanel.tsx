'use client';

import { useState } from 'react';
import {
  GENERATOR_FUEL_OPTIONS,
  GENERATOR_MODE_OPTIONS,
  type GeneratorFuel,
  type GeneratorMode,
  type OnsiteGenerator,
} from '../utils/generators';
import type { LatLng } from '../utils/area';

const inputClass =
  'mt-1.5 block w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm text-foreground shadow-sm transition placeholder:text-foreground/40 focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30';

const labelClass =
  'block text-xs font-semibold uppercase tracking-[0.12em] text-foreground/60 whitespace-nowrap';

interface Props {
  generators: OnsiteGenerator[];
  pendingGeneratorId: string | null;
  centroid: LatLng | null;
  polygonComplete: boolean;
  onAdd: () => void;
  onUpdate: (id: string, patch: Partial<OnsiteGenerator>) => void;
  onDelete: (id: string) => void;
  onPlace: (id: string) => void;
}

const GeneratorsPanel: React.FC<Props> = ({
  generators,
  pendingGeneratorId,
  centroid,
  polygonComplete,
  onAdd,
  onUpdate,
  onDelete,
  onPlace,
}) => {
  const [open, setOpen] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const toggleExpand = (id: string) =>
    setExpandedId((prev) => (prev === id ? null : id));

  const useCentroid = (id: string) => {
    if (!centroid) return;
    onUpdate(id, { lat: centroid[0], lon: centroid[1] });
  };

  return (
    <div
      className="absolute left-3 top-3 z-[400] w-72 rounded-xl border border-border bg-surface/95 shadow-lg backdrop-blur"
      style={{ pointerEvents: 'auto' }}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left"
        aria-expanded={open}
      >
        <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-foreground/70">
          Onsite Generators
        </span>
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-semibold text-foreground/45">
            {generators.length}
          </span>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className={`h-3.5 w-3.5 text-foreground/60 transition-transform ${open ? 'rotate-180' : ''}`}
            aria-hidden="true"
          >
            <path d="m6 9 6 6 6-6" />
          </svg>
        </div>
      </button>

      {open && (
        <div className="border-t border-border p-2">
          <ul className="space-y-2">
            {generators.map((gen, idx) => {
              const placed = gen.lat !== null && gen.lon !== null;
              const isPending = pendingGeneratorId === gen.id;
              const isExpanded = expandedId === gen.id;
              return (
                <li
                  key={gen.id}
                  className={`rounded-lg border bg-surface shadow-sm ${
                    isPending ? 'border-accent ring-2 ring-accent/30' : 'border-border'
                  }`}
                >
                  <div className="flex items-center gap-2 px-2 py-1.5">
                    <span
                      className="inline-block h-2.5 w-2.5 flex-shrink-0 rounded-full"
                      style={{ backgroundColor: placed ? '#dc2626' : '#cbd5e1' }}
                      aria-hidden="true"
                    />
                    <button
                      type="button"
                      onClick={() => toggleExpand(gen.id)}
                      className="flex-1 truncate text-left"
                    >
                      <span className="block truncate text-xs font-semibold text-foreground">
                        Generator {idx + 1}
                      </span>
                      <span className="block truncate text-[11px] text-foreground/55">
                        {gen.fuel}
                        {gen.powerMW > 0 ? ` · ${gen.powerMW} MW` : ''}
                        {' · '}
                        {gen.mode === 'backup' ? `backup ${gen.runHours} hr/yr` : 'prime'}
                        {' · '}
                        {placed ? 'placed' : 'not placed'}
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={() => onDelete(gen.id)}
                      className="rounded p-1 text-foreground/40 transition hover:text-rose-600"
                      aria-label="Remove generator"
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
                  </div>

                  {isExpanded && (
                    <div className="space-y-2 border-t border-border px-3 pb-3 pt-2">
                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <label className={labelClass} htmlFor={`p-fuel-${gen.id}`}>
                            Fuel type
                          </label>
                          <select
                            id={`p-fuel-${gen.id}`}
                            value={gen.fuel}
                            onChange={(e) =>
                              onUpdate(gen.id, { fuel: e.target.value as GeneratorFuel })
                            }
                            className={inputClass}
                          >
                            {GENERATOR_FUEL_OPTIONS.map((opt) => (
                              <option key={opt} value={opt}>
                                {opt}
                              </option>
                            ))}
                          </select>
                        </div>
                        <div>
                          <label className={labelClass} htmlFor={`p-mode-${gen.id}`}>
                            Mode
                          </label>
                          <select
                            id={`p-mode-${gen.id}`}
                            value={gen.mode}
                            onChange={(e) =>
                              onUpdate(gen.id, { mode: e.target.value as GeneratorMode })
                            }
                            className={inputClass}
                          >
                            {GENERATOR_MODE_OPTIONS.map((opt) => (
                              <option key={opt.value} value={opt.value}>
                                {opt.label}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>

                      {gen.mode === 'backup' && (
                        <div>
                          <label className={labelClass} htmlFor={`p-runhrs-${gen.id}`}>
                            Annual run hours
                          </label>
                          <input
                            id={`p-runhrs-${gen.id}`}
                            type="number"
                            inputMode="numeric"
                            min={0}
                            max={8760}
                            step="10"
                            placeholder="100"
                            value={gen.runHours || ''}
                            onChange={(e) =>
                              onUpdate(gen.id, {
                                runHours: e.target.value === '' ? 0 : Number(e.target.value),
                              })
                            }
                            className={inputClass}
                          />
                        </div>
                      )}

                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <label className={labelClass} htmlFor={`p-lat-${gen.id}`}>
                            Lat
                          </label>
                          <input
                            id={`p-lat-${gen.id}`}
                            type="number"
                            inputMode="decimal"
                            step="0.00001"
                            placeholder="38.97"
                            value={gen.lat ?? ''}
                            onChange={(e) =>
                              onUpdate(gen.id, {
                                lat: e.target.value === '' ? null : Number(e.target.value),
                              })
                            }
                            className={inputClass}
                          />
                        </div>
                        <div>
                          <label className={labelClass} htmlFor={`p-lon-${gen.id}`}>
                            Lon
                          </label>
                          <input
                            id={`p-lon-${gen.id}`}
                            type="number"
                            inputMode="decimal"
                            step="0.00001"
                            placeholder="-77.45"
                            value={gen.lon ?? ''}
                            onChange={(e) =>
                              onUpdate(gen.id, {
                                lon: e.target.value === '' ? null : Number(e.target.value),
                              })
                            }
                            className={inputClass}
                          />
                        </div>
                      </div>

                      <div>
                        <label className={labelClass} htmlFor={`p-mw-${gen.id}`}>
                          Power (MW)
                        </label>
                        <input
                          id={`p-mw-${gen.id}`}
                          type="number"
                          inputMode="decimal"
                          step="0.1"
                          min={0}
                          placeholder="0"
                          value={gen.powerMW || ''}
                          onChange={(e) =>
                            onUpdate(gen.id, {
                              powerMW: e.target.value === '' ? 0 : Number(e.target.value),
                            })
                          }
                          className={inputClass}
                        />
                      </div>

                      <div className="grid grid-cols-2 gap-2">
                        <button
                          type="button"
                          onClick={() => onPlace(gen.id)}
                          disabled={!polygonComplete}
                          className="inline-flex items-center justify-center gap-1 rounded-md bg-accent px-2 py-1.5 text-[11px] font-semibold text-accent-foreground transition hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="h-3 w-3"
                            aria-hidden="true"
                          >
                            <path d="M12 21s-7-7.5-7-12a7 7 0 1 1 14 0c0 4.5-7 12-7 12Z" />
                            <circle cx="12" cy="9" r="2.5" />
                          </svg>
                          {isPending ? 'Click on map…' : placed ? 'Move on map' : 'Place on map'}
                        </button>
                        <button
                          type="button"
                          onClick={() => useCentroid(gen.id)}
                          disabled={!polygonComplete}
                          className="inline-flex items-center justify-center rounded-md bg-accent-soft px-2 py-1.5 text-[11px] font-semibold text-accent ring-1 ring-accent/30 transition hover:bg-accent hover:text-accent-foreground disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          Centroid
                        </button>
                      </div>
                    </div>
                  )}
                </li>
              );
            })}
          </ul>

          <button
            type="button"
            onClick={onAdd}
            className="mt-2 inline-flex w-full items-center justify-center gap-1.5 rounded-lg border border-dashed border-border bg-surface px-3 py-2 text-xs font-semibold text-foreground/70 transition hover:border-border-strong hover:bg-surface-muted"
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
            Add generator
          </button>

          {!polygonComplete && (
            <p className="mt-2 px-1 text-[11px] text-foreground/50">
              Draw the site boundary first to place generators.
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default GeneratorsPanel;
