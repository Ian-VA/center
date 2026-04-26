'use client';

import { useEffect, useMemo, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON, ZoomControl } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import type { Feature, FeatureCollection, Geometry } from 'geojson';
import type { Layer, PathOptions } from 'leaflet';

interface Props {
  costByFips: Record<string, number>;
  highlightFips?: string[];
}

const COUNTY_GEOJSON_URL =
  'https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json';
const CACHE_KEY = 'center.us-counties-geojson.v1';

interface CountyProps {
  GEO_ID?: string;
  NAME?: string;
  STATE?: string;
  LSAD?: string;
}

async function loadCountiesGeoJson(): Promise<FeatureCollection<Geometry, CountyProps>> {
  if (typeof window !== 'undefined') {
    try {
      const cached = window.sessionStorage.getItem(CACHE_KEY);
      if (cached) return JSON.parse(cached);
    } catch {}
  }
  const res = await fetch(COUNTY_GEOJSON_URL);
  if (!res.ok) throw new Error(`GeoJSON fetch failed: ${res.status}`);
  const data = (await res.json()) as FeatureCollection<Geometry, CountyProps>;
  if (typeof window !== 'undefined') {
    try {
      window.sessionStorage.setItem(CACHE_KEY, JSON.stringify(data));
    } catch {}
  }
  return data;
}

const usdFormatter = new Intl.NumberFormat('en-US', {
  style: 'currency',
  currency: 'USD',
  maximumFractionDigits: 0,
});

const RAMP = [
  '#fef3c7',
  '#fde68a',
  '#fbbf24',
  '#f59e0b',
  '#d97706',
  '#b45309',
  '#9a3412',
  '#7c2d12',
];

interface LinearScale {
  max: number;
}

function buildLinearScale(values: number[]): LinearScale {
  // Cap the ramp at the SECOND-largest county. The user's home county usually dominates
  // total cost (it's the emission source); pinning the legend to that single outlier
  // squashes every other county into the lightest swatch. Capping at #2 keeps the rest
  // of the country visible; the #1 county just clamps to the darkest swatch.
  let first = 0;
  let second = 0;
  for (const v of values) {
    if (!Number.isFinite(v)) continue;
    const abs = Math.abs(v);
    if (abs > first) {
      second = first;
      first = abs;
    } else if (abs > second) {
      second = abs;
    }
  }
  return { max: second > 0 ? second : first };
}

function colorFor(value: number, scale: LinearScale): string {
  if (!Number.isFinite(value) || value <= 0 || scale.max <= 0) return RAMP[0];
  const t = Math.min(1, value / scale.max);
  // Map [0, 1] across the ramp; t === 1 must hit the darkest swatch.
  if (t >= 1) return RAMP[RAMP.length - 1];
  return RAMP[Math.min(RAMP.length - 1, Math.floor(t * RAMP.length))];
}

const CountyChoroplethMap: React.FC<Props> = ({ costByFips, highlightFips }) => {
  const [geo, setGeo] = useState<FeatureCollection<Geometry, CountyProps> | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    loadCountiesGeoJson()
      .then((g) => {
        if (!cancelled) setGeo(g);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load counties');
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const positiveValues = useMemo(() => {
    const out: number[] = [];
    for (const v of Object.values(costByFips)) {
      if (Number.isFinite(v) && Math.abs(v) > 0) out.push(Math.abs(v));
    }
    return out;
  }, [costByFips]);

  const scale = useMemo(() => buildLinearScale(positiveValues), [positiveValues]);
  const highlightSet = useMemo(() => new Set(highlightFips ?? []), [highlightFips]);

  const styleFn = useMemo(
    () => (feature?: Feature<Geometry, CountyProps>): PathOptions => {
      const fips = feature?.id != null ? String(feature.id).padStart(5, '0') : '';
      const v = costByFips[fips];
      const isNaaqs = highlightSet.has(fips);
      if (v == null || !Number.isFinite(v)) {
        return {
          color: '#cbd5e1',
          weight: 0.4,
          fillColor: '#f8fafc',
          fillOpacity: 0.6,
        };
      }
      return {
        color: isNaaqs ? '#7f1d1d' : '#475569',
        weight: isNaaqs ? 1.4 : 0.4,
        fillColor: colorFor(Math.abs(v), scale),
        fillOpacity: 0.85,
      };
    },
    [costByFips, scale, highlightSet],
  );

  const onEachFeature = useMemo(
    () => (feature: Feature<Geometry, CountyProps>, layer: Layer) => {
      const fips = feature?.id != null ? String(feature.id).padStart(5, '0') : '';
      const v = costByFips[fips];
      const props = feature.properties ?? {};
      const stateFips = fips.slice(0, 2);
      const name = props.NAME
        ? `${props.NAME}${props.LSAD ? ' ' + props.LSAD : ''}, ${stateFips}`
        : `FIPS ${fips}`;
      const isNaaqs = highlightSet.has(fips);
      const tag = isNaaqs ? ' · NAAQS exceedance' : '';
      const cost =
        typeof v === 'number' && Number.isFinite(v)
          ? `${usdFormatter.format(v)}/yr`
          : 'no data';
      layer.bindTooltip(`<strong>${name}</strong><br/>${cost}${tag}`, {
        sticky: false,
        direction: 'top',
        opacity: 0.95,
      });
      // Close any open tooltip when the user starts a drag — otherwise leaflet leaves
      // stale tooltips floating across the screen as you pan.
      layer.on('mousedown', () => layer.closeTooltip());
      layer.on('mouseout', () => layer.closeTooltip());
    },
    [costByFips, highlightSet],
  );

  return (
    <div className="relative h-full w-full">
      <MapContainer
        center={[39, -97]}
        zoom={4}
        zoomControl={false}
        style={{ height: '100%', width: '100%' }}
        worldCopyJump={false}
      >
        <ZoomControl position="bottomleft" />
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; OpenStreetMap contributors'
          opacity={0.4}
        />
        {geo && (
          <GeoJSON
            data={geo}
            style={styleFn as never}
            onEachFeature={onEachFeature}
          />
        )}
      </MapContainer>

      {!geo && !error && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-surface/40 text-xs font-medium text-foreground/70">
          Loading county boundaries…
        </div>
      )}
      {error && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-surface/80 px-4 text-center text-xs font-medium text-rose-700 dark:text-rose-300">
          County map unavailable: {error}
        </div>
      )}

      <ChoroplethLegend scale={scale} highlightCount={highlightSet.size} />
    </div>
  );
};

const ChoroplethLegend: React.FC<{ scale: LinearScale; highlightCount: number }> = ({
  scale,
  highlightCount,
}) => {
  const ticks = [0, scale.max / 2, scale.max];
  return (
    <div
      className="absolute right-3 bottom-3 z-[400] w-60 rounded-xl border border-border bg-surface/95 p-3 shadow-lg backdrop-blur"
      style={{ pointerEvents: 'auto' }}
    >
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-foreground/65">
        Health cost / yr
      </div>
      <div className="flex h-3 w-full overflow-hidden rounded">
        {RAMP.map((c) => (
          <div key={c} className="flex-1" style={{ backgroundColor: c }} />
        ))}
      </div>
      <div className="mt-1 flex justify-between text-[10px] font-mono tabular-nums text-foreground/65">
        {ticks.map((t, i) => (
          <span key={i}>{formatShort(t)}</span>
        ))}
      </div>
      {highlightCount > 0 && (
        <div className="mt-2 flex items-center gap-1.5 text-[10px] text-foreground/70">
          <span
            aria-hidden="true"
            className="inline-block h-2.5 w-2.5 rounded-sm border-2"
            style={{ borderColor: '#7f1d1d', background: 'transparent' }}
          />
          <span>NAAQS PM2.5 exceedance ({highlightCount})</span>
        </div>
      )}
    </div>
  );
};

function formatShort(v: number): string {
  const abs = Math.abs(v);
  const sign = v < 0 ? '-' : '';
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(1)}M`;
  if (abs >= 1e3) return `${sign}$${(abs / 1e3).toFixed(1)}K`;
  return `${sign}$${abs.toFixed(0)}`;
}

export default CountyChoroplethMap;
