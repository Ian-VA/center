import type { LatLng } from './area';
import type { SensitiveCategory, SensitiveFeature } from '../data/sensitive-layers';

interface RawFeature {
  type?: string;
  geometry?:
    | { type: 'Polygon'; coordinates: number[][][] }
    | { type: 'MultiPolygon'; coordinates: number[][][][] }
    | { type: string; coordinates: unknown };
  properties?: Record<string, unknown>;
}

interface RawCollection {
  type?: string;
  features?: RawFeature[];
}

const NAME_KEYS = [
  'UNIT_NAME',
  'PARKNAME',
  'NAME',
  'NAME10',
  'NAME20',
  'name',
  'GEOID',
  'LARNAME',
  'LARName',
];

const pickName = (props: Record<string, unknown> | undefined): string => {
  if (!props) return 'Unknown';
  for (const key of NAME_KEYS) {
    const value = props[key];
    if (typeof value === 'string' && value.trim().length > 0) return value;
  }
  return 'Unknown';
};

const ringToLatLng = (ring: number[][]): LatLng[] =>
  ring.map(([lng, lat]) => [lat, lng] as LatLng);

interface ParsedFeature {
  name: string;
  polygons: LatLng[][];
}

export function parseGeoJson(raw: unknown): ParsedFeature[] {
  const collection = raw as RawCollection;
  if (!collection || !Array.isArray(collection.features)) return [];

  return collection.features.flatMap((feature) => {
    const geom = feature.geometry;
    if (!geom) return [];
    const polygons: LatLng[][] = [];

    if (geom.type === 'Polygon') {
      const rings = geom.coordinates as number[][][];
      if (rings[0]) polygons.push(ringToLatLng(rings[0]));
    } else if (geom.type === 'MultiPolygon') {
      const polys = geom.coordinates as number[][][][];
      polys.forEach((poly) => {
        if (poly[0]) polygons.push(ringToLatLng(poly[0]));
      });
    }

    if (polygons.length === 0) return [];
    return [{ name: pickName(feature.properties), polygons }];
  });
}

const CACHE_PREFIX = 'center.geo-cache.v1.';
const CACHE_TTL_MS = 24 * 60 * 60 * 1000;

function readCache(key: string): ParsedFeature[] | null {
  if (typeof window === 'undefined' || !key) return null;
  try {
    const raw = window.localStorage.getItem(CACHE_PREFIX + key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { ts?: number; data?: ParsedFeature[] };
    if (
      typeof parsed.ts !== 'number' ||
      !Array.isArray(parsed.data) ||
      Date.now() - parsed.ts > CACHE_TTL_MS
    ) {
      return null;
    }
    return parsed.data;
  } catch {
    return null;
  }
}

function writeCache(key: string, data: ParsedFeature[]): void {
  if (typeof window === 'undefined' || !key) return;
  try {
    window.localStorage.setItem(
      CACHE_PREFIX + key,
      JSON.stringify({ ts: Date.now(), data }),
    );
  } catch {
    // Quota exceeded — caching is best-effort.
  }
}

async function fetchPage(url: string): Promise<{ data: unknown; exceeded: boolean }> {
  const response = await fetch(url, {
    headers: { Accept: 'application/geo+json,application/json' },
  });
  if (!response.ok) {
    throw new Error(`fetch ${url} failed with ${response.status}`);
  }
  const data = await response.json();
  const exceeded =
    (data && typeof data === 'object' && 'exceededTransferLimit' in data
      ? (data as { exceededTransferLimit?: boolean }).exceededTransferLimit
      : false) ?? false;
  return { data, exceeded };
}

const PAGE_SIZE = 500;
const MAX_PAGES = 12;

async function fetchTotalCount(url: string): Promise<number | null> {
  try {
    const sep = url.includes('?') ? '&' : '?';
    const countUrl = `${url.replace(/f=geojson/, 'f=json')}${sep}returnCountOnly=true`;
    const response = await fetch(countUrl, {
      headers: { Accept: 'application/json' },
    });
    if (!response.ok) return null;
    const json = (await response.json()) as { count?: number };
    return typeof json.count === 'number' ? json.count : null;
  } catch {
    return null;
  }
}

export async function fetchGeoFeatures(
  url: string,
  cacheKey?: string,
): Promise<ParsedFeature[]> {
  if (cacheKey) {
    const cached = readCache(cacheKey);
    if (cached) return cached;
  }

  const sep = url.includes('?') ? '&' : '?';
  const total = await fetchTotalCount(url);
  const pages =
    total !== null && total > 0
      ? Math.min(Math.ceil(total / PAGE_SIZE), MAX_PAGES)
      : MAX_PAGES;

  const pageUrls = Array.from(
    { length: pages },
    (_, i) =>
      `${url}${sep}resultOffset=${i * PAGE_SIZE}&resultRecordCount=${PAGE_SIZE}`,
  );

  const settled = await Promise.allSettled(pageUrls.map(fetchPage));
  const out: ParsedFeature[] = [];
  for (const result of settled) {
    if (result.status === 'fulfilled') {
      out.push(...parseGeoJson(result.value.data));
    }
  }

  if (cacheKey && out.length > 0) writeCache(cacheKey, out);
  return out;
}

export function featuresFromParsed(
  parsed: ParsedFeature[],
  category: SensitiveCategory,
  source: string,
  prefix: string,
): SensitiveFeature[] {
  return parsed.flatMap((item, i) =>
    item.polygons.map((polygon, j) => ({
      id: `${prefix}-${i}-${j}`,
      name: item.name,
      category,
      source,
      polygon,
    })),
  );
}

export interface LiveLayerSource {
  category: SensitiveCategory;
  url: string;
  source: string;
  prefix: string;
}

// Public Census TIGERweb endpoints that return GeoJSON with detailed boundaries.
// Verified live and CORS-enabled. If any endpoint moves or rate-limits, the
// bundled bbox dataset acts as the fallback.
const TIGER = 'https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb';
const queryTail = (precision: number, maxAllowableOffset?: number) => {
  const offsetParam =
    maxAllowableOffset !== undefined
      ? `&maxAllowableOffset=${maxAllowableOffset}`
      : '';
  return `query?where=1%3D1&outFields=NAME&geometryPrecision=${precision}&outSR=4326&f=geojson${offsetParam}`;
};

export const LIVE_SOURCES: Partial<Record<SensitiveCategory, LiveLayerSource>> = {
  park: {
    category: 'park',
    url: `${TIGER}/Special_Land_Use_Areas/MapServer/0/${queryTail(5)}`,
    source: 'Census TIGERweb · NPS Areas (live)',
    prefix: 'park-live',
  },
  tribal: {
    category: 'tribal',
    url: `${TIGER}/AIANNHA/MapServer/2/${queryTail(5)}`,
    source: 'Census TIGERweb · Federal AIR (live)',
    prefix: 'tribal-live',
  },
  populated: {
    category: 'populated',
    url: `${TIGER}/tigerWMS_Census2020/MapServer/88/${queryTail(4, 0.002)}`,
    source: 'Census TIGERweb · 2020 Urban Areas Corrected (live)',
    prefix: 'pop-live',
  },
};
