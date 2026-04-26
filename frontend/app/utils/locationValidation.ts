import type { LatLng } from './area';

const US_REGIONS: { minLat: number; maxLat: number; minLng: number; maxLng: number }[] = [
  { minLat: 24.4, maxLat: 49.5, minLng: -125.0, maxLng: -66.5 },
  { minLat: 51.0, maxLat: 71.5, minLng: -180.0, maxLng: -129.0 },
  { minLat: 18.5, maxLat: 22.3, minLng: -160.5, maxLng: -154.5 },
  { minLat: 17.5, maxLat: 18.6, minLng: -67.5, maxLng: -65.2 },
];

const WATER_TYPES = new Set([
  'water',
  'bay',
  'sea',
  'ocean',
  'lake',
  'river',
  'reservoir',
  'wetland',
  'pond',
  'strait',
  'lagoon',
  'fjord',
  'stream',
  'canal',
]);

export interface ValidationResult {
  ok: boolean;
  reason?: string;
}

export function isWithinUSBounds([lat, lng]: LatLng): boolean {
  return US_REGIONS.some(
    (b) => lat >= b.minLat && lat <= b.maxLat && lng >= b.minLng && lng <= b.maxLng,
  );
}

interface NominatimReverseResponse {
  class?: string;
  type?: string;
  address?: { country_code?: string };
}

export async function validatePoint(
  point: LatLng,
  signal?: AbortSignal,
): Promise<ValidationResult> {
  if (!isWithinUSBounds(point)) {
    return { ok: false, reason: 'Site must be within the United States.' };
  }
  try {
    const [lat, lng] = point;
    const response = await fetch(
      `https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json&zoom=14`,
      { signal, headers: { Accept: 'application/json' } },
    );
    if (!response.ok) return { ok: true };
    const data = (await response.json()) as NominatimReverseResponse;
    const country = data.address?.country_code?.toLowerCase();
    if (!country) {
      return { ok: false, reason: 'Site appears to be over open water.' };
    }
    if (country !== 'us') {
      return { ok: false, reason: 'Site must be within the United States.' };
    }
    const klass = String(data.class ?? '').toLowerCase();
    const type = String(data.type ?? '').toLowerCase();
    if (klass === 'water' || WATER_TYPES.has(type)) {
      return { ok: false, reason: 'Site cannot be placed over water.' };
    }
  } catch {
    // Network/CORS — don't block on transient failures.
  }
  return { ok: true };
}
