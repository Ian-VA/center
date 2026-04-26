const EARTH_RADIUS_M = 6378137;
const SQ_M_PER_SQ_FT = 0.09290304;

const toRad = (deg: number) => (deg * Math.PI) / 180;

export type LatLng = [number, number];

export function polygonAreaSquareMeters(vertices: LatLng[]): number {
  if (vertices.length < 3) return 0;
  let area = 0;
  const n = vertices.length;
  for (let i = 0; i < n; i++) {
    const [lat1, lng1] = vertices[i];
    const [lat2, lng2] = vertices[(i + 1) % n];
    area +=
      toRad(lng2 - lng1) *
      (2 + Math.sin(toRad(lat1)) + Math.sin(toRad(lat2)));
  }
  return Math.abs((area * EARTH_RADIUS_M * EARTH_RADIUS_M) / 2);
}

export function polygonAreaSquareFeet(vertices: LatLng[]): number {
  return polygonAreaSquareMeters(vertices) / SQ_M_PER_SQ_FT;
}

export function polygonCentroid(vertices: LatLng[]): LatLng | null {
  if (vertices.length === 0) return null;
  const sum = vertices.reduce(
    (acc, [lat, lng]) => [acc[0] + lat, acc[1] + lng] as LatLng,
    [0, 0] as LatLng,
  );
  return [sum[0] / vertices.length, sum[1] / vertices.length];
}

export function formatSquareFeet(sqft: number): string {
  if (!Number.isFinite(sqft) || sqft <= 0) return '0 ft²';
  const rounded = sqft.toLocaleString(undefined, { maximumFractionDigits: 0 });
  return `${rounded} ft²`;
}

export function encodePolygon(vertices: LatLng[]): string {
  return vertices.map(([lat, lng]) => `${lat},${lng}`).join(';');
}

const POWER_DENSITY_W_PER_SQFT = 100;

export function estimateMegawatts(sqft: number): number {
  if (!Number.isFinite(sqft) || sqft <= 0) return 0;
  return (sqft * POWER_DENSITY_W_PER_SQFT) / 1_000_000;
}

export function decodePolygon(value: string | null | undefined): LatLng[] {
  if (!value) return [];
  return value
    .split(';')
    .map((pair) => {
      const [lat, lng] = pair.split(',').map(Number);
      return [lat, lng] as LatLng;
    })
    .filter(([lat, lng]) => Number.isFinite(lat) && Number.isFinite(lng));
}
