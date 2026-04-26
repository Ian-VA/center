export type GeneratorFuel = 'Diesel' | 'Natural Gas';
export type GeneratorMode = 'prime' | 'backup';

export const GENERATOR_FUEL_OPTIONS: GeneratorFuel[] = ['Diesel', 'Natural Gas'];
export const GENERATOR_MODE_OPTIONS: { value: GeneratorMode; label: string }[] = [
  { value: 'backup', label: 'Backup' },
  { value: 'prime', label: 'Prime power' },
];

export const DEFAULT_BACKUP_RUN_HOURS = 100;

export interface OnsiteGenerator {
  id: string;
  fuel: GeneratorFuel;
  lat: number | null;
  lon: number | null;
  powerMW: number;
  mode: GeneratorMode;
  runHours: number;
}

export function newGenerator(): OnsiteGenerator {
  const id =
    typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
      ? crypto.randomUUID()
      : `g-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  return {
    id,
    fuel: 'Diesel',
    lat: null,
    lon: null,
    powerMW: 0,
    mode: 'backup',
    runHours: DEFAULT_BACKUP_RUN_HOURS,
  };
}

export function encodeGenerators(generators: OnsiteGenerator[]): string {
  return encodeURIComponent(JSON.stringify(generators));
}

export function decodeGenerators(value: string | null | undefined): OnsiteGenerator[] {
  if (!value) return [];
  try {
    const parsed = JSON.parse(decodeURIComponent(value));
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((g): g is OnsiteGenerator => g && typeof g === 'object' && 'id' in g && 'fuel' in g)
      .map((g) => ({
        id: String(g.id),
        fuel: g.fuel === 'Natural Gas' ? 'Natural Gas' : 'Diesel',
        lat: typeof g.lat === 'number' ? g.lat : null,
        lon: typeof g.lon === 'number' ? g.lon : null,
        powerMW: typeof g.powerMW === 'number' ? g.powerMW : 0,
        mode: g.mode === 'prime' ? 'prime' : 'backup',
        runHours:
          typeof g.runHours === 'number' && Number.isFinite(g.runHours)
            ? g.runHours
            : DEFAULT_BACKUP_RUN_HOURS,
      }));
  } catch {
    return [];
  }
}
