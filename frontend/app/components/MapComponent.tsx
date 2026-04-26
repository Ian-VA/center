'use client';

import {
  MapContainer,
  TileLayer,
  useMapEvents,
  useMap,
  Polygon,
  Polyline,
  CircleMarker,
  Tooltip,
  ZoomControl,
} from 'react-leaflet';
import { useEffect, useMemo, useState } from 'react';
import 'leaflet/dist/leaflet.css';
import type { LatLng } from '../utils/area';
import {
  SENSITIVE_LAYERS,
  type SensitiveCategory,
  type SensitiveFeature,
} from '../data/sensitive-layers';
import type { LiveStatus } from '../hooks/useLiveLayers';
import type { OnsiteGenerator } from '../utils/generators';

interface MapComponentProps {
  polygon?: LatLng[];
  onPolygonChange?: (polygon: LatLng[]) => void;
  centerOn?: LatLng;
  isInteractive?: boolean;
  initialLayers?: SensitiveCategory[];
  liveData?: Partial<Record<SensitiveCategory, SensitiveFeature[]>>;
  liveStatus?: Record<SensitiveCategory, LiveStatus>;
  generators?: OnsiteGenerator[];
  pendingGeneratorId?: string | null;
  placementError?: string | null;
  onPlaceGenerator?: (id: string, point: LatLng) => void;
  onCancelPlacement?: () => void;
}

const FlyTo: React.FC<{ target?: LatLng }> = ({ target }) => {
  const map = useMap();
  useEffect(() => {
    if (target) {
      map.flyTo(target, Math.max(map.getZoom(), 14), { duration: 0.6 });
    }
  }, [map, target]);
  return null;
};

const ClickHandler: React.FC<{
  enabled: boolean;
  onAdd: (point: LatLng) => void;
  placementMode: boolean;
  onPlace: (point: LatLng) => void;
}> = ({ enabled, onAdd, placementMode, onPlace }) => {
  useMapEvents({
    click(e) {
      const point: LatLng = [e.latlng.lat, e.latlng.lng];
      if (placementMode) {
        onPlace(point);
        return;
      }
      if (!enabled) return;
      onAdd(point);
    },
  });
  return null;
};

const SVG_NS = 'http://www.w3.org/2000/svg';

const buildPattern = (
  id: string,
  baseFill: string,
  stripeColor: string,
  rotation = 45,
): SVGPatternElement => {
  const pattern = document.createElementNS(SVG_NS, 'pattern');
  pattern.setAttribute('id', id);
  pattern.setAttribute('patternUnits', 'userSpaceOnUse');
  pattern.setAttribute('width', '10');
  pattern.setAttribute('height', '10');
  pattern.setAttribute('patternTransform', `rotate(${rotation})`);

  const rect = document.createElementNS(SVG_NS, 'rect');
  rect.setAttribute('width', '10');
  rect.setAttribute('height', '10');
  rect.setAttribute('fill', baseFill);
  rect.setAttribute('fill-opacity', '0.35');
  pattern.appendChild(rect);

  const line = document.createElementNS(SVG_NS, 'line');
  line.setAttribute('x1', '0');
  line.setAttribute('y1', '0');
  line.setAttribute('x2', '0');
  line.setAttribute('y2', '10');
  line.setAttribute('stroke', stripeColor);
  line.setAttribute('stroke-width', '5');
  line.setAttribute('stroke-opacity', '0.95');
  pattern.appendChild(line);

  return pattern;
};

const PatternDefs: React.FC = () => {
  const map = useMap();
  useEffect(() => {
    const inject = (): boolean => {
      const overlayPane = map.getPane('overlayPane');
      if (!overlayPane) return false;
      const svg = overlayPane.querySelector('svg');
      if (!svg) return false;
      if (svg.querySelector('#sensitive-defs')) return true;

      const defs = document.createElementNS(SVG_NS, 'defs');
      defs.setAttribute('id', 'sensitive-defs');
      defs.appendChild(buildPattern('sensitive-stripe-yellow', '#fde68a', '#ca8a04', 45));
      defs.appendChild(buildPattern('sensitive-stripe-red', '#fecaca', '#b91c1c', 45));
      defs.appendChild(buildPattern('sensitive-stripe-red-back', '#fecaca', '#b91c1c', -45));
      svg.insertBefore(defs, svg.firstChild);
      return true;
    };

    if (inject()) return;
    const handler = () => {
      if (inject()) map.off('layeradd', handler);
    };
    map.on('layeradd', handler);
    return () => {
      map.off('layeradd', handler);
    };
  }, [map]);
  return null;
};

const LayerPanel: React.FC<{
  visible: Record<SensitiveCategory, boolean>;
  setVisible: React.Dispatch<
    React.SetStateAction<Record<SensitiveCategory, boolean>>
  >;
}> = ({ visible, setVisible }) => {
  const [open, setOpen] = useState(true);
  return (
    <div
      className="absolute right-3 top-3 z-[400] w-64 rounded-xl border border-border bg-surface/95 shadow-lg backdrop-blur"
      style={{ pointerEvents: 'auto' }}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left"
      >
        <span className="text-[11px] font-semibold uppercase tracking-[0.16em] text-foreground/70">
          Sensitive Areas
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
      </button>
      {open && (
        <ul className="border-t border-border px-1 py-1">
          {SENSITIVE_LAYERS.map((layer) => (
            <li key={layer.id}>
              <label className="flex cursor-pointer items-start gap-2 rounded-lg px-2 py-1.5 transition hover:bg-surface-muted">
                <input
                  type="checkbox"
                  className="mt-0.5 h-4 w-4 rounded border-border text-accent focus:ring-accent/30"
                  checked={visible[layer.id]}
                  onChange={(e) =>
                    setVisible((prev) => ({
                      ...prev,
                      [layer.id]: e.target.checked,
                    }))
                  }
                />
                <span
                  aria-hidden="true"
                  className="mt-1 inline-block h-3 w-5 flex-shrink-0 rounded border border-border-strong/60"
                  style={{ backgroundImage: layer.swatch.startsWith('repeating') ? layer.swatch : undefined, backgroundColor: layer.swatch.startsWith('repeating') ? '#fff' : layer.swatch }}
                />
                <span className="flex-1">
                  <span className="block text-xs font-semibold text-foreground">
                    {layer.label}
                  </span>
                  <span className="block text-[11px] text-foreground/55">
                    {layer.description}
                  </span>
                </span>
              </label>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

const MapComponent: React.FC<MapComponentProps> = ({
  polygon = [],
  onPolygonChange,
  centerOn,
  isInteractive = true,
  initialLayers = ['populated', 'park', 'tribal', 'ban', 'pushback'],
  liveData = {},
  liveStatus = {
    populated: 'idle',
    park: 'idle',
    tribal: 'idle',
    ban: 'idle',
    pushback: 'idle',
  },
  generators = [],
  pendingGeneratorId = null,
  placementError = null,
  onPlaceGenerator,
  onCancelPlacement,
}) => {
  const initialCenter: LatLng = polygon[0] ?? centerOn ?? [37.0902, -95.7129];
  const initialZoom = polygon.length > 0 || centerOn ? 14 : 4;

  const [visibleLayers, setVisibleLayers] = useState<
    Record<SensitiveCategory, boolean>
  >({
    populated: initialLayers.includes('populated'),
    park: initialLayers.includes('park'),
    tribal: initialLayers.includes('tribal'),
    ban: initialLayers.includes('ban'),
    pushback: initialLayers.includes('pushback'),
  });

  const layerFeatures = useMemo(() => {
    const out: Record<SensitiveCategory, SensitiveFeature[]> = {
      populated: [],
      park: [],
      tribal: [],
      ban: [],
      pushback: [],
    };
    SENSITIVE_LAYERS.forEach((layer) => {
      out[layer.id] = liveData[layer.id] ?? layer.features;
    });
    return out;
  }, [liveData]);

  const handleAdd = (point: LatLng) => {
    if (!onPolygonChange) return;
    onPolygonChange([...polygon, point]);
  };

  const handleRemove = (index: number) => {
    if (!onPolygonChange) return;
    onPolygonChange(polygon.filter((_, i) => i !== index));
  };

  const accent = '#4f46e5';
  const generatorColor = '#dc2626';

  const placementMode = pendingGeneratorId !== null;
  const pendingIndex = placementMode
    ? generators.findIndex((g) => g.id === pendingGeneratorId)
    : -1;

  useEffect(() => {
    if (!placementMode || !onCancelPlacement) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancelPlacement();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [placementMode, onCancelPlacement]);

  return (
    <div className="relative h-full w-full">
      <MapContainer
        center={initialCenter}
        zoom={initialZoom}
        zoomControl={false}
        style={{
          height: '100%',
          width: '100%',
          cursor: placementMode ? 'crosshair' : undefined,
        }}
      >
        <ZoomControl position="bottomleft" />
        <PatternDefs />
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        />

        {SENSITIVE_LAYERS.map((layer) =>
          visibleLayers[layer.id]
            ? layerFeatures[layer.id].map((feature) => (
                <Polygon
                  key={feature.id}
                  positions={feature.polygon}
                  pathOptions={{
                    color: layer.stroke,
                    weight: 1.5,
                    fillColor: layer.fill,
                    fillOpacity: layer.fill.startsWith('url(') ? 1 : 0.35,
                    interactive: true,
                  }}
                >
                  <Tooltip direction="center" sticky opacity={0.95}>
                    <span className="text-xs font-semibold">{feature.name}</span>
                    <br />
                    <span className="text-[10px] opacity-70">{feature.source}</span>
                  </Tooltip>
                </Polygon>
              ))
            : null,
        )}

        {polygon.length >= 3 && (
          <Polygon
            positions={polygon}
            pathOptions={{
              color: accent,
              weight: 2,
              fillColor: accent,
              fillOpacity: 0.18,
            }}
          />
        )}

        {polygon.length === 2 && (
          <Polyline
            positions={polygon}
            pathOptions={{ color: accent, weight: 2, dashArray: '4 4' }}
          />
        )}

        {polygon.map((point, i) => (
          <CircleMarker
            key={i}
            center={point}
            radius={6}
            pathOptions={{
              color: '#ffffff',
              weight: 2,
              fillColor: accent,
              fillOpacity: 1,
              bubblingMouseEvents: false,
              interactive: isInteractive,
            }}
            eventHandlers={
              isInteractive
                ? {
                    click: () => handleRemove(i),
                  }
                : undefined
            }
          />
        ))}

        {generators.map((gen, i) =>
          gen.lat !== null && gen.lon !== null ? (
            <CircleMarker
              key={gen.id}
              center={[gen.lat, gen.lon]}
              radius={8}
              pathOptions={{
                color: '#ffffff',
                weight: 2,
                fillColor: generatorColor,
                fillOpacity: 1,
                bubblingMouseEvents: false,
              }}
            >
              <Tooltip direction="top" opacity={0.95}>
                <span className="text-xs font-semibold">
                  Generator {i + 1} · {gen.fuel}
                </span>
                {gen.powerMW > 0 && (
                  <>
                    <br />
                    <span className="text-[10px] opacity-70">{gen.powerMW} MW</span>
                  </>
                )}
              </Tooltip>
            </CircleMarker>
          ) : null,
        )}

        <ClickHandler
          enabled={isInteractive}
          onAdd={handleAdd}
          placementMode={placementMode}
          onPlace={(point) => {
            if (pendingGeneratorId && onPlaceGenerator) {
              onPlaceGenerator(pendingGeneratorId, point);
            }
          }}
        />
        <FlyTo target={centerOn} />
      </MapContainer>

      {placementMode && (
        <div
          className={`absolute left-1/2 top-3 z-[400] -translate-x-1/2 rounded-full border px-4 py-2 shadow-lg backdrop-blur ${
            placementError
              ? 'border-rose-300 bg-rose-50/95 dark:border-rose-900 dark:bg-rose-950/80'
              : 'border-border bg-surface/95'
          }`}
          style={{ pointerEvents: 'auto' }}
        >
          <div className="flex items-center gap-3 text-xs">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: placementError ? '#dc2626' : generatorColor }}
            />
            <span
              className={`font-semibold ${
                placementError ? 'text-rose-800 dark:text-rose-200' : 'text-foreground'
              }`}
            >
              {placementError
                ? placementError
                : pendingIndex >= 0
                  ? `Click inside the site boundary to place Generator ${pendingIndex + 1}`
                  : 'Click inside the site boundary to place generator'}
            </span>
            <button
              type="button"
              onClick={onCancelPlacement}
              className="rounded-md border border-border bg-surface-muted px-2 py-0.5 text-[11px] font-semibold text-foreground/70 transition hover:border-border-strong hover:bg-surface"
            >
              Cancel (Esc)
            </button>
          </div>
        </div>
      )}

      <LayerPanel visible={visibleLayers} setVisible={setVisibleLayers} />
    </div>
  );
};

export default MapComponent;
