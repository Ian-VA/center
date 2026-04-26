'use client';

import { useEffect, useState } from 'react';
import {
  LIVE_SOURCES,
  fetchGeoFeatures,
  featuresFromParsed,
} from '../utils/geojson';
import type {
  SensitiveCategory,
  SensitiveFeature,
} from '../data/sensitive-layers';

export type LiveStatus = 'idle' | 'loading' | 'ready' | 'failed';

export interface LiveLayersResult {
  liveData: Partial<Record<SensitiveCategory, SensitiveFeature[]>>;
  liveStatus: Record<SensitiveCategory, LiveStatus>;
}

export function useLiveLayers(): LiveLayersResult {
  const [liveData, setLiveData] = useState<
    Partial<Record<SensitiveCategory, SensitiveFeature[]>>
  >({});
  const [liveStatus, setLiveStatus] = useState<
    Record<SensitiveCategory, LiveStatus>
  >({
    populated: 'idle',
    park: 'idle',
    tribal: 'idle',
    ban: 'idle',
    pushback: 'idle',
  });

  useEffect(() => {
    let cancelled = false;
    (Object.keys(LIVE_SOURCES) as SensitiveCategory[]).forEach((category) => {
      const source = LIVE_SOURCES[category];
      if (!source) return;
      setLiveStatus((s) => ({ ...s, [category]: 'loading' }));
      fetchGeoFeatures(source.url, source.prefix)
        .then((parsed) => {
          if (cancelled) return;
          const features = featuresFromParsed(
            parsed,
            source.category,
            source.source,
            source.prefix,
          );
          if (features.length === 0) {
            setLiveStatus((s) => ({ ...s, [category]: 'failed' }));
            return;
          }
          setLiveData((d) => ({ ...d, [category]: features }));
          setLiveStatus((s) => ({ ...s, [category]: 'ready' }));
        })
        .catch(() => {
          if (cancelled) return;
          setLiveStatus((s) => ({ ...s, [category]: 'failed' }));
        });
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return { liveData, liveStatus };
}
