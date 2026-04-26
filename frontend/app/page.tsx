'use client';

import { useCallback, useEffect, useState } from 'react';
import dynamic from 'next/dynamic';

const MapComponent = dynamic(() => import('./components/MapComponent'), { ssr: false });
import SidebarForm from './components/SidebarForm';
import GeneratorsPanel from './components/GeneratorsPanel';
import NameInputModal from './components/NameInputModal';
import { polygonCentroid, type LatLng } from './utils/area';
import { pointInPolygon } from './utils/spatial';
import { useLiveLayers } from './hooks/useLiveLayers';
import { isWithinUSBounds, validatePoint } from './utils/locationValidation';
import { newGenerator, type OnsiteGenerator } from './utils/generators';
import {
  generateAnalysisId,
  loadSavedAnalyses,
  persistSavedAnalyses,
  type SavedAnalysis,
} from './utils/savedAnalyses';

const DRAFT_KEY = 'center.dashboard-draft.v1';

interface DashboardDraft {
  polygon: LatLng[];
  mwUsage: number;
  gridUsage: number;
  onsiteUsage: number;
  generators: OnsiteGenerator[];
  currentAnalysisId: string | null;
}

function loadDraft(): DashboardDraft | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.sessionStorage.getItem(DRAFT_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || !Array.isArray(parsed.polygon)) return null;
    return {
      polygon: parsed.polygon,
      mwUsage: typeof parsed.mwUsage === 'number' ? parsed.mwUsage : 0,
      gridUsage: typeof parsed.gridUsage === 'number' ? parsed.gridUsage : 50,
      onsiteUsage: typeof parsed.onsiteUsage === 'number' ? parsed.onsiteUsage : 50,
      generators: Array.isArray(parsed.generators) ? parsed.generators : [],
      currentAnalysisId:
        typeof parsed.currentAnalysisId === 'string' ? parsed.currentAnalysisId : null,
    };
  } catch {
    return null;
  }
}

function saveDraft(draft: DashboardDraft): void {
  if (typeof window === 'undefined') return;
  try {
    window.sessionStorage.setItem(DRAFT_KEY, JSON.stringify(draft));
  } catch {}
}

function clearDraft(): void {
  if (typeof window === 'undefined') return;
  try {
    window.sessionStorage.removeItem(DRAFT_KEY);
  } catch {}
}

export default function Dashboard() {
  const [polygon, setPolygon] = useState<LatLng[]>([]);
  const [centerOn, setCenterOn] = useState<LatLng | undefined>();
  const [mwUsage, setMwUsage] = useState<number>(0);
  const [gridUsage, setGridUsage] = useState<number>(50);
  const [onsiteUsage, setOnsiteUsage] = useState<number>(50);
  const [generators, setGenerators] = useState<OnsiteGenerator[]>([]);
  const [pendingGeneratorId, setPendingGeneratorId] = useState<string | null>(null);
  const [placementError, setPlacementError] = useState<string | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [saveNameModalOpen, setSaveNameModalOpen] = useState(false);
  const [savedAnalyses, setSavedAnalyses] = useState<SavedAnalysis[]>([]);
  const [currentAnalysisId, setCurrentAnalysisId] = useState<string | null>(null);

  const { liveData, liveStatus } = useLiveLayers();
  const [draftHydrated, setDraftHydrated] = useState(false);

  useEffect(() => {
    setSavedAnalyses(loadSavedAnalyses());
    const draft = loadDraft();
    if (draft) {
      setPolygon(draft.polygon);
      setMwUsage(draft.mwUsage);
      setGridUsage(draft.gridUsage);
      setOnsiteUsage(draft.onsiteUsage);
      setGenerators(draft.generators);
      setCurrentAnalysisId(draft.currentAnalysisId);
    }
    setDraftHydrated(true);
  }, []);

  useEffect(() => {
    if (!draftHydrated) return;
    saveDraft({ polygon, mwUsage, gridUsage, onsiteUsage, generators, currentAnalysisId });
  }, [draftHydrated, polygon, mwUsage, gridUsage, onsiteUsage, generators, currentAnalysisId]);

  // Auto-sync the loaded saved analysis with the current form. When you load an analysis,
  // tweak (or don't tweak) anything, then navigate away and back, the saved entry in
  // localStorage stays in lockstep. Skip writes when the form already matches the saved
  // version so we don't bump updatedAt for nothing.
  useEffect(() => {
    if (!draftHydrated || !currentAnalysisId) return;
    setSavedAnalyses((prev) => {
      let mutated = false;
      const next = prev.map((a) => {
        if (a.id !== currentAnalysisId) return a;
        const samePolygon =
          a.polygon.length === polygon.length &&
          a.polygon.every((p, i) => p[0] === polygon[i][0] && p[1] === polygon[i][1]);
        const sameGens =
          a.generators.length === generators.length &&
          a.generators.every((g, i) => {
            const o = generators[i];
            return (
              g.id === o.id &&
              g.fuel === o.fuel &&
              g.lat === o.lat &&
              g.lon === o.lon &&
              g.powerMW === o.powerMW
            );
          });
        if (
          a.mwUsage === mwUsage &&
          a.gridUsage === gridUsage &&
          a.onsiteUsage === onsiteUsage &&
          samePolygon &&
          sameGens
        ) {
          return a;
        }
        mutated = true;
        return {
          ...a,
          polygon,
          mwUsage,
          gridUsage,
          onsiteUsage,
          generators,
          updatedAt: Date.now(),
        };
      });
      if (!mutated) return prev;
      persistSavedAnalyses(next);
      return next;
    });
  }, [draftHydrated, currentAnalysisId, polygon, mwUsage, gridUsage, onsiteUsage, generators]);

  useEffect(() => {
    if (!validationError) return;
    const t = setTimeout(() => setValidationError(null), 4500);
    return () => clearTimeout(t);
  }, [validationError]);

  const handlePolygonChange = useCallback(
    (next: LatLng[]) => {
      if (next.length <= polygon.length) {
        setPolygon(next);
        return;
      }
      const newPoint = next[next.length - 1];
      if (!isWithinUSBounds(newPoint)) {
        setValidationError('Site must be within the United States.');
        return;
      }
      setPolygon(next);
      validatePoint(newPoint).then((result) => {
        if (!result.ok) {
          setValidationError(result.reason ?? 'Invalid location.');
          setPolygon((prev) => prev.filter((p) => p !== newPoint));
        }
      });
    },
    [polygon],
  );

  const currentAnalysis =
    savedAnalyses.find((a) => a.id === currentAnalysisId) ?? null;

  const handleSaveAnalysis = useCallback(() => {
    if (polygon.length < 3) {
      setValidationError('Draw a polygon before saving.');
      return;
    }
    if (currentAnalysisId) {
      const next = savedAnalyses.map((a) =>
        a.id === currentAnalysisId
          ? {
              ...a,
              polygon,
              mwUsage,
              gridUsage,
              onsiteUsage,
              generators,
              updatedAt: Date.now(),
            }
          : a,
      );
      setSavedAnalyses(next);
      persistSavedAnalyses(next);
      return;
    }
    setSaveNameModalOpen(true);
  }, [polygon, currentAnalysisId, savedAnalyses, mwUsage, gridUsage, onsiteUsage, generators]);

  const handleConfirmSaveName = useCallback(
    (name: string) => {
      const id = generateAnalysisId();
      const now = Date.now();
      const item: SavedAnalysis = {
        id,
        name,
        createdAt: now,
        updatedAt: now,
        polygon,
        mwUsage,
        gridUsage,
        onsiteUsage,
        generators,
      };
      const next = [item, ...savedAnalyses];
      setSavedAnalyses(next);
      persistSavedAnalyses(next);
      setCurrentAnalysisId(id);
      setSaveNameModalOpen(false);
    },
    [polygon, mwUsage, gridUsage, onsiteUsage, generators, savedAnalyses],
  );

  const handleLoadAnalysis = useCallback(
    (id: string) => {
      const item = savedAnalyses.find((a) => a.id === id);
      if (!item) return;
      setPolygon(item.polygon);
      setMwUsage(item.mwUsage);
      setGridUsage(item.gridUsage);
      setOnsiteUsage(item.onsiteUsage);
      setGenerators(item.generators ?? []);
      setCurrentAnalysisId(id);
      if (item.polygon.length > 0) setCenterOn(item.polygon[0]);
    },
    [savedAnalyses],
  );

  const handleDeleteAnalysis = useCallback(
    (id: string) => {
      const next = savedAnalyses.filter((a) => a.id !== id);
      setSavedAnalyses(next);
      persistSavedAnalyses(next);
      if (currentAnalysisId === id) setCurrentAnalysisId(null);
    },
    [savedAnalyses, currentAnalysisId],
  );

  const handleNewAnalysis = useCallback(() => {
    setPolygon([]);
    setMwUsage(0);
    setGridUsage(50);
    setOnsiteUsage(50);
    setGenerators([]);
    setPendingGeneratorId(null);
    setCurrentAnalysisId(null);
    setCenterOn(undefined);
    clearDraft();
  }, []);

  const setGridUsageLinked = useCallback((value: number) => {
    const clamped = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));
    setGridUsage(clamped);
    setOnsiteUsage(100 - clamped);
  }, []);

  const setOnsiteUsageLinked = useCallback((value: number) => {
    const clamped = Math.max(0, Math.min(100, Number.isFinite(value) ? value : 0));
    setOnsiteUsage(clamped);
    setGridUsage(100 - clamped);
  }, []);

  const handlePlaceGenerator = useCallback(
    (id: string, point: LatLng) => {
      if (polygon.length < 3 || !pointInPolygon(point, polygon)) {
        setPlacementError('Generator must be inside the site boundary.');
        return;
      }
      setPlacementError(null);
      setGenerators((prev) =>
        prev.map((g) => (g.id === id ? { ...g, lat: point[0], lon: point[1] } : g)),
      );
      setPendingGeneratorId(null);
    },
    [polygon],
  );

  const handleCancelPlacement = useCallback(() => {
    setPendingGeneratorId(null);
    setPlacementError(null);
  }, []);

  const handleAddGenerator = useCallback(() => {
    const gen = newGenerator();
    setGenerators((prev) => [...prev, gen]);
    if (polygon.length >= 3) setPendingGeneratorId(gen.id);
  }, [polygon.length]);

  const handleUpdateGenerator = useCallback(
    (id: string, patch: Partial<OnsiteGenerator>) => {
      setGenerators((prev) => prev.map((g) => (g.id === id ? { ...g, ...patch } : g)));
    },
    [],
  );

  const handleDeleteGenerator = useCallback((id: string) => {
    setGenerators((prev) => prev.filter((g) => g.id !== id));
    setPendingGeneratorId((prev) => (prev === id ? null : prev));
  }, []);

  const handleStartPlacement = useCallback((id: string) => {
    setPendingGeneratorId(id);
    setPlacementError(null);
  }, []);

  useEffect(() => {
    if (!placementError) return;
    const t = setTimeout(() => setPlacementError(null), 2500);
    return () => clearTimeout(t);
  }, [placementError]);

  const handleRenameAnalysis = useCallback(
    (id: string, name: string) => {
      const trimmed = name.trim();
      if (!trimmed) return;
      const next = savedAnalyses.map((a) =>
        a.id === id ? { ...a, name: trimmed, updatedAt: Date.now() } : a,
      );
      setSavedAnalyses(next);
      persistSavedAnalyses(next);
    },
    [savedAnalyses],
  );

  return (
    <div className="flex flex-1 min-h-0" suppressHydrationWarning>
      <aside
        className="sidebar-surface w-1/4 min-w-[320px] border-r border-border p-6 overflow-y-auto"
        suppressHydrationWarning
      >
        <SidebarForm
          polygon={polygon}
          setPolygon={setPolygon}
          setCenterOn={setCenterOn}
          mwUsage={mwUsage}
          gridUsage={gridUsage}
          onsiteUsage={onsiteUsage}
          setMwUsage={setMwUsage}
          setGridUsage={setGridUsageLinked}
          setOnsiteUsage={setOnsiteUsageLinked}
          generators={generators}
          liveData={liveData}
          validationError={validationError}
          savedAnalyses={savedAnalyses}
          currentAnalysis={currentAnalysis}
          onSaveAnalysis={handleSaveAnalysis}
          onLoadAnalysis={handleLoadAnalysis}
          onDeleteAnalysis={handleDeleteAnalysis}
          onNewAnalysis={handleNewAnalysis}
          onRenameAnalysis={handleRenameAnalysis}
        />
      </aside>
      <div className="w-3/4 relative">
        <MapComponent
          polygon={polygon}
          onPolygonChange={handlePolygonChange}
          centerOn={centerOn}
          liveData={liveData}
          liveStatus={liveStatus}
          generators={generators}
          pendingGeneratorId={pendingGeneratorId}
          placementError={placementError}
          onPlaceGenerator={handlePlaceGenerator}
          onCancelPlacement={handleCancelPlacement}
        />
        <GeneratorsPanel
          generators={generators}
          pendingGeneratorId={pendingGeneratorId}
          centroid={polygonCentroid(polygon)}
          polygonComplete={polygon.length >= 3}
          onAdd={handleAddGenerator}
          onUpdate={handleUpdateGenerator}
          onDelete={handleDeleteGenerator}
          onPlace={handleStartPlacement}
        />
      </div>
      <NameInputModal
        open={saveNameModalOpen}
        title="Save analysis"
        description="Give this draft a name so you can return to it later."
        initialValue="Untitled site"
        placeholder="Untitled site"
        submitLabel="Save"
        onClose={() => setSaveNameModalOpen(false)}
        onSubmit={handleConfirmSaveName}
      />
    </div>
  );
}
