import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { MapContainer, TileLayer, Polyline, CircleMarker, Tooltip, useMap } from "react-leaflet";
import PlotComponent from "react-plotly.js";
import { fetchSnappedRoute } from "./routeService.js";

// Handle Vite CommonJS to ESM interop for react-plotly.js
const Plot = PlotComponent.default || PlotComponent;
const API_URL = "http://127.0.0.1:5000/optimize";

const ROUTE_COLORS = [
  "#ef4444",
  "#22c55e",
  "#3b82f6",
  "#eab308",
  "#a855f7",
  "#06b6d4",
  "#f97316",
  "#84cc16",
];

const SCENARIOS = [
  { id: "nyc", label: "New York (approx.)" },
  { id: "london", label: "London (approx.)" },
  { id: "bangalore", label: "Bengaluru (approx.)" },
  { id: "sf", label: "San Francisco (approx.)" },
  { id: "pittsburgh", label: "Pittsburgh (Land-locked)" },
  { id: "denver", label: "Denver (High Plains)" },
  { id: "kansas_city", label: "Kansas City (Heartland)" },
];



const FLEET_COLORS = [
  "#22c55e", // Emerald
  "#06b6d4", // Cyan
  "#f43f5e", // Rose
  "#eab308", // Amber
  "#8b5cf6", // Violet
  "#f97316", // Orange
  "#d946ef", // Fuchsia
  "#14b8a6"  // Teal
];


/** Mirrors backend SCENARIOS for map fit before any API response */
const SCENARIO_BBOX = {
  // Strictly Land-locked: Central Park / Midtown East
  nyc: { south: 40.760, west: -73.980, north: 40.800, east: -73.950 },
  // Strictly Land-locked: Hyde Park / Mayfair
  london: { south: 51.500, west: -0.160, north: 51.520, east: -0.120 },
  bangalore: { south: 12.898, west: 77.548, north: 13.048, east: 77.688 },
  sf: { south: 37.740, west: -122.440, north: 37.800, east: -122.380 },
  pittsburgh: { south: 40.435, west: -79.965, north: 40.455, east: -79.915 },
  denver: { south: 39.720, west: -105.020, north: 39.770, east: -104.970 },
  kansas_city: { south: 39.080, west: -94.600, north: 39.120, east: -94.560 },
};



function haversineKm(a, b) {
  const R = 6371;
  const dLat = ((b[0] - a[0]) * Math.PI) / 180;
  const dLng = ((b[1] - a[1]) * Math.PI) / 180;
  const lat1 = (a[0] * Math.PI) / 180;
  const lat2 = (b[0] * Math.PI) / 180;
  const x =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLng / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(x));
}

function interpolateAlongPolyline(path, t) {
  if (!path.length) return null;
  if (path.length === 1) return { pos: path[0], segmentIndex: 0 };
  const segLens = [];
  let total = 0;
  for (let i = 0; i < path.length - 1; i++) {
    const d = haversineKm(path[i], path[i + 1]);
    segLens.push(d);
    total += d;
  }
  if (total <= 0) return { pos: path[0], segmentIndex: 0 };
  let dist = Math.min(1, Math.max(0, t)) * total;
  for (let i = 0; i < segLens.length; i++) {
    if (dist <= segLens[i]) {
      const f = segLens[i] === 0 ? 0 : dist / segLens[i];
      return {
        pos: [
          path[i][0] + f * (path[i + 1][0] - path[i][0]),
          path[i][1] + f * (path[i + 1][1] - path[i][1]),
        ],
        segmentIndex: i
      };
    }
    dist -= segLens[i];
  }
  return { pos: path[path.length - 1], segmentIndex: path.length > 1 ? path.length - 2 : 0 };
}

function FitBounds({ bounds }) {
  const map = useMap();
  useEffect(() => {
    if (bounds) {
      map.fitBounds(bounds, { padding: [28, 28], maxZoom: 14 });
    }
  }, [bounds, map]);
  return null;
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
    </label>
  );
}

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, errorInfo: null };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    console.error("ErrorBoundary caught an error", error, errorInfo);
    this.setState({ errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: "20px", background: "white", color: "red" }}>
          <h2>Something went wrong.</h2>
          <details style={{ whiteSpace: "pre-wrap" }}>
            {this.state.error && this.state.error.toString()}
            <br />
            {this.state.errorInfo && this.state.errorInfo.componentStack}
          </details>
        </div>
      );
    }
    return this.props.children;
  }
}

function MainApp() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [payload, setPayload] = useState(null);

  const [scenario, setScenario] = useState("nyc");
  const [numDepots, setNumDepots] = useState(3);
  const [maxCustomers, setMaxCustomers] = useState(100);
  const [syntheticExtra, setSyntheticExtra] = useState(0);
  const [vehicleCapacity, setVehicleCapacity] = useState(200);
  const [numAnts, setNumAnts] = useState(20);
  const [alpha, setAlpha] = useState(1);
  const [beta, setBeta] = useState(2);
  const [evaporation, setEvaporation] = useState(0.5);
  const [iterations, setIterations] = useState(30);
  const [vehiclePenalty, setVehiclePenalty] = useState(100);

  // New Swarm Intelligence params
  const [explorationBias, setExplorationBias] = useState(0.5);
  const [failureMode, setFailureMode] = useState(false);
  const [localOptimumTrap, setLocalOptimumTrap] = useState(false);
  const [useLocalSearch, setUseLocalSearch] = useState(true);
  const [incremental, setIncremental] = useState(false);
  const [showPheromones, setShowPheromones] = useState(false);
  const [showcaseMode, setShowcaseMode] = useState(false);


  const [showCustomers, setShowCustomers] = useState(true);
  const [animDepot, setAnimDepot] = useState(0);
  const [animRoute, setAnimRoute] = useState(0);
  const [animIteration, setAnimIteration] = useState(0); // For timeline playback
  const [animPlaying, setAnimPlaying] = useState(false);
  const [animSpeed, setAnimSpeed] = useState(1.0);
  const [isPaused, setIsPaused] = useState(false);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [animT, setAnimT] = useState(0);
  const [targetVehicleIdx, setTargetVehicleIdx] = useState(0); 
  const animRef = useRef(null);
  const lastFrameTime = useRef(null);
  const currentAccumulatedT = useRef(0);



  const [snappedRoutes, setSnappedRoutes] = useState({});
  const [snappedSwarm, setSnappedSwarm] = useState({});
  const [snappedFleetIteration, setSnappedFleetIteration] = useState({});
  const [snappedNodes, setSnappedNodes] = useState({});

  const results = payload?.results ?? [];
  const customersGeo = payload?.customers_geo ?? [];

  const [paramSnapshot, setParamSnapshot] = useState(null);

  useEffect(() => {
    if (showcaseMode) {
      // Save current industrial parameters
      setParamSnapshot({
        numDepots,
        maxCustomers,
        alpha,
        beta,
        evaporation,
        numAnts,
        iterations,
      });

      // Inject educational presets
      setNumDepots(1);
      setMaxCustomers(12);
      setAlpha(1.0);
      setBeta(2.0);
      setEvaporation(0.4);
      setNumAnts(10);
      setIterations(30);
      setScenario("kansas_city");
    } else if (paramSnapshot) {

      // Restore previous settings
      setNumDepots(paramSnapshot.numDepots);
      setMaxCustomers(paramSnapshot.maxCustomers);
      setAlpha(paramSnapshot.alpha);
      setBeta(paramSnapshot.beta);
      setEvaporation(paramSnapshot.evaporation);
      setNumAnts(paramSnapshot.numAnts);
      setIterations(paramSnapshot.iterations);
      setParamSnapshot(null);
    }
  }, [showcaseMode]);

  useEffect(() => {
    let active = true;
    setSnappedRoutes({});
    setSnappedNodes({});
    
    if (payload?.results && payload.results.length > 0) {
      payload.results.forEach((depotResult, depotIdx) => {
        (depotResult?.routes || []).forEach((route, routeIdx) => {
          if (!route?.lat || !route?.lng) return;
          const originalPositions = route.lat.map((lat, i) => [lat, route.lng[i]]);
          
          fetchSnappedRoute(originalPositions).then(res => {
            if (active && res && res.snapped && res.snapped.length > 0) {
              setSnappedRoutes(prev => ({
                ...prev,
                [`${depotIdx}-${routeIdx}`]: res.snapped
              }));
              
              if (res.waypoints) {
                const newNodes = {};
                res.waypoints.forEach(wp => {
                  if (wp && wp.original && wp.snapped) {
                    newNodes[`${wp.original[0]},${wp.original[1]}`] = wp.snapped;
                  }
                });
                setSnappedNodes(prev => ({ ...prev, ...newNodes }));
              }
            }
          });
        });
      });
    }

    return () => { active = false; };
  }, [payload]);

  // High-Fidelity Swarm Sync: Snapping Ghost Ants in Showcase Mode
  useEffect(() => {
    if (!showcaseMode || !payload?.results?.[animDepot]?.timeline) return;
    const itData = payload.results[animDepot].timeline[animIteration];
    if (!itData?.swarm_samples) return;

    const r = payload.results[animDepot];
    const custMap = r.customer_mapping || [];

    Object.entries(itData.swarm_samples).forEach(([key, s]) => {
      const originalPositions = s.route.map(node => {
        if (node === 0) return [r.depot.lat, r.depot.lng];
        const globalIdx = custMap[node - 1];
        const c = Object.values(customersGeo).find(cg => cg.id === globalIdx) || customersGeo[globalIdx];
        return [c.lat, c.lng];
      });

      fetchSnappedRoute(originalPositions).then(res => {
        if (active && res?.snapped) {
          setSnappedSwarm(prev => ({ ...prev, [`${animIteration}-${key}`]: res.snapped }));
        }
      });
    });
  }, [payload, animDepot, animIteration, showcaseMode, customersGeo]);

  // High-Fidelity Fleet Sync: Snapping the active iteration's full fleet
  useEffect(() => {
    let active = true;
    if (!payload?.results?.[animDepot]?.timeline) return;
    const itData = payload.results[animDepot].timeline[animIteration];
    if (!itData?.fleet) return;

    const r = payload.results[animDepot];
    const custMap = r.customer_mapping || [];

    itData.fleet.forEach((routeNodes, vIdx) => {
      const originalPositions = routeNodes.map(node => {
        if (node === 0) return [r.depot.lat, r.depot.lng];
        const globalIdx = custMap[node - 1];
        const c = Object.values(customersGeo).find(cg => cg.id === globalIdx) || customersGeo[globalIdx];
        return [c.lat, c.lng];
      });

      fetchSnappedRoute(originalPositions).then(res => {
        if (active && res?.snapped) {
          setSnappedFleetIteration(prev => ({
            ...prev,
            [`${animDepot}-${animIteration}-${vIdx}`]: res.snapped
          }));
        }
      });
    });
    return () => { active = false; };
  }, [payload, animDepot, animIteration, customersGeo]);

  const scenarioBBox = useMemo(() => {
    if (payload?.scenario?.id === scenario && payload?.scenario?.bbox) {
      return payload.scenario.bbox;
    }
    return SCENARIO_BBOX[scenario] ?? SCENARIO_BBOX.nyc;
  }, [payload, scenario]);

  const mapBounds = useMemo(() => {
    if (!scenarioBBox) return null;
    return [
      [scenarioBBox.south, scenarioBBox.west],
      [scenarioBBox.north, scenarioBBox.east],
    ];
  }, [scenarioBBox]);

  const fitBounds = useMemo(() => {
    if (!results?.length) return mapBounds;
    const pts = [];
    customersGeo?.forEach((c) => {
      if (c && c.lat != null && c.lng != null) {
        pts.push([c.lat, c.lng]);
      }
    });
    results.forEach((r) => {
      if (r?.depot?.lat != null && r?.depot?.lng != null) {
        pts.push([r.depot.lat, r.depot.lng]);
      }
      r?.routes?.forEach((route) => {
        if (route?.lat && route?.lng) {
          for (let i = 0; i < route.lat.length; i++) {
            if (route.lat[i] != null && route.lng[i] != null) {
              pts.push([route.lat[i], route.lng[i]]);
            }
          }
        }
      });
    });
    if (!pts.length) return mapBounds;
    const lats = pts.map((p) => p[0]);
    const lngs = pts.map((p) => p[1]);
    return [
      [Math.min(...lats), Math.min(...lngs)],
      [Math.max(...lats), Math.max(...lngs)],
    ];
  }, [results, customersGeo, mapBounds]);

  const center = useMemo(() => {
    const b = mapBounds;
    if (!b) return [40.76, -73.98];
    return [(b[0][0] + b[1][0]) / 2, (b[0][1] + b[1][1]) / 2];
  }, [mapBounds]);

  const fleetPaths = useMemo(() => {
    if (!results?.[animDepot]?.timeline) return [];
    const r = results[animDepot];
    const iterationData = r.timeline[animIteration];
    if (!iterationData?.fleet) return [];

    const custMap = r.customer_mapping || [];
    return iterationData.fleet.map((routeNodes, vIdx) => {
      // Priority 1: High-fidelity snapped fleet route
      const snapped = snappedFleetIteration[`${animDepot}-${animIteration}-${vIdx}`];
      if (snapped && snapped.length >= 2) {
        return { color: FLEET_COLORS[vIdx % FLEET_COLORS.length], pts: snapped };
      }

      // Fallback: Node-to-node (snapped nodes)
      const pts = routeNodes.map(node => {
        if (node === 0) return [r.depot.lat, r.depot.lng];
        const globalIdx = custMap[node - 1];
        const c = Object.values(customersGeo).find(cg => cg.id === globalIdx) || customersGeo[globalIdx];
        if (c && c.lat != null && c.lng != null) {
          const key = `${c.lat},${c.lng}`;
          return snappedNodes[key] || [c.lat, c.lng];
        }
        return [r.depot.lat, r.depot.lng];
      });
      return { 
        pts: pts.length >= 2 ? pts : [], 
        color: FLEET_COLORS[vIdx % FLEET_COLORS.length] 
      };
    });
  }, [results, animDepot, animIteration, customersGeo, snappedNodes, snappedFleetIteration]);

  const animPath = useMemo(() => {
    if (!fleetPaths.length) return [];
    const idx = Math.min(targetVehicleIdx, fleetPaths.length - 1);
    return fleetPaths[idx]?.pts || [];
  }, [fleetPaths, targetVehicleIdx]);


  const swarmPaths = useMemo(() => {
    if (!showcaseMode || !payload?.results?.[animDepot]?.timeline) return {};
    const itData = payload.results[animDepot].timeline[animIteration];
    if (!itData?.swarm_samples) return {};
    
    const custMap = payload.results[animDepot].customer_mapping || [];
    const r = payload.results[animDepot];

    const processed = {};
    Object.entries(itData.swarm_samples).forEach(([key, s]) => {
      // Priority 1: High-fidelity snapped road path
      const snapped = snappedSwarm[`${animIteration}-${key}`];
      if (snapped && snapped.length >= 2) {
        processed[key] = snapped;
        return;
      }
      
      // Fallback: Node-to-node jumping (while snapping nodes)
      processed[key] = s.route.map(node => {
        if (node === 0) return [r.depot.lat, r.depot.lng];
        const globalIdx = custMap[node - 1];
        const c = Object.values(customersGeo).find(cg => cg.id === globalIdx) || customersGeo[globalIdx];
        const raw = c ? [c.lat, c.lng] : [r.depot.lat, r.depot.lng];
        return snappedNodes[`${raw[0]},${raw[1]}`] || raw;
      });
    });
    return processed;
  }, [payload, animDepot, animIteration, showcaseMode, customersGeo, snappedNodes, snappedSwarm]);



  const activePheromones = useMemo(() => {
    if (!showPheromones || !results.length) return [];
    const allLayers = [];
    results.forEach((r) => {
        if (!r.timeline || r.timeline.length === 0) return;
        const custMap = r.customer_mapping || [];
        const iterationData = r.timeline[Math.min(animIteration, r.timeline.length - 1)];
        if (!iterationData || !iterationData.pheromones) return;
        
        const ph = iterationData.pheromones.map(p => {
           let pt1, pt2;
           if (p.edge[0] === 0) pt1 = [r.depot.lat, r.depot.lng];
           else {
               const gIdx1 = custMap[p.edge[0] - 1];
               const c1 = customersGeo[gIdx1];
               pt1 = c1 ? [c1.lat, c1.lng] : [0,0];
           }

           if (p.edge[1] === 0) pt2 = [r.depot.lat, r.depot.lng];
           else {
               const gIdx2 = custMap[p.edge[1] - 1];
               const c2 = customersGeo[gIdx2];
               pt2 = c2 ? [c2.lat, c2.lng] : [0,0];
           }
           
           return { path: [pt1, pt2], opacity: Math.min(1.0, p.intensity / 50.0) };
        });
        allLayers.push(...ph);
    });
    return allLayers;
  }, [results, animIteration, showPheromones, customersGeo]);

  useEffect(() => {
    if (animPlaying && animPath.length >= 2) {
      lastFrameTime.current = performance.now();
      
      const frame = (now) => {
        if (!isPaused) {
          const delta = now - lastFrameTime.current;
          // Base speed: 14000ms for a full loop
          const increment = (delta / 14000) * animSpeed;
          currentAccumulatedT.current = (currentAccumulatedT.current + increment) % 1;
          setAnimT(currentAccumulatedT.current);
        }
        lastFrameTime.current = now;
        animRef.current = requestAnimationFrame(frame);
      };
      
      animRef.current = requestAnimationFrame(frame);
    } else {
      currentAccumulatedT.current = 0;
      setAnimT(0);
      if (animRef.current) cancelAnimationFrame(animRef.current);
    }
    return () => { if (animRef.current) cancelAnimationFrame(animRef.current); };
  }, [animPlaying, isPaused, animSpeed, animPath]);


  const vehiclePos = useMemo(() => interpolateAlongPolyline(animPath, animT), [animPath, animT]);

  const runOptimization = async () => {
    setLoading(true);
    setError("");
    setAnimPlaying(false);
    try {
      const body = {
        scenario: showcaseMode ? "kansas_city" : scenario,
        num_depots: numDepots,
        max_customers: maxCustomers,
        synthetic_extra: syntheticExtra,
        vehicle_capacity: vehicleCapacity,
        num_ants: numAnts,
        alpha,
        beta,
        evaporation,
        iterations,
        vehicle_penalty: vehiclePenalty,
        exploration_bias: explorationBias,
        failure_mode: failureMode,
        local_optimum_trap: localOptimumTrap,
        incremental: incremental,
        showcase: showcaseMode
      };

      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error(`Backend error: ${response.status}`);
      }
      const data = await response.json();
      if (data.error) {
        throw new Error(data.error);
      }
      if (data.status !== "ok" || !Array.isArray(data.results)) {
        throw new Error("Invalid backend response");
      }
      setPayload(data);
      setAnimDepot(0);
      setAnimRoute(0);
      setAnimIteration(0);
    } catch (err) {
      setError(err.message || "Failed to run optimization");
      setPayload(null);
    } finally {
      setLoading(false);
    }
  };

  const plotData = useMemo(() => {
    if (!payload?.convergence) return [];
    const per = payload.convergence.per_depot || [];
    const traces = per
      .map((series, i) =>
        series?.length
          ? {
            x: series.map((_, j) => j + 1),
            y: series,
            mode: "lines",
            name: `Depot ${i}`,
            line: { width: 2, color: ROUTE_COLORS[i % ROUTE_COLORS.length] },
          }
          : null,
      )
      .filter(Boolean);
    if (payload.convergence.combined?.length) {
      traces.push({
        x: payload.convergence.combined.map((_, j) => j + 1),
        y: payload.convergence.combined,
        mode: "lines",
        name: "Sum (all depots)",
        line: { width: 2, dash: "dash", color: "#94a3b8" },
      });
    }
    return traces;
  }, [payload]);

  const onSliderAnimDepot = useCallback(
    (e) => {
      setAnimDepot(Number(e.target.value));
      setAnimT(0);
    },
    [setAnimDepot],
  );

  const onSliderAnimRoute = useCallback(
    (e) => {
      setAnimRoute(Number(e.target.value));
      setAnimT(0);
    },
    [setAnimRoute],
  );

  const onSliderAnimIteration = useCallback(
    (e) => {
      setAnimIteration(Number(e.target.value));
      setAnimT(0);
    },
    [setAnimIteration],
  );

  const activeTimelineState = useMemo(() => {
     if (!results[animDepot] || !results[animDepot].timeline) return null;
     return results[animDepot].timeline[Math.min(animIteration, results[animDepot].timeline.length - 1)];
  }, [results, animDepot, animIteration]);

  const stats = payload?.stats;
  const totalRoutes = useMemo(
    () => results.reduce((sum, d) => sum + (d.routes?.length || 0), 0),
    [results],
  );

  return (
    <div className="layout-shell">
      <aside className="glass-panel">
        <div>
          <h1 className="title">VRP Explorer</h1>
          <div className="mode-toggle-container">
            <button 
              className={`mode-btn ${!showcaseMode ? 'active' : ''}`}
              onClick={() => setShowcaseMode(false)}
            >
              Industrial MDVRP
            </button>
            <button 
              className={`mode-btn ${showcaseMode ? 'active' : ''}`}
              onClick={() => setShowcaseMode(true)}
            >
              Educational Showcase
            </button>
          </div>
          <p className="subtitle">
            {showcaseMode 
              ? "Understand the core mechanics: Single depot, small swarm, and live ant decision rationales."
              : "Advanced Swarm Intelligence: Parallel optimization on large-scale multi-depot clusters."
            }
          </p>
          {showcaseMode && activeTimelineState?.first_decision_probs && (
            <div className="decision-hud">
              <h3 className="hud-title">Next-Node Probabilities (Ant Brain)</h3>
              <Plot 
                data={[
                  {
                    x: activeTimelineState.first_decision_probs.nodes.map(n => `Node ${n}`),
                    y: activeTimelineState.first_decision_probs.probs,
                    type: 'bar',
                    marker: { color: '#8b5cf6' }
                  }
                ]}
                layout={{
                  width: 280,
                  height: 180,
                  margin: { t: 10, b: 40, l: 30, r: 10 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#fff', size: 10 },
                  xaxis: { tickangle: -45 }
                }}
                config={{ displayModeBar: false }}
              />
              <p className="hud-caption">How Alpha & Beta shift the pick chance.</p>
            </div>
          )}

          {payload?.technical_rationale && (
            <div className="technical-rationale-hud">
              <span className="rationale-icon">ℹ️</span>
              <p className="rationale-text">{payload.technical_rationale}</p>
            </div>
          )}
        </div>

        <div className="section">
          <h2 className="section-title">Scenario & data</h2>
          <Field label="Map region">
            <select value={scenario} onChange={(e) => setScenario(e.target.value)} className="glass-input">
              {SCENARIOS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </Field>
          <Field label={`Depots (K-means): ${numDepots}`}>
            <div className="field-info">
              {showcaseMode 
                ? "Locked at 1 for mechanical baseline."
                : `K-means clusters the single Solomon depot into ${numDepots} virtual hubs.`
              }
            </div>
            <input
              type="range"
              min={1}
              max={8}
              value={numDepots}
              disabled={showcaseMode}
              onChange={(e) => setNumDepots(Number(e.target.value))}
              style={{ opacity: showcaseMode ? 0.4 : 1, cursor: showcaseMode ? 'not-allowed' : 'pointer' }}
            />
          </Field>
          <Field label={`Customers sampled: ${maxCustomers}`}>
            <div className="field-info">Synthetic X/Y coordinates from c101.txt projected into real-world grid.</div>
            <input
              type="range"
              min={10}
              max={200}
              value={maxCustomers}
              onChange={(e) => setMaxCustomers(Number(e.target.value))}
            />
          </Field>
          <Field label={`Synthetic extra stops: ${syntheticExtra}`}>
            <input
              type="range"
              min={0}
              max={150}
              step={5}
              value={syntheticExtra}
              onChange={(e) => setSyntheticExtra(Number(e.target.value))}
            />
          </Field>
        </div>

        <div className="section">
          <h2 className="section-title">ACO parameters</h2>
          <Field label={`Vehicle capacity: ${vehicleCapacity}`}>
            <input
              type="range"
              min={50}
              max={300}
              step={10}
              value={vehicleCapacity}
              onChange={(e) => setVehicleCapacity(Number(e.target.value))}
            />
          </Field>
          <Field label={`Ants per iteration: ${numAnts}`}>
            <input type="range" min={5} max={60} value={numAnts} onChange={(e) => setNumAnts(Number(e.target.value))} />
          </Field>
          <Field label={`α (pheromone): ${alpha.toFixed(2)}`}>
            <input
              type="range"
              min={0.5}
              max={3}
              step={0.05}
              value={alpha}
              onChange={(e) => setAlpha(Number(e.target.value))}
            />
          </Field>
          <Field label={`β (heuristic): ${beta.toFixed(2)}`}>
            <input
              type="range"
              min={1}
              max={5}
              step={0.05}
              value={beta}
              onChange={(e) => setBeta(Number(e.target.value))}
            />
          </Field>
          <Field label={`ρ (evaporation): ${evaporation.toFixed(2)}`}>
            <input
              type="range"
              min={0.1}
              max={0.9}
              step={0.02}
              value={evaporation}
              onChange={(e) => setEvaporation(Number(e.target.value))}
            />
          </Field>
          <Field label={`Iterations: ${iterations}`}>
            <input
              type="range"
              min={5}
              max={80}
              value={iterations}
              onChange={(e) => setIterations(Number(e.target.value))}
            />
          </Field>
          <Field label={`Exploration Bias (ratio): ${explorationBias.toFixed(2)}`}>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={explorationBias}
              onChange={(e) => setExplorationBias(Number(e.target.value))}
            />
          </Field>
          <label className="toggle-label" style={{marginTop: '10px'}}>
            <input type="checkbox" checked={useLocalSearch} onChange={(e) => setUseLocalSearch(e.target.checked)} />
            Enable 2-Opt Local Search
          </label>
          <label className="toggle-label">
            <input type="checkbox" checked={incremental} onChange={(e) => setIncremental(e.target.checked)} />
            Incremental Warm-Start
          </label>
          <label className="toggle-label" style={{color: '#f87171'}}>
            <input type="checkbox" checked={failureMode} onChange={(e) => setFailureMode(e.target.checked)} />
            ⚠️ Simulate Algorithm Failure
          </label>
          <label className="toggle-label" style={{color: '#fcd34d'}}>
            <input type="checkbox" checked={localOptimumTrap} onChange={(e) => setLocalOptimumTrap(e.target.checked)} />
            🚨 Aha Moment: Force Local Optimum Trap
          </label>
        </div>

        <button type="button" onClick={runOptimization} disabled={loading} className="btn-primary">
          {loading ? "Running optimization…" : "Run optimization"}
        </button>

        {error ? <div className="error-box">{error}</div> : null}

        {stats ? (
          <div className="stats-grid">
            <div className="stat-card">
              <span className="stat-label">Total distance</span>
              <span className="stat-value">{stats.total_distance?.toFixed(1) ?? "—"}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Vehicles</span>
              <span className="stat-value">{stats.total_vehicles ?? "—"}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Customers</span>
              <span className="stat-value">{stats.customer_count ?? "—"}</span>
            </div>
            <div className="stat-card">
              <span className="stat-label">Routes</span>
              <span className="stat-value">{totalRoutes}</span>
            </div>
          </div>
        ) : null}

        {results.length ? (
          <div className="section">
            <h2 className="section-title">Timeline & Animation</h2>
            <div style={{ display: 'flex', gap: '15px' }}>
              <label className="toggle-label">
                <input type="checkbox" checked={showCustomers} onChange={(e) => setShowCustomers(e.target.checked)} />
                Customers
              </label>
              <label className="toggle-label" style={{color: '#d8b4fe'}}>
                <input type="checkbox" checked={showPheromones} onChange={(e) => setShowPheromones(e.target.checked)} />
                Pheromones Layer
              </label>
            </div>
            
            <Field label={`Depot: ${animDepot}`}>
              <input
                type="range"
                min={0}
                max={Math.max(0, results.length - 1)}
                value={animDepot}
                onChange={onSliderAnimDepot}
              />
            </Field>

            {activeTimelineState && (
              <div className="concept-hud">
                 <div className="hud-metric">
                    <span className="hud-label">Phase:</span>
                    <span className="hud-value" style={{color: activeTimelineState.phase?.includes('Exploration') ? '#f472b6' : '#a3e635'}}>
                       {activeTimelineState.phase}
                    </span>
                 </div>
                 <div className="hud-metric">
                    <span className="hud-label">Dominant Constraint Rationale:</span>
                    <span className="hud-value">{activeTimelineState.dominant_rationale}</span>
                 </div>
                 <div className="hud-metric">
                    <span className="hud-label">Diversity Entropy:</span>
                    <span className="hud-value">{activeTimelineState.entropy?.toFixed(3)} (lower = converged)</span>
                 </div>
                 {activeTimelineState.hybrid_savings > 0 && (
                 <div className="hud-metric">
                    <span className="hud-label">2-Opt Hybrid Savings:</span>
                    <span className="hud-value" style={{color: '#60a5fa'}}>{activeTimelineState.hybrid_savings?.toFixed(2)} units</span>
                 </div>
                 )}
              </div>
            )}

            <Field label={`Iteration Playback: ${animIteration}`}>
              <input
                type="range"
                min={0}
                max={Math.max(0, (results[animDepot]?.timeline?.length || 1) - 1)}
                value={animIteration}
                onChange={onSliderAnimIteration}
              />
            </Field>

            <div className="field">
              <button type="button" className="btn-secondary" onClick={() => setAnimPlaying((p) => !p)}>
                {animPlaying ? "Pause Playback" : "Animate Fleet"}
              </button>
              <span className="subtitle">Vehicles follow iteration paths. Select a specific route below to audit.</span>
            </div>

            {fleetPaths.length > 0 && (
              <div className="fleet-audit-panel">
                <h3 className="section-title" style={{fontSize: '11px', marginTop: '10px'}}>Fleet Composition (Iteration {animIteration})</h3>
                <div className="fleet-grid">
                  {fleetPaths.map((f, i) => (
                    <div 
                      key={`fleet-v-${i}`} 
                      className={`fleet-pill ${targetVehicleIdx === i ? 'active' : ''}`}
                      onClick={() => setTargetVehicleIdx(i)}
                      style={{ borderLeft: `4px solid ${f.color}` }}
                    >
                      <div className="v-id">Vehicle #{i+1}</div>
                      <div className="v-nodes">{f.pts.length - 2} Nodes</div>
                    </div>
                  ))}
                </div>
              </div>
            )}

          </div>
        ) : null}

        {plotData.length ? (
          <div className="section" style={{ marginTop: "8px" }}>
            <h2 className="section-title">Convergence (best distance)</h2>
            <Plot
              data={plotData}
              layout={{
                paper_bgcolor: "transparent",
                plot_bgcolor: "transparent",
                margin: { l: 48, r: 12, t: 8, b: 40 },
                xaxis: { title: "Iteration", gridcolor: "rgba(148,163,184,0.25)" },
                yaxis: { title: "Distance", gridcolor: "rgba(148,163,184,0.25)" },
                font: { color: "#cbd5e1", family: "system-ui, sans-serif" },
                showlegend: true,
                legend: { orientation: "h" },
                height: 260,
              }}
              config={{ displayModeBar: false }}
              style={{ width: "100%" }}
            />
          </div>
        ) : null}

        <p className="footer-code">
          Backend: <code>python backend/app.py</code> · Frontend:{" "}
          <code>npm run dev</code>
        </p>
      </aside>

      <main className="map-container">
        <MapContainer center={center} zoom={12} style={{ height: "100%", width: "100%", zIndex: 1 }} scrollWheelZoom>
          <TileLayer
            attribution='&copy; <a href="https://carto.com/attributions">CARTO</a>'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          <FitBounds bounds={fitBounds} />

          {showPheromones
            ? activePheromones.map((ph, idx) => (
                <Polyline
                  key={`phero-${idx}`}
                  positions={ph.path}
                  pathOptions={{ color: "#d8b4fe", weight: 3, opacity: ph.opacity }}
                />
              ))
            : results.map((depotResult, depotIdx) =>
              (depotResult?.routes || []).map((route, routeIdx) => {
                if (!route?.lat || !route?.lng) return null;
                const originalPositions = route.lat.map((lat, i) => [lat, route.lng[i]]);
                const positions = snappedRoutes[`${depotIdx}-${routeIdx}`] || originalPositions;
                const color = ROUTE_COLORS[(depotIdx + routeIdx) % ROUTE_COLORS.length];
                
                // Active visual chaos rendering
                const isExploring = activeTimelineState?.phase?.includes("Exploration") || activeTimelineState?.phase?.includes("Trap");
                const lineDash = isExploring ? "5, 10" : null;
                const lineOp = isExploring ? 0.5 : 0.85;

                return (
                  <Polyline
                    key={`${depotIdx}-${routeIdx}`}
                    positions={positions}
                    pathOptions={{ color, weight: 4, opacity: lineOp, dashArray: lineDash }}
                  />
                );
              }),
            )}

          {showCustomers
            ? customersGeo.map((c) => {
              const key = `${c.lat},${c.lng}`;
              const center = snappedNodes[key] || [c.lat, c.lng];
              return (
              <CircleMarker
                key={c.id}
                center={center}
                radius={4}
                pathOptions={{ color: "#0f172a", weight: 1, fillColor: "#38bdf8", fillOpacity: 0.9 }}
              >
                <Tooltip direction="top" offset={[0, -6]} opacity={0.95}>
                  Customer {c.id}
                  <br />
                  Demand: {c.demand}
                </Tooltip>
              </CircleMarker>
              );
            })
            : null}

          {results.map((depotResult, depotIdx) => {
            const dp = depotResult.depot;
            const key = `${dp.lat},${dp.lng}`;
            const center = snappedNodes[key] || [dp.lat, dp.lng];
            return (
            <CircleMarker
              key={`depot-${depotIdx}`}
              center={center}
              radius={9}
              pathOptions={{ color: "#0f172a", weight: 2, fillColor: "#fbbf24", fillOpacity: 1 }}
            >
              <Tooltip direction="top">Depot {depotIdx}</Tooltip>
            </CircleMarker>
            );
          })}

          {/* Full Fleet Visuals */}
          {fleetPaths.map((f, i) => (
            <Polyline
              key={`fleet-line-${i}`}
              positions={f.pts}
              pathOptions={{
                color: f.color,
                weight: targetVehicleIdx === i ? 5 : 2,
                opacity: targetVehicleIdx === i ? 1 : 0.4,
                dashArray: targetVehicleIdx === i ? null : "5, 10"
              }}
            >
              <Tooltip sticky>Vehicle {i+1} Route</Tooltip>
            </Polyline>
          ))}

          {vehiclePos && animPlaying ? (
            <>
              {/* Main Champion Ant */}
              <CircleMarker
                center={vehiclePos.pos}
                radius={8}
                pathOptions={{ color: "#000", weight: 2, fillColor: "#22c55e", fillOpacity: 1 }}
              >
                <Tooltip direction="top" permanent offset={[0, -10]} className="live-rationale-tooltip">
                  <div className="live-rationale-hud">
                    <div className="live-rationale-title">Champion Agent</div>
                    <div className="live-rationale-text">
                      {activeTimelineState?.route_rationales?.[vehiclePos.segmentIndex] || "Traversing Segment"}
                    </div>
                  </div>
                </Tooltip>
              </CircleMarker>

              {/* Ghost Explorer Ants for Showcase */}
              {showcaseMode && Object.entries(swarmPaths).map(([key, path]) => {
                if (key === "champion") return null;
                const ghostPos = interpolateAlongPolyline(path, animT);
                if (!ghostPos) return null;
                return (
                  <CircleMarker
                    key={`ghost-${key}`}
                    center={ghostPos.pos}
                    radius={5}
                    pathOptions={{ 
                      color: key === "explorer" ? "#a855f7" : "#ef4444", 
                      fillOpacity: 0.6,
                      fillColor: key === "explorer" ? "#a855f7" : "#ef4444",
                      weight: 1
                    }}
                  >
                    <Tooltip direction="bottom" opacity={0.7}>
                      {key.toUpperCase()} Sub-swarm
                    </Tooltip>
                  </CircleMarker>
                );
              })}
            </>
          ) : null}
        </MapContainer>

        {/* Master Simulation Controls Overlay */}
        {showcaseMode && (
          <div className="showcase-controls-overlay">
            <div className="controls-group">
              <button 
                className="control-btn main-play" 
                onClick={() => {
                  if (!results.length) {
                    // Logic to automatically trigger optimization if no results
                    const runBtn = document.querySelector('.run-btn');
                    if (runBtn) runBtn.click();
                  }
                  setAnimPlaying(!animPlaying);
                  if (isPaused) setIsPaused(false);
                }}
              >
                {animPlaying && !isPaused ? "⏸ PAUSE" : "▶ PLAY WALKTHROUGH"}
              </button>
              
              {animPlaying && (
                <button className="control-btn step-btn" onClick={() => setIsPaused(!isPaused)}>
                  {isPaused ? "RESUME" : "PAUSE"}
                </button>
              )}
            </div>

            <div className="controls-group speed-control">
              <span className="control-label">Simulation Speed: {animSpeed.toFixed(1)}x</span>
              <input 
                type="range" 
                min={0.2} 
                max={5.0} 
                step={0.1} 
                value={animSpeed} 
                onChange={(e) => setAnimSpeed(parseFloat(e.target.value))}
              />
            </div>
            
            <div className="concept-ribbon">
              {animT < 0.1 ? "CONCEPT: DEPOT INITIALIZATION" : 
               animT < 0.4 ? "CONCEPT: PROBABILISTIC SEARCH STATE" :
               animT < 0.8 ? "CONCEPT: PHEROMONE UPDATE" : 
               "CONCEPT: LOCAL IMPROVEMENT (2-OPT)"}
            </div>
          </div>
        )}
        {/* Floating Action Button for Analytics */}
        {results.length > 0 && (
          <button className="analytics-fab" onClick={() => setShowAnalytics(true)}>
            <span className="fab-sigma">Σ</span>
            <span className="fab-text">Optimization Analytics</span>
          </button>
        )}

        {/* The Academic Analytics Modal */}
        {showAnalytics && (
          <div className="analytics-modal-overlay">
            <div className="analytics-modal">
              <div className="modal-header">
                <div>
                  <h2 className="modal-title">Academy: Optimization Analytics Hub</h2>
                  <p className="modal-subtitle">Mathematical verification of swarm intelligence & convergence metrics.</p>
                </div>
                <div className="audit-summary">
                  <div className="audit-item">
                    <span className="audit-label">Initial Baseline</span>
                    <span className="audit-value">{(payload.convergence.combined[0] || 0).toFixed(1)} km</span>
                  </div>
                  <div className="audit-item accent">
                    <span className="audit-label">Swarm Optimized</span>
                    <span className="audit-value">{(Math.min(...payload.convergence.combined)).toFixed(1)} km</span>
                  </div>
                  <div className="audit-item highlight">
                    <span className="audit-label">Improvement Rate</span>
                    <span className="audit-value">
                      {((1 - (Math.min(...payload.convergence.combined) / payload.convergence.combined[0])) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                <button className="close-modal" onClick={() => setShowAnalytics(false)}>✕</button>

              </div>

              <div className="analytics-grid">
                {/* Chart 1: Convergence */}
                <div className="analytics-card">
                  <h3 className="card-title">A. Global Convergence Curve</h3>
                  <Plot 
                    data={[
                      {
                        x: payload.convergence.combined.map((_, i) => i),
                        y: payload.convergence.combined,
                        type: 'scatter',
                        mode: 'lines+markers',
                        line: { color: '#10b981', width: 3, shape: 'spline' },
                        marker: { color: '#10b981', size: 4 },
                        name: 'Total Fleet Distance'
                      },
                      {
                        x: [0, payload.convergence.combined.length - 1],
                        y: [Math.min(...payload.convergence.combined) * 0.95, Math.min(...payload.convergence.combined) * 0.95],
                        type: 'scatter',
                        mode: 'lines',
                        line: { color: 'rgba(255,255,255,0.2)', dash: 'dash', width: 1 },
                        name: 'Reference Min'
                      }
                    ]}
                    layout={{
                      autosize: true, height: 250, margin: { t: 10, b: 40, l: 50, r: 10 },
                      paper_bgcolor: 'transparent', plot_bgcolor: 'rgba(0,0,0,0.2)',
                      font: { color: '#fff', size: 10 },
                      xaxis: { title: 'Iteration', gridcolor: '#334155' },
                      yaxis: { title: 'Total Distance', gridcolor: '#334155' },
                      shapes: [{
                        type: 'line', x0: animIteration, x1: animIteration, y0: 0, y1: 1, 
                        yref: 'paper', line: { color: '#fcd34d', width: 2, dash: 'dot' }
                      }]
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                  />
                </div>

                {/* Chart 2: Shannon Entropy (Learning Curve) */}
                <div className="analytics-card">
                  <h3 className="card-title">B. Swarm Diversity (Shannon Entropy)</h3>
                  <p className="card-formula">H(S) = -Σ P(i) log P(i)</p>
                  <Plot 
                    data={[{
                      x: results[animDepot].timeline.map((_, i) => i),
                      y: results[animDepot].timeline.map(it => it.entropy),
                      type: 'scatter',
                      line: { color: '#f472b6', width: 2, shape: 'spline' },
                      fill: 'tozeroy',
                      fillcolor: 'rgba(244, 114, 182, 0.1)',
                      name: 'Search Entropy'
                    }]}
                    layout={{
                      autosize: true, height: 250, margin: { t: 10, b: 40, l: 50, r: 10 },
                      paper_bgcolor: 'transparent', plot_bgcolor: 'rgba(0,0,0,0.2)',
                      font: { color: '#fff', size: 10 },
                      xaxis: { title: 'Iteration', gridcolor: '#334155' },
                      yaxis: { title: 'Entropy (Exploration)', gridcolor: '#334155' },
                      shapes: [{
                        type: 'line', x0: animIteration, x1: animIteration, y0: 0, y1: 1, 
                        yref: 'paper', line: { color: '#fcd34d', width: 2, dash: 'dot' }
                      }]
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                  />
                </div>

                {/* Chart 3: Logic Distribution */}
                <div className="analytics-card">
                  <h3 className="card-title">C. Heuristic Intelligence Balance</h3>
                  <p className="card-formula">Decision ≈ (τ^α) * (η^β)</p>
                  <Plot 
                    data={[{
                      values: [
                        results[animDepot].timeline[animIteration].route_rationales.filter(r => r.includes("Pheromone")).length,
                        results[animDepot].timeline[animIteration].route_rationales.filter(r => r.includes("Heuristic")).length,
                        results[animDepot].timeline[animIteration].route_rationales.filter(r => r.includes("Blend") || r.includes("Diversity")).length,
                      ],
                      labels: ['Pheromone Flow', 'Greedy Heuristic', 'Exploration/Blend'],
                      type: 'pie',
                      marker: { colors: ['#a855f7', '#0ea5e9', '#f472b6'] },
                      hole: 0.4
                    }]}
                    layout={{
                      autosize: true, height: 250, margin: { t: 10, b: 10, l: 10, r: 10 },
                      paper_bgcolor: 'transparent',
                      font: { color: '#fff', size: 10 },
                      showlegend: true,
                      legend: { orientation: 'h', y: -0.2 }
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                  />
                </div>

                {/* Chart 4: Hybrid Optimization Savings */}
                <div className="analytics-card">
                  <h3 className="card-title">D. Local Search Gain (2-Opt)</h3>
                  <p className="card-formula">ΔD = D(old) - D(new)</p>
                  <Plot 
                    data={[{
                      x: results[animDepot].timeline.map((_, i) => i),
                      y: results[animDepot].timeline.map(it => it.hybrid_savings),
                      type: 'bar',
                      marker: { color: '#60a5fa' },
                      name: 'Hybrid Savings'
                    }]}
                    layout={{
                      autosize: true, height: 250, margin: { t: 10, b: 40, l: 50, r: 10 },
                      paper_bgcolor: 'transparent', plot_bgcolor: 'rgba(0,0,0,0.2)',
                      font: { color: '#fff', size: 10 },
                      xaxis: { title: 'Iteration', gridcolor: '#334155' },
                      yaxis: { title: 'Distance Saved', gridcolor: '#334155' }
                    }}
                    config={{ responsive: true, displayModeBar: false }}
                  />
                </div>
              </div>

              <div className="modal-footer">
                <div className="legend-pills">
                  <span className="pill green">Convergence</span>
                  <span className="pill pink">Entropy</span>
                  <span className="pill purple">Intelligence</span>
                  <span className="pill blue">Local Search</span>
                </div>
                <p className="footer-note">Educational Mode Sync: All units are normalized to Solomon Problem Benchmarks.</p>
              </div>
            </div>
          </div>
        )}

      </main>

    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <MainApp />
    </ErrorBoundary>
  );
}
