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
];

/** Mirrors backend SCENARIOS for map fit before any API response */
const SCENARIO_BBOX = {
  nyc: { south: 40.698, west: -74.028, north: 40.822, east: -73.918 },
  london: { south: 51.472, west: -0.172, north: 51.532, east: -0.048 },
  bangalore: { south: 12.898, west: 77.548, north: 13.048, east: 77.688 },
  sf: { south: 37.738, west: -122.448, north: 37.802, east: -122.378 },
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
  if (path.length === 1) return path[0];
  const segLens = [];
  let total = 0;
  for (let i = 0; i < path.length - 1; i++) {
    const d = haversineKm(path[i], path[i + 1]);
    segLens.push(d);
    total += d;
  }
  if (total <= 0) return path[0];
  let dist = Math.min(1, Math.max(0, t)) * total;
  for (let i = 0; i < segLens.length; i++) {
    if (dist <= segLens[i]) {
      const f = segLens[i] === 0 ? 0 : dist / segLens[i];
      return [
        path[i][0] + f * (path[i + 1][0] - path[i][0]),
        path[i][1] + f * (path[i + 1][1] - path[i][1]),
      ];
    }
    dist -= segLens[i];
  }
  return path[path.length - 1];
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

  const [showCustomers, setShowCustomers] = useState(true);
  const [animDepot, setAnimDepot] = useState(0);
  const [animRoute, setAnimRoute] = useState(0);
  const [animPlaying, setAnimPlaying] = useState(false);
  const [animT, setAnimT] = useState(0);
  const animRef = useRef(null);

  const [snappedRoutes, setSnappedRoutes] = useState({});
  const [snappedNodes, setSnappedNodes] = useState({});

  const results = payload?.results ?? [];
  const customersGeo = payload?.customers_geo ?? [];

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

  const animPath = useMemo(() => {
    const r = results[animDepot];
    if (!r?.routes?.length) return [];
    
    const routeIdx = Math.min(animRoute, r.routes.length - 1);
    const snapped = snappedRoutes[`${animDepot}-${routeIdx}`];
    if (snapped && snapped.length >= 2) return snapped;

    const rt = r.routes[routeIdx];
    if (!rt?.lat?.length) return [];
    return rt.lat.map((lat, i) => [lat, rt.lng[i]]);
  }, [results, animDepot, animRoute, snappedRoutes]);

  useEffect(() => {
    if (!animPlaying || animPath.length < 2) return undefined;
    const durationMs = 14000;
    const start = performance.now();
    const tick = (now) => {
      const u = ((now - start) % durationMs) / durationMs;
      setAnimT(u);
      animRef.current = requestAnimationFrame(tick);
    };
    animRef.current = requestAnimationFrame(tick);
    return () => {
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [animPlaying, animPath]);

  const vehiclePos = useMemo(() => interpolateAlongPolyline(animPath, animT), [animPath, animT]);

  const runOptimization = async () => {
    setLoading(true);
    setError("");
    setAnimPlaying(false);
    try {
      const body = {
        scenario,
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
          <p className="subtitle">
            Ant Colony Optimization on Solomon-style instances, shown on OpenStreetMap. Tune parameters and watch
            routes and convergence.
          </p>
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
            <input
              type="range"
              min={1}
              max={8}
              value={numDepots}
              onChange={(e) => setNumDepots(Number(e.target.value))}
            />
          </Field>
          <Field label={`Customers sampled: ${maxCustomers}`}>
            <input
              type="range"
              min={10}
              max={100}
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
          <Field label={`Depot-return penalty: ${vehiclePenalty}`}>
            <input
              type="range"
              min={10}
              max={400}
              step={5}
              value={vehiclePenalty}
              onChange={(e) => setVehiclePenalty(Number(e.target.value))}
            />
          </Field>
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
            <h2 className="section-title">Animation</h2>
            <label className="toggle-label">
              <input type="checkbox" checked={showCustomers} onChange={(e) => setShowCustomers(e.target.checked)} />
              Show customer stops
            </label>
            <Field label={`Depot ${animDepot}`}>
              <input
                type="range"
                min={0}
                max={Math.max(0, results.length - 1)}
                value={animDepot}
                onChange={onSliderAnimDepot}
              />
            </Field>
            <Field label={`Route ${animRoute}`}>
              <input
                type="range"
                min={0}
                max={Math.max(0, (results[animDepot]?.routes?.length || 1) - 1)}
                value={animRoute}
                onChange={onSliderAnimRoute}
              />
            </Field>
            <div className="field">
              <button type="button" className="btn-secondary" onClick={() => setAnimPlaying((p) => !p)}>
                {animPlaying ? "Pause vehicle" : "Play vehicle"}
              </button>
              <span className="subtitle">Vehicle follows the selected route on the map.</span>
            </div>
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
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <FitBounds bounds={fitBounds} />

          {results.map((depotResult, depotIdx) =>
            (depotResult?.routes || []).map((route, routeIdx) => {
              if (!route?.lat || !route?.lng) return null;
              const originalPositions = route.lat.map((lat, i) => [lat, route.lng[i]]);
              const positions = snappedRoutes[`${depotIdx}-${routeIdx}`] || originalPositions;
              const color = ROUTE_COLORS[(depotIdx + routeIdx) % ROUTE_COLORS.length];
              return (
                <Polyline
                  key={`${depotIdx}-${routeIdx}`}
                  positions={positions}
                  pathOptions={{ color, weight: 4, opacity: 0.85 }}
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

          {vehiclePos && animPlaying ? (
            <CircleMarker
              center={vehiclePos}
              radius={7}
              pathOptions={{ color: "#fff", weight: 3, fillColor: "#22c55e", fillOpacity: 1 }}
            >
              <Tooltip direction="top" permanent={false}>
                Active vehicle
              </Tooltip>
            </CircleMarker>
          ) : null}
        </MapContainer>
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
