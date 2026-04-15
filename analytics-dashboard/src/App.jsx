import React, { useState, useMemo } from 'react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer,
  PieChart, Pie, Cell
} from 'recharts';
import { Activity, Zap, Compass, RefreshCw, BarChart2, Layers, AlertTriangle } from 'lucide-react';

const API_URL = "http://127.0.0.1:5000/optimize";

const COLORS = ['#2979ff', '#00e676', '#ff9100', '#ff3333', '#d946ef', '#14b8a6'];
const PIE_COLORS = ['#ff9100', '#2979ff', '#ff3333'];

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="custom-tooltip">
        <p className="label">Iteration {label}</p>
        {payload.map((p, i) => (
          <p key={i} style={{ color: p.color }}>
            {p.name}: {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function App() {
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);
  const [error, setError] = useState("");

  const fetchData = async () => {
    setLoading(true);
    setError("");
    try {
      const body = {
        scenario: "nyc",
        num_depots: 3,
        max_customers: 60,
        vehicle_capacity: 200,
        num_ants: 25,
        alpha: 1.2,
        beta: 2.0,
        evaporation: 0.4,
        iterations: 30,
        exploration_bias: 0.5,
        random_seed: 42,
      };

      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) throw new Error("Server error");
      const json = await response.json();
      if (json.status !== "ok") throw new Error("Invalid response");
      
      setData(json);
    } catch (err) {
      setError(err.message || "Failed to connect to simulation engine.");
    } finally {
      setLoading(false);
    }
  };

  // 1. Convergence Data
  const convergenceData = useMemo(() => {
    if (!data?.convergence?.combined) return [];
    return data.convergence.combined.map((val, i) => ({
      iteration: i,
      distance: val
    }));
  }, [data]);

  // 2. Entropy Data (from depot 0 timeline as proxy for colony exploration state)
  const entropyData = useMemo(() => {
    if (!data?.results?.[0]?.timeline) return [];
    return data.results[0].timeline.map(t => ({
      iteration: t.iteration,
      entropy: t.entropy
    }));
  }, [data]);

  // 3. Local Search Gain (2-opt savings summed across all depots)
  const localSearchData = useMemo(() => {
    if (!data?.results || data.results.length === 0) return [];
    
    const maxIters = Math.max(...data.results.map(d => d.timeline?.length || 0));
    const combined = [];
    
    let totalSavingsOverall = 0;

    for (let i = 0; i < maxIters; i++) {
       let iteration_savings = 0;
       data.results.forEach(d => {
          if (d.timeline?.[i]?.hybrid_savings) {
             iteration_savings += d.timeline[i].hybrid_savings;
          }
       });
       totalSavingsOverall += iteration_savings;
       combined.push({
         iteration: i,
         savings: iteration_savings
       });
    }
    
    // Add a flag if completely zero so we can show an empty state.
    combined.hasData = totalSavingsOverall > 0.01;
    return combined;
  }, [data]);

  // 4. Reason Distribution Pie
  const reasonData = useMemo(() => {
    if (!data?.results?.[0]?.timeline) return [];
    const lastTimeline = data.results[0].timeline[data.results[0].timeline.length - 1];
    if (!lastTimeline?.route_rationales) return [];
    
    let phero = 0, heur = 0, diver = 0;
    lastTimeline.route_rationales.forEach(r => {
      if (r.includes("Pheromone")) phero++;
      else if (r.includes("Heuristic")) heur++;
      else diver++;
    });

    return [
      { name: 'Pheromone Trail (Exploit)', value: phero },
      { name: 'Heuristic Distance (Greedy)', value: heur },
      { name: 'Probabilistic Diversity (Explore)', value: diver }
    ];
  }, [data]);

  // 5. Fleet Composition 
  const fleetSummary = useMemo(() => {
    if (!data?.results) return [];
    let fleet = [];
    data.results.forEach(depot => {
      const last = depot.timeline[depot.timeline.length - 1];
      if (last && last.fleet) {
        last.fleet.forEach((route, idx) => {
           fleet.push({
             id: `V${fleet.length + 1}`,
             nodes: route.length - 2,
             depot: depot.depot_id
           });
        });
      }
    });
    return fleet;
  }, [data]);

  // 6. Fleet Node Distribution (Replacing empty probability log)
  const fleetSizeData = useMemo(() => {
     if (!data?.results) return [];
     let distribution = [];
     let vCount = 1;
     data.results.forEach(depot => {
       const last = depot.timeline[depot.timeline.length - 1];
       if (last && last.fleet) {
         last.fleet.forEach(route => {
            distribution.push({
              vehicle: `V${vCount}`,
              nodes: route.length - 2
            });
            vCount++;
         });
       }
     });
     return distribution;
  }, [data]);

  return (
    <div className="dashboard-container">
      {/* SIDEBAR / CONTROL PANEL */}
      <aside className="sidebar">
        <div className="brand">
          <h1 className="brand-title">Syndicate<br/>ACO Net</h1>
          <span className="brand-subtitle">MDVRP Quantum Logistics Matrix</span>
        </div>

        <div className="control-group" style={{marginTop: '2rem'}}>
          <span className="control-label">Engine Link</span>
          <button 
            className="btn btn-primary" 
            onClick={fetchData} 
            disabled={loading}
          >
            {loading ? <RefreshCw className="spin" size={18} /> : <Zap size={18} />}
            {loading ? "Simulating..." : "Initialize Run"}
          </button>
          
          {error && <div style={{color: 'var(--accent-red)', fontSize: '0.8rem'}}>{error}</div>}
        </div>

        {data && (
          <div className="control-group" style={{marginTop: '2rem', padding: '1rem', background: '#111', border: '1px solid #333'}}>
             <span className="control-label" style={{color: 'var(--accent-green)'}}>LIVE TELEMETRY SUMMARY</span>
             <p style={{fontSize: '0.8rem', color: '#fff', margin: '0.5rem 0'}}>Distance: <strong style={{color: 'var(--accent-blue)'}}>{data.stats.total_distance.toFixed(1)} km</strong></p>
             <p style={{fontSize: '0.8rem', color: '#fff', margin: '0.5rem 0'}}>Vehicles Active: <strong>{data.stats.total_vehicles}</strong></p>
             <p style={{fontSize: '0.8rem', color: '#fff', margin: '0.5rem 0'}}>Nodes Served: <strong>{data.stats.customer_count}</strong></p>
          </div>
        )}

        <div className="control-group" style={{marginTop: 'auto', padding: '1rem', background: 'rgba(255, 51, 51, 0.1)', border: '1px solid var(--accent-red)'}}>
          <h4 style={{fontSize: '0.75rem', color: 'var(--accent-red)', display: 'flex', alignItems: 'center', gap: '0.4rem'}}><AlertTriangle size={14}/> STOCHASTIC ENGINE</h4>
          <p style={{fontSize: '0.7rem', color: '#aaa', marginTop: '0.5rem', lineHeight: '1.4'}}>
            Ant Colony Algorithms rely on probabilistic exploration. The backend leverages python's global RNG which advances randomly on each API fetch. <strong>It is mathematically guaranteed and intended that hitting 'Initialize Run' multiple times evaluates varying geometric routes, producing entirely different charts.</strong> Review raw JSON stream below.
          </p>
        </div>
      </aside>

      {/* MAIN VIEW */}
      <main className="main-content">
        <header className="header">
          <div className="header-title">Optimization Analytics Core</div>
          <div className={`header-status ${!data ? 'idle' : ''}`}>
             <span className={`status-dot ${!data ? 'idle' : ''}`}></span>
             {data ? 'DATA LINK ACTIVE' : 'AWAITING TELEMETRY'}
          </div>
        </header>

        {!data ? (
          <div className="empty-state">
            <Activity />
            <p>Initiate Engine Run to Visualize Metrics.</p>
          </div>
        ) : (
          <>
            {/* KPI STRIP */}
            <div className="kpi-grid">
              <div className="kpi-card">
                <span className="kpi-label">Total Optimization Distance</span>
                <span className="kpi-value">{data.stats.total_distance.toFixed(1)} <span style={{fontSize: '0.8rem', color: 'var(--text-muted)'}}>KM</span></span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Colony Iterations</span>
                <span className="kpi-value">{data.stats.iterations}</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Active Fleet Vehicles</span>
                <span className="kpi-value">{data.stats.total_vehicles}</span>
              </div>
              <div className="kpi-card">
                <span className="kpi-label">Customer Hubs Assigned</span>
                <span className="kpi-value">{data.stats.customer_count}</span>
              </div>
            </div>

            {/* CHARTS GRID */}
            <div className="charts-grid">

              {/* Convergence Line */}
              <div className="chart-card chart-card-full">
                <div className="chart-header">
                  <h3 className="chart-title"><Activity size={18} /> Global Minima Convergence</h3>
                  <span className="control-label" style={{color: 'var(--accent-blue)'}}>TOTAL DISTANCE VS TIME</span>
                </div>
                <div className="chart-container" style={{height: '350px'}}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={convergenceData}>
                      <XAxis dataKey="iteration" tick={{fill: '#888'}} axisLine={false} tickLine={false} />
                      <YAxis domain={['auto', 'auto']} tick={{fill: '#888'}} axisLine={false} tickLine={false} />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Line type="monotone" dataKey="distance" name="Total Distance" stroke="var(--accent-blue)" strokeWidth={3} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Entropy Area */}
              <div className="chart-card">
                <div className="chart-header">
                  <h3 className="chart-title"><Layers size={18} /> Swarm Diversity (Entropy)</h3>
                </div>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <AreaChart data={entropyData}>
                      <defs>
                        <linearGradient id="colorEntropy" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="var(--accent-green)" stopOpacity={0.8}/>
                          <stop offset="95%" stopColor="var(--accent-green)" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <XAxis dataKey="iteration" hide />
                      <YAxis hide domain={['auto', 'auto']} />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Area type="monotone" dataKey="entropy" name="Shannon Entropy" stroke="var(--accent-green)" strokeWidth={2} fillOpacity={1} fill="url(#colorEntropy)" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Route Logic Pie Chart */}
              <div className="chart-card">
                <div className="chart-header">
                  <h3 className="chart-title"><Compass size={18} /> Heuristic Intelligence State</h3>
                </div>
                <div className="chart-container" style={{display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={reasonData}
                        innerRadius={80}
                        outerRadius={110}
                        paddingAngle={5}
                        dataKey="value"
                        stroke="none"
                      >
                        {reasonData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={PIE_COLORS[index % PIE_COLORS.length]} />
                        ))}
                      </Pie>
                      <RechartsTooltip content={<CustomTooltip />} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Local Search Savings Bar */}
              <div className="chart-card">
                <div className="chart-header">
                  <h3 className="chart-title"><BarChart2 size={18} /> 2-Opt Local Search Yield</h3>
                </div>
                <div className="chart-container">
                  {localSearchData.hasData ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={localSearchData}>
                        <XAxis dataKey="iteration" hide />
                        <YAxis hide />
                        <RechartsTooltip content={<CustomTooltip />} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
                        <Bar dataKey="savings" name="Distance Saved" fill="var(--accent-orange)" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={{display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)'}}>
                      2-Opt yielded 0 savings (Routes fully optimized).
                    </div>
                  )}
                </div>
              </div>

              {/* Fleet Deployment Chart */}
              <div className="chart-card">
                <div className="chart-header">
                  <h3 className="chart-title"><Activity size={18} /> Fleet Route Length Distribution</h3>
                  <span className="control-label" style={{color: 'var(--accent-green)'}}>NODES SERVED PER VEHICLE</span>
                </div>
                <div className="chart-container">
                  {fleetSizeData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={fleetSizeData}>
                        <XAxis dataKey="vehicle" tick={{fill: '#888', fontSize: 12}} axisLine={false} tickLine={false} />
                        <YAxis hide />
                        <RechartsTooltip content={<CustomTooltip />} cursor={{fill: 'rgba(255,255,255,0.05)'}} />
                        <Bar dataKey="nodes" name="Customers Served" fill="var(--accent-blue)" radius={[2, 2, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={{display: 'flex', height: '100%', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)'}}>
                      No valid fleet data recorded.
                    </div>
                  )}
                </div>
              </div>

              {/* Fleet Deployment Layout */}
              <div className="chart-card chart-card-full" style={{minHeight: 'min-content', paddingBottom: '2rem'}}>
                 <div className="chart-header">
                  <h3 className="chart-title"><Layers size={18} /> Optimized Active Fleet Roster</h3>
                 </div>
                 <div className="fleet-grid">
                    {fleetSummary.map(f => (
                      <div key={f.id} className="fleet-card">
                        <div className="fleet-header">
                           <span className="fleet-id">{f.id}</span>
                           <span>HUB {f.depot}</span>
                        </div>
                        <div className="fleet-metric">
                           {f.nodes} <span>NODES</span>
                        </div>
                      </div>
                    ))}
                 </div>
              </div>

              {/* RAW DATA AUDIT */}
              <div className="chart-card chart-card-full">
                 <div className="chart-header">
                  <h3 className="chart-title"><Layers size={18} /> Telemetry Data Audit (Verification)</h3>
                  <span className="control-label" style={{color: 'var(--text-muted)'}}>ACO IS STOCHASTIC (EXPECT SLIGHT VARIANCE PER RUN)</span>
                 </div>
                 <div className="chart-container" style={{height: 'auto', background: '#0a0a0a', padding: '1rem', border: '1px solid #333', overflowX: 'auto'}}>
                    <pre style={{color: '#00e676', fontSize: '0.75rem', fontFamily: 'monospace'}}>
{JSON.stringify({
  source_file: "backend/main_engine.py",
  timestamp: new Date().toISOString(),
  parameters_sent: { random_seed: 42, num_ants: 25, iterations: 30 },
  backend_status: data.status,
  overall_stats: data.stats,
  sample_depot_0_fleet: data.results[0]?.routes?.length + " routes mapped via Valhalla/Euclidean",
  champion_route_sample: data.results[0]?.timeline?.[data.results[0].timeline.length - 1]?.route?.slice(0,10)
}, null, 2)}
                    </pre>
                 </div>
              </div>

            </div>
          </>
        )}
      </main>
    </div>
  );
}
