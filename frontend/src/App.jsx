import { useState, useRef, useEffect } from "react";

function App() {
  const canvasRef = useRef(null);

  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);

  const [phase, setPhase] = useState(0);

  // -------- CITY --------
  const depots = [
    { x: 200, y: 150 },
    { x: 650, y: 150 },
    { x: 300, y: 450 },
    { x: 700, y: 450 }
  ];

  let customers = [];
  depots.forEach(d => {
    for (let i = 0; i < 25; i++) {
      customers.push({
        x: d.x + (Math.random() - 0.5) * 150,
        y: d.y + (Math.random() - 0.5) * 150,
        depot: null,
        visited: false
      });
    }
  });

  useEffect(() => {
    if (!running) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    let frame;
    let timer = 0;

    let vehicles = depots.map(d => ({
      x: d.x,
      y: d.y,
      depot: d,
      target: null,
      prev: null,
      route: []
    }));

    // -------- ASSIGN CUSTOMERS --------
    const assignCustomers = () => {
      customers.forEach(c => {
        let bestDepot = depots.reduce((best, d) => {
          const d1 = Math.hypot(c.x - d.x, c.y - d.y);
          const d2 = Math.hypot(c.x - best.x, c.y - best.y);
          return d1 < d2 ? d : best;
        });
        c.depot = bestDepot;
      });
    };

    const getNextCustomer = (vehicle) => {
      let options = customers.filter(
        c => c.depot === vehicle.depot && !c.visited
      );

      if (options.length === 0) return null;

      return options.reduce((best, c) => {
        const d1 = Math.hypot(vehicle.x - c.x, vehicle.y - c.y);
        const d2 = Math.hypot(vehicle.x - best.x, vehicle.y - best.y);
        return d1 < d2 ? c : best;
      });
    };

    assignCustomers();

    const animate = () => {
      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      timer += 0.01 * speed;

      // -------- PHASE 1: SHOW CITY --------
      if (phase === 0) {
        drawCity(ctx);

        if (timer > 1.5) {
          setPhase(1);
          timer = 0;
        }
      }

      // -------- PHASE 2: ASSIGNMENT --------
      else if (phase === 1) {
        drawCity(ctx);

        customers.forEach(c => {
          ctx.beginPath();
          ctx.moveTo(c.x, c.y);
          ctx.lineTo(c.depot.x, c.depot.y);
          ctx.strokeStyle = "rgba(0,255,159,0.2)";
          ctx.stroke();
        });

        if (timer > 2) {
          setPhase(2);
          timer = 0;
        }
      }

      // -------- PHASE 3: ROUTING --------
      else if (phase === 2) {
        drawCity(ctx);

        vehicles.forEach((v, idx) => {
          const color = ["#ff4d4d", "#00ff9f", "#4da6ff", "#facc15"][idx];

          if (!v.target) {
            v.target = getNextCustomer(v);
            v.prev = { x: v.x, y: v.y };
          }

          if (v.target) {
            const dx = v.target.x - v.x;
            const dy = v.target.y - v.y;

            v.x += dx * 0.02 * speed;
            v.y += dy * 0.02 * speed;

            // 🔥 DRAW CURRENT EDGE ONLY
            ctx.beginPath();
            ctx.moveTo(v.prev.x, v.prev.y);
            ctx.lineTo(v.x, v.y);
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.stroke();

            if (Math.hypot(v.x - v.target.x, v.y - v.target.y) < 5) {
              v.target.visited = true;
              v.route.push(v.target);
              v.prev = { x: v.target.x, y: v.target.y };
              v.target = null;
            }
          }

          // vehicle
          ctx.beginPath();
          ctx.fillStyle = "#fff";
          ctx.arc(v.x, v.y, 5, 0, 2 * Math.PI);
          ctx.fill();
        });
      }

      frame = requestAnimationFrame(animate);
    };

    const drawCity = (ctx) => {
      // customers
      customers.forEach(c => {
        ctx.fillStyle = c.visited ? "#22c55e" : "#666";
        ctx.beginPath();
        ctx.arc(c.x, c.y, 3, 0, 2 * Math.PI);
        ctx.fill();
      });

      // depots
      depots.forEach(d => {
        ctx.fillStyle = "#00e5ff";
        ctx.beginPath();
        ctx.arc(d.x, d.y, 7, 0, 2 * Math.PI);
        ctx.fill();
      });
    };

    animate();
    return () => cancelAnimationFrame(frame);

  }, [running, speed, phase]);

  return (
    <div style={styles.container}>
      <canvas ref={canvasRef} width={900} height={600} />

      <div style={styles.panel}>
        <button onClick={() => {
          setPhase(0);
          setRunning(true);
        }}>Start</button>

        <button onClick={() => setRunning(false)}>Stop</button>

        <div>
          Speed: {speed.toFixed(2)}
          <input
            type="range"
            min="0.2"
            max="3"
            step="0.1"
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
          />
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    background: "#020617"
  },
  panel: {
    width: "250px",
    padding: "20px",
    background: "#0f172a",
    color: "white"
  }
};

export default App;