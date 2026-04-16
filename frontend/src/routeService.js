// Service to snap paths and nodes to roads using Valhalla Routing Engine
// Handles >20 coordinates by chunking, with fallback mechanisms.

const routeCache = new Map();
const requestQueue = [];
let isProcessing = false;

// We delay slightly to be polite to the public Valhalla API
const VALHALLA_DELAY = 1000;
const MAX_RETRIES = 3;
const CHUNK_SIZE = 19; // Valhalla public API imposes a ~20 location limit

function decodePolyline(str, precision = 6) {
  let index = 0, lat = 0, lng = 0, coordinates = [], shift = 0, result = 0, byte = null, latitude_change, longitude_change, factor = Math.pow(10, precision);
  while (index < str.length) {
      byte = null; shift = 0; result = 0;
      do { byte = str.charCodeAt(index++) - 63; result |= (byte & 0x1f) << shift; shift += 5; } while (byte >= 0x20);
      latitude_change = ((result & 1) ? ~(result >> 1) : (result >> 1));
      shift = result = 0;
      do { byte = str.charCodeAt(index++) - 63; result |= (byte & 0x1f) << shift; shift += 5; } while (byte >= 0x20);
      longitude_change = ((result & 1) ? ~(result >> 1) : (result >> 1));
      lat += latitude_change; lng += longitude_change;
      coordinates.push([lat / factor, lng / factor]);
  }
  return coordinates;
}

function generateMockPath(path) {
  if (path.length < 2) {
    return {
      snapped: path,
      waypoints: path.map(p => ({ original: p, snapped: p }))
    };
  }
  const snapped = [path[0]];
  for (let i = 0; i < path.length - 1; i++) {
    const p1 = path[i];
    const p2 = path[i + 1];
    if ((i % 2) === 0) {
      snapped.push([p1[0], p2[1]]);
    } else {
      snapped.push([p2[0], p1[1]]);
    }
    snapped.push(p2);
  }
  const waypoints = path.map(p => ({ original: p, snapped: p }));
  return { snapped, waypoints };
}

function delay(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function fetchValhallaWithRetry(locations) {
  const url = `https://valhalla1.openstreetmap.de/route`;
  const body = JSON.stringify({ locations, costing: 'auto' });
  
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const response = await fetch(url, { method: 'POST', body, headers: { 'Content-Type': 'application/json' }});
      if (response.status === 429 || response.status >= 500) {
        throw new Error(`Rate limited or server error (${response.status})`);
      }
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Valhalla request failed: ${response.status} - ${text}`);
      }
      const data = await response.json();
      if (data.trip && data.trip.legs) {
        return data;
      }
      throw new Error(`Invalid Valhalla response`);
    } catch (err) {
      if (attempt === MAX_RETRIES - 1) {
        throw err;
      }
      const backoffMs = VALHALLA_DELAY * Math.pow(2, attempt);
      console.warn(`Valhalla fetch failed, retrying in ${backoffMs}ms...`, err.message);
      await delay(backoffMs);
    }
  }
  throw new Error("Failed to fetch from Valhalla after max retries");
}

async function processQueue() {
  if (isProcessing) return;
  isProcessing = true;

  while (requestQueue.length > 0) {
    const { path, resolve } = requestQueue.shift();
    const key = path.map(p => `${p[0].toFixed(5)},${p[1].toFixed(5)}`).join('|');
    
    if (routeCache.has(key)) {
      resolve(routeCache.get(key));
      continue;
    }

    try {
      let combinedSnapped = [];
      let combinedWaypoints = [];
      
      // Valhalla has a strict 20 coordinate limit. Chunk with 1 overlap.
      for (let i = 0; i < path.length; i += CHUNK_SIZE - 1) {
        const chunk = path.slice(i, i + CHUNK_SIZE);
        if (chunk.length < 2) {
            if (chunk.length === 1 && path.length > 1) continue;
        }
        
        const locs = chunk.map(p => ({ lat: p[0], lon: p[1] }));
        const data = await fetchValhallaWithRetry(locs);
        
        let chunkSnapped = [];
        data.trip.legs.forEach(leg => {
           let decoded = decodePolyline(leg.shape, 6);
           if (chunkSnapped.length > 0 && decoded.length > 0) {
               // Prevent exact duplicate of the boundary node
               chunkSnapped.push(...decoded.slice(1));
           } else {
               chunkSnapped.push(...decoded);
           }
        });

        const chunkWaypoints = data.trip.locations.map((w, wIdx) => ({
          original: chunk[wIdx],
          snapped: w ? [w.lat, w.lon] : chunk[wIdx]
        }));

        if (i === 0) {
          combinedSnapped.push(...chunkSnapped);
          combinedWaypoints.push(...chunkWaypoints);
        } else {
          // Drop overlapping first point for polyline and waypoints
          if (combinedSnapped.length > 0 && chunkSnapped.length > 0) {
              combinedSnapped.push(...chunkSnapped.slice(1));
          }
          combinedWaypoints.push(...chunkWaypoints.slice(1));
        }

        await delay(VALHALLA_DELAY);
      }
      
      const result = { snapped: combinedSnapped, waypoints: combinedWaypoints };
      routeCache.set(key, result);
      resolve(result);

    } catch (err) {
      console.warn("Failed to route, falling back to mock road mapping:", err.message);
      const mockResult = generateMockPath(path);
      routeCache.set(key, mockResult);
      resolve(mockResult);
      await delay(VALHALLA_DELAY);
    }
  }

  isProcessing = false;
}

export function fetchSnappedRoute(path) {
  return new Promise((resolve) => {
    requestQueue.push({ path, resolve });
    processQueue();
  });
}
