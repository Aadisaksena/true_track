// ============================================================
// Map setup
// ============================================================
const map = L.map('map').setView([20.5937, 78.9629], 5); // default: India, until we get a real fix
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors',
}).addTo(map);

function dot(color) {
  return L.circleMarker([0, 0], { radius: 8, color: '#fff', weight: 2, fillColor: color, fillOpacity: 1 });
}
const style = (color, dashed) => ({ color, weight: 4, opacity: 0.85, dashArray: dashed ? '6 6' : null });

// Live-mode layers
const liveMarker = dot('#1E2761');       // current fused/GPS position
const destMarker = dot('#C62828');       // chosen destination
const routeLine = L.polyline([], style('#1E7A34')).addTo(map); // planned route
const liveTrail = L.polyline([], style('#1E2761'));            // where we've actually been

// Replay-mode layers
const gtLine = L.polyline([], style('#1E7A34'));
const naiveLine = L.polyline([], style('#C62828', true));
const aiLine = L.polyline([], style('#1E2761'));
const gtMarker = dot('#1E7A34');
const naiveMarker = dot('#C62828');
const aiMarker = dot('#1E2761');

function setStatus(mode, speedKmh, gpsText) {
  const modeEl = document.getElementById('status-mode');
  modeEl.textContent = mode || '—';
  modeEl.className = mode || '';
  document.getElementById('status-speed').textContent = speedKmh != null ? `${speedKmh.toFixed(1)} km/h` : '—';
  document.getElementById('status-gps').textContent = gpsText != null ? gpsText : '—';
}

function haversineM(lat1, lon1, lat2, lon2) {
  const R = 6371000, toRad = Math.PI / 180;
  const dLat = (lat2 - lat1) * toRad, dLon = (lon2 - lon1) * toRad;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

// ============================================================
// Tabs
// ============================================================
document.getElementById('tab-replay').onclick = () => switchTab('replay');
document.getElementById('tab-live').onclick = () => switchTab('live');
function switchTab(which) {
  document.getElementById('tab-replay').classList.toggle('active', which === 'replay');
  document.getElementById('tab-live').classList.toggle('active', which === 'live');
  document.getElementById('panel-replay').classList.toggle('hidden', which !== 'replay');
  document.getElementById('panel-live').classList.toggle('hidden', which !== 'live');

  if (which === 'replay') {
    liveMarker.remove(); destMarker.remove(); routeLine.setLatLngs([]); liveTrail.remove();
    gtLine.addTo(map); naiveLine.addTo(map); aiLine.addTo(map);
  } else {
    gtLine.remove(); naiveLine.remove(); aiLine.remove();
    gtLine.setLatLngs([]); naiveLine.setLatLngs([]); aiLine.setLatLngs([]);
  }
}
switchTab('live'); // start on Live Mode

// ============================================================
// Replay mode (unchanged behaviour — plays back a real recorded trip)
// ============================================================
let replayTimer = null;
let replaySessionId = null;

async function loadTrips() {
  const res = await fetch('/api/replay/trips');
  const trips = await res.json();
  const sel = document.getElementById('trip-select');
  sel.innerHTML = trips.map(t => `<option value="${t.id}">${t.label}</option>`).join('');
}
loadTrips();

document.getElementById('btn-start-replay').onclick = async () => {
  gtLine.setLatLngs([]); naiveLine.setLatLngs([]); aiLine.setLatLngs([]);
  const tripId = document.getElementById('trip-select').value;
  const res = await fetch(`/api/replay/${tripId}/start`, { method: 'POST' });
  const data = await res.json();
  replaySessionId = data.session_id;
  document.getElementById('btn-start-replay').disabled = true;
  document.getElementById('btn-pause-replay').disabled = false;
  document.getElementById('btn-pause-replay').textContent = 'Pause';
  runReplayLoop();
};

document.getElementById('btn-pause-replay').onclick = () => {
  if (replayTimer) {
    clearInterval(replayTimer);
    replayTimer = null;
    document.getElementById('btn-pause-replay').textContent = 'Resume';
  } else {
    runReplayLoop();
    document.getElementById('btn-pause-replay').textContent = 'Pause';
  }
};

function runReplayLoop() {
  replayTimer = setInterval(async () => {
    const simulateOutage = document.getElementById('toggle-outage-replay').checked;
    const res = await fetch(`/api/replay/session/${replaySessionId}/next`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ simulate_outage: simulateOutage }),
    });
    const d = await res.json();
    if (d.done) {
      clearInterval(replayTimer); replayTimer = null;
      document.getElementById('btn-start-replay').disabled = false;
      document.getElementById('btn-pause-replay').disabled = true;
      return;
    }

    if (d.phase === 'warmup') {
      aiMarker.setLatLng([d.lat, d.lon]).addTo(map);
      map.panTo([d.lat, d.lon]);
      setStatus('GNSS', null, 'demo trip');
      return;
    }

    gtMarker.setLatLng([d.ground_truth.lat, d.ground_truth.lon]).addTo(map);
    gtLine.addLatLng([d.ground_truth.lat, d.ground_truth.lon]);

    if (d.mode === 'DR') {
      aiMarker.setLatLng([d.ai_fused.lat, d.ai_fused.lon]).addTo(map);
      aiLine.addLatLng([d.ai_fused.lat, d.ai_fused.lon]);
      naiveMarker.setLatLng([d.naive.lat, d.naive.lon]).addTo(map);
      naiveLine.addLatLng([d.naive.lat, d.naive.lon]);
      const drift = haversineM(d.ground_truth.lat, d.ground_truth.lon, d.ai_fused.lat, d.ai_fused.lon);
      setStatus('DR', d.speed_kmh, `drift ${drift.toFixed(0)}m`);
      map.panTo([d.ai_fused.lat, d.ai_fused.lon]);
    } else {
      aiMarker.setLatLng([d.lat, d.lon]).addTo(map);
      map.panTo([d.lat, d.lon]);
      setStatus('GNSS', null, 'demo trip');
    }
  }, 100);
}

// ============================================================
// Live mode — behaves like a normal map app first, AI dead-reckoning
// only takes over when real GPS/network signal is actually lost.
// ============================================================
let liveSessionId = null;
let liveTimer = null;
let tracking = false;

let latestAccel = null;
let latestGravityEst = { x: 0, y: 0, z: 9.81 };
let latestGyro = { yaw: 0, pitch: 0, roll: 0 };
const GRAVITY_ALPHA = 0.85;

let latestGps = null;          // {lat, lon, accuracy} from the most recent GOOD fix
let lastFixTime = 0;           // ms timestamp of the most recent GOOD fix
let haveEverFixed = false;
const MAX_ACCEPTABLE_ACCURACY_M = 60; // reject wildly inaccurate fixes (cell-tower-only, etc.)
const STALE_AFTER_MS = 4000;          // no good fix for this long -> treat as signal lost

function onDeviceMotion(e) {
  const a = e.accelerationIncludingGravity;
  if (!a || a.x == null) return;
  latestAccel = { x: a.x, y: a.y, z: a.z };
  // Low-pass filter to separate gravity out of the combined reading -- the
  // same technique a phone's own gravity sensor uses. We don't rely on the
  // browser's separate "acceleration" (gravity-removed) field because it
  // isn't reliably available across devices/browsers.
  latestGravityEst = {
    x: GRAVITY_ALPHA * latestGravityEst.x + (1 - GRAVITY_ALPHA) * a.x,
    y: GRAVITY_ALPHA * latestGravityEst.y + (1 - GRAVITY_ALPHA) * a.y,
    z: GRAVITY_ALPHA * latestGravityEst.z + (1 - GRAVITY_ALPHA) * a.z,
  };
  if (e.rotationRate) {
    const d2r = Math.PI / 180;
    latestGyro = {
      yaw: (e.rotationRate.alpha || 0) * d2r,
      pitch: (e.rotationRate.beta || 0) * d2r,
      roll: (e.rotationRate.gamma || 0) * d2r,
    };
  }
}

function onGeoSuccess(pos) {
  const acc = pos.coords.accuracy;
  if (acc != null && acc > MAX_ACCEPTABLE_ACCURACY_M) {
    // Fix arrived but is too inaccurate to trust (e.g. wifi/cell-only lookup)
    // -- don't use it, but don't treat it as a total loss either; just wait
    // for a better one. If nothing better shows up, STALE_AFTER_MS below
    // will correctly trigger dead reckoning anyway.
    return;
  }
  latestGps = { lat: pos.coords.latitude, lon: pos.coords.longitude, accuracy: acc };
  lastFixTime = Date.now();
  if (!haveEverFixed) {
    haveEverFixed = true;
    map.setView([latestGps.lat, latestGps.lon], 17);
    liveMarker.setLatLng([latestGps.lat, latestGps.lon]).addTo(map);
    liveTrail.addTo(map);
    document.getElementById('btn-recenter').disabled = false;
  }
}

function onGeoError(err) {
  console.warn('Geolocation error:', err.message);
  // Explicit error -- lastFixTime is simply not refreshed, so the staleness
  // check below will correctly flip into dead-reckoning mode.
}

document.getElementById('btn-recenter').onclick = () => {
  if (latestGps) map.setView([latestGps.lat, latestGps.lon], 17);
};

document.getElementById('btn-start-live').onclick = async () => {
  if (tracking) return;

  // iOS 13+ requires an explicit permission prompt triggered by a user gesture
  if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
    try {
      const perm = await DeviceMotionEvent.requestPermission();
      if (perm !== 'granted') { alert('Motion sensor permission denied — AI dead reckoning needs this to work during a signal outage.'); return; }
    } catch (err) { alert('Could not request motion permission: ' + err); return; }
  }
  window.addEventListener('devicemotion', onDeviceMotion);

  if (!navigator.geolocation) {
    alert('This browser does not support geolocation.');
    return;
  }
  navigator.geolocation.watchPosition(onGeoSuccess, onGeoError, {
    enableHighAccuracy: true, maximumAge: 0, timeout: 5000,
  });

  const res = await fetch('/api/live/session/start', { method: 'POST' });
  const data = await res.json();
  liveSessionId = data.session_id;
  tracking = true;
  document.getElementById('btn-start-live').disabled = true;
  document.getElementById('btn-start-live').textContent = 'Tracking…';

  liveTimer = setInterval(async () => {
    if (!latestAccel) return; // no motion data yet, nothing to send

    const manualOutage = document.getElementById('toggle-outage-live').checked;
    const signalStale = haveEverFixed && (Date.now() - lastFixTime > STALE_AFTER_MS);
    const useGps = haveEverFixed && !manualOutage && !signalStale;

    const body = {
      accel: latestAccel, gravity: latestGravityEst, gyro: latestGyro,
      gps: useGps ? { lat: latestGps.lat, lon: latestGps.lon } : null,
      simulate_outage: manualOutage || signalStale || !haveEverFixed,
      dt: 0.1,
    };
    const r = await fetch(`/api/live/session/${liveSessionId}/sample`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const d = await r.json();

    if (d.mode === 'NO_FIX' || d.lat == null) {
      setStatus('waiting for GPS…', null, 'no fix yet');
      return;
    }

    liveMarker.setLatLng([d.lat, d.lon]).addTo(map);
    liveTrail.addLatLng([d.lat, d.lon]);
    map.panTo([d.lat, d.lon]);

    let gpsText;
    if (d.mode === 'GNSS') {
      gpsText = latestGps && latestGps.accuracy != null ? `±${Math.round(latestGps.accuracy)}m` : 'ok';
    } else {
      gpsText = manualOutage ? 'forced outage' : 'signal lost — using AI';
    }
    setStatus(d.mode, d.speed_kmh, gpsText);
  }, 100);
};

// ============================================================
// Destination search (Nominatim geocoding) + routing (OSRM)
// Both are free public demo services -- fine for hackathon/demo use,
// not for production-scale traffic (Nominatim in particular asks for
// max ~1 request/second).
// ============================================================
let selectedDestination = null;

async function geocode(query) {
  const url = `https://nominatim.openstreetmap.org/search?format=json&limit=5&q=${encodeURIComponent(query)}`;
  const res = await fetch(url, { headers: { 'Accept-Language': 'en' } });
  if (!res.ok) throw new Error('Geocoding failed');
  return res.json();
}

document.getElementById('btn-route').onclick = async () => {
  const q = document.getElementById('destination-search').value.trim();
  if (!q) return;
  let results;
  try {
    results = await geocode(q);
  } catch (e) {
    alert('Could not search for that place: ' + e.message);
    return;
  }
  const box = document.getElementById('geocode-results');
  if (!results.length) {
    box.innerHTML = '<div class="geocode-item">No results found.</div>';
    box.classList.remove('hidden');
    return;
  }
  box.innerHTML = results.map((r, i) =>
    `<div class="geocode-item" data-i="${i}">${r.display_name}</div>`
  ).join('');
  box.classList.remove('hidden');
  box.querySelectorAll('.geocode-item[data-i]').forEach(el => {
    el.onclick = () => {
      const r = results[parseInt(el.dataset.i, 10)];
      selectedDestination = { lat: parseFloat(r.lat), lon: parseFloat(r.lon), label: r.display_name };
      destMarker.setLatLng([selectedDestination.lat, selectedDestination.lon]).addTo(map);
      box.classList.add('hidden');
      document.getElementById('destination-search').value = r.display_name;
      drawRoute();
    };
  });
};

async function drawRoute() {
  if (!selectedDestination) return;
  if (!latestGps) {
    alert('Waiting for your current location first — allow location access, then try again.');
    return;
  }
  const { lat: olat, lon: olon } = latestGps;
  const { lat: dlat, lon: dlon } = selectedDestination;
  const url = `https://router.project-osrm.org/route/v1/driving/${olon},${olat};${dlon},${dlat}?overview=full&geometries=geojson`;
  let data;
  try {
    const res = await fetch(url);
    data = await res.json();
  } catch (e) {
    alert('Routing service unavailable: ' + e.message);
    return;
  }
  if (!data.routes || !data.routes.length) {
    alert('No route found between your location and that destination.');
    return;
  }
  const coords = data.routes[0].geometry.coordinates.map(([lon, lat]) => [lat, lon]);
  routeLine.setLatLngs(coords).addTo(map);
  map.fitBounds(routeLine.getBounds(), { padding: [30, 30] });
}