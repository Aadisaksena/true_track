// ---- Map setup ----
const map = L.map('map').setView([52.5, -1.5], 15);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors',
}).addTo(map);

const style = (color) => ({ color, weight: 4, opacity: 0.85 });
const gtLine = L.polyline([], style('#1E7A34')).addTo(map);
const naiveLine = L.polyline([], style('#C62828')).addTo(map);
const aiLine = L.polyline([], style('#1E2761')).addTo(map);

function dot(color) {
  return L.circleMarker([0, 0], { radius: 7, color: '#fff', weight: 2, fillColor: color, fillOpacity: 1 });
}
const gtMarker = dot('#1E7A34');
const naiveMarker = dot('#C62828');
const aiMarker = dot('#1E2761').addTo(map);

const legend = L.control({ position: 'bottomright' });
legend.onAdd = () => {
  const div = L.DomUtil.create('div', 'legend');
  div.style.background = 'white'; div.style.padding = '6px 10px'; div.style.borderRadius = '8px';
  div.style.fontSize = '12px'; div.style.lineHeight = '1.6em';
  div.innerHTML = `
    <div><span class="legend-dot" style="background:#1E7A34"></span>Ground truth (GPS)</div>
    <div><span class="legend-dot" style="background:#C62828"></span>Naive integration</div>
    <div><span class="legend-dot" style="background:#1E2761"></span>TrueTrack (AI-fused)</div>`;
  return div;
};
legend.addTo(map);

function setStatus(mode, speedKmh, driftM) {
  const modeEl = document.getElementById('status-mode');
  modeEl.textContent = mode || '—';
  modeEl.className = mode || '';
  document.getElementById('status-speed').textContent = speedKmh != null ? `${speedKmh.toFixed(1)} km/h` : '—';
  document.getElementById('status-drift').textContent = driftM != null ? `${driftM.toFixed(0)} m` : '—';
}

function haversineM(lat1, lon1, lat2, lon2) {
  const R = 6371000, toRad = Math.PI / 180;
  const dLat = (lat2 - lat1) * toRad, dLon = (lon2 - lon1) * toRad;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

// ---- Tabs ----
document.getElementById('tab-replay').onclick = () => switchTab('replay');
document.getElementById('tab-live').onclick = () => switchTab('live');
function switchTab(which) {
  document.getElementById('tab-replay').classList.toggle('active', which === 'replay');
  document.getElementById('tab-live').classList.toggle('active', which === 'live');
  document.getElementById('panel-replay').classList.toggle('hidden', which !== 'replay');
  document.getElementById('panel-live').classList.toggle('hidden', which !== 'live');
}

// ---- Replay mode ----
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
      setStatus('GNSS', null, null);
      return;
    }

    // blackout phase
    gtMarker.setLatLng([d.ground_truth.lat, d.ground_truth.lon]).addTo(map);
    gtLine.addLatLng([d.ground_truth.lat, d.ground_truth.lon]);

    if (d.mode === 'DR') {
      aiMarker.setLatLng([d.ai_fused.lat, d.ai_fused.lon]);
      aiLine.addLatLng([d.ai_fused.lat, d.ai_fused.lon]);
      naiveMarker.setLatLng([d.naive.lat, d.naive.lon]).addTo(map);
      naiveLine.addLatLng([d.naive.lat, d.naive.lon]);
      const drift = haversineM(d.ground_truth.lat, d.ground_truth.lon, d.ai_fused.lat, d.ai_fused.lon);
      setStatus('DR', d.speed_kmh, drift);
      map.panTo([d.ai_fused.lat, d.ai_fused.lon]);
    } else {
      aiMarker.setLatLng([d.lat, d.lon]);
      map.panTo([d.lat, d.lon]);
      setStatus('GNSS', null, 0);
    }
  }, 100);
}

// ---- Live mode ----
let liveSessionId = null;
let liveTimer = null;
let latestAccel = null, latestGravityEst = { x: 0, y: 0, z: 9.81 }, latestGyro = { yaw: 0, pitch: 0, roll: 0 };
let latestGps = null;
const GRAVITY_ALPHA = 0.85;

function onDeviceMotion(e) {
  const a = e.accelerationIncludingGravity;
  if (!a || a.x == null) return;
  latestAccel = { x: a.x, y: a.y, z: a.z };
  // low-pass filter to separate out gravity, same technique Android's own
  // gravity sensor uses -- we don't rely on a raw "linear acceleration" field
  // because it isn't reliably available across browsers/devices.
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

document.getElementById('btn-start-live').onclick = async () => {
  // iOS 13+ requires an explicit permission prompt triggered by a user gesture
  if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
    try {
      const perm = await DeviceMotionEvent.requestPermission();
      if (perm !== 'granted') { alert('Motion sensor permission denied.'); return; }
    } catch (err) { alert('Could not request motion permission: ' + err); return; }
  }
  window.addEventListener('devicemotion', onDeviceMotion);

  if (navigator.geolocation) {
    navigator.geolocation.watchPosition(
      (pos) => { latestGps = { lat: pos.coords.latitude, lon: pos.coords.longitude }; },
      (err) => { console.warn('Geolocation error:', err.message); },
      { enableHighAccuracy: true, maximumAge: 500 },
    );
  }

  const res = await fetch('/api/live/session/start', { method: 'POST' });
  const data = await res.json();
  liveSessionId = data.session_id;
  document.getElementById('btn-start-live').disabled = true;
  document.getElementById('btn-start-live').textContent = 'Tracking…';

  liveTimer = setInterval(async () => {
    if (!latestAccel) return;
    const simulateOutage = document.getElementById('toggle-outage-live').checked;
    const body = {
      accel: latestAccel, gravity: latestGravityEst, gyro: latestGyro,
      gps: latestGps, simulate_outage: simulateOutage, dt: 0.1,
    };
    const r = await fetch(`/api/live/session/${liveSessionId}/sample`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const d = await r.json();
    if (d.mode === 'NO_FIX' || d.lat == null) return;
    aiMarker.setLatLng([d.lat, d.lon]).addTo(map);
    aiLine.addLatLng([d.lat, d.lon]);
    map.panTo([d.lat, d.lon]);
    setStatus(d.mode, d.speed_kmh, null);
  }, 100);
};
