// ============================================================
// S.A.F.A.R. / TrueTrack — Frontend Navigation Controller
// ============================================================

// --- Map Initialization ---
const map = L.map('map', {
  zoomControl: false // custom floating controls
}).setView([20.5937, 78.9629], 5); // Default view until GPS fix

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors',
  maxZoom: 19
}).addTo(map);

// Leaflet Marker Helpers with Heading Cone
function createDirectionalMarker(color) {
  const icon = L.divIcon({
    className: 'custom-vehicle-marker',
    html: `<div style="width: 20px; height: 20px; background: ${color}; border: 2.5px solid #ffffff; border-radius: 50%; box-shadow: 0 0 10px ${color};"></div>`,
    iconSize: [20, 20],
    iconAnchor: [10, 10]
  });
  return L.marker([0, 0], { icon });
}

const style = (color, dashed) => ({ color, weight: 4, opacity: 0.85, dashArray: dashed ? '6 6' : null });

// Map Layers & Markers
const liveMarker = createDirectionalMarker('#38bdf8');
const destMarker = L.circleMarker([0, 0], { radius: 8, color: '#fff', weight: 2, fillColor: '#f43f5e', fillOpacity: 1 });
const routeLine = L.polyline([], style('#38bdf8')).addTo(map);
const liveTrail = L.polyline([], style('#38bdf8'));

const gtLine = L.polyline([], style('#10b981'));
const naiveLine = L.polyline([], style('#f43f5e', true));
const aiLine = L.polyline([], style('#38bdf8'));

const gtMarker = createDirectionalMarker('#10b981');
const naiveMarker = createDirectionalMarker('#f43f5e');
const aiMarker = createDirectionalMarker('#38bdf8');

// --- UI Element References ---
const navModeBadge = document.getElementById('nav-mode-badge');
const navModeText = document.getElementById('nav-mode-text');
const hudSpeed = document.getElementById('hud-speed');
const hudConfidencePct = document.getElementById('hud-confidence-pct');
const hudConfidenceFill = document.getElementById('hud-confidence-fill');

const sheetStatus = document.getElementById('sheet-status');
const sheetUncertainty = document.getElementById('sheet-uncertainty');
const sheetHeading = document.getElementById('sheet-heading');

const telHeading = document.getElementById('tel-heading');
const telUncertainty = document.getElementById('tel-uncertainty');
const telSpeedSource = document.getElementById('tel-speed-source');
const telState = document.getElementById('tel-state');

const diagAccel = document.getElementById('diag-accel');
const diagGrav = document.getElementById('diag-grav');
const diagGyro = document.getElementById('diag-gyro');
const diagFwdAcc = document.getElementById('diag-fwd-acc');
const diagAiSpeed = document.getElementById('diag-ai-speed');
const diagPhysSpeed = document.getElementById('diag-phys-speed');
const diagFusedSpeed = document.getElementById('diag-fused-speed');
const diagGnssSpeed = document.getElementById('diag-gnss-speed');
const diagDrHeading = document.getElementById('diag-dr-heading');
const diagGnssCourse = document.getElementById('diag-gnss-course');
const diagDrift = document.getElementById('diag-drift');
const diagMode = document.getElementById('diag-mode');

// Camera Follow State
let autoCameraFollow = true;
map.on('dragstart', () => { autoCameraFollow = false; });

function updateUIState(mode, speedKmh, headingDeg, confidence, uncertaintyM, detailText) {
  // Mode Badge
  if (mode === 'DR') {
    navModeBadge.className = 'nav-mode-badge dr-active';
    navModeText.textContent = 'S.A.F.A.R. IDR';
    sheetStatus.textContent = 'GNSS-Denied';
    telSpeedSource.textContent = 'AI + Physics Fusion';
    telState.textContent = 'GNSS_DENIED (IDR)';
  } else if (mode === 'RECOVERING') {
    navModeBadge.className = 'nav-mode-badge dr-active';
    navModeText.textContent = 'RECOVERING';
    sheetStatus.textContent = 'Reacquiring GNSS';
    telSpeedSource.textContent = 'Smooth Blending';
    telState.textContent = 'RECOVERING';
  } else {
    navModeBadge.className = 'nav-mode-badge gnss-active';
    navModeText.textContent = 'GNSS ACTIVE';
    sheetStatus.textContent = 'GNSS Locked';
    telSpeedSource.textContent = 'GNSS Filter';
    telState.textContent = 'GNSS_LOCKED';
  }

  // Speed HUD
  const displaySpeed = speedKmh != null && !isNaN(speedKmh) ? Math.round(speedKmh) : 0;
  hudSpeed.textContent = displaySpeed;

  // Confidence
  const confVal = confidence != null ? Math.round(confidence) : 95;
  hudConfidencePct.textContent = `${confVal}%`;
  hudConfidenceFill.style.width = `${confVal}%`;

  // Telemetry details
  const headingVal = headingDeg != null ? `${headingDeg.toFixed(1)}°` : '0.0°';
  const uncertVal = uncertaintyM != null ? `± ${uncertaintyM.toFixed(1)} m` : '± 0.5 m';
  
  sheetHeading.textContent = headingVal;
  sheetUncertainty.textContent = uncertVal;
  telHeading.textContent = headingVal;
  telUncertainty.textContent = uncertVal;

  // Diagnostics modal updates
  diagFusedSpeed.textContent = `${displaySpeed} km/h`;
  diagDrHeading.textContent = headingVal;
  diagMode.textContent = mode || 'GNSS';
  diagDrift.textContent = detailText || '0.0 m';
}

function haversineM(lat1, lon1, lat2, lon2) {
  const R = 6371000, toRad = Math.PI / 180;
  const dLat = (lat2 - lat1) * toRad, dLon = (lon2 - lon1) * toRad;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * toRad) * Math.cos(lat2 * toRad) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

// --- Bottom Sheet & Modal Controls ---
const bottomSheet = document.getElementById('bottom-sheet');
const sheetHandle = document.getElementById('sheet-handle');
sheetHandle.onclick = () => {
  bottomSheet.classList.toggle('collapsed');
};

const diagModal = document.getElementById('diag-modal');
document.getElementById('btn-diag-toggle').onclick = () => {
  diagModal.classList.remove('hidden');
};
document.getElementById('btn-close-diag').onclick = () => {
  diagModal.classList.add('hidden');
};

// Mode Switcher (Live vs Replay)
const tabLive = document.getElementById('tab-live');
const tabReplay = document.getElementById('tab-replay');
const panelLive = document.getElementById('panel-live');
const panelReplay = document.getElementById('panel-replay');
document.getElementById('btn-tab-toggle').onclick = () => {
  if (panelLive.classList.contains('hidden')) {
    switchTab('live');
  } else {
    switchTab('replay');
  }
};

tabLive.onclick = () => switchTab('live');
tabReplay.onclick = () => switchTab('replay');

function switchTab(which) {
  tabLive.classList.toggle('active', which === 'live');
  tabReplay.classList.toggle('active', which === 'replay');
  panelLive.classList.toggle('hidden', which !== 'live');
  panelReplay.classList.toggle('hidden', which !== 'replay');

  if (which === 'replay') {
    liveMarker.remove(); destMarker.remove(); routeLine.setLatLngs([]); liveTrail.remove();
    gtLine.addTo(map); naiveLine.addTo(map); aiLine.addTo(map);
  } else {
    gtLine.remove(); naiveLine.remove(); aiLine.remove();
    gtLine.setLatLngs([]); naiveLine.setLatLngs([]); aiLine.setLatLngs([]);
  }
}
switchTab('live');

// --- Replay Mode ---
let replayTimer = null;
let replaySessionId = null;

async function loadTrips() {
  try {
    const res = await fetch('/api/replay/trips');
    const trips = await res.json();
    const sel = document.getElementById('trip-select');
    sel.innerHTML = trips.map(t => `<option value="${t.id}">${t.label}</option>`).join('');
  } catch (e) {
    console.warn('Failed to load trips:', e);
  }
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
  document.getElementById('btn-pause-replay').innerHTML = '<i class="fa-solid fa-pause"></i> Pause';
  autoCameraFollow = true;
  runReplayLoop();
};

document.getElementById('btn-pause-replay').onclick = () => {
  if (replayTimer) {
    clearInterval(replayTimer);
    replayTimer = null;
    document.getElementById('btn-pause-replay').innerHTML = '<i class="fa-solid fa-play"></i> Resume';
  } else {
    runReplayLoop();
    document.getElementById('btn-pause-replay').innerHTML = '<i class="fa-solid fa-pause"></i> Pause';
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
      if (autoCameraFollow) map.panTo([d.lat, d.lon]);
      updateUIState('GNSS', 0, 0, 95, 0.5, 'warmup');
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
      
      updateUIState('DR', d.speed_kmh, 0, 85, drift * 0.1, `drift ${drift.toFixed(0)}m`);
      if (autoCameraFollow) map.panTo([d.ai_fused.lat, d.ai_fused.lon]);
    } else {
      aiMarker.setLatLng([d.lat, d.lon]).addTo(map);
      if (autoCameraFollow) map.panTo([d.lat, d.lon]);
      updateUIState('GNSS', 0, 0, 95, 0.5, 'GNSS demo');
    }
  }, 100);
}

// --- Live Sensors Mode ---
let liveSessionId = null;
let liveTimer = null;
let tracking = false;

let latestAccel = null;
let latestGravityEst = { x: 0, y: 0, z: 9.81 };
let latestGyro = { yaw: 0, pitch: 0, roll: 0 };
const GRAVITY_ALPHA = 0.85;

let latestGps = null;
let lastFixTime = 0;
let haveEverFixed = false;
const MAX_ACCEPTABLE_ACCURACY_M = 60;
const STALE_AFTER_MS = 4000;

function onDeviceMotion(e) {
  const a = e.accelerationIncludingGravity;
  if (!a || a.x == null) return;
  latestAccel = { x: a.x, y: a.y, z: a.z };
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

  // Diagnostics
  if (diagAccel) {
    const accMag = Math.sqrt(a.x**2 + a.y**2 + a.z**2);
    diagAccel.textContent = `${accMag.toFixed(2)} m/s²`;
    diagGrav.textContent = `${Math.sqrt(latestGravityEst.x**2 + latestGravityEst.y**2 + latestGravityEst.z**2).toFixed(2)} m/s²`;
    diagGyro.textContent = `${latestGyro.yaw.toFixed(2)} rad/s`;
  }
}

function onGeoSuccess(pos) {
  const acc = pos.coords.accuracy;
  if (acc != null && acc > MAX_ACCEPTABLE_ACCURACY_M) return;
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
}

document.getElementById('btn-recenter').onclick = () => {
  autoCameraFollow = true;
  if (latestGps) map.setView([latestGps.lat, latestGps.lon], 17);
};

document.getElementById('btn-start-live').onclick = async () => {
  if (tracking) return;

  if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
    try {
      const perm = await DeviceMotionEvent.requestPermission();
      if (perm !== 'granted') { alert('Motion sensor permission denied.'); return; }
    } catch (err) { alert('Could not request motion permission: ' + err); return; }
  }
  window.addEventListener('devicemotion', onDeviceMotion);

  if (!navigator.geolocation) {
    alert('Browser does not support geolocation.');
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
  document.getElementById('btn-start-live').innerHTML = '<i class="fa-solid fa-signal"></i> Tracking Active';

  liveTimer = setInterval(async () => {
    if (!latestAccel) return;

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
      updateUIState('NO_FIX', 0, 0, 0, 99, 'no fix yet');
      return;
    }

    liveMarker.setLatLng([d.lat, d.lon]).addTo(map);
    liveTrail.addLatLng([d.lat, d.lon]);
    if (autoCameraFollow) map.panTo([d.lat, d.lon]);

    updateUIState(d.mode, d.speed_kmh, d.heading_deg, d.confidence, d.pos_uncertainty_m, d.mode);
  }, 100);
};

// --- Destination Geocoding & Routing ---
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
    alert('Could not search for location: ' + e.message);
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
    alert('Waiting for your location fix first.');
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
    alert('No route found.');
    return;
  }
  const coords = data.routes[0].geometry.coordinates.map(([lon, lat]) => [lat, lon]);
  routeLine.setLatLngs(coords).addTo(map);
  map.fitBounds(routeLine.getBounds(), { padding: [40, 40] });
}