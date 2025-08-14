import { useEffect, useRef } from "react";
import * as faceapi from "face-api.js";
import "./App.css";

/* ======================== CONFIG (tune) ======================== */
// Asset base (works on Netlify sub-paths)
const BASE = (import.meta?.env?.BASE_URL) ? import.meta.env.BASE_URL : "/";

// Camera
const CAM_W = 640;
const CAM_H = 480;
const CAM_FPS = 30;

// Distance model
const FACE_WIDTH_M = 0.15;   // avg face width (m)
const FOCAL_PX     = 500;    // tune per camera if needed
const GREEN_M      = 0.8;    // <= 0.8 m = GREEN, else RED

// Detector & scheduler
const SCORE_TH     = 0.3;                 // lower = more sensitive
const INPUT_STEPS  = [320, 416, 512];     // adaptive, trimmed for speed
const TICK_MS      = 120;                 // ~8 FPS compute
const HEAVY_EVERY  = 3;                   // run age/gender/descriptors every Nth tick

// Tracking (frame-to-frame box stability)
const MATCH_THRESHOLD_PX = 60;
const TRACK_TIMEOUT_MS   = 1200;

// Ephemeral recognition (RAM-only, non-persistent)
const ENABLE_EPHEMERAL_RECOG       = true;
const MAX_GALLERY                  = 50;
const DESCRIPTOR_MATCH_THRESHOLD   = 0.55;
const GALLERY_TTL_MS               = 12 * 60 * 60 * 1000; // 12h

// Known-face recognition (labels.json + folders)
const ENABLE_KNOWN_RECOG  = true;
const LABELS_MANIFEST_URL = `${BASE}labels/labels.json`;
const KNOWN_MATCH_THRESHOLD = 0.52;

// Webhook (minimal payload)
const N8N_WEBHOOK_URL = "https://n8n.srv954455.hstgr.cloud/webhook-test/camera";

/* ======================== helpers ======================== */
const center = (b) => ({ cx: b.x + b.width / 2, cy: b.y + b.height / 2 });
const euclid = (a, b) => Math.hypot(a.cx - b.cx, a.cy - b.cy);
const topExpression = (exp = {}) =>
  Object.entries(exp)
    .map(([expression, probability]) => ({ expression, probability }))
    .sort((a, b) => b.probability - a.probability)[0] || { expression: "neutral", probability: 0 };

function classifyAgeGroup(ageNum) { return ageNum >= 18 ? "adult" : "young"; }

function shrinkBox(box, ratio = 0.9) {
  const nw = box.width  * ratio;
  const nh = box.height * ratio;
  const nx = box.x + (box.width  - nw) / 2;
  const ny = box.y + (box.height - nh) / 2;
  return new faceapi.Rect(nx, ny, nw, nh);
}

function prepCanvasForDisplay(video, canvas) {
  const width  = video.clientWidth  || video.videoWidth  || CAM_W;
  const height = video.clientHeight || video.videoHeight || CAM_H;
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width  = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width, height };
}

function setStatus(text, color) {
  const el = document.getElementById("statusLight");
  if (!el) return;
  el.textContent = text;
  el.style.color = color;
}
const setOK = () => setStatus("OK", "#17c964");
const setERR = () => setStatus("ERR", "#f31260");

/* ========================== APP ========================= */
export default function App() {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);

  // timers
  const loopTimerRef = useRef(null);
  const tickIdxRef   = useRef(0);

  // tracks for overlay
  const tracksRef = useRef([]); // [{id, detection, age, gender, expressions, cx, cy, lastSeen, _knownName?, _ephemeralLabel?}]
  const nextIdRef = useRef(1);

  // adaptive state
  const lastGoodInputRef = useRef(INPUT_STEPS[0]);
  const emptyStreakRef   = useRef(0);

  // recognition
  const faceMatcherRef   = useRef(null); // known
  const galleryRef       = useRef([]);   // ephemeral
  const nextLabelRef     = useRef(1);

  // webhook dedupe
  const lastPayloadHashRef = useRef("");

  useEffect(() => {
    (async () => {
      try {
        // 1) Camera
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width:  { ideal: CAM_W, max: CAM_W },
            height: { ideal: CAM_H, max: CAM_H },
            frameRate: { ideal: CAM_FPS }
          },
          audio: false
        });
        const v = videoRef.current;
        v.srcObject = stream;
        await new Promise(res => { if (v.readyState >= 2) res(); else v.onloadedmetadata = () => res(); });
        await v.play().catch(() => {});

        // 2) Models
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(`${BASE}models`),
          faceapi.nets.faceLandmark68Net.loadFromUri(`${BASE}models`),
          faceapi.nets.faceExpressionNet.loadFromUri(`${BASE}models`),
          faceapi.nets.ageGenderNet.loadFromUri(`${BASE}models`),
          (ENABLE_EPHEMERAL_RECOG || ENABLE_KNOWN_RECOG)
            ? faceapi.nets.faceRecognitionNet.loadFromUri(`${BASE}models`)
            : Promise.resolve(),
        ]);

        // 3) Backend: try WebGL, fallback CPU
        try {
          await faceapi.tf.setBackend("webgl");
          await faceapi.tf.ready();
          console.log("Using WebGL");
        } catch (e) {
          console.log("WebGL unavailable, using CPU:", e?.message || e);
          await faceapi.tf.setBackend("cpu");
          await faceapi.tf.ready();
        }

        // 4) Known-face matcher (optional)
        if (ENABLE_KNOWN_RECOG) {
          faceMatcherRef.current = await loadKnownMatcher();
        }

        // 5) Start loop
        startLoop();
      } catch (e) {
        console.error("[init] failed:", e);
        setERR();
      }
    })();

    return () => {
      if (loopTimerRef.current) clearInterval(loopTimerRef.current);
      const s = videoRef.current?.srcObject;
      if (s) s.getTracks().forEach(t => t.stop());
    };
  }, []);

  /* ---------------- Known faces loader ---------------- */
  async function loadKnownMatcher() {
    try {
      const res = await fetch(LABELS_MANIFEST_URL, { cache: "no-store" });
      if (!res.ok) throw new Error("labels.json not found");
      const data = await res.json();
      const entries = Array.isArray(data.people) ? data.people : [];
      const labeled = [];

      for (const { name, images } of entries) {
        if (!name) continue;
        const count = Math.max(1, Number(images || 1));
        const ds = [];
        for (let i = 1; i <= count; i++) {
          try {
            const img = await faceapi.fetchImage(`${BASE}labels/${name}/${i}.jpg`);
            const det = await faceapi
              .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: SCORE_TH }))
              .withFaceLandmarks()
              .withFaceDescriptor();
            if (det?.descriptor) ds.push(det.descriptor);
          } catch { /* ignore */ }
        }
        if (ds.length) labeled.push(new faceapi.LabeledFaceDescriptors(name, ds));
      }
      return new faceapi.FaceMatcher(labeled, KNOWN_MATCH_THRESHOLD);
    } catch (e) {
      console.warn("[labels] skip known faces:", e?.message || e);
      return new faceapi.FaceMatcher([], KNOWN_MATCH_THRESHOLD);
    }
  }

  /* ---------------------- Main loop ---------------------- */
  function startLoop() {
    if (loopTimerRef.current) clearInterval(loopTimerRef.current);
    loopTimerRef.current = setInterval(tick, TICK_MS);
  }

  async function tick() {
    try {
      const idx = ++tickIdxRef.current;
      const heavy = (idx % HEAVY_EVERY) === 0;

      const detections = await detectAdaptive(heavy);
      const people = updateTracks(detections, heavy);
      drawOverlay(people);
      drawTable(people);
      sendToN8N_Minimal(people); // non-blocking

      setOK();
    } catch (e) {
      console.error("[tick] error:", e);
      setERR();
    }
  }

  /* -------------- Detection (adaptive + cadence) -------------- */
  async function detectAdaptive(heavy) {
    const sizes = (emptyStreakRef.current >= 3) ? INPUT_STEPS : [INPUT_STEPS[0]];
    for (const sz of sizes) {
      const result = await detectOnce(sz, heavy);
      if (result.length > 0) {
        emptyStreakRef.current = 0;
        lastGoodInputRef.current = sz;
        return result;
      }
    }
    emptyStreakRef.current++;
    return [];
  }

  async function detectOnce(inputSize, heavy) {
    const v = videoRef.current;
    if (!v) return [];
    const opts = new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold: SCORE_TH });

    const base = faceapi.detectAllFaces(v, opts).withFaceLandmarks();

    let det;
    if (heavy) {
      const chain = base.withFaceExpressions().withAgeAndGender();
      det = (ENABLE_EPHEMERAL_RECOG || ENABLE_KNOWN_RECOG) ? await chain.withFaceDescriptors() : await chain;
    } else {
      // light: just boxes + landmarks (fast)
      det = await base;
    }

    const displaySize = prepCanvasForDisplay(v, canvasRef.current);
    const resized = faceapi.resizeResults(det, displaySize);

    // attach known/ephemeral labels only on heavy frames (we have descriptors)
    if (heavy) {
      resized.forEach(d => {
        let name = null;
        if (ENABLE_KNOWN_RECOG && faceMatcherRef.current && d.descriptor) {
          const best = faceMatcherRef.current.findBestMatch(d.descriptor);
          if (best && best.label && best.label !== "unknown" && best.distance <= KNOWN_MATCH_THRESHOLD) {
            name = best.label;
          }
        }
        d._knownName = name;
        if (!name && ENABLE_EPHEMERAL_RECOG && d.descriptor) {
          d._ephemeralLabel = upsertEphemeralIdentity(d.descriptor);
        }
      });
    }
    return resized;
  }

  /* ---------------- Ephemeral gallery (RAM) ---------------- */
  function purgeStaleFromGallery(now = Date.now()) {
    galleryRef.current = galleryRef.current.filter(e => now - e.lastSeen < GALLERY_TTL_MS);
    while (galleryRef.current.length > MAX_GALLERY) {
      let oldestIdx = 0;
      for (let i = 1; i < galleryRef.current.length; i++) {
        if (galleryRef.current[i].lastSeen < galleryRef.current[oldestIdx].lastSeen) oldestIdx = i;
      }
      galleryRef.current.splice(oldestIdx, 1);
    }
  }
  function findBestGalleryMatch(descriptor) {
    let best = { idx: -1, dist: Infinity };
    const gal = galleryRef.current;
    for (let i = 0; i < gal.length; i++) {
      const d = faceapi.euclideanDistance(gal[i].descriptor, descriptor);
      if (d < best.dist) best = { idx: i, dist: d };
    }
    return best;
  }
  function upsertEphemeralIdentity(descriptor) {
    const now = Date.now();
    purgeStaleFromGallery(now);
    const best = findBestGalleryMatch(descriptor);
    if (best.idx >= 0 && best.dist <= DESCRIPTOR_MATCH_THRESHOLD) {
      galleryRef.current[best.idx].lastSeen = now;
      return galleryRef.current[best.idx].label;
    }
    const label = `G${nextLabelRef.current++}`;
    galleryRef.current.push({ label, descriptor, lastSeen: now, createdAt: now });
    purgeStaleFromGallery(now);
    return label;
  }

  /* -------------------- Tracking & fuse -------------------- */
  function updateTracks(resized, heavy) {
    const now  = Date.now();
    const prev = tracksRef.current;
    const next = [];

    resized.forEach(d => {
      const ctr = center(d.detection.box);
      let bestIdx = -1, best = Infinity;
      prev.forEach((p, i) => {
        const dd = euclid(ctr, { cx: p.cx, cy: p.cy });
        if (dd < MATCH_THRESHOLD_PX && dd < best) { best = dd; bestIdx = i; }
      });

      if (bestIdx >= 0) {
        const p = { ...prev[bestIdx] };
        p.detection = d.detection;
        p.cx = ctr.cx; p.cy = ctr.cy;
        p.lastSeen = now;

        // only overwrite attrs on heavy frames
        if (heavy) {
          if (d.age != null)        p.age = d.age;
          if (d.gender != null)     p.gender = d.gender;
          if (d.expressions != null)p.expressions = d.expressions;
          p._knownName      = d._knownName ?? p._knownName ?? null;
          p._ephemeralLabel = (!p._knownName) ? (d._ephemeralLabel ?? p._ephemeralLabel ?? null) : p._ephemeralLabel;
        }

        next.push(p);
      } else {
        next.push({
          id: nextIdRef.current++,
          detection: d.detection,
          age: d.age,
          gender: d.gender,
          expressions: d.expressions,
          cx: ctr.cx, cy: ctr.cy,
          lastSeen: now,
          _knownName: d._knownName ?? null,
          _ephemeralLabel: (!d._knownName) ? (d._ephemeralLabel ?? null) : null,
        });
      }
    });

    const kept = next.filter(p => now - p.lastSeen < TRACK_TIMEOUT_MS);
    kept.sort((a, b) => a.id - b.id);
    tracksRef.current = kept;
    return kept;
  }

  /* ---------------------- Drawing ---------------------- */
  function drawOverlay(people) {
    const v  = videoRef.current;
    const displayW = v?.clientWidth  || CAM_W;
    const displayH = v?.clientHeight || CAM_H;

    const c   = canvasRef.current;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, displayW, displayH);

    people.forEach(p => {
      const rawBox = p.detection.box;
      const box = shrinkBox(rawBox, 0.9);
      const distanceM = (FOCAL_PX * FACE_WIDTH_M) / rawBox.width;
      const zone = distanceM <= GREEN_M ? "green" : "red";
      const expr = topExpression(p.expressions || { neutral: 1 });

      const who = p._knownName ? p._knownName : (p._ephemeralLabel || "P?");
      const label = `${who} • ${zone} • ${Math.round(p.age ?? 0)} ${p.gender ?? "?"} • ${expr.expression}`;

      new faceapi.draw.DrawBox(box, {
        label,
        boxColor: zone === "green" ? "#17c964" : "#f31260",
        lineWidth: 2,
      }).draw(c);
    });

    const el = document.getElementById("peopleCount");
    if (el) el.innerText = `People on screen: ${people.length}`;
  }

  function drawTable(people) {
    const body = document.getElementById("dataBodyindex");
    if (!body) return;
    body.innerHTML = "";
    people.forEach(p => {
      const rawBox = p.detection.box;
      const distanceM = (FOCAL_PX * FACE_WIDTH_M) / rawBox.width;
      const zone = distanceM <= GREEN_M ? "green" : "red";
      const expr = topExpression(p.expressions || { neutral: 1 });
      const who  = p._knownName ? p._knownName : (p._ephemeralLabel || "—");

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${who}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${p.gender ?? "-"}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${p.age != null ? Math.round(p.age) : "-"}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${expr.expression}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${isFinite(distanceM) ? distanceM.toFixed(2) : "-"}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${zone}</td>
      `;
      body.appendChild(tr);
    });
  }

  /* ---------------------- Webhook ---------------------- */
  function sendToN8N_Minimal(people) {
    if (!N8N_WEBHOOK_URL) return;

    const minimal = {
      timestamp: new Date().toISOString(),
      peopleCount: people.length,
      anyInGreen: people.some(p => {
        const b = p.detection.box;
        const distanceM = (FOCAL_PX * FACE_WIDTH_M) / b.width;
        return distanceM <= GREEN_M;
      }),
      people: people.map(p => {
        const b = p.detection.box;
        const distanceM = (FOCAL_PX * FACE_WIDTH_M) / b.width;
        const zone = distanceM <= GREEN_M ? "green" : "red";
        return {
          gender: p.gender === "female" ? "female" : "male",
          ageGroup: classifyAgeGroup(Math.round(p.age ?? 0)),
          zone,
          name: p._knownName || null
        };
      })
    };

    const h = JSON.stringify(minimal);
    if (h === lastPayloadHashRef.current) return;
    lastPayloadHashRef.current = h;

    try {
      if (navigator.sendBeacon) {
        const blob = new Blob([JSON.stringify(minimal)], { type: "application/json" });
        navigator.sendBeacon(N8N_WEBHOOK_URL, blob);
        return;
      }
      fetch(N8N_WEBHOOK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(minimal),
        keepalive: true,
      }).catch(() => {});
    } catch { /* swallow */ }
  }

  /* ------------------------- UI ------------------------- */
  return (
    <div style={{ padding: 12 }}>
      <h3>Fast Face Tracker (Optimized)</h3>

      <div style={{ position: "relative", width: CAM_W, height: CAM_H }}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{ display: "block", width: `${CAM_W}px`, height: `${CAM_H}px` }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute", left: 0, top: 0,
            width: `${CAM_W}px`, height: `${CAM_H}px`, pointerEvents: "none"
          }}
        />
      </div>

      <div style={{ marginTop: 8 }}>
        People: <span id="peopleCount">0</span> • Status: <span id="statusLight" style={{ fontWeight: 600 }}>…</span>
      </div>

      <div style={{ maxHeight: "40vh", overflowY: "auto", marginTop: 8 }}>
        <table style={{ borderCollapse: "collapse", width: "100%", textAlign: "center" }}>
          <thead>
            <tr>
              <th style={{ border: "1px solid #fff", padding: 6 }}>Name</th>
              <th style={{ border: "1px solid #fff", padding: 6 }}>Gender</th>
              <th style={{ border: "1px solid #fff", padding: 6 }}>Age</th>
              <th style={{ border: "1px solid #fff", padding: 6 }}>Emotion</th>
              <th style={{ border: "1px solid #fff", padding: 6 }}>Distance (m)</th>
              <th style={{ border: "1px solid #fff", padding: 6 }}>Zone</th>
            </tr>
          </thead>
          <tbody id="dataBodyindex"></tbody>
        </table>
      </div>
    </div>
  );
}