import { useEffect, useRef } from "react";
import * as faceapi from "face-api.js";
import "./App.css";

/* ======================== CONFIG ======================== */
// Base path (works for Netlify subpaths)
const BASE = (import.meta?.env?.BASE_URL) ? import.meta.env.BASE_URL : "/";

// Camera
const CAM_W = 640, CAM_H = 480, CAM_FPS = 30;

// Distance model (tune FOCAL_PX per camera if needed)
const FACE_WIDTH_M = 0.15;
const FOCAL_PX     = 500;
const GREEN_M      = 0.8; // <= 0.8 m = green; otherwise red

// Detection cadence
const SCORE_TH     = 0.3;
const INPUT_STEPS  = [320, 416, 512]; // try bigger only after consecutive misses
const TICK_MS      = 120;              // compute interval (~8 fps)
const HEAVY_EVERY  = 3;                // heavy pass (age/gender/desc) every Nth tick

// Tracking & identity
const MATCH_THRESHOLD_PX = 60;
const TRACK_TIMEOUT_MS   = 1200;

// Ephemeral (RAM-only) recognition
const ENABLE_EPHEMERAL_RECOG     = true;
const MAX_GALLERY                = 50;
const DESCRIPTOR_MATCH_THRESHOLD = 0.55;
const GALLERY_TTL_MS             = 12 * 60 * 60 * 1000; // 12h

// Known faces via manifest
const ENABLE_KNOWN_RECOG  = true;
const LABELS_MANIFEST_URL = `${BASE}labels/labels.json`;
const KNOWN_MATCH_THRESHOLD = 0.52;

// n8n webhook (through Netlify proxy → same origin)
const N8N_WEBHOOK_URL = "/api/n8n/camera";
/* ======================================================== */

const center = (b) => ({ cx: b.x + b.width / 2, cy: b.y + b.height / 2 });
const euclid = (a, b) => Math.hypot(a.cx - b.cx, a.cy - b.cy);
const topExpression = (exp = {}) =>
  Object.entries(exp)
    .map(([expression, probability]) => ({ expression, probability }))
    .sort((a, b) => b.probability - a.probability)[0] || { expression: "neutral", probability: 0 };
const classifyAgeGroup = (age) => (age >= 18 ? "adult" : "young");

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

export default function App() {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);

  const loopTimerRef = useRef(null);
  const tickIdxRef   = useRef(0);

  const tracksRef = useRef([]); // stable boxes per frame
  const nextIdRef = useRef(1);

  const lastGoodInputRef = useRef(INPUT_STEPS[0]);
  const emptyStreakRef   = useRef(0);

  const faceMatcherRef   = useRef(null); // known faces
  const galleryRef       = useRef([]);   // ephemeral
  const nextLabelRef     = useRef(1);

  const lastPayloadHashRef = useRef("");

  useEffect(() => {
    (async () => {
      try {
        // 1) start camera
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

        // 2) load models (only what we need)
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(`${BASE}models`),
          faceapi.nets.faceLandmark68Net.loadFromUri(`${BASE}models`),
          faceapi.nets.faceExpressionNet.loadFromUri(`${BASE}models`),
          faceapi.nets.ageGenderNet.loadFromUri(`${BASE}models`),
          (ENABLE_EPHEMERAL_RECOG || ENABLE_KNOWN_RECOG)
            ? faceapi.nets.faceRecognitionNet.loadFromUri(`${BASE}models`)
            : Promise.resolve(),
        ]);

        // 3) backend: try WebGL, fallback CPU; allow ?forceCpu=1
        const FORCE_CPU = /forceCpu=1/.test(window.location.search);
        try {
          if (FORCE_CPU) throw new Error("Forced CPU via ?forceCpu=1");
          await faceapi.tf.setBackend("webgl");
          await faceapi.tf.ready();
          console.log("Using WebGL");
        } catch (e) {
          console.log("WebGL unavailable → CPU:", e?.message || e);
          await faceapi.tf.setBackend("cpu");
          await faceapi.tf.ready();
        }

        // 4) known faces (safe loader; may return null)
        if (ENABLE_KNOWN_RECOG) faceMatcherRef.current = await loadKnownMatcherSafe();

        // 5) start loop
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

  async function loadKnownMatcherSafe() {
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
            const url = `${BASE}labels/${name}/${i}.jpg`;
            const img = await faceapi.fetchImage(url);
            const det = await faceapi
              .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: SCORE_TH }))
              .withFaceLandmarks()
              .withFaceDescriptor();
            if (det?.descriptor) ds.push(det.descriptor);
            else console.warn(`[labels] No face in ${url} — skipped`);
          } catch (e) {
            console.warn(`[labels] Failed ${name}/${i}.jpg — skipped`, e?.message || e);
          }
        }
        if (ds.length) labeled.push(new faceapi.LabeledFaceDescriptors(name, ds));
        else console.warn(`[labels] "${name}" had 0 usable images — omitted`);
      }
      if (!labeled.length) {
        console.warn("[labels] No valid labeled faces — running without known matching.");
        return null;
      }
      return new faceapi.FaceMatcher(labeled, KNOWN_MATCH_THRESHOLD);
    } catch (e) {
      console.warn("[labels] manifest error — skip known faces:", e?.message || e);
      return null;
    }
  }

  function startLoop() {
    if (loopTimerRef.current) clearInterval(loopTimerRef.current);
    loopTimerRef.current = setInterval(tick, TICK_MS);
  }

  async function tick() {
    try {
      const heavy = (tickIdxRef.current++ % HEAVY_EVERY) === 0;
      const detections = await detectAdaptive(heavy);
      const people = updateTracks(detections, heavy);
      drawOverlay(people);
      drawTable(people);
      sendToN8N_Minimal(people); // sendBeacon (non-blocking)
      setOK();
    } catch (e) {
      console.error("[tick] error:", e);
      setERR();
    }
  }

  async function detectAdaptive(heavy) {
    const sizes = (emptyStreakRef.current >= 3) ? INPUT_STEPS : [INPUT_STEPS[0]];
    for (const sz of sizes) {
      const r = await detectOnce(sz, heavy);
      if (r.length) {
        emptyStreakRef.current = 0;
        lastGoodInputRef.current = sz;
        return r;
      }
    }
    emptyStreakRef.current++;
    return [];
  }

  async function detectOnce(inputSize, heavy) {
    const v = videoRef.current; if (!v) return [];
    const opts = new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold: SCORE_TH });
    const base = faceapi.detectAllFaces(v, opts).withFaceLandmarks();

    let det;
    if (heavy) {
      const chain = base.withFaceExpressions().withAgeAndGender();
      det = (ENABLE_EPHEMERAL_RECOG || ENABLE_KNOWN_RECOG) ? await chain.withFaceDescriptors() : await chain;
    } else {
      det = await base; // light pass
    }

    const displaySize = prepCanvasForDisplay(v, canvasRef.current);
    const resized = faceapi.resizeResults(det, displaySize);

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

  function purgeStaleFromGallery(now = Date.now()) {
    galleryRef.current = galleryRef.current.filter(e => now - e.lastSeen < GALLERY_TTL_MS);
    while (galleryRef.current.length > MAX_GALLERY) {
      let oldest = 0;
      for (let i = 1; i < galleryRef.current.length; i++) {
        if (galleryRef.current[i].lastSeen < galleryRef.current[oldest].lastSeen) oldest = i;
      }
      galleryRef.current.splice(oldest, 1);
    }
  }
  function upsertEphemeralIdentity(descriptor) {
    const now = Date.now();
    purgeStaleFromGallery(now);
    // find best
    let bestIdx = -1, bestDist = Infinity;
    for (let i = 0; i < galleryRef.current.length; i++) {
      const d = faceapi.euclideanDistance(galleryRef.current[i].descriptor, descriptor);
      if (d < bestDist) { bestDist = d; bestIdx = i; }
    }
    if (bestIdx >= 0 && bestDist <= DESCRIPTOR_MATCH_THRESHOLD) {
      galleryRef.current[bestIdx].lastSeen = now;
      return galleryRef.current[bestIdx].label;
    }
    // new
    const label = `G${nextLabelRef.current++}`;
    galleryRef.current.push({ label, descriptor, lastSeen: now, createdAt: now });
    purgeStaleFromGallery(now);
    return label;
  }

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
      const who  = p._knownName ? p._knownName : (p._ephemeralLabel || "P?");
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

  // --- sendBeacon (preferred) with JSON body; fetch keepalive fallback ---
  function sendToN8N_Minimal(people) {
    if (!N8N_WEBHOOK_URL) return;

    const payload = {
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
        return {
          gender: p.gender === "female" ? "female" : "male",
          ageGroup: classifyAgeGroup(Math.round(p.age ?? 0)),
          zone: distanceM <= GREEN_M ? "green" : "red",
          name: p._knownName || null
        };
      })
    };

    const h = JSON.stringify(payload);
    if (h === lastPayloadHashRef.current) return;
    lastPayloadHashRef.current = h;

    try {
      if (navigator.sendBeacon) {
        const blob = new Blob([h], { type: "application/json" });
        navigator.sendBeacon(N8N_WEBHOOK_URL, blob);
        return;
      }
      fetch(N8N_WEBHOOK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: h,
        keepalive: true
      }).catch(() => {});
    } catch { /* ignore */ }
  }

  return (
    <div style={{ padding: 12 }}>
      <h3>Face Tracker</h3>

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
          style={{ position: "absolute", left: 0, top: 0, width: `${CAM_W}px`, height: `${CAM_H}px`, pointerEvents: "none" }}
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