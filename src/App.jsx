import { useEffect, useRef } from "react";
import * as faceapi from "face-api.js";
import "./App.css";

/* ======================== CONFIG ======================== */
// Distance model
const FACE_WIDTH_M = 0.15;                 // average face width (m)
const FOCAL_PX     = 500;                  // tune per camera if needed
const GREEN_M      = 0.8;                  // <= 0.8 m = GREEN (greet), > 0.8 m = RED

// Detector sensitivity & far-face support
const SCORE_TH     = 0.3;                  // lower = more sensitive
const INPUT_STEPS  = [320, 416, 512, 608]; // adaptive sizes for small/far faces

// Tracking (frame-to-frame for stable overlay)
const MATCH_THRESHOLD_PX = 60;             // proximity to match same person
const TRACK_TIMEOUT_MS   = 1500;           // drop if unseen this long

// Ephemeral recognition (RAM-only, non-persistent)
const ENABLE_EPHEMERAL_RECOG       = true;
const MAX_GALLERY                  = 50;
const DESCRIPTOR_MATCH_THRESHOLD   = 0.55; // for ephemeral matching
const GALLERY_TTL_MS               = 24 * 60 * 60 * 1000; // 24h TTL

// Known-face recognition (files in /public/labels/<Name>/<1..N>.jpg)
const ENABLE_KNOWN_RECOG           = true;
const KNOWN_LABELS                 = ["Jokowi", "Raisa"]; // <-- put your folder names here
const KNOWN_IMAGES_PER_LABEL       = 4;  // how many images per label to load
const KNOWN_MATCH_THRESHOLD        = 0.52; // stricter than ephemeral

// n8n webhook (minimal payload only)
const N8N_WEBHOOK_URL = "https://n8n.srv954455.hstgr.cloud/webhook-test/camera";
/* ======================================================== */

/* ----------------------- helpers ------------------------ */
const center = (b) => ({ cx: b.x + b.width / 2, cy: b.y + b.height / 2 });
const euclid = (a, b) => Math.hypot(a.cx - b.cx, a.cy - b.cy);
const topExpression = (exp = {}) =>
  Object.entries(exp)
    .map(([expression, probability]) => ({ expression, probability }))
    .sort((a, b) => b.probability - a.probability)[0] || { expression: "neutral", probability: 0 };

function classifyAgeGroup(ageNum) {
  return ageNum >= 18 ? "adult" : "young";
}

// DPR-aware canvas sizing aligned to the video’s displayed size
function prepCanvasForDisplay(video, canvas) {
  const width  = video.clientWidth  || video.videoWidth  || 640;
  const height = video.clientHeight || video.videoHeight || 480;
  const dpr = window.devicePixelRatio || 1;
  canvas.width  = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  canvas.style.width  = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { width, height };
}

/* ========================== APP ========================= */
export default function App() {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const rafRef    = useRef(null);

  // frame-to-frame tracker (for stable boxes)
  const tracksRef = useRef([]); // [{id, detection, age, gender, expressions, cx, cy, lastSeen, _ephemeralLabel?, _knownName?}]
  const nextIdRef = useRef(1);

  // ephemeral recognition gallery (RAM-only)
  const galleryRef     = useRef([]); // [{ label, descriptor: Float32Array, lastSeen, createdAt }]
  const nextLabelRef   = useRef(1);

  // known recognition matcher
  const faceMatcherRef = useRef(null);

  // CPU backend flag
  const cpuSetRef      = useRef(false);

  // far-face adaptive memory
  const lastGoodInputRef = useRef(INPUT_STEPS[0]);

  // webhook change detection
  const lastPayloadHashRef = useRef("");

  /* -------------- lifecycle: init & cleanup -------------- */
  useEffect(() => {
    (async () => {
      try {
        // 1) start camera
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        const v = videoRef.current;
        v.srcObject = stream;
        await new Promise(res => { if (v.readyState >= 2) res(); else v.onloadedmetadata = () => res(); });
        await v.play().catch(() => {});

        // 2) load models
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
          faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
          faceapi.nets.faceExpressionNet.loadFromUri("/models"),
          faceapi.nets.ageGenderNet.loadFromUri("/models"),
          (ENABLE_EPHEMERAL_RECOG || ENABLE_KNOWN_RECOG)
            ? faceapi.nets.faceRecognitionNet.loadFromUri("/models")
            : Promise.resolve(),
        ]);

        // 3) CPU backend (CSP-safe)
        if (!cpuSetRef.current) {
          await faceapi.tf.setBackend("cpu");
          await faceapi.tf.ready();
          cpuSetRef.current = true;
        }

        // 4) load known faces (optional)
        if (ENABLE_KNOWN_RECOG && KNOWN_LABELS.length > 0) {
          faceMatcherRef.current = await buildKnownFaceMatcher(KNOWN_LABELS, KNOWN_IMAGES_PER_LABEL);
        }

        // 5) start loop
        startLoop();
      } catch (e) {
        console.error("[init] failed:", e);
      }
    })();

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      const s = videoRef.current?.srcObject;
      if (s) s.getTracks().forEach(t => t.stop());
    };
  }, []);

  /* -------- load labeled faces and create a FaceMatcher -------- */
  async function buildKnownFaceMatcher(labels, imagesPerLabel) {
    const descriptors = [];
    for (const label of labels) {
      const ds = [];
      for (let i = 1; i <= imagesPerLabel; i++) {
        try {
          const img = await faceapi.fetchImage(`/labels/${label}/${i}.jpg`);
          const det = await faceapi
            .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: SCORE_TH }))
            .withFaceLandmarks()
            .withFaceDescriptor();
          if (det && det.descriptor) ds.push(det.descriptor);
        } catch (e) {
          // ignore missing images
        }
      }
      if (ds.length) descriptors.push(new faceapi.LabeledFaceDescriptors(label, ds));
    }
    return new faceapi.FaceMatcher(descriptors, KNOWN_MATCH_THRESHOLD);
  }

  /* ------------------------- main loop ------------------------- */
  function startLoop() {
    const tick = async () => {
      const detections = await detectAdaptive();
      const people = updateTracksAndLabels(detections); // stable boxes + names/ephemeral labels
      drawOverlay(people);
      drawTable(people);
      await sendToN8N_Minimal(people);
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }

  /* ------------------- detection (adaptive) ------------------- */
  async function detectOnce(inputSize) {
    const v = videoRef.current;
    if (!v) return [];
    const opts = new faceapi.TinyFaceDetectorOptions({
      inputSize,
      scoreThreshold: SCORE_TH,
    });

    // chain models (add descriptors for recognition)
    const chain = faceapi
      .detectAllFaces(v, opts)
      .withFaceLandmarks()
      .withFaceExpressions()
      .withAgeAndGender();

    const det = (ENABLE_EPHEMERAL_RECOG || ENABLE_KNOWN_RECOG) ? await chain.withFaceDescriptors() : await chain;

    const displaySize = prepCanvasForDisplay(v, canvasRef.current);
    const resized = faceapi.resizeResults(det, displaySize);

    // attach known names / ephemeral labels
    resized.forEach(d => {
      // known recognition first
      let name = null;
      if (ENABLE_KNOWN_RECOG && faceMatcherRef.current && d.descriptor) {
        const best = faceMatcherRef.current.findBestMatch(d.descriptor);
        if (best && best.label && best.label !== "unknown" && best.distance <= KNOWN_MATCH_THRESHOLD) {
          name = best.label;
        }
      }
      d._knownName = name;

      // ephemeral RAM-only label if not known
      if (!name && ENABLE_EPHEMERAL_RECOG && d.descriptor) {
        d._ephemeralLabel = upsertEphemeralIdentity(d.descriptor);
      }
    });

    return resized;
  }

  // try last good size first; if no faces, try larger sizes to catch far/small faces
  async function detectAdaptive() {
    const sizes = [lastGoodInputRef.current, ...INPUT_STEPS.filter(s => s !== lastGoodInputRef.current)];
    for (const sz of sizes) {
      const result = await detectOnce(sz);
      if (result.length > 0) {
        lastGoodInputRef.current = sz;
        return result;
      }
    }
    return [];
  }

  /* ------------------ ephemeral recognition ------------------ */
  function purgeStaleFromGallery(now = Date.now()) {
    // TTL
    galleryRef.current = galleryRef.current.filter(e => now - e.lastSeen < GALLERY_TTL_MS);
    // LRU capacity
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
      return galleryRef.current[best.idx].label; // e.g., "G12"
    }
    const label = `G${nextLabelRef.current++}`;
    galleryRef.current.push({
      label,
      descriptor,
      lastSeen: now,
      createdAt: now
    });
    purgeStaleFromGallery(now);
    return label;
  }

  /* --------------- frame-to-frame tracking ------------------- */
  function updateTracksAndLabels(resized) {
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
        p.detection   = d.detection;
        p.age         = d.age;
        p.gender      = d.gender;
        p.expressions = d.expressions;
        p.cx = ctr.cx; p.cy = ctr.cy;
        p.lastSeen    = now;
        // names/labels
        p._knownName      = d._knownName ?? p._knownName ?? null;
        p._ephemeralLabel = (!p._knownName) ? (d._ephemeralLabel ?? p._ephemeralLabel ?? null) : p._ephemeralLabel;
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

    // drop stale tracks
    const kept = next.filter(p => now - p.lastSeen < TRACK_TIMEOUT_MS);
    // sort by creation id (stable row order)
    kept.sort((a, b) => a.id - b.id);

    tracksRef.current = kept;
    return kept;
  }

  /* --------------------- drawing overlay --------------------- */
  function drawOverlay(people) {
    const v  = videoRef.current;
    const displayW = v?.clientWidth  || 640;
    const displayH = v?.clientHeight || 480;

    const c   = canvasRef.current;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, displayW, displayH);

    people.forEach(p => {
      const b = p.detection.box;
      const distanceM = (FOCAL_PX * FACE_WIDTH_M) / b.width;
      const zone = distanceM <= GREEN_M ? "green" : "red";
      const expr = topExpression(p.expressions);

      const who =
        p._knownName ? p._knownName :
        (p._ephemeralLabel ? p._ephemeralLabel : "P?");
      const label = `${who} • ${zone} • ${Math.round(p.age)} ${p.gender} • ${expr.expression}`;

      const drawBox = new faceapi.draw.DrawBox(b, {
        label,
        boxColor: zone === "green" ? "#17c964" : "#f31260",
        lineWidth: 2,
      });
      drawBox.draw(c);
    });

    const el = document.getElementById("peopleCount");
    if (el) el.innerText = `People on screen: ${people.length}`;
  }

  /* --------------------- live table output -------------------- */
  function drawTable(people) {
    const body = document.getElementById("dataBodyindex");
    if (!body) return;
    body.innerHTML = "";

    people.forEach(p => {
      const b = p.detection.box;
      const distanceM = (FOCAL_PX * FACE_WIDTH_M) / b.width;
      const zone = distanceM <= GREEN_M ? "green" : "red";
      const expr = topExpression(p.expressions);
      const who =
        p._knownName ? p._knownName :
        (p._ephemeralLabel ? p._ephemeralLabel : "—");

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${who}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${p.gender}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${Math.round(p.age)}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${expr.expression}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${distanceM.toFixed(2)}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${zone}</td>
      `;
      body.appendChild(tr);
    });
  }

  /* ------------------- webhook to n8n (minimal) ------------------- */
  function simpleHash(obj) {
    try { return JSON.stringify(obj); } catch { return String(Math.random()); }
  }

  async function sendToN8N_Minimal(people) {
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
          ageGroup: classifyAgeGroup(Math.round(p.age)), // "adult" | "young"
          zone,                                         // "green" | "red"
          // emotion: topExpression(p.expressions).expression, // keep if you want
          // name: p._knownName || undefined, // you said you don't need names in webhook; uncomment if you do
        };
      })
    };

    const h = simpleHash(minimal);
    if (h === lastPayloadHashRef.current) return; // only send on change
    lastPayloadHashRef.current = h;

    try {
      await fetch(N8N_WEBHOOK_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(minimal)
      });
    } catch (e) {
      console.warn("n8n webhook failed:", e);
    }
  }

  /* ---------------------------- UI ---------------------------- */
  return (
    <div style={{ padding: 12 }}>
      <h3>Face Tracker (Table + Known Names)</h3>

      <div style={{ position: "relative", width: 940, height: 650 }}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{ display: "block", width: "940px", height: "650px" }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute", left: 0, top: 0,
            width: "940px", height: "650px", pointerEvents: "none"
          }}
        />
      </div>

      <div id="peopleCount" style={{ marginTop: 8 }}>People on screen: 0</div>

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