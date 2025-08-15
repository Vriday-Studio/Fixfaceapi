import { useEffect, useRef, useState } from "react";
import * as faceapi from "face-api.js";

/* ──────────────────────────────────────────────────────────────────────────────
   SIMPLE CONFIG (test-only, direct to n8n)
   ────────────────────────────────────────────────────────────────────────────── */

// face-api models
const MODEL_URL = "/models";

// distance estimation
const FACE_WIDTH_M = 0.15;   // avg human face width in meters
let   FOCAL_PX     = 500;    // will be auto-tuned after camera starts
const GREEN_MAX_M  = 0.8;    // <= 0.4m => green, else red

// detector tuned for smaller/far faces
const TINY_OPTS = new faceapi.TinyFaceDetectorOptions({
  inputSize: 512,       // try 384 if you still lose far faces (slower)
  scoreThreshold: 0.3, // a bit more permissive to keep people at distance
});

// session pacing
const START_FRAMES    = 8;     // need N consecutive frames with faces to start
const END_AFTER_MS    = 8000;  // stop after 8s of no faces
const SNAPSHOT_EVERY  = 900;   // send at most once per 900ms
const DETECTOR_STEPMS = 120;   // ~8 FPS

// box appearance
const BOX_SHRINK      = 0.7;   // 0.6 tighter, 0.8 looser
const BOX_LINE_WIDTH  = 5;

// face recognition strictness + smoothing (to avoid flicker)
const MATCH_STRICT      = 0.50; // stricter than default 0.6
const MATCH_MARGIN      = 0.06; // best must beat 2nd-best by this margin
const STABILIZE_FRAMES  = 3;    // require N consecutive frames to switch
const MAX_SLOTS         = 5;    // how many left→right slots to stabilize

// FORCE TEST (simple and reliable for n8n community)
const N8N = {
  start:   "/api/n8n-test/camera/start",
  snapshot:"/api/n8n-test/camera/snapshot",
  stop:    "/api/n8n-test/camera/stop",
};

// optional debug flag: await responses and show OK/ERR
const DEBUG_FETCH = /[?&]debugPost=1\b/.test(window.location.search);

/* ──────────────────────────────────────────────────────────────────────────────
   HELPERS
   ────────────────────────────────────────────────────────────────────────────── */

const uuid = () =>
  (crypto?.randomUUID ? crypto.randomUUID()
   : Math.random().toString(36).slice(2) + Date.now().toString(36));

const estimateDistanceM = (faceBoxWidthPx) =>
  faceBoxWidthPx ? (FOCAL_PX * FACE_WIDTH_M) / faceBoxWidthPx : null;

const ageGroupOf = (age) => {
  if (age == null) return "unknown";
  const a = Math.round(age);
  if (a >= 18) return "adult";
  if (a >= 12) return "teen";
  return "child";
};

const zoneOf = (distanceM) =>
  distanceM != null && distanceM <= GREEN_MAX_M ? "green" : "red";

const topExpression = (expressions) => {
  if (!expressions) return "neutral";
  return Object.entries(expressions).reduce((a, b) => (a[1] > b[1] ? a : b))[0];
};

// make a smaller rectangle centered inside the original
function shrinkBox(b, factor = BOX_SHRINK) {
  const w = b.width * factor;
  const h = b.height * factor;
  return {
    x: b.x + (b.width - w) / 2,
    y: b.y + (b.height - h) / 2,
    width: w,
    height: h,
  };
}

// compute best & second-best label distances for a descriptor (for margin)
function bestTwoMatches(matcher, queryDesc) {
  let best = { label: null, dist: 1 };
  let second = { label: null, dist: 1 };
  for (const ld of matcher.labeledDescriptors) {
    for (const d of ld.descriptors) {
      const dist = faceapi.euclideanDistance(queryDesc, d);
      if (dist < best.dist) {
        second = best;
        best = { label: ld.label, dist };
      } else if (dist < second.dist) {
        second = { label: ld.label, dist };
      }
    }
  }
  return { best, second };
}

// stabilization slots (left→right)
const slotsRefInit = () =>
  Array.from({ length: MAX_SLOTS }, () => ({
    shown: null,       // currently displayed name
    candidate: null,   // name being considered
    streak: 0,         // consecutive frames won
  }));

/* ──────────────────────────────────────────────────────────────────────────────
   APP
   ────────────────────────────────────────────────────────────────────────────── */

export default function App() {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [backend, setBackend] = useState("cpu");

  const [sessionStatus, setSessionStatus] = useState("IDLE"); // IDLE | ACTIVE
  const [sessionId, setSessionId] = useState(null);
  const [posts, setPosts] = useState({ start: 0, snapshot: 0, stop: 0 });
  const [lastSent, setLastSent] = useState({ start: "-", snapshot: "-", stop: "-" });
  const [lastHttp, setLastHttp] = useState({ start: "", snapshot: "", stop: "" });

  const [table, setTable] = useState([]); // only GREEN faces (max 5)
  const [totals, setTotals] = useState({ all: 0, green: 0, red: 0 });

  const faceMatcherRef = useRef(null);
  const slotsRef = useRef(slotsRefInit());

  // FSM scratch
  const S = useRef({
    id: null,
    seenFrames: 0,
    lastFaceTs: 0,
    lastSnapshotTs: 0,
  });

  /* ── init: backend, models, labels, camera ──────────────────────────────── */
  useEffect(() => {
    (async () => {
      // backend fallback: webgl → wasm → cpu
      try {
        await faceapi.tf.setBackend("webgl");
        await faceapi.tf.ready();
        setBackend("webgl");
      } catch {
        try {
          await faceapi.tf.setBackend("wasm");
          await faceapi.tf.ready();
          setBackend("wasm");
        } catch {
          await faceapi.tf.setBackend("cpu");
          await faceapi.tf.ready();
          setBackend("cpu");
        }
      }

      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
        faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      ]);

      // optional name recognition via /labels/labels.json
      try {
        const res = await fetch("/labels/labels.json", { cache: "no-store" });
        if (res.ok) {
          const manifest = await res.json(); // { people:[{name,images}] }
          const labeled = [];
          for (const person of manifest.people || []) {
            const descs = [];
            for (let i = 1; i <= Number(person.images || 0); i++) {
              try {
                const img = await faceapi.fetchImage(`/labels/${person.name}/${i}.jpg`);
                const d = await faceapi
                  .detectSingleFace(img, TINY_OPTS)
                  .withFaceLandmarks()
                  .withFaceDescriptor();
                if (d?.descriptor) descs.push(d.descriptor);
              } catch {}
            }
            if (descs.length) labeled.push(new faceapi.LabeledFaceDescriptors(person.name, descs));
          }
          if (labeled.length) faceMatcherRef.current = new faceapi.FaceMatcher(labeled, MATCH_STRICT);
        }
      } catch {}

      // camera
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1920 }, height: { ideal: 1080 } }, frameRate: { ideal: 30 },
        audio: false,
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        // auto-tune focal length after metadata loads (720p ≈ 900, 1080p ≈ 1350)
        videoRef.current.onloadedmetadata = () => {
          const w = videoRef.current.videoWidth || 1280;
          FOCAL_PX = w >= 1920 ? 1350 : 900;
          // console.log("Auto focalPx =", FOCAL_PX, "for width", w);
        };
      }

      setReady(true);
    })();

    return () => {
      const s = videoRef.current?.srcObject;
      if (s) s.getTracks()?.forEach(t => t.stop());
    };
  }, []);

  /* ── detection loop ─────────────────────────────────────────────────────── */
  useEffect(() => {
    if (!ready) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const resize = () => {
      if (!video) return;
      canvas.width = video.videoWidth || 940;
      canvas.height = video.videoHeight || 650;
    };
    video.addEventListener("loadedmetadata", resize);
    resize();

    let raf = 0;
    let lastRun = 0;

    const loop = async (ts) => {
      raf = requestAnimationFrame(loop);
      if (ts - lastRun < DETECTOR_STEPMS) return;
      lastRun = ts;
      if (!video.videoWidth) return;

      const dets = await faceapi
        .detectAllFaces(video, TINY_OPTS)
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender()
        .withFaceDescriptors();

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const resized = faceapi
        .resizeResults(dets, { width: canvas.width, height: canvas.height })
        .sort((a, b) => a.detection.box.x - b.detection.box.x);

      const matcher = faceMatcherRef.current;
      const allRows = [];
      const peopleForPost = [];

      for (let i = 0; i < resized.length; i++) {
        const det = resized[i];
        const box = det.detection.box;
        const distM = estimateDistanceM(box.width);
        const zone  = zoneOf(distM);
        const color = zone === "green" ? "#22c55e" : "#ef4444";

        // strict match + margin + temporal stabilization
        let name = null;
        if (matcher && det.descriptor) {
          const { best, second } = bestTwoMatches(matcher, det.descriptor);
          const strict = zone === "red" ? (MATCH_STRICT - 0.02) : MATCH_STRICT; // a bit stricter when far
          const passes =
            best.label &&
            best.label !== "unknown" &&
            best.dist <= strict &&
            (second.dist - best.dist) >= MATCH_MARGIN;

          const proposed = passes ? best.label : null;

          const slot = slotsRef.current[i] || (slotsRef.current[i] = { shown: null, candidate: null, streak: 0 });
          if (proposed === slot.shown) {
            slot.candidate = proposed;
            slot.streak = STABILIZE_FRAMES;
          } else {
            if (proposed === slot.candidate) {
              slot.streak += 1;
            } else {
              slot.candidate = proposed;
              slot.streak = 1;
            }
            if (slot.streak >= STABILIZE_FRAMES) {
              slot.shown = slot.candidate;
            }
          }
          name = slot.shown;
        }

        // draw smaller, centered box + label
        const dbox = shrinkBox(box, BOX_SHRINK);

        ctx.strokeStyle = color;
        ctx.lineWidth = BOX_LINE_WIDTH;
        ctx.strokeRect(dbox.x, dbox.y, dbox.width, dbox.height);

        const label =
          `${zone} • ${Math.max(0, Math.round(det.age))} ${det.gender}` +
          ` • ${topExpression(det.expressions)}` +
          (name ? ` • ${name}` : "");

        ctx.font = "14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
        const padX = 6, padY = 4;
        const textW = ctx.measureText(label).width + padX * 2;
        const textH = 18 + padY * 2;
        const labelX = Math.max(0, Math.min(dbox.x, canvas.width - textW));
        const labelY = Math.max(0, dbox.y - textH - 4);

        ctx.fillStyle = color;
        ctx.fillRect(labelX, labelY, textW, textH);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, labelX + padX, labelY + 14);

        // rows for UI (we’ll filter to green later)
        allRows.push({
          idx: i + 1,
          gender: (det.gender || "").toLowerCase(),
          ageGroup: ageGroupOf(det.age),
          zone,
          name: name || "-",
          distance: distM ? distM.toFixed(2) + " m" : "-",
        });

        // payload for n8n (send ALL faces)
        peopleForPost.push({
          gender: (det.gender || "").toLowerCase(),
          ageGroup: ageGroupOf(det.age),
          zone,
          name: name || null,
        });
      }

      // totals
      const green = allRows.filter(r => r.zone === "green").length;
      const red   = allRows.filter(r => r.zone === "red").length;
      setTotals({ all: allRows.length, green, red });

      // UI table: only green faces (max 5), pad to 5 rows
      const greenRows = allRows.filter(r => r.zone === "green").slice(0, 5);
      while (greenRows.length < 5) {
        greenRows.push({
          idx: greenRows.length + 1,
          gender: "-",
          ageGroup: "-",
          zone: "-",
          name: "-",
          distance: "-",
        });
      }
      setTable(greenRows);

      // session update (ALL faces, green + red)
      await updateSession(peopleForPost);
    };

    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      video?.removeEventListener("loadedmetadata", resize);
    };
  }, [ready]);

  /* ── posting helpers (sendBeacon by default; fetch fallback; debug option) ─ */
  const post = async (which, payload) => {
    const url = N8N[which];
    const nowStr = new Date().toLocaleTimeString();

    if (!DEBUG_FETCH) {
      try {
        const ok = navigator.sendBeacon(url, new Blob([JSON.stringify(payload)], { type: "application/json" }));
        if (!ok) {
          const r = await fetch(url, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(payload),
          });
          setLastHttp(h => ({ ...h, [which]: r.ok ? "OK (fetch)" : `HTTP ${r.status}` }));
        } else {
          setLastHttp(h => ({ ...h, [which]: "sent (beacon)" }));
        }
      } catch (e) {
        setLastHttp(h => ({ ...h, [which]: `ERR ${String(e)}` }));
      }
      setPosts(p => ({ ...p, [which]: p[which] + 1 }));
      setLastSent(s => ({ ...s, [which]: nowStr }));
      return;
    }

    // debug path: await the response
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      setPosts(p => ({ ...p, [which]: p[which] + 1 }));
      setLastSent(s => ({ ...s, [which]: nowStr }));
      setLastHttp(h => ({ ...h, [which]: r.ok ? "OK" : `HTTP ${r.status}` }));
    } catch (e) {
      setLastHttp(h => ({ ...h, [which]: `ERR ${String(e)}` }));
    }
  };

  /* ── tiny session FSM ──────────────────────────────────────────────────── */
  async function updateSession(people) {
    const now = Date.now();
    const anyFace = Array.isArray(people) && people.length > 0;

    if (!S.current.id) {
      if (anyFace) {
        S.current.seenFrames++;
        if (S.current.seenFrames >= START_FRAMES) {
          S.current.id = uuid();
          S.current.lastFaceTs = now;
          S.current.lastSnapshotTs = 0;
          setSessionId(S.current.id);
          setSessionStatus("ACTIVE");
          await post("start", { sessionId: S.current.id, ts: now });
        }
      } else {
        S.current.seenFrames = 0;
      }
      return;
    }

    if (anyFace) {
      S.current.lastFaceTs = now;
      if (now - S.current.lastSnapshotTs >= SNAPSHOT_EVERY) {
        S.current.lastSnapshotTs = now;
        await post("snapshot", { sessionId: S.current.id, ts: now, people });
      }
    } else {
      if (now - S.current.lastFaceTs >= END_AFTER_MS) {
        await post("stop", { sessionId: S.current.id, ts: now });
        S.current.id = null;
        S.current.seenFrames = 0;
        setSessionId(null);
        setSessionStatus("IDLE");
        // reset name stabilization (optional)
        slotsRef.current = slotsRefInit();
      }
    }
  }

  /* ── UI ────────────────────────────────────────────────────────────────── */
  const statusDot = (s) => ({
    display: "inline-block",
    width: 10, height: 10, borderRadius: 999, marginRight: 6,
    background: s === "ACTIVE" ? "#22c55e" : "#fbbf24",
  });

  return (
    <main style={{ background: "#0b0b0b", color: "#ebebeb", minHeight: "100vh", padding: 10 }}>
      {/* Status bar */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 8,
          maxWidth: 960,
          margin: "0 auto 10px",
          fontSize: 14,
        }}
      >
        <div style={{ background:"#1a1a1a", borderRadius:8, padding:10, display:"flex", flexDirection:"column", gap:4 }}>
          <div><strong>Backend:</strong> {backend}</div>
          <div><strong>Models:</strong> {ready ? "loaded" : "loading…"}</div>
          <div><span style={statusDot(sessionStatus)} /><strong>Session:</strong> {sessionStatus}{sessionId ? ` (${sessionId.slice(0,8)})` : ""}</div>
          <div><strong>Posts:</strong> start {posts.start} · snap {posts.snapshot} · stop {posts.stop}</div>
        </div>
        <div style={{ background:"#1a1a1a", borderRadius:8, padding:10, display:"flex", flexDirection:"column", gap:4 }}>
          <div><strong>Last Sent:</strong> start {lastSent.start} · snap {lastSent.snapshot} · stop {lastSent.stop}</div>
          <div style={{ opacity:.85 }}>
            <strong>HTTP:</strong> start {lastHttp.start || "-"} · snap {lastHttp.snapshot || "-"} · stop {lastHttp.stop || "-"}
            {DEBUG_FETCH ? " (debug)" : " (beacon)"}
          </div>
          <div style={{ opacity:.75 }}><strong>Faces:</strong> total {totals.all} • green {totals.green} • red {totals.red}</div>
        </div>
      </div>

      {/* Video + overlay */}
      <div style={{ position:"relative", maxWidth:960, margin:"0 auto" }}>
        <video ref={videoRef} autoPlay muted playsInline style={{ width:"100%", borderRadius:8, background:"#000" }} />
        <canvas ref={canvasRef} style={{ position:"absolute", inset:0, width:"100%", height:"100%" }} />
      </div>

      {/* Table (only GREEN rows) */}
      <div style={{ maxWidth:960, margin:"10px auto 40px" }}>
        <div style={{ margin:"8px 4px", fontSize:14, opacity:.9 }}>
          <strong>Shown in table:</strong> up to 5 GREEN faces (n8n still receives ALL)
        </div>
        <div style={{ overflowX:"auto" }}>
          <table style={{ width:"100%", borderCollapse:"collapse", fontSize:14 }}>
            <thead>
              <tr style={{ background:"#1f1f1f" }}>
                {["#", "Gender", "AgeGroup", "Zone", "Name", "Distance"].map((h) => (
                  <th key={h} style={{ textAlign:"left", padding:"8px 10px" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.map((r) => (
                <tr key={r.idx} style={{ borderTop:"1px solid #222" }}>
                  <td style={{ padding:"8px 10px" }}>{r.idx}</td>
                  <td style={{ padding:"8px 10px" }}>{r.gender}</td>
                  <td style={{ padding:"8px 10px" }}>{r.ageGroup}</td>
                  <td style={{ padding:"8px 10px", color: r.zone === "green" ? "#22c55e" : r.zone === "red" ? "#ef4444" : "#aaa" }}>{r.zone}</td>
                  <td style={{ padding:"8px 10px" }}>{r.name}</td>
                  <td style={{ padding:"8px 10px" }}>{r.distance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop:6, fontSize:12, opacity:.7 }}>
          Optional: add <code>?debugPost=1</code> to the URL to await HTTP responses and show OK/ERR above.
        </div>
      </div>
    </main>
  );
}