import { useEffect, useRef, useState } from "react";
import * as faceapi from "face-api.js";

// ====== CONFIG ======
const MODEL_URL = "/models";                 // face-api models path
const FACE_WIDTH_M = 0.15;                   // avg face width (m)
const FOCAL_PX = 500;                        // tune per camera if needed
const GREEN_MAX_M = 0.8;                     // <= 0.8m → green

const TINY_OPTS = new faceapi.TinyFaceDetectorOptions({
  inputSize: 224,
  scoreThreshold: 0.5,
});

// n8n (proxied by netlify.toml)
const IS_LOCALHOST = /^(localhost|127\.0\.0\.1|\d+\.\d+\.\d+\.\d+)$/.test(location.hostname);
const FORCE_TEST   = /[?&]forceTest=1\b/.test(location.search); // manual override via URL param

const N8N = (IS_LOCALHOST || FORCE_TEST)
  ? { // test (webhook-test): works only while node is "Listening…"
      start: "/api/n8n-test/camera/start",
      snapshot: "/api/n8n-test/camera/snapshot",
      stop: "/api/n8n-test/camera/stop",
    }
  : { // production (webhook): requires workflow Activated
      start: "/api/n8n/start",
      snapshot: "/api/n8n/snapshot",
      stop: "/api/n8n/stop",
    };

// session timings
const START_FRAMES = 8;      // need N consecutive frames w/ faces to start
const END_AFTER_MS = 8000;   // stop after 8s of no faces
const SNAPSHOT_EVERY = 600;  // snapshot at most once per 600ms

// ====== HELPERS ======
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

// ====== APP ======
export default function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [backend, setBackend] = useState("cpu");

  const [sessionStatus, setSessionStatus] = useState("IDLE"); // IDLE | ACTIVE
  const [sessionId, setSessionId] = useState(null);
  const [posts, setPosts] = useState({ start: 0, snapshot: 0, stop: 0 });
  const [lastSent, setLastSent] = useState({ start: "-", snapshot: "-", stop: "-" });
  const [lastHttp, setLastHttp] = useState({ start: "", snapshot: "", stop: "" }); // OK / error (debug)
  const [table, setTable] = useState([]); // [{idx, gender, ageGroup, zone, name}…]

  const faceMatcherRef = useRef(null);

  // FSM scratch
  const S = useRef({
    id: null,
    seenFrames: 0,
    lastFaceTs: 0,
    lastSnapshotTs: 0,
  });

  const DEBUG_FETCH = /[?&]debugPost=1\b/.test(window.location.search);

  // --- init: backend, models, labels, camera ---
  useEffect(() => {
    (async () => {
      // backend fallback (webgl → wasm → cpu)
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
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
        faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      ]);

      // optional face recognition via /labels/labels.json
      try {
        const res = await fetch("/labels/labels.json", { cache: "no-store" });
        if (res.ok) {
          const manifest = await res.json(); // { people:[{name,images}] }
          const labeled = [];
          for (const person of manifest.people || []) {
            const descs = [];
            for (let i = 1; i <= Number(person.images || 0); i++) {
              const img = await faceapi.fetchImage(`/labels/${person.name}/${i}.jpg`);
              const d = await faceapi
                .detectSingleFace(img, TINY_OPTS)
                .withFaceLandmarks()
                .withFaceDescriptor();
              if (d?.descriptor) descs.push(d.descriptor);
            }
            if (descs.length) labeled.push(new faceapi.LabeledFaceDescriptors(person.name, descs));
          }
          if (labeled.length) faceMatcherRef.current = new faceapi.FaceMatcher(labeled, 0.6);
        }
      } catch {
        // no labels/manifest.json available → ignore
      }

      // start camera
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      if (videoRef.current) videoRef.current.srcObject = stream;

      setReady(true);
    })();
  }, []);

  // --- detection loop ---
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
    const STEP_MS = 120; // ~8 FPS detector

    const loop = async (ts) => {
      raf = requestAnimationFrame(loop);
      if (ts - lastRun < STEP_MS) return;
      lastRun = ts;
      if (!video.videoWidth) return;

      const dets = await faceapi
        .detectAllFaces(video, TINY_OPTS)
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender()
        .withFaceDescriptors();

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const resized = faceapi.resizeResults(dets, { width: canvas.width, height: canvas.height });
      const matcher = faceMatcherRef.current;

      // Sort by x (left→right) for stable table ordering
      resized.sort((a, b) => a.detection.box.x - b.detection.box.x);

      const rows = [];
      const peopleForPost = [];

      resized.forEach((det, i) => {
        const box = det.detection.box;
        const distM = estimateDistanceM(box.width);
        const zone = zoneOf(distM);
        const color = zone === "green" ? "#22c55e" : "#ef4444";

        // optional name via faceMatcher
        const name =
          matcher && det.descriptor
            ? (() => {
                const best = matcher.findBestMatch(det.descriptor);
                return best?.label && best.label !== "unknown" ? best.label : null;
              })()
            : null;

        // draw box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(box.x, box.y, box.width, box.height);

        // label
        const lbl =
          `${zone} • ${Math.max(0, Math.round(det.age))} ${det.gender}` +
          ` • ${topExpression(det.expressions)}` +
          (name ? ` • ${name}` : "");
        ctx.font = "16px system-ui, sans-serif";
        const pad = 6;
        const textW = ctx.measureText(lbl).width + pad * 2;
        const textH = 22;
        const y = Math.max(0, box.y - textH - 4);
        ctx.fillStyle = color;
        ctx.fillRect(box.x, y, textW, textH);
        ctx.fillStyle = "#fff";
        ctx.fillText(lbl, box.x + pad, y + 16);

        rows.push({
          idx: i + 1,
          gender: (det.gender || "").toLowerCase(),
          ageGroup: ageGroupOf(det.age),
          zone,
          name: name || "-",
          distance: distM ? distM.toFixed(2) + " m" : "-",
        });

        peopleForPost.push({
          gender: (det.gender || "").toLowerCase(),
          ageGroup: ageGroupOf(det.age),
          zone,
          name: name || null,
        });
      });

      // Fill table to 5 rows for a stable view
      const filled = rows.slice(0, 5);
      for (let i = filled.length; i < 5; i++) {
        filled.push({ idx: i + 1, gender: "-", ageGroup: "-", zone: "-", name: "-", distance: "-" });
      }
      setTable(filled);

      // update session FSM
      await updateSession(peopleForPost);
    };

    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      video?.removeEventListener("loadedmetadata", resize);
    };
  }, [ready]);

  // --- posting helpers (beacon by default, fetch if ?debugPost=1) ---
  const post = async (which, payload) => {
    const url = N8N[which];
    const nowStr = new Date().toLocaleTimeString();
    if (!DEBUG_FETCH) {
      // fire-and-forget
      const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
      const ok = navigator.sendBeacon(url, blob);
      setPosts((p) => ({ ...p, [which]: p[which] + 1 }));
      setLastSent((s) => ({ ...s, [which]: nowStr }));
      setLastHttp((h) => ({ ...h, [which]: ok ? "sent (beacon)" : "fallback sent" }));
      if (!ok) {
        // fallback
        try {
          const r = await fetch(url, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(payload),
          });
          setLastHttp((h) => ({ ...h, [which]: r.ok ? "OK (fetch)" : `HTTP ${r.status}` }));
        } catch (e) {
          setLastHttp((h) => ({ ...h, [which]: `ERR ${String(e)}` }));
        }
      }
      return;
    }

    // debug path: await response & show OK/ERR
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
      });
      setPosts((p) => ({ ...p, [which]: p[which] + 1 }));
      setLastSent((s) => ({ ...s, [which]: nowStr }));
      setLastHttp((h) => ({ ...h, [which]: r.ok ? "OK" : `HTTP ${r.status}` }));
    } catch (e) {
      setLastHttp((h) => ({ ...h, [which]: `ERR ${String(e)}` }));
    }
  };

  // --- tiny session FSM ---
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
      }
    }
  }

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
        <div
          style={{
            background: "#1a1a1a",
            borderRadius: 8,
            padding: 10,
            display: "flex",
            flexDirection: "column",
            gap: 4,
          }}
        >
          <div><strong>Backend:</strong> {backend}</div>
          <div><strong>Models:</strong> {ready ? "loaded" : "loading…"}</div>
          <div><strong>Session:</strong> {sessionStatus}{sessionId ? ` (${sessionId.slice(0, 8)})` : ""}</div>
          <div><strong>Posts:</strong> start {posts.start} · snap {posts.snapshot} · stop {posts.stop}</div>
        </div>

        <div
          style={{
            background: "#1a1a1a",
            borderRadius: 8,
            padding: 10,
            display: "flex",
            flexDirection: "column",
            gap: 4,
          }}
        >
          <div><strong>Last Sent:</strong> start {lastSent.start} · snap {lastSent.snapshot} · stop {lastSent.stop}</div>
          <div style={{ opacity: 0.8 }}>
            <strong>HTTP:</strong>{" "}
            start {lastHttp.start || "-"} · snap {lastHttp.snapshot || "-"} · stop {lastHttp.stop || "-"}
            {DEBUG_FETCH ? " (debug)" : " (beacon)"}
          </div>
        </div>
      </div>

      {/* Video + overlay */}
      <div style={{ position: "relative", maxWidth: 960, margin: "0 auto" }}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{ width: "100%", borderRadius: 8, background: "#000" }}
        />
        <canvas
          ref={canvasRef}
          style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
        />
      </div>

      {/* Table */}
      <div style={{ maxWidth: 960, margin: "10px auto 40px" }}>
        <div style={{ margin: "8px 4px", fontSize: 14, opacity: 0.9 }}>
          <strong>Total on screen:</strong> {table.filter(r => r.zone === "green" || r.zone === "red").length}
        </div>
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
            <thead>
              <tr style={{ background: "#1f1f1f" }}>
                {["#", "Gender", "AgeGroup", "Zone", "Name", "Distance"].map((h) => (
                  <th key={h} style={{ textAlign: "left", padding: "8px 10px" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.map((r) => (
                <tr key={r.idx} style={{ borderTop: "1px solid #222" }}>
                  <td style={{ padding: "8px 10px" }}>{r.idx}</td>
                  <td style={{ padding: "8px 10px" }}>{r.gender}</td>
                  <td style={{ padding: "8px 10px" }}>{r.ageGroup}</td>
                  <td style={{ padding: "8px 10px", color: r.zone === "green" ? "#22c55e" : r.zone === "red" ? "#ef4444" : "#aaa" }}>
                    {r.zone}
                  </td>
                  <td style={{ padding: "8px 10px" }}>{r.name}</td>
                  <td style={{ padding: "8px 10px" }}>{r.distance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop: 6, fontSize: 12, opacity: 0.7 }}>
          Tip: add <code>?debugPost=1</code> to the URL to await HTTP responses and see “OK/ERR” above.
        </div>
      </div>
    </main>
  );
}