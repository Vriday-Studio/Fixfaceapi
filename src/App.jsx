import { useEffect, useRef, useState } from "react";
import * as faceapi from "face-api.js";

/* -------------------- CONFIG (simple & stable) -------------------- */
const MODEL_URL = "/models";
const LABELS_URL = "/labels/labels.json";

const FACE_WIDTH_M = 0.15;
let   FOCAL_PX     = 500;
const GREEN_MAX_M  = 0.8;

const TINY_OPTS = new faceapi.TinyFaceDetectorOptions({
  inputSize: 512,
  scoreThreshold: 0.3,
});

// recognition strictness
const MATCH_THRESHOLD   = 0.52; // tighten if needed (0.50–0.55)
const MATCH_MARGIN      = 0.06; // best must beat 2nd-best by this
const STABILIZE_FRAMES  = 4;    // frames of disagreement before switching

// drawing
const BOX_SHRINK     = 0.7;
const BOX_LINE_WIDTH = 5;

// session pacing
const START_FRAMES   = 8;
const END_AFTER_MS   = 8000;
const SNAPSHOT_EVERY = 900;
const LOOP_STEP_MS   = 120;

// n8n (Netlify function proxy)
const N8N = {
  start:   "/api/n8n-test/camera/start",
  snapshot:"/api/n8n-test/camera/snapshot",
  stop:    "/api/n8n-test/camera/stop",
};

const DEBUG_FETCH = /[?&]debugPost=1\b/.test(window.location.search);

/* -------------------- Guest ID memory (per page session) -------------------- */
const guestSeqRef = { current: 1 };                 // Guest01, Guest02, ...
const guestMemRef = { current: [] };                // [{ id, desc: Float32Array }]
const GUEST_TOL   = 0.56;                           // reuse a guest if <= this dist

function assignGuestIdFor(descriptor) {
  if (!descriptor || !descriptor.length) {
    const id = `Guest${String(guestSeqRef.current++).padStart(2, "0")}`;
    return id;
  }
  const mem = guestMemRef.current;
  let bestIdx = -1, bestDist = 1;
  for (let i = 0; i < mem.length; i++) {
    const d = faceapi.euclideanDistance(descriptor, mem[i].desc);
    if (d < bestDist) { bestDist = d; bestIdx = i; }
  }
  if (bestIdx >= 0 && bestDist <= GUEST_TOL) return mem[bestIdx].id;
  const id = `Guest${String(guestSeqRef.current++).padStart(2, "0")}`;
  mem.push({ id, desc: Float32Array.from(descriptor) });
  return id;
}

/* -------------------- helpers -------------------- */
const uuid = () => (crypto?.randomUUID ? crypto.randomUUID()
  : Math.random().toString(36).slice(2) + Date.now().toString(36));

const estimateDistanceM = (wPx) => (wPx ? (FOCAL_PX * FACE_WIDTH_M) / wPx : null);
const ageGroupOf = (age) => (age == null ? "unknown" : (Math.round(age) >= 18 ? "adult" : (Math.round(age) >= 12 ? "teen" : "child")));
const zoneOf = (d) => (d != null && d <= GREEN_MAX_M ? "green" : "red");
const topExpression = (e) => (!e ? "neutral" : Object.entries(e).reduce((a,b)=>a[1]>b[1]?a:b)[0]);

const shrinkBox = (b, f = BOX_SHRINK) => {
  const w = b.width * f, h = b.height * f;
  return { x: b.x + (b.width - w) / 2, y: b.y + (b.height - h) / 2, width: w, height: h };
};

// compute best & second-best so we can enforce a margin
function bestTwoMatches(matcher, queryDesc){
  let best={label:null,dist:1}, second={label:null,dist:1};
  for (const ld of matcher.labeledDescriptors){
    for (const d of ld.descriptors){
      const dist = faceapi.euclideanDistance(queryDesc, d);
      if (dist < best.dist){ second = best; best = { label: ld.label, dist }; }
      else if (dist < second.dist){ second = { label: ld.label, dist }; }
    }
  }
  return { best, second };
}

/* ------------------------------ App ------------------------------ */
export default function App(){
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [backend, setBackend] = useState("cpu");

  const [sessionStatus, setSessionStatus] = useState("IDLE");
  const [sessionId, setSessionId] = useState(null);
  const [posts, setPosts] = useState({ start:0, snapshot:0, stop:0 });
  const [lastSent, setLastSent] = useState({ start:"-", snapshot:"-", stop:"-" });
  const [lastHttp, setLastHttp] = useState({ start:"", snapshot:"", stop:"" });
  const [totals, setTotals] = useState({ all:0, green:0, red:0 });

  const [table, setTable] = useState(
    Array.from({length:5}, (_,i)=>({ idx:i+1, gender:"-", ageGroup:"-", zone:"-", name:"-", distance:"-" }))
  );

  const faceMatcherRef = useRef(null);
  const [knownCount, setKnownCount] = useState(0);

  // per-face (approx box) debounce cache to reduce flicker
  const recentMapRef = useRef({}); // key -> { name, count }

  // session state
  const S = useRef({ id:null, seenFrames:0, lastFaceTs:0, lastSnapshotTs:0 });

  /* -------------------- init: backend, models, labels, camera -------------------- */
  useEffect(() => {
    (async () => {
      // backend
      try { await faceapi.tf.setBackend("webgl"); await faceapi.tf.ready(); setBackend("webgl"); }
      catch { try { await faceapi.tf.setBackend("wasm"); await faceapi.tf.ready(); setBackend("wasm"); }
              catch { await faceapi.tf.setBackend("cpu"); await faceapi.tf.ready(); setBackend("cpu"); } }

      // nets
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
        faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      ]);

      // known faces via labels.json (Tiny pipeline for parity)
      try {
        let total = 0;
        const res = await fetch(LABELS_URL, { cache:"no-store" });
        const manifest = res.ok ? await res.json() : { people:[] };
        const labeled = [];

        for (const p of manifest.people || []) {
          const name = String(p.name||"").trim();
          const n = Number(p.images||0);
          if (!name || !n) continue;

          const descs = [];
          for (let i=1;i<=n;i++){
            const url = `/labels/${encodeURIComponent(name)}/${i}.jpg`;
            try{
              const img = await faceapi.fetchImage(url);
              const d = await faceapi
                .detectSingleFace(img, TINY_OPTS)
                .withFaceLandmarks()
                .withFaceDescriptor();
              if (d?.descriptor) descs.push(d.descriptor);
            }catch{}
          }
          if (descs.length){
            labeled.push(new faceapi.LabeledFaceDescriptors(name, descs));
            total += descs.length;
          }
        }
        if (labeled.length){
          faceMatcherRef.current = new faceapi.FaceMatcher(labeled, MATCH_THRESHOLD);
          setKnownCount(total);
        } else {
          faceMatcherRef.current = null;
          setKnownCount(0);
        }
      } catch {
        faceMatcherRef.current = null;
        setKnownCount(0);
      }

      // camera
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode:"user", width:{ideal:1920}, height:{ideal:1080}, frameRate:{ideal:30} },
        audio: false,
      });
      if (videoRef.current){
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          const w = videoRef.current.videoWidth || 1280;
          FOCAL_PX = w >= 1920 ? 1350 : 900;
        };
      }

      setReady(true);
    })();

    return () => {
      const s = videoRef.current?.srcObject;
      if (s) s.getTracks()?.forEach(t=>t.stop());
    };
  }, []);

  /* ------------------------------ loop ------------------------------ */
  useEffect(() => {
    if (!ready) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const resize = () => {
      if (!video) return;
      canvas.width  = video.videoWidth  || 940;
      canvas.height = video.videoHeight || 650;
    };
    video.addEventListener("loadedmetadata", resize);
    resize();

    let raf = 0, lastRun = 0;

    const loop = async (ts) => {
      raf = requestAnimationFrame(loop);
      if (ts - lastRun < LOOP_STEP_MS) return;
      lastRun = ts;
      if (!video.videoWidth) return;

      const dets = await faceapi
        .detectAllFaces(video, TINY_OPTS)
        .withFaceLandmarks()
        .withFaceExpressions()
        .withAgeAndGender()
        .withFaceDescriptors();

      ctx.clearRect(0,0,canvas.width,canvas.height);

      const resized = faceapi
        .resizeResults(dets, { width: canvas.width, height: canvas.height })
        .sort((a,b)=>a.detection.box.x - b.detection.box.x);

      const matcher = faceMatcherRef.current;
      const rows = [];
      const peopleForPost = [];
      let total=0, green=0, red=0;

      for (const det of resized){
        const box  = det.detection.box;
        const dist = estimateDistanceM(box.width);
        const zone = zoneOf(dist);
        const color = zone === "green" ? "#22c55e" : "#ef4444";
        const gender = (det.gender||"").toLowerCase();
        const expr = topExpression(det.expressions);

        // known match with margin check
        let name = null;
        if (matcher && det.descriptor) {
          // First, use FaceMatcher’s own threshold (simple + robust)
          const bestOnly = matcher.findBestMatch(det.descriptor);
          if (bestOnly && bestOnly.label !== "unknown" && bestOnly.distance <= MATCH_THRESHOLD) {
            name = bestOnly.label;
          } else if (bestOnly && bestOnly.label !== "unknown" && bestOnly.distance <= (MATCH_THRESHOLD + 0.05)) {
            // Soft accept path: allow slightly looser distance if it also clears a small margin
            const { best, second } = bestTwoMatches(matcher, det.descriptor);
            // Use a SMALL margin so we don’t kill legit matches
            const SMALL_MARGIN = 0.03; 
            if (best.label && best.label !== "unknown" && (second.dist - best.dist) >= SMALL_MARGIN) {
              name = best.label;
            }
          }
        }

        

        // stable GuestXX when not known
        let guestId = null;
        if (!name) {
          guestId = assignGuestIdFor(det.descriptor);
        }

        // debounced display label (reduce flicker)
        let displayName = name || guestId || "Guest";
        const key = [
          Math.round(box.x),
          Math.round(box.y),
          Math.round(box.width),
          Math.round(box.height),
        ].join("-");
        const cache = recentMapRef.current;
        const entry = cache[key];

        if (entry) {
          if (entry.name !== displayName) {
            if ((entry.count || 0) < STABILIZE_FRAMES) {
              displayName = entry.name;            // keep old until stable
              entry.count = (entry.count || 0) + 1;
            } else {
              cache[key] = { name: displayName, count: 0 }; // accept new
            }
          } else {
            entry.count = 0; // same label → reset counter
          }
        } else {
          cache[key] = { name: displayName, count: 0 };
        }

        // draw
        const dbox = shrinkBox(box);
        ctx.strokeStyle = color;
        ctx.lineWidth = BOX_LINE_WIDTH;
        ctx.strokeRect(dbox.x, dbox.y, dbox.width, dbox.height);

        const label = `${displayName} • ${zone} • ${Math.max(0,Math.round(det.age))} ${gender} • ${expr}`;
        ctx.font = "14px system-ui,-apple-system,Segoe UI,Roboto,sans-serif";
        const padX=6, padY=4;
        const tw = ctx.measureText(label).width + padX*2;
        const th = 18 + padY*2;
        const lx = Math.max(0, Math.min(dbox.x, canvas.width - tw));
        const ly = Math.max(0, dbox.y - th - 4);
        ctx.fillStyle = color; ctx.fillRect(lx, ly, tw, th);
        ctx.fillStyle = "#fff"; ctx.fillText(label, lx + padX, ly + 14);

        // table (first 5)
        if (rows.length < 5){
          rows.push({
            idx: rows.length+1,
            gender,
            ageGroup: ageGroupOf(det.age),
            zone,
            name: name || "-", // show GuestXX or known name
            distance: dist ? dist.toFixed(2)+" m" : "-",
          });
        }

        // payload (ALL faces)
        peopleForPost.push({
          gender,
          ageGroup: ageGroupOf(det.age),
          zone,
          name: name || null,   // known name if any
          gid: guestId || null, // Guest01… when unknown
        });

        total++; if (zone==="green") green++; else if (zone==="red") red++;
      }

      while (rows.length < 5) rows.push({ idx: rows.length+1, gender:"-", ageGroup:"-", zone:"-", name:"-", distance:"-" });
      setTable(rows);
      setTotals({ all: total, green, red });

      await updateSession(peopleForPost);
    };

    raf = requestAnimationFrame(loop);
    return () => { cancelAnimationFrame(raf); video?.removeEventListener("loadedmetadata", resize); };
  }, [ready]);

  /* ------------------------------ posting ------------------------------ */
  const post = async (which, payload) => {
    const url = N8N[which];
    const nowStr = new Date().toLocaleTimeString();

    if (!DEBUG_FETCH){
      try{
        const ok = navigator.sendBeacon(url, new Blob([JSON.stringify(payload)], { type:"application/json" }));
        if (!ok){
          const r = await fetch(url, { method:"POST", headers:{ "content-type":"application/json" }, body: JSON.stringify(payload) });
          setLastHttp(h=>({ ...h, [which]: r.ok ? "OK (fetch)" : `HTTP ${r.status}` }));
        } else {
          setLastHttp(h=>({ ...h, [which]: "sent (beacon)" }));
        }
      }catch(e){ setLastHttp(h=>({ ...h, [which]: `ERR ${String(e)}` })); }
      setPosts(p=>({ ...p, [which]: p[which]+1 })); setLastSent(s=>({ ...s, [which]: nowStr }));
      return;
    }

    try{
      const r = await fetch(url, { method:"POST", headers:{ "content-type":"application/json" }, body: JSON.stringify(payload) });
      setPosts(p=>({ ...p, [which]: p[which]+1 })); setLastSent(s=>({ ...s, [which]: nowStr }));
      setLastHttp(h=>({ ...h, [which]: r.ok ? "OK" : `HTTP ${r.status}` }));
    }catch(e){ setLastHttp(h=>({ ...h, [which]: `ERR ${String(e)}` })); }
  };

  /* ------------------------------ session FSM ------------------------------ */
  async function updateSession(people){
    const now = Date.now();
    const any = Array.isArray(people) && people.length > 0;

    if (!S.current.id){
      if (any){
        S.current.seenFrames++;
        if (S.current.seenFrames >= START_FRAMES){
          S.current.id = uuid();
          S.current.lastFaceTs = now;
          S.current.lastSnapshotTs = 0;
          setSessionId(S.current.id); setSessionStatus("ACTIVE");
          await post("start", { sessionId: S.current.id, ts: now });
        }
      } else {
        S.current.seenFrames = 0;
      }
      return;
    }

    if (any){
      S.current.lastFaceTs = now;
      if (now - S.current.lastSnapshotTs >= SNAPSHOT_EVERY){
        S.current.lastSnapshotTs = now;
        await post("snapshot", { sessionId: S.current.id, ts: now, people });
      }
    } else {
      if (now - S.current.lastFaceTs >= END_AFTER_MS){
        await post("stop", { sessionId: S.current.id, ts: now });
        S.current.id = null; S.current.seenFrames = 0;
        setSessionId(null); setSessionStatus("IDLE");
        recentMapRef.current = {}; // clear debounce cache on stop
      }
    }
  }

  /* ------------------------------ UI ------------------------------ */
  const statusDot = (s) => ({
    display:"inline-block", width:10, height:10, borderRadius:999, marginRight:6,
    background: s === "ACTIVE" ? "#22c55e" : "#fbbf24",
  });

  return (
    <main style={{ background:"#0b0b0b", color:"#ebebeb", minHeight:"100vh", padding:10 }}>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8, maxWidth:960, margin:"0 auto 10px", fontSize:14 }}>
        <div style={{ background:"#1a1a1a", borderRadius:8, padding:10, display:"flex", flexDirection:"column", gap:4 }}>
          <div><strong>Backend:</strong> {backend}</div>
          <div><strong>Models:</strong> {ready ? "loaded" : "loading…"}</div>
          <div><span style={statusDot(sessionStatus)} /><strong>Session:</strong> {sessionStatus}{sessionId ? ` (${sessionId.slice(0,8)})` : ""}</div>
          <div><strong>Known faces loaded:</strong> {knownCount}</div>
          <div><strong>Posts:</strong> start {posts.start} · snap {posts.snapshot} · stop {posts.stop}</div>
        </div>
        <div style={{ background:"#1a1a1a", borderRadius:8, padding:10, display:"flex", flexDirection:"column", gap:4 }}>
          <div><strong>Last Sent:</strong> start {lastSent.start} · snap {lastSent.snapshot} · stop {lastSent.stop}</div>
          <div style={{ opacity:.85 }}>
            <strong>HTTP:</strong> start {lastHttp.start || "-"} · snap {lastHttp.snapshot || "-"} · stop {lastHttp.stop || "-"} {DEBUG_FETCH ? " (debug)" : " (beacon)"}
          </div>
          <div style={{ opacity:.75 }}><strong>Faces:</strong> total {totals.all} • green {totals.green} • red {totals.red}</div>
        </div>
      </div>

      <div style={{ position:"relative", maxWidth:960, margin:"0 auto" }}>
        <video ref={videoRef} autoPlay muted playsInline style={{ width:"100%", borderRadius:8, background:"#000" }} />
        <canvas ref={canvasRef} style={{ position:"absolute", inset:0, width:"100%", height:"100%" }} />
      </div>

      <div style={{ maxWidth:960, margin:"10px auto 40px" }}>
        <div style={{ overflowX:"auto" }}>
          <table style={{ width:"100%", borderCollapse:"collapse", fontSize:14 }}>
            <thead>
              <tr style={{ background:"#1f1f1f" }}>
                {["#", "Gender", "AgeGroup", "Zone", "Name", "Distance"].map(h => (
                  <th key={h} style={{ textAlign:"left", padding:"8px 10px" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {table.map(r => (
                <tr key={r.idx} style={{ borderTop:"1px solid #222" }}>
                  <td style={{ padding:"8px 10px" }}>{r.idx}</td>
                  <td style={{ padding:"8px 10px" }}>{r.gender}</td>
                  <td style={{ padding:"8px 10px" }}>{r.ageGroup}</td>
                  <td style={{ padding:"8px 10px", color: r.zone==="green" ? "#22c55e" : r.zone==="red" ? "#ef4444" : "#aaa" }}>{r.zone}</td>
                  <td style={{ padding:"8px 10px" }}>{r.name}</td>
                  <td style={{ padding:"8px 10px" }}>{r.distance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </main>
  );
}