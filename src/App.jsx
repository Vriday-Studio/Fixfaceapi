import { useEffect, useRef, useState } from "react";
import * as faceapi from "face-api.js";

/* ──────────────────────────────────────────────────────────────────────────────
   CONFIG (n8n Community test + sticky 5-row table)
   ────────────────────────────────────────────────────────────────────────────── */

const MODEL_URL = "/models";

// distance → zone
const FACE_WIDTH_M = 0.15;
let   FOCAL_PX     = 500;      // auto-tuned after camera starts
const GREEN_MAX_M  = 0.8;      // <= 0.8 m → green (distance color only)

// runtime detector (keeps farther faces)
const TINY_OPTS = new faceapi.TinyFaceDetectorOptions({
  inputSize: 512,
  scoreThreshold: 0.30,
});

// box style
const BOX_SHRINK     = 0.7;
const BOX_LINE_WIDTH = 5;

// recognition thresholds (tight to avoid false positives)
const MATCH_STRICT     = 0.52; // smaller = stricter
const MATCH_MARGIN     = 0.06; // best must beat second by this much
const ABS_MAX          = 0.60; // never accept above this
const STABILIZE_FRAMES = 3;

// 5 fixed UI rows (“seats”)
const NUM_SLOTS = 5;

// guest IDs (Guest01…) persistence (per day)
const GUEST_TTL_MS     = 8 * 60 * 60 * 1000;
const GUEST_DESC_TOL   = 0.60;
const GUEST_CENTER_TOL = 140;

// session pacing
const START_FRAMES   = 8;
const END_AFTER_MS   = 8000;
const SNAPSHOT_EVERY = 900;
const LOOP_STEP_MS   = 120;

// n8n TEST (Netlify proxy)
const N8N = {
  start:   "/api/n8n-test/camera/start",
  snapshot:"/api/n8n-test/camera/snapshot",
  stop:    "/api/n8n-test/camera/stop",
};

// optional: await HTTP responses
const DEBUG_FETCH = /[?&]debugPost=1\b/.test(window.location.search);

/* ──────────────────────────────────────────────────────────────────────────────
   HELPERS
   ────────────────────────────────────────────────────────────────────────────── */

const uuid = () =>
  (crypto?.randomUUID ? crypto.randomUUID()
   : Math.random().toString(36).slice(2) + Date.now().toString(36));

const estimateDistanceM = (wPx) => (wPx ? (FOCAL_PX * FACE_WIDTH_M) / wPx : null);
const ageGroupOf = (age) => (age == null ? "unknown" : (Math.round(age) >= 18 ? "adult" : (Math.round(age) >= 12 ? "teen" : "child")));
const zoneOf = (distM) => (distM != null && distM <= GREEN_MAX_M ? "green" : "red");
const topExpression = (e) => (!e ? "neutral" : Object.entries(e).reduce((a,b)=>a[1]>b[1]?a:b)[0]);

const shrinkBox = (b, f=BOX_SHRINK) => {
  const w = b.width * f, h = b.height * f;
  return { x: b.x + (b.width-w)/2, y: b.y + (b.height-h)/2, width: w, height: h };
};
const centerOf = (b) => [b.x + b.width/2, b.y + b.height/2];
const l2 = (a,b) => { let s=0; for (let i=0;i<a.length;i++){ const d=a[i]-b[i]; s+=d*d; } return Math.sqrt(s); };

function bestTwoMatches(matcher, desc){
  let best={label:null,dist:1}, second={label:null,dist:1};
  for (const ld of matcher.labeledDescriptors){
    for (const d of ld.descriptors){
      const dist = faceapi.euclideanDistance(desc, d);
      if (dist < best.dist){ second=best; best={label:ld.label,dist}; }
      else if (dist < second.dist){ second={label:ld.label,dist}; }
    }
  }
  return { best, second };
}

/* ── Guest IDs (Guest01…) per day ─────────────────────────────────────────── */
const guestDB = { day: new Date().toDateString(), list: [], next: 1 };
function resetGuestsIfNewDay(){
  const today = new Date().toDateString();
  if (guestDB.day !== today){ guestDB.day = today; guestDB.list = []; guestDB.next = 1; }
}
function getGuestId(desc, box){
  resetGuestsIfNewDay();
  const [cx, cy] = centerOf(box);
  const now = Date.now();
  guestDB.list = guestDB.list.filter(g => (now - g.last) < GUEST_TTL_MS);

  let best=null;
  for (const g of guestDB.list){
    const dc = Math.hypot(g.cx - cx, g.cy - cy);
    if (dc > 3*GUEST_CENTER_TOL) continue;
    const d = l2(desc, g.desc);
    if (d <= GUEST_DESC_TOL && (!best || d < best.d)) best = { g, d };
  }
  if (best){
    best.g.last = now; best.g.cx = cx; best.g.cy = cy;
    return best.g.id;
  }
  const id = `Guest${String(guestDB.next++).padStart(2,"0")}`;
  guestDB.list.push({ id, desc: Float32Array.from(desc), last: now, cx, cy });
  return id;
}

// Force a fresh guest list whenever the page fully loads
if (typeof window !== "undefined") {
  window.addEventListener("pageshow", () => {
    guestDB.day = new Date().toDateString();
    guestDB.list = [];
    guestDB.next = 1;
  });
}
// If Vite HMR replaces the module, force a full reload so memory resets
if (import.meta && import.meta.hot) {
  import.meta.hot.accept(() => window.location.reload());
}

/* ── 5 sticky UI slots ────────────────────────────────────────────────────── */
function makeSlots(){
  return Array.from({length: NUM_SLOTS}, () => ({
    id: null,         // "Raisa" or "Guest01"
    isKnown: false,
    zone: "-",
    gender: "-",
    ageGroup: "-",
    distance: "-",
    lastSeen: 0,
    // for recognition flicker smoothing per slot:
    shownName: null, candidate: null, streak: 0,
    _center: null,
  }));
}

/* ──────────────────────────────────────────────────────────────────────────────
   APP
   ────────────────────────────────────────────────────────────────────────────── */

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
    Array.from({length: NUM_SLOTS}, (_,i)=>({ idx:i+1, gender:"-", ageGroup:"-", zone:"-", name:"-", distance:"-" }))
  );

  const slotsRef = useRef(makeSlots());
  const faceMatcherRef = useRef(null);

  const S = useRef({ id:null, seenFrames:0, lastFaceTs:0, lastSnapshotTs:0 });

  /* ── init ──────────────────────────────────────────────────────────────── */
  useEffect(() => {
    (async () => {
      // backend chain: webgl → wasm → cpu
      try { await faceapi.tf.setBackend("webgl"); await faceapi.tf.ready(); setBackend("webgl"); }
      catch { try { await faceapi.tf.setBackend("wasm"); await faceapi.tf.ready(); setBackend("wasm"); }
              catch { await faceapi.tf.setBackend("cpu"); await faceapi.tf.ready(); setBackend("cpu"); } }

      // load nets
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
        faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),   // needed for label encoding
        faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
        faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
        faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
        faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
      ]);

      // --- load known faces from /labels/labels.json (USE SSD FOR ENCODING) ---
      try {
        const r = await fetch("/labels/labels.json", { cache: "no-store" });
        if (r.ok) {
          const j = await r.json(); // { people:[{name, images}] }
          const labeled = [];

          const ssdOpts = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });

          for (const p of j.people || []) {
            const descs = [];
            for (let i = 1; i <= Number(p.images || 0); i++) {
              try {
                const img = await faceapi.fetchImage(`/labels/${p.name}/${i}.jpg`);
                const d = await faceapi
                  .detectSingleFace(img, ssdOpts)           // SSD here (stable)
                  .withFaceLandmarks()
                  .withFaceDescriptor();
                if (d?.descriptor) descs.push(d.descriptor);
              } catch {}
            }
            if (descs.length) {
              labeled.push(new faceapi.LabeledFaceDescriptors(p.name, descs));
            }
          }

          if (labeled.length) {
            faceMatcherRef.current = new faceapi.FaceMatcher(labeled);
            console.log("Loaded labels:", labeled.map(l => `${l.label}(${l.descriptors.length})`));
          } else {
            console.warn("No label descriptors built. Check /labels/labels.json and image files.");
          }
        }
      } catch (e) {
        console.warn("Label load error:", e);
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
          FOCAL_PX = w >= 1920 ? 1350 : 900; // coarse auto-tune
        };
      }

      setReady(true);
    })();

    return () => {
      const s = videoRef.current?.srcObject;
      if (s) s.getTracks()?.forEach(t=>t.stop());
    };
  }, []);

  /* ── assign detections → 5 sticky slots (known → slot 0 priority) ──────── */
  function updateSlotsWithDetections(resized){
    const matcher = faceMatcherRef.current;
    const now = Date.now();
    const prev = slotsRef.current;
    const next = makeSlots();

    // build identities for each detection (left→right)
    const items = resized.map(det => {
      const box = det.detection.box;
      const distM = estimateDistanceM(box.width);
      const zone  = zoneOf(distM);
      const gender = (det.gender || "").toLowerCase();

      let name=null, isKnown=false;
      if (matcher && det.descriptor){
        const { best, second } = bestTwoMatches(matcher, det.descriptor);
        const strict = zone === "red" ? (MATCH_STRICT - 0.02) : MATCH_STRICT;
        const ok = best.label && best.label !== "unknown" &&
                   best.dist <= strict &&
                   best.dist <= ABS_MAX &&
                   (second.dist - best.dist) >= MATCH_MARGIN;
        if (ok){ name = best.label; isKnown = true; }
      }

      let id = name;
      if (!id && det.descriptor) id = getGuestId(det.descriptor, det.detection.box);

      return {
        id, isKnown,
        zone, gender, ageGroup: ageGroupOf(det.age),
        distance: distM ? distM.toFixed(2)+" m" : "-",
        _center: centerOf(det.detection.box),
      };
    });

    // stick by exact ID
    for (const it of items){
      let placed = false;
      for (let i=0;i<NUM_SLOTS;i++){
        const s = prev[i];
        if (s.id && s.id === it.id){
          next[i] = { ...it, lastSeen: now, shownName: s.shownName, candidate: s.candidate, streak: s.streak };
          placed = true; break;
        }
      }
      if (placed) continue;

      // else: first free slot
      const free = next.findIndex(s => !s.id);
      if (free >= 0) next[free] = { ...it, lastSeen: now };
    }

    // carry centers for continuity (optional)
    for (let i=0;i<NUM_SLOTS;i++){
      const n = next[i], p = prev[i];
      if (n.id && !n._center) n._center = p._center || null;
    }

    // Promote any KNOWN to slot 0 (row 1)
    const knownIdx = next.findIndex(s => s.id && s.isKnown);
    if (knownIdx > 0){
      const tmp = next[0]; next[0] = next[knownIdx]; next[knownIdx] = tmp;
    }

    // Name smoothing per slot
    for (let i=0;i<NUM_SLOTS;i++){
      const n = next[i], p = prev[i];
      if (!n.id){ next[i] = { ...n, shownName:null, candidate:null, streak:0 }; continue; }
      const proposed = n.id;
      if (proposed === p.shownName){
        next[i].shownName = proposed;
        next[i].candidate = proposed;
        next[i].streak = STABILIZE_FRAMES;
      } else {
        next[i].candidate = proposed;
        next[i].streak = (proposed === p.candidate) ? Math.min(STABILIZE_FRAMES, (p.streak || 0)+1) : 1;
        if (next[i].streak >= STABILIZE_FRAMES) next[i].shownName = proposed;
        else next[i].shownName = p.shownName || proposed;
      }
    }

    slotsRef.current = next;
  }

  /* ── draw boxes + build posting payload ─────────────────────────────────── */
  function drawAndBuild(ctx, canvas, resized){
    const matcher = faceMatcherRef.current;
    const peopleForPost = [];
    let total=0, green=0, red=0;

    for (const det of resized){
      const box = det.detection.box;
      const distM = estimateDistanceM(box.width);
      const zone  = zoneOf(distM);
      const color = zone === "green" ? "#22c55e" : "#ef4444";
      const gender = (det.gender || "").toLowerCase();
      const expr = topExpression(det.expressions);

      // resolve display name the same way as slots (for label consistency)
      let name=null, isKnown=false;
      if (matcher && det.descriptor){
        const { best, second } = bestTwoMatches(matcher, det.descriptor);
        const strict = zone === "red" ? (MATCH_STRICT - 0.02) : MATCH_STRICT;
        const ok = best.label && best.label !== "unknown" &&
                   best.dist <= strict &&
                   best.dist <= ABS_MAX &&
                   (second.dist - best.dist) >= MATCH_MARGIN;
        if (ok){ name = best.label; isKnown = true; }
      }
      let id = name;
      if (!id && det.descriptor) id = getGuestId(det.descriptor, box);
      const shown = id || "Guest??";

      // draw
      const dbox = shrinkBox(box, BOX_SHRINK);
      ctx.strokeStyle = color;
      ctx.lineWidth = BOX_LINE_WIDTH;
      ctx.strokeRect(dbox.x, dbox.y, dbox.width, dbox.height);

      const label = `${shown} • ${zone} • ${Math.max(0, Math.round(det.age))} ${gender} • ${expr}`;
      ctx.font = "14px system-ui, -apple-system, Segoe UI, Roboto, sans-serif";
      const padX = 6, padY = 4;
      const textW = ctx.measureText(label).width + padX*2;
      const textH = 18 + padY*2;
      const lx = Math.max(0, Math.min(dbox.x, canvas.width - textW));
      const ly = Math.max(0, dbox.y - textH - 4);
      ctx.fillStyle = color; ctx.fillRect(lx, ly, textW, textH);
      ctx.fillStyle = "#fff"; ctx.fillText(label, lx + padX, ly + 14);

      // payload
      peopleForPost.push({
        gender,
        ageGroup: ageGroupOf(det.age),
        zone,
        name: isKnown ? name : null,
        gid: !isKnown ? shown : null,
      });

      total++; if (zone === "green") green++; else if (zone === "red") red++;
    }

    // Sticky table from slots
    const rows = [];
    const slots = slotsRef.current;
    for (let i=0;i<NUM_SLOTS;i++){
      const s = slots[i];
      if (s.id){
        rows.push({
          idx: i+1,
          gender: s.gender,
          ageGroup: s.ageGroup,
          zone: s.zone,
          name: s.id,
          distance: s.distance,
        });
      } else {
        rows.push({ idx: i+1, gender:"-", ageGroup:"-", zone:"-", name:"-", distance:"-" });
      }
    }

    setTotals({ all: total, green, red });
    return { rows, peopleForPost };
  }

  /* ── main detection loop ────────────────────────────────────────────────── */
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

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const resized = faceapi
        .resizeResults(dets, { width: canvas.width, height: canvas.height })
        .sort((a,b)=>a.detection.box.x - b.detection.box.x);

      updateSlotsWithDetections(resized);                   // sticky seats first
      const { rows, peopleForPost } = drawAndBuild(ctx, canvas, resized);
      setTable(rows);

      await updateSession(peopleForPost);                   // posting
    };

    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      video?.removeEventListener("loadedmetadata", resize);
    };
  }, [ready]);

  /* ── posting helpers ────────────────────────────────────────────────────── */
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

  /* ── session FSM ────────────────────────────────────────────────────────── */
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
        // also reset guest slots when session ends (optional)
        slotsRef.current = makeSlots();
      }
    }
  }

  /* ── UI ─────────────────────────────────────────────────────────────────── */
  const statusDot = (s) => ({
    display:"inline-block", width:10, height:10, borderRadius:999, marginRight:6,
    background: s === "ACTIVE" ? "#22c55e" : "#fbbf24",
  });

  return (
    <main style={{ background:"#0b0b0b", color:"#ebebeb", minHeight:"100vh", padding:10 }}>
      {/* Status */}
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8, maxWidth:960, margin:"0 auto 10px", fontSize:14 }}>
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

      {/* Video */}
      <div style={{ position:"relative", maxWidth:960, margin:"0 auto" }}>
        <video ref={videoRef} autoPlay muted playsInline style={{ width:"100%", borderRadius:8, background:"#000" }} />
        <canvas ref={canvasRef} style={{ position:"absolute", inset:0, width:"100%", height:"100%" }} />
      </div>

      {/* Sticky table (5 rows) */}
      <div style={{ maxWidth:960, margin:"10px auto 40px" }}>
        <div style={{ margin:"8px 4px", fontSize:14, opacity:.9 }}>
          <strong>Rows are sticky:</strong> 5 seats; a known person (if any) is always in <strong>row 1</strong>.
        </div>
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
                  <td style={{ padding:"8px 10px", color: r.zone === "green" ? "#22c55e" : r.zone === "red" ? "#ef4444" : "#aaa" }}>{r.zone}</td>
                  <td style={{ padding:"8px 10px" }}>{r.name}</td>
                  <td style={{ padding:"8px 10px" }}>{r.distance}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div style={{ marginTop:6, fontSize:12, opacity:.7 }}>
          Add <code>?debugPost=1</code> to await HTTP responses and see “OK/ERR”.
        </div>
      </div>
    </main>
  );
}