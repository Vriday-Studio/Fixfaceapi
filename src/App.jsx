import { useEffect, useRef, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-wasm";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import * as faceapi from "face-api.js";
import './App.css';

/* ====================== CONFIG ====================== */
const MODEL_URL  = "/models";
const LABELS_URL = "/labels/labels.json";

const FACE_WIDTH_M  = 0.15;
let   FOCAL_PX      = 500;     // will be adjusted after video loads
const DEFAULT_GREEN_MAX_M = 0.8;

// recognition strictness
const MATCH_THRESHOLD  = 0.50; // tighten/loosen (0.48–0.55)
const MATCH_MARGIN     = 0.03; // how much best must beat 2nd-best
const STABILIZE_FRAMES = 5;    // frames before switching label

// drawing
const BOX_SHRINK     = 0.7;
const BOX_LINE_WIDTH = 5;
const LABEL_FONT     = "16px system-ui,-apple-system,Segoe UI,Roboto,sans-serif";
const LABEL_PAD_X    = 8;
const LABEL_PAD_Y    = 6;

// pacing
const START_FRAMES   = 8;      // ~5s @120ms/frame would be ~40 if you want to gate session start harder
const END_AFTER_MS   = 10000;
const SNAPSHOT_EVERY = 1500;
const LOOP_STEP_MS   = 120;

// n8n (Netlify proxy)
const N8N = {
  start:    "/api/n8n/camera/start",
  snapshot: "/api/n8n/camera/snapshot",
  stop:     "/api/n8n/camera/stop",
  stt:      "/api/n8n/stt/utterance",
  say:      "/api/n8n/ai/say",
  speaking: "/api/n8n/speaking",
};

const DEBUG_FETCH = /[?&]debugPost=1\b/.test(window.location.search);

/* ====================== Auto-calibrate policy (added) ====================== */
const MAX_EMPTY_BEFORE_AUTOCAL = 3;     // consecutive empty transcripts before auto-cal
const FAILED_STARTS_BEFORE_AUTOCAL = 4; // short voice bursts that fail to reach listenMs
const AUTO_CAL_MIN_GAP_MS      = 30000; // min gap between auto-cals (cooldown)
const MIN_NONEMPTY_CHARS       = 2;     // treat <2 chars as empty/no-op

/* ====================== Guest ID memory ====================== */
const guestSeqRef = { current: 1 };
const guestMemRef = { current: [] };
const GUEST_TOL   = 0.60;

let guestSavePending = false;
  function scheduleGuestSave() {
    if (guestSavePending) return;
    guestSavePending = true;
    setTimeout(() => {
      saveGuestMem({});
      guestSavePending = false;
    }, 750);
  }

function assignGuestIdFor(descriptor) {
  if (!descriptor || !descriptor.length) {
    const id = `Guest${String(guestSeqRef.current++).padStart(2, "0")}`;
    scheduleGuestSave();
    return id;
  }
  const mem = guestMemRef.current;
  let bestIdx = -1, bestDist = 1;
  for (let i = 0; i < mem.length; i++) {
    const d = faceapi.euclideanDistance(descriptor, mem[i].desc);
    if (d < bestDist) { bestDist = d; bestIdx = i; }
  }

  if (bestIdx >= 0 && bestDist <= GUEST_TOL) {
    mem[bestIdx].ts = Date.now();   // keep fresh for multi-day retention
    scheduleGuestSave();
    return mem[bestIdx].id;
  }

  const id = `Guest${String(guestSeqRef.current++).padStart(2, "0")}`;
  mem.push({ id, ts: Date.now(), desc: Float32Array.from(descriptor) });
  scheduleGuestSave();
  return id;
}

/* ====================== Guest persistence ====================== */
const GUEST_STORE_KEY = "ika:guestMem.v1";
const GUEST_RETENTION_DAYS = 1; // set >1 if you want multi-day retention

const dayKey = (d = new Date()) =>
  d.toLocaleDateString("en-CA", { timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone }); // YYYY-MM-DD

const msToNextMidnight = () => {
  const now = new Date();
  const next = new Date(now);
  next.setHours(24,0,0,0); // local midnight
  return next - now;
};

// quantize [-1,1] float32 descriptor -> Uint8Array (0..255)
function encodeDescFloat32ToU8(descF32) {
  const out = new Uint8Array(descF32.length);
  for (let i=0;i<descF32.length;i++){
    const clamped = Math.max(-1, Math.min(1, descF32[i]));
    out[i] = Math.round((clamped + 1) * 127.5); // -1 -> 0, 1 -> 255
  }
  return out;
}
function decodeDescU8ToFloat32(u8) {
  const out = new Float32Array(u8.length);
  for (let i=0;i<u8.length;i++){
    out[i] = (u8[i] / 127.5) - 1;
  }
  return out;
}
function u8ToB64(u8){ // compact-ish
  let bin = "";
  const CHUNK = 0x8000;
  for (let i=0;i<u8.length;i+=CHUNK){
    bin += String.fromCharCode.apply(null, u8.subarray(i, i+CHUNK));
  }
  return btoa(bin);
}
function b64ToU8(b64){
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i=0;i<bin.length;i++) out[i] = bin.charCodeAt(i);
  return out;
}

const GUEST_MAX = 500; // tune for your venue

function saveGuestMem({
    day = dayKey(),
    seq = guestSeqRef.current,
    mem = guestMemRef.current
  }) {
    try {
      // defensive copy + prune to most-recent GUEST_MAX
      const list = [...mem].sort((a,b)=>(b.ts||0) - (a.ts||0));
      if (list.length > GUEST_MAX) list.length = GUEST_MAX;

      const items = list.map(m => ({
        id: m.id,
        ts: m.ts || Date.now(),
        desc: u8ToB64(encodeDescFloat32ToU8(m.desc))
      }));

      const payload = { day, seq, items, savedAt: Date.now() };
      localStorage.setItem(GUEST_STORE_KEY, JSON.stringify(payload));
    } catch {}
  }

function loadGuestMem() {
  try {
    const raw = localStorage.getItem(GUEST_STORE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (Array.isArray(data.items) && data.items.length > GUEST_MAX) {
      data.items = data.items
        .sort((a,b) => (b.ts||0) - (a.ts||0))
        .slice(0, GUEST_MAX);
    }
    return data;
  } catch { return null; }
}

function pruneByRetention(data) {
  if (!data) return null;
  if (GUEST_RETENTION_DAYS <= 0) return null;

  // daily reset mode
  if (GUEST_RETENTION_DAYS === 1) {
    if (data.day !== dayKey()) return null;  // new day => reset
    return data;
  }

  // multi-day mode
  const cutoff = Date.now() - GUEST_RETENTION_DAYS * 86400_000;
  data.items = (data.items || []).filter(it => (it.ts || 0) >= cutoff);
  return data;
}

/* ====================== helpers ====================== */
const uuid = () =>
  (crypto?.randomUUID ? crypto.randomUUID()
   : Math.random().toString(36).slice(2) + Date.now().toString(36));

const estimateDistanceM = (wPx) => (wPx ? (FOCAL_PX * FACE_WIDTH_M) / wPx : null);
const ageGroupOf = (age) => (age == null ? "unknown"
  : (Math.round(age) >= 18 ? "adult" : (Math.round(age) >= 12 ? "teen" : "child")));
const zoneOf = (d, greenMaxM) => (d != null && d <= greenMaxM ? "green" : "red");
const topExpression = (e) => {
  if (!e) return "neutral";
  const entries = Object.entries(e);
  if (!entries.length) return "neutral";
  return entries.reduce((a, b) => (a[1] > b[1] ? a : b))[0];
};

const shrinkBox = (b, f = BOX_SHRINK) => {
  const w = b.width * f, h = b.height * f;
  return { x: b.x + (b.width - w) / 2, y: b.y + (b.height - h) / 2, width: w, height: h };
};

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

// --- Safe base64 for large blobs (avoid "maximum call stack" on big arrays)
function toBase64(uint8) {
  let bin = "";
  const CHUNK = 0x8000; // 32k
  for (let i = 0; i < uint8.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, uint8.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

// POST with timeout + safe JSON parsing (handles empty bodies)
async function postJSON(url, payload, timeoutMs = 8000) {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", "accept": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
      signal: ctrl.signal,
    });
    const txt = await res.text(); // ← handle empty bodies safely
    let json = {};
    try { json = txt ? JSON.parse(txt) : {}; } catch { /* keep {} */ }
    return { ok: res.ok, status: res.status, text: txt, json };
  } finally {
    clearTimeout(t);
  }
}

function isCamLive() {
  const s = camRef.current?.stream;
  if (!s) return false;
  const tracks = s.getVideoTracks?.() || [];
  if (!tracks.length) return false;
  return tracks.some(t => t.readyState === "live" && t.enabled !== false);
}

/* ====================== TTS hooks ====================== */
async function notifyTTSStart(sessionId, meta = {}) {
  try {
    await fetch("/api/n8n/ai/tts/start", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ sessionId, ...meta })
    });
  } catch {}
}

async function notifyTTSEnd(sessionId, meta = {}) {
  try {
    await fetch("/api/n8n/ai/tts/end", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ sessionId, ...meta })
    });
  } catch {}
}

// Example: play an audio blob (from ElevenLabs or Gemini)
export async function playTTS(sessionId, arrayBuffer, mime="audio/mpeg") {
  const blob = new Blob([arrayBuffer], { type: mime });
  const url  = URL.createObjectURL(blob);

  // we likely don't have the reply text here, so just send meta without text
  await notifyTTSStart(sessionId, { source: "client-audio" });

  const audio = new Audio(url);
  audio.play().catch(()=>{});

  audio.addEventListener("ended", () => {
    notifyTTSEnd(sessionId, { source: "client-audio" });
    URL.revokeObjectURL(url);
  });
}

/* ====================== [AUDIO] tiny meter ====================== */
function LevelMeter({ levelDbfs = -60, thresholdDbfs = -45, bars = 20, height = 18 }) {
  const norm = (db, min=-60, max=-20) => {
    const x = (db - min) / (max - min);
    return Math.min(1, Math.max(0, x));
  };
  const fill = norm(levelDbfs);
  const tpos = norm(thresholdDbfs);
  return (
    <div style={{ position:'relative', background:'#0a0a0a', padding:'10px 12px', borderRadius:10 }}>
      <div style={{ color:'#9ef99f', fontWeight:600, marginBottom:6 }}>Volume</div>
      <div style={{ display:'grid', gridTemplateColumns:`repeat(${bars}, 1fr)`, gap:4 }}>
        {Array.from({length:bars}).map((_,i) => {
          const lit = i / (bars-1) <= fill;
          return <div key={i} style={{ height, borderRadius:4, background: lit ? '#22c55e' : '#184a1d', transition:'background 80ms linear' }}/>;
        })}
      </div>
      <div style={{ position:'absolute', left:`${tpos*100}%`, top:34, transform:'translateX(0%)', display:'flex', alignItems:'center', gap:6, pointerEvents:'none' }}>
        <div style={{ width:6, height:height+10, background:'#9ef99f', borderRadius:2, opacity:.9 }}/>
        <div style={{ color:'#9ef99f', fontSize:12, opacity:.9 }}>{Math.round(thresholdDbfs)} dB</div>
      </div>
    </div>
  );
}

/* ====================== Small UI parts (added) ====================== */
function ToggleSwitch({ checked, onChange, label }) {
  const onClr = '#0ea5e9'; // blue
  const offClr = '#475569'; // gray
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={()=>onChange(!checked)}
      title={label}
      style={{
        display:'inline-flex', alignItems:'center', gap:8,
        background:'transparent', border:'none', cursor:'pointer', padding:0
      }}
    >
      <span style={{ color:'#ebebeb', fontSize:13 }}>{label}</span>
      <span
        aria-hidden
        style={{
          width:44, height:24, borderRadius:999,
          background: checked ? onClr : offClr,
          position:'relative', transition:'background .15s ease'
        }}
      >
        <span style={{
          position:'absolute', top:3, left: checked ? 24 : 3,
          width:18, height:18, borderRadius:'50%', background:'#fff',
          transition:'left .15s ease'
        }}/>
      </span>
    </button>
  );
}

/* ====================== App ====================== */
export default function App(){
  const speakingRef = useRef(false);
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const lastSnapRef = useRef("");
  const lastFrameTsRef = useRef(0);

  const MIC_STOP_GRACE_MS = 5000; // 3–5 seconds
  const lastAllGreenRef = useRef(0);

  const userMicOffRef = useRef(false);

  const [ready, setReady] = useState(false);
  const [backend, setBackend] = useState("cpu");

  // Green zone distance (meters) — persisted
  const [greenMaxM, setGreenMaxM] = useState(DEFAULT_GREEN_MAX_M);
  const [videoId, setVideoId] = useState("");
  const greenMaxMRef = useRef(greenMaxM);
  useEffect(() => { greenMaxMRef.current = greenMaxM; }, [greenMaxM]);

  // choose input size based on current video width
  const pickInputSize = (w) => (w >= 1920 ? 512 : w >= 1280 ? 416 : 320);

  // detector options as a ref so we can tweak inputSize later
  const tinyOptsRef = useRef(new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.3 }));

  // one-time TFJS backend fallback flag
  const triedFallbackRef = useRef(false);

  /* === Auto-detect state & refs (moved up) === */
  const [autoDetectOn, setAutoDetectOn] = useState(true);
  const autoDetectOnRef = useRef(autoDetectOn);
  useEffect(() => { autoDetectOnRef.current = autoDetectOn; }, [autoDetectOn]);

  const [emptyTranscripts, setEmptyTranscripts] = useState(0);
  const lastAutoCalRef = useRef(0);
  const [isPressed, setIsPressed] = useState(false); // press-blue for Calibrate

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;

    const updateOpts = () => {
      const w = v.videoWidth || 1280;
      tinyOptsRef.current = new faceapi.TinyFaceDetectorOptions({
        inputSize: pickInputSize(w),
        scoreThreshold: 0.3,
      });
    };

    v.addEventListener("loadedmetadata", updateOpts);
    v.addEventListener("resize", updateOpts);
    updateOpts();

    return () => {
      v.removeEventListener("loadedmetadata", updateOpts);
      v.removeEventListener("resize", updateOpts);
    };
  }, [ready, videoId]);

  // when camera changes, load that camera’s greenMaxM
  useEffect(() => {
    try {
      const gm = localStorage.getItem(`ika:greenMaxM:${videoId || 'default'}`);
      if (gm != null) {
        const v = parseFloat(gm);
        if (Number.isFinite(v)) setGreenMaxM(Math.min(2.0, Math.max(0.3, v)));
      }
    } catch {}
  }, [videoId]);

  // init: backend, models, camera
  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        // 1) TFJS backend selection
        //    Set paths BEFORE selecting the wasm backend
        setWasmPaths("/tfjs-backend-wasm/");

        const tryBackend = async (name) => {
          try {
            await tf.setBackend(name);
            await tf.ready();
            return tf.getBackend() === name;
          } catch {
            return false;
          }
        };

        // Prefer GPU → WASM → CPU
        let ok = await tryBackend("webgl");
        if (!ok) ok = await tryBackend("wasm");
        if (!ok) await tryBackend("cpu");

        if (!cancelled) setBackend(tf.getBackend()); // "webgl" | "wasm" | "cpu"

        // 2) Load face-api models
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
          faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        ]);

        // 3) Camera + ready
        await startCamera();

        // warm-up: one tiny pass so first real frame isn't JIT-laggy
        try {
          const off = document.createElement("canvas");
          off.width = 128; off.height = 128;
          await faceapi.detectAllFaces(
            off,
            new faceapi.TinyFaceDetectorOptions({ inputSize: 128, scoreThreshold: 0.5 })
          );
        } catch {}

        // kick an OPTIONS preflight early (non-fatal if it fails)
        try { fetch(N8N.stt, { method: "OPTIONS" }).catch(() => {}); } catch {}

        if (!cancelled) setReady(true);
      } catch (e) {
        console.error("[init]", e);
        if (!cancelled) setBackend(tf.getBackend?.() || "cpu");
      }
    })();

    // cleanup
    return () => {
      cancelled = true;
      userMicOffRef.current = true;
      stopAll({ reset: true, reason: "unmount" });
    };
  }, []);

  // Load labels and build FaceMatcher
  useEffect(() => {
    if (!ready) return;

    let cancelled = false;

    (async () => {
      try {
        const res = await fetch(LABELS_URL, { cache: "no-store" });
        const data = await res.json();

        // Accept a few common shapes:
        // 1) [{ label, descriptors: [ [..128], [..128], ... ] }]
        // 2) { "Alice": [ [..128], [..128] ], "Bob": [ [..128] ] }
        // 3) [{ label, descriptors_b64: ["...","..."] }]  // Uint8 -> base64 (0..255) using your quantizer

        const entries = Array.isArray(data)
          ? data
          : Object.entries(data).map(([label, descriptors]) => ({ label, descriptors }));

        const labeled = await Promise.all(
          entries.map(async (e) => {
            let descs = [];

            if (Array.isArray(e.descriptors_b64)) {
              // quantized uint8 → Float32 using your helpers
              descs = e.descriptors_b64.map((b64) => decodeDescU8ToFloat32(b64ToU8(b64)));
            } else if (Array.isArray(e.descriptors)) {
              // arrays of numbers
              descs = e.descriptors.map((arr) =>
                arr instanceof Float32Array ? arr : new Float32Array(arr)
              );
            } else {
              descs = [];
            }

            // filter bad lengths; face-api expects 128
            descs = descs.filter((d) => d && d.length === 128);

            return new faceapi.LabeledFaceDescriptors(e.label, descs);
          })
        );

        // Only keep labels that have at least one descriptor
        const usable = labeled.filter((l) => l.descriptors?.length);
        const matcher = new faceapi.FaceMatcher(usable, MATCH_THRESHOLD);

        if (!cancelled) {
          faceMatcherRef.current = matcher;
          setKnownCount(usable.reduce((acc, l) => acc + l.descriptors.length, 0));
        }
      } catch (err) {
        console.warn("[labels] failed to load/build matcher:", err);
        if (!cancelled) {
          faceMatcherRef.current = null;
          setKnownCount(0);
        }
      }
    })();

    return () => { cancelled = true; };
  }, [ready]);

  // ------------------------------ STT result hook ------------------------------
  async function handleTranscriptResult(resp) {
    const txt = (resp?.text ?? "").trim();
    if (!txt || txt.length < MIN_NONEMPTY_CHARS) {
      setEmptyTranscripts(c => {
        const next = c + 1;
        if (autoDetectOn && next >= MAX_EMPTY_BEFORE_AUTOCAL) {
          setTimeout(() => setEmptyTranscripts(0), 0);
          maybeAutoCalibrate("consecutive-empty");
        }
        return next;
      });
      return;
    }

    setEmptyTranscripts(0);

    try {
      await fetch(N8N.say, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          sessionId: S.current.id || "default",
          text: txt
        })
      });
    } catch {}
  }

  // Stable autoCalibrate (unchanged)
  const autoCalibrate = useCallback(async () => {
    const a = audioRef.current;
    if (!a?.analyser || a.analyser.fftSize <= 0) return;

    const analyser = a.analyser;
    const origSmoothing = analyser.smoothingTimeConstant ?? 0;
    analyser.smoothingTimeConstant = 0;

    const buf = new Float32Array(analyser.fftSize || 2048);
    const start = performance.now();
    const windowMs = 2000;
    const readings = [];

    while (performance.now() - start < windowMs) {
      analyser.getFloatTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
      const rms = Math.sqrt(sum / buf.length) || 0.0000001;
      let d = 20 * Math.log10(rms);
      if (!isFinite(d)) d = -120;
      readings.push(d);
      await new Promise((r) => requestAnimationFrame(r));
    }

    readings.sort((x, y) => x - y);
    const keep = Math.max(1, Math.floor(readings.length * 0.35));
    const lowSlice = readings.slice(0, keep);
    const mid = lowSlice[Math.floor(lowSlice.length / 2)] ?? -60;
    const target = Math.max(-60, Math.min(-20, mid + 3));
    setThreshold(Math.round(target));

    analyser.smoothingTimeConstant = origSmoothing;
  }, []);

  // Track previous value so we only react to a true toggle (false -> true)
  const prevAutoDetectOn = useRef(autoDetectOn);

  // If user turns Auto-calibrate ON while mic is already running, run one calibration immediately
  useEffect(() => {
    const was = prevAutoDetectOn.current;
    prevAutoDetectOn.current = autoDetectOn; // update for next run

    // Only when toggled OFF -> ON (not on first mount)
    if (!was && autoDetectOn && micOnRef.current && audioRef.current?.analyser && !speakingRef.current) {
      // Also skip if we just auto-calibrated very recently (cooldown)
      if (Date.now() - (lastAutoCalRef.current || 0) < 1000) return;

      (async () => {
        setIsPressed(true);
        try {
          await autoCalibrate();
          lastAutoCalRef.current = Date.now();
        } finally {
          setTimeout(() => setIsPressed(false), 250);
        }
      })();
    }
  }, [autoDetectOn, autoCalibrate]);

  async function maybeAutoCalibrate(reason = "auto"){
    if (!autoDetectOnRef.current) return;
    if (!micOnRef.current) return;
    const isRecording = !!audioRef.current?.vad?.recording;
    if (isRecording) return;
    const now = Date.now();
    if (now - lastAutoCalRef.current < AUTO_CAL_MIN_GAP_MS) return;
    lastAutoCalRef.current = now;
    setIsPressed(true);
    try { await autoCalibrate(); } finally { setTimeout(()=>setIsPressed(false), 250); }
  }

  const [sessionStatus, setSessionStatus] = useState("IDLE");
  const [sessionId, setSessionId] = useState(null);
  const [posts, setPosts] = useState({ start:0, snapshot:0, stop:0 });
  const [lastSent, setLastSent] = useState({ start:"-", snapshot:"-", stop:"-" });
  const [lastHttp, setLastHttp] = useState({ start:"", snapshot:"", stop:"" });
  const [totals, setTotals] = useState({ all:0, green:0, red:0 });

  /* ---------- Devices (audio + video) ---------- */
  const [micOn, setMicOn] = useState(false);
  const micOnRef = useRef(false);
  useEffect(() => { micOnRef.current = micOn; }, [micOn]);

  const [dbfs, setDbfs] = useState(-60);
  const [threshold, setThreshold] = useState(-45);
  const [listenMs, setListenMs]   = useState(400);
  const [silenceMs, setSilenceMs] = useState(500);

  const [audioDevs, setAudioDevs] = useState([]);
  const [videoDevs, setVideoDevs] = useState([]);
  const [audioId, setAudioId] = useState("");

  const audioRef = useRef({
    ctx:null, stream:null, source:null, analyser:null, raf:0,
    vad: { highSince:0, lowSince:0, recording:false, recorder:null, chunks:[], startTs:0, failedStarts:0 }
  });
  const camRef   = useRef({ stream:null });

  // Auto-control mic based on session status
  useEffect(() => {
    if (!ready) return;

    if (sessionStatus === "ACTIVE") {
      if (!micOn && !userMicOffRef.current) startMic().catch(() => {});
    } else {
      if (micOn)  stopMic().catch(() => {});
    }
  }, [sessionStatus, ready, micOn]);

  const [table, setTable] = useState(
    Array.from({length:5}, (_,i)=>({ idx:i+1, gender:"-", ageGroup:"-", zone:"-", name:"-", distance:"-" }))
  );

  const faceMatcherRef = useRef(null);
  const [knownCount, setKnownCount] = useState(0);

  const recentMapRef = useRef({});
  // Session state (ID, frame counters, timers)
  const S = useRef({ id: null, seenFrames: 0, lastFaceTs: 0, lastSnapshotTs: 0 });

  // --- keyboard nudges for green zone distance ---
    useEffect(() => {
      const onKey = (e) => {
        const tag = (document.activeElement?.tagName || "").toUpperCase();
        if (["INPUT","TEXTAREA","SELECT"].includes(tag)) return;
        if (e.ctrlKey || e.metaKey || e.altKey) return;

        if (e.key === '[') {
          setGreenMaxM(v => Math.max(0.3, +(v - 0.05).toFixed(2)));
        } else if (e.key === ']') {
          setGreenMaxM(v => Math.min(2.0, +(v + 0.05).toFixed(2)));
        }
      };
      window.addEventListener('keydown', onKey);
      return () => window.removeEventListener('keydown', onKey);
    }, []);

  // live refs so tick sees latest slider values (prevents stale closures in tick)
  const thresholdRef = useRef(threshold);
  const listenMsRef  = useRef(listenMs);
  const silenceMsRef = useRef(silenceMs);

  useEffect(() => { thresholdRef.current = threshold; }, [threshold]);
  useEffect(() => { listenMsRef.current  = listenMs;  }, [listenMs]);
  useEffect(() => { silenceMsRef.current = silenceMs; }, [silenceMs]);

  useEffect(() => {
    try { localStorage.setItem(`ika:greenMaxM:${videoId || 'default'}`, String(greenMaxM)); } catch {}
  }, [greenMaxM, videoId]);

  useEffect(() => {
  // Restore guest memory
  const data = pruneByRetention(loadGuestMem());
  if (data && Array.isArray(data.items)) {
    guestSeqRef.current = Math.max(1, Number(data.seq) || 1);
    guestMemRef.current = data.items.map(it => ({
      id: it.id,
      ts: it.ts || Date.now(),
      desc: decodeDescU8ToFloat32(b64ToU8(it.desc))
    }));
  } else {
    guestSeqRef.current = 1;
    guestMemRef.current = [];
    saveGuestMem({}); // write fresh day
  }

  // Schedule automatic reset at local midnight if daily mode
  if (GUEST_RETENTION_DAYS === 1) {
    const t = setTimeout(() => {
      guestSeqRef.current = 1;
      guestMemRef.current = [];
      saveGuestMem({ day: dayKey(), seq: 1, mem: [] });
    }, msToNextMidnight());
    return () => clearTimeout(t);
  }
}, []);

  /* ====================== tap to enable audio ====================== */
const [audioUnlocked, setAudioUnlocked] = useState(false);
      useEffect(() => {
        if (audioUnlocked) return;
        const unlock = async () => {
          try {
            const ctx = new (window.AudioContext || window.webkitAudioContext)({ latencyHint: "interactive" });
            await ctx.resume();  // gesture-initiated
            await ctx.close();
            setAudioUnlocked(true);
            window.removeEventListener("touchend", unlock);
            window.removeEventListener("click", unlock);
          } catch {}
        };
        window.addEventListener("touchend", unlock, { once:true });
        window.addEventListener("click",    unlock, { once:true });
        return () => {
          window.removeEventListener("touchend", unlock);
          window.removeEventListener("click", unlock);
        };
      }, [audioUnlocked]);

  // init: load prefs once, using the *stored* videoId to pick per-camera greenMaxM
  useEffect(() => {
    (async () => {
      try {
        const t   = localStorage.getItem("ika:threshold");
        const vId = localStorage.getItem("ika:videoId") || ""; // read first
        const gm  = (
          localStorage.getItem(`ika:greenMaxM:${vId || 'default'}`) ??
          localStorage.getItem("ika:greenMaxM") // legacy fallback
        );
        const l   = localStorage.getItem("ika:listenMs");
        const s   = localStorage.getItem("ika:silenceMs");
        const a   = localStorage.getItem("ika:audioId");
        const aut = localStorage.getItem("ika:autoDetectOn");

        if (t !== null) setThreshold(Number(t));
        if (gm !== null) {
          const val = parseFloat(gm);
          if (Number.isFinite(val)) setGreenMaxM(Math.min(2.0, Math.max(0.3, val)));
        }
        if (l !== null) setListenMs(Number(l));
        if (s !== null) setSilenceMs(Number(s));
        if (a) setAudioId(a);
        // set videoId *after* we used vId to load per-camera greenMaxM
        if (vId) setVideoId(vId);
        if (aut != null) setAutoDetectOn(aut === "true");
      } catch {}

      try {
        const list = await navigator.mediaDevices.enumerateDevices();
        setAudioDevs(list.filter(d => d.kind === "audioinput"));
        setVideoDevs(list.filter(d => d.kind === "videoinput"));
      } catch {}
    })();
  }, []);

  // React to device hot-plug
  useEffect(() => {
    const onChange = async () => {
      try {
        const list = await navigator.mediaDevices.enumerateDevices();
        setAudioDevs(list.filter(d => d.kind === "audioinput"));
        setVideoDevs(list.filter(d => d.kind === "videoinput"));
      } catch {}
    };

    if (navigator.mediaDevices && typeof navigator.mediaDevices.addEventListener === "function") {
      navigator.mediaDevices.addEventListener("devicechange", onChange);
    }

    return () => {
      if (navigator.mediaDevices && typeof navigator.mediaDevices.removeEventListener === "function") {
        navigator.mediaDevices.removeEventListener("devicechange", onChange);
      }
    };
  }, []);

  // --- POLL IKA "speaking" (read-only; VAD uses this to pause) ---
  useEffect(() => {
  if (!sessionId) return;
  let alive = true;
  let timer = 0;

  const INTERVAL_ON_MS  = 900;
  const INTERVAL_OFF_MS = 2500;
  const jitter = () => 40 + Math.floor(Math.random() * 120);

  const tick = async () => {
    if (!alive) return;
    const everyMs = (micOn ? INTERVAL_ON_MS : INTERVAL_OFF_MS) + jitter();

    try {
      const url = `${N8N.speaking}?sessionId=${encodeURIComponent(sessionId)}&t=${Date.now()}`;
      const r = await fetch(url, { cache: "no-store" });
      const j = await r.json().catch(() => ({}));
      if (!alive) return;
      speakingRef.current = !!j.speaking;
    } catch {
      speakingRef.current = false;
    }

    if (alive) timer = window.setTimeout(tick, everyMs);
  };

  timer = window.setTimeout(tick, 100);
  return () => { alive = false; window.clearTimeout(timer); };
}, [sessionId, micOn]);

  // Persist prefs
  useEffect(()=>{ try{ localStorage.setItem("ika:threshold", String(threshold)); }catch{} },[threshold]);
  useEffect(()=>{ try{ localStorage.setItem("ika:listenMs", String(listenMs)); }catch{} },[listenMs]);
  useEffect(()=>{ try{ localStorage.setItem("ika:silenceMs", String(silenceMs)); }catch{} },[silenceMs]);
  useEffect(()=>{ try{ localStorage.setItem("ika:audioId", audioId||""); }catch{} },[audioId]);
  useEffect(()=>{ try{ localStorage.setItem("ika:videoId", videoId||""); }catch{} },[videoId]);
  useEffect(()=>{ try{ localStorage.setItem("ika:autoDetectOn", String(autoDetectOn)); }catch{} },[autoDetectOn]);

  async function stopMic() {
    // If we’re already stopped, just ensure UI is quiet and bail
    if (!audioRef.current?.stream && !micOn) {
      setMicOn(false);
      setDbfs(-60);
      return;
    }

    try {
      if (audioRef.current.raf) cancelAnimationFrame(audioRef.current.raf);
    } catch {}

    try {
      const rec = audioRef.current?.vad?.recorder;
      if (rec && rec.state === "recording") {
        try { rec.requestData?.(); } catch {}
        try { rec.stop(); } catch {}
      }
    } catch {}

    try { audioRef.current.stream?.getTracks()?.forEach(t => t.stop()); } catch {}
    try { await audioRef.current.ctx?.close(); } catch {}
    try { audioRef.current.source?.disconnect?.(); } catch {}
    try { audioRef.current.analyser?.disconnect?.(); } catch {}

    // Help GC + make sure the next cancelAnimationFrame wouldn't miss
    audioRef.current.source   = null;
    audioRef.current.analyser = null;

    audioRef.current = {
      ctx: null,
      stream: null,
      source: null,
      analyser: null,
      raf: 0,
      deviceId: "",
      vad: { highSince:0, lowSince:0, recording:false, recorder:null, chunks:[], startTs:0, failedStarts:0 }
    };

    setMicOn(false);
    setDbfs(-60);
  }

  async function startMic(id = audioId, { force = false } = {}) {
  // if already running with same device, skip; otherwise restart
  if (audioRef.current?.stream) {
    const same = (id || "") === (audioRef.current.deviceId || "");
    if (!force && same) {
      console.log("[Mic] already running on selected device");
      return;
    }
    await stopMic();
  }

  // get mic
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        deviceId: id ? { exact: id } : undefined,
        noiseSuppression: true,
        echoCancellation: true,
        autoGainControl: true,
        channelCount: 1,
        sampleRate: 48000,
        sampleSize: 16,
      },
      video: false
    });
  } catch (e) {
    console.error("[Mic] permission/error:", e);
    setMicOn(false);
    return; // bail if user blocks mic or no device
  }

  // wire up audio graph
  const ctx = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: 48000,
    latencyHint: "interactive"
 });
  try { await ctx.resume(); } catch {}
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 512; // still stable, half the work
  source.connect(analyser);

  // store refs
  audioRef.current = {
    ctx,
    stream,
    source,
    analyser,
    raf: 0,
    deviceId: id || "",
    vad: {
      highSince: 0, lowSince: 0, recording: false,
      recorder: null, chunks: [], startTs: 0, failedStarts: 0
    }
  };
  setMicOn(true);
  // Auto-calibrate once on mic start if enabled
  if (autoDetectOnRef.current && !speakingRef.current) {
    setTimeout(async () => {
      // double-check stream/analyser still exist
      if (!audioRef.current?.stream || !audioRef.current?.analyser) return;
      setIsPressed(true);
      try {
        await autoCalibrate();
        lastAutoCalRef.current = Date.now();
      } finally {
        setTimeout(() => setIsPressed(false), 250);
      }
    }, 350);
  }

  // choose recorder MIME
  let mimeType = "";
  const ua = navigator.userAgent || "";
  const isiOS = /iPad|iPhone|iPod/.test(ua) || (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  const isSafari = /^((?!chrome|android).)*safari/i.test(ua);

  if (typeof MediaRecorder !== "undefined") {
    
    if (!isiOS && MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
      mimeType = "audio/webm;codecs=opus";
    } else if (!isiOS && MediaRecorder.isTypeSupported("audio/webm")) {
      mimeType = "audio/webm";
    } else if (!isiOS && MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
      mimeType = "audio/ogg;codecs=opus";
    } else if ((isiOS || isSafari) && MediaRecorder.isTypeSupported("audio/mp4")) {
      // iOS Safari tends to only allow MP4/AAC
      mimeType = "audio/mp4";
    } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
      mimeType = "audio/mp4";
    } else {
      // last resort: let the browser pick; it may still work
      mimeType = "";
    }
  }

  // VAD buffers and smoothing
  const buf = new Float32Array(analyser.fftSize);
  const EMA = 0.25;
  let smooth = -60;

  const tick = () => {
    analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length) || 0.0000001;
    let d = 20 * Math.log10(rms);
    if (!isFinite(d)) d = -120;

    // UI smoothing
    smooth = smooth === -60 ? d : (EMA * d + (1 - EMA) * smooth);
    setDbfs(smooth);

    const vad   = audioRef.current.vad;
    const nowTs = performance.now();

    // live refs (avoid stale closure)
    const thr        = thresholdRef.current;
    const minKeepMs  = Math.max(400, listenMsRef.current);
    const silenceDur = silenceMsRef.current;

    // choose raw for edges; use 'smooth' if you prefer extra inertia
    const edgeLevel = d;
    const isLoud    = edgeLevel >= thr;

    // pause VAD while bot is speaking
    if (speakingRef.current) {
      if (vad.recording) {
        const rec = vad.recorder;
        if (rec && rec.state === "recording") {
          try { rec.requestData?.(); } catch {}
          try { rec.stop(); } catch {}
        }
        vad.recording = false;
        vad.recorder  = null;
        vad.chunks    = [];
        vad.startTs   = 0;
      }
      audioRef.current.raf = requestAnimationFrame(tick);
      return;
    }

    // hysteresis
    const HYSTERESIS_DB = 2;
    const QUICK_STOP_DB = 8;      // stop if we fall this far below thr
    const QUICK_STOP_MS = 220;    // ...for at least this long
    const riseEdge = edgeLevel >=  thr;
    const fallEdge = edgeLevel <  (thr - HYSTERESIS_DB);
    if (isLoud && !vad.highSince) { vad.highSince = nowTs; vad.lowSince = 0; }

    // START
    if (riseEdge && !vad.recording) {
      try {
        if (typeof MediaRecorder !== "undefined") {
          const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
          vad.recorder  = rec;
          vad.chunks    = [];
          vad.startTs   = nowTs;
          vad.recording = true;
          vad.lowSince  = 0;

          rec.ondataavailable = (e) => { if (e.data?.size) vad.chunks.push(e.data); };

          rec.onstop = async () => {
            try {
              const blob = new Blob(vad.chunks, { type: rec.mimeType || mimeType || "audio/webm" });
              const dur  = performance.now() - vad.startTs;

              if (dur < minKeepMs) {
                vad.failedStarts = (vad.failedStarts || 0) + 1;
                if (autoDetectOnRef.current && vad.failedStarts >= FAILED_STARTS_BEFORE_AUTOCAL) {
                  vad.failedStarts = 0;
                  maybeAutoCalibrate("failed-starts");
                }
                return;
              }
              vad.failedStarts = 0;

              const fd = new FormData();
              fd.append("sessionId", S.current.id || "default");
              fd.append("language", "en-US");
              fd.append("thresholdDb", String(Math.round(thr)));
              fd.append("listenMs", String(listenMsRef.current));
              fd.append("silenceMs", String(silenceMsRef.current));
              fd.append("audio", blob, `utterance.${blob.type.includes("webm") ? "webm" : "mp4"}`);
              const res = await fetch(N8N.stt, { method: "POST", body: fd, keepalive: true, cache: "no-store" });
              const json = await res.json().catch(()=> ({}));

              if (json && typeof json === "object") await handleTranscriptResult(json);
            } catch (e) {
              console.error("[STT] onstop error:", e);
            } finally {
              vad.recording = false;
              vad.recorder  = null;
              vad.chunks    = [];
              vad.startTs   = 0;
            }
          };

          try { rec.start(250); } catch { rec.start(); }
        } else {
          // No recorder; treat as non-fatal and don't attempt to start
          console.warn("[VAD] MediaRecorder not supported");
          audioRef.current.raf = requestAnimationFrame(tick);
          return;
        }
      } catch (e) {
        console.error("[VAD] start failed:", e);
      }
    }

    // STOP
    if (vad.recording) {
      const utterMs = nowTs - vad.startTs;

      if (fallEdge) {
       vad.lowSince = vad.lowSince || nowTs;
       const lowDur = nowTs - vad.lowSince;
       if (lowDur >= silenceDur || (edgeLevel < (thr - QUICK_STOP_DB) && lowDur >= QUICK_STOP_MS)) {
          const rec = vad.recorder;
          if (rec && rec.state === "recording") {
            try { rec.requestData?.(); } catch {}
            try { rec.stop(); } catch {}
          }
        }
      } else {
        vad.lowSince = 0;
      }

      const MAX_UTTER_MS = 15000;
      if (utterMs >= MAX_UTTER_MS) {
        const rec = vad.recorder;
        if (rec && rec.state === "recording") {
          try { rec.requestData?.(); } catch {}
          try { rec.stop(); } catch {}
        }
      }
    } else {
      // keep edge timers tidy when idle
      if (!isLoud) {
        vad.lowSince  = vad.lowSince || nowTs;
        vad.highSince = 0;
      } else {
        if (!vad.highSince) vad.highSince = nowTs;
        vad.lowSince = 0;
      }
    }

    // schedule next frame (only here)
    audioRef.current.raf = requestAnimationFrame(tick);
  };

  // kick the loop exactly once
  audioRef.current.raf = requestAnimationFrame(tick);
  }

  // Stop mic + camera + end session + reset UI and (optionally) POST /camera/stop
  async function stopAll({ reset = true, reason = "nav", skipPost = false } = {}) {
    // Freeze UI immediately even if stopMic throws or is delayed
    setDbfs(-60);
    try { await stopMic(); } catch {}
    try { camRef.current.stream?.getTracks()?.forEach(t => t.stop()); } catch {}
    
    camRef.current.stream = null;

    const active = !!S.current.id;
    const sid = S.current.id;

    if (active && !skipPost) {
      try { await post("stop", { sessionId: sid, ts: Date.now(), reset, reason }); } catch {}
    }

    speakingRef.current = false;
    S.current = { id:null, seenFrames:0, lastFaceTs:0, lastSnapshotTs:0 };
    setSessionId(null);
    setSessionStatus("IDLE");
    recentMapRef.current = {};
    lastSnapRef.current = "";
  }

  /* ------------------------------ Expose hooks for n8n / UE ------------------------------ */

  // Effect that installs the global + listener
  useEffect(() => {
    window.ikaCalibrate = async () => {
      setIsPressed(true);
      try { await autoCalibrate(); }
      finally { setTimeout(() => setIsPressed(false), 250); }
    };

    const onMsg = (e) => {
      const m = e?.data;
      if (!m) return;
      if (m.type === "CALIBRATE") window.ikaCalibrate();
      if (m.type === "SET_AUTODETECT") setAutoDetectOn(!!m.value);
    };
    window.addEventListener("message", onMsg);

    return () => {
      window.removeEventListener("message", onMsg);
      try { delete window.ikaCalibrate; } catch {}
    };
  }, [autoCalibrate]);

    useEffect(() => {
      const onVisibility = () => {
        if (document.hidden) {
          userMicOffRef.current = true;
          stopAll({ reset: true, reason: "visibility" });
        }
      };

      const onPageHide = () => {
        userMicOffRef.current = true;
        stopAll({ reset: true, reason: "pagehide" });
      };

      const onBeforeUnload = () => {
        try {
          const rec = audioRef.current?.vad?.recorder;
          if (rec && rec.state === "recording") rec.stop();
        } catch {}
        try {
          if (audioRef.current.raf) cancelAnimationFrame(audioRef.current.raf);
        } catch {}
      };

      document.addEventListener("visibilitychange", onVisibility);
      window.addEventListener("pagehide", onPageHide);
      window.addEventListener("beforeunload", onBeforeUnload);

      return () => {
        document.removeEventListener("visibilitychange", onVisibility);
        window.removeEventListener("pagehide", onPageHide);
        window.removeEventListener("beforeunload", onBeforeUnload);
      };
    }, []);

  async function startCamera(id = videoId) {
    // Stop previous cam tracks
    try { camRef.current.stream?.getTracks()?.forEach(t => t.stop()); } catch {}

    let stream = null;

    // Try specific camera first
    if (id) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: id }, width:{ ideal:1920 }, height:{ ideal:1080 }, frameRate:{ ideal:30 } },
          audio: false
        });
      } catch (e) {
        console.warn("[Cam] chosen device failed; falling back to default:", e.name);
        if (e.name === "OverconstrainedError" || e.name === "NotFoundError") {
          setVideoId("");
          try { localStorage.setItem("ika:videoId", ""); } catch {}
        }
      }
    }

    // Fallback to default camera
    if (!stream) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode:"user", width:{ ideal:1920 }, height:{ ideal:1080 }, frameRate:{ ideal:30 } },
          audio: false
        });
      } catch (e) {
        console.error("[Cam] default device failed:", e);
        alert("Camera not available or permission denied.");
        return;
      }
    }

    camRef.current.stream = stream;

    // Tie mic to camera lifecycle: if cam pauses/ends, kill the mic
    try {
      const onCamGone = () => {
        userMicOffRef.current = true;            // behave like Stop
        stopMic().catch(()=>{});
      }; // ← CLOSE the function

      const vTracks = stream.getVideoTracks();
      vTracks.forEach(t => {
        t.addEventListener("ended", onCamGone);
        t.addEventListener("mute", onCamGone);
        t.onended = onCamGone;
      });

      if (typeof stream.addEventListener === "function") {
        stream.addEventListener("inactive", onCamGone);
      }
    } catch {} // ← CLOSE the try

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        const w = videoRef.current.videoWidth || 1280;
        FOCAL_PX = w >= 1920 ? 1350 : 900;

        tinyOptsRef.current = new faceapi.TinyFaceDetectorOptions({
          inputSize: pickInputSize(w),
          scoreThreshold: 0.3,
        });
      };
    }

    // Refresh device lists after permission
    try {
      const list = await navigator.mediaDevices.enumerateDevices();
      setAudioDevs(list.filter(d => d.kind === "audioinput"));
      setVideoDevs(list.filter(d => d.kind === "videoinput"));
    } catch {}
  }

  /* ------------------------------ loop ------------------------------ */
  useEffect(() => {
    if (!ready) return;

    const video  = videoRef.current;
    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");

    const resize = () => {
      if (!video) return;
      canvas.width  = video.videoWidth  || 940;
      canvas.height = video.videoHeight || 650;
    };
    video.addEventListener("loadedmetadata", resize);
    video.addEventListener("resize", resize);
    resize();

    let raf = 0;
    let lastRun = 0;
    let detecting = false; // <-- guard

    let frameCount = 0;
    const loop = () => {
      raf = requestAnimationFrame(loop);

      const now = performance.now();
      if (now - lastRun < LOOP_STEP_MS || !video.videoWidth || detecting) return;
      frameCount++;
      if (audioRef.current?.vad?.recording && (frameCount % 2 === 0)) return; // half-rate during speech
      lastRun = now;

      detecting = true;
      (async () => {
        lastFrameTsRef.current = performance.now();

        // … then inside the loop:
        let dets = [];
        try {
          dets = await faceapi
            .detectAllFaces(video, tinyOptsRef.current)
            .withFaceLandmarks()
            .withFaceExpressions()
            .withAgeAndGender()
            .withFaceDescriptors();
        } catch (e) {
          console.warn("faceapi detect error:", e);

          // one-time fallback from webgl → wasm if webgl misbehaves
          if (!triedFallbackRef.current && tf.getBackend?.() === "webgl") {
            triedFallbackRef.current = true;
            try {
              await tf.setBackend("wasm");
              await tf.ready();
              setBackend(tf.getBackend()); // shows 'wasm' in your status panel
              console.log("[tfjs] fell back to WASM backend");
            } catch {}
          }

          detecting = false; // release and bail from this frame
          return;
        }

        ctx.clearRect(0,0,canvas.width,canvas.height);

        // set common font + baseline once per frame
        ctx.font = LABEL_FONT;
        ctx.textBaseline = "top";

        const resized = faceapi
          .resizeResults(dets, { width: canvas.width, height: canvas.height })
          .sort((a,b)=>a.detection.box.x - b.detection.box.x);

        const matcher = faceMatcherRef.current;
        const rows = [];
        const peopleForPost = [];
        let total=0, green=0, red=0;

        const tracks = recentMapRef.current;

        for (let i = 0; i < resized.length; i++) {
          const det  = resized[i];
          const box  = det.detection.box;
          const dist = estimateDistanceM(box.width);
          const zone = zoneOf(dist, greenMaxMRef.current);
          const color = zone === "green" ? "#22c55e" : "#ef4444";
          const gender = (det.gender || "").toLowerCase();
          const expr = topExpression(det.expressions);

          let name = null;
          if (matcher && det.descriptor) {
            const best = matcher.findBestMatch(det.descriptor);
            if (best && best.label !== "unknown" && best.distance <= MATCH_THRESHOLD) {
              name = best.label;
            } else if (best && best.label !== "unknown" && best.distance <= (MATCH_THRESHOLD + 0.03)) {
              const { best: b, second: s } = bestTwoMatches(matcher, det.descriptor);
              if (b.label && b.label !== "unknown" && (s.dist - b.dist) >= MATCH_MARGIN) {
                name = b.label;
              }
            }
          }

          let guestId = null;
          if (!name) guestId = assignGuestIdFor(det.descriptor);

          let displayName = name || guestId || "Guest";
          const t = tracks[i];
          if (t && t.name !== displayName) {
            if ((t.count || 0) < STABILIZE_FRAMES) {
              displayName = t.name;
              t.count = (t.count || 0) + 1;
            } else {
              tracks[i] = { name: displayName, count: 0 };
            }
          } else {
            tracks[i] = { name: displayName, count: 0 };
          }

          const dbox = shrinkBox(box);
          ctx.strokeStyle = color;
          ctx.lineWidth = BOX_LINE_WIDTH;
          ctx.strokeRect(dbox.x, dbox.y, dbox.width, dbox.height);

          const label = `${displayName} • ${zone} • ${Math.max(0,Math.round(det.age))} ${gender} • ${expr}`;

          const tw = ctx.measureText(label).width + LABEL_PAD_X * 2;
          const th = 18 + LABEL_PAD_Y * 2;
          const lx = Math.max(0, Math.min(dbox.x, canvas.width - tw));
          const ly = Math.max(0, dbox.y - th - 4);
          ctx.fillStyle = color; ctx.fillRect(lx, ly, tw, th);
          ctx.fillStyle = "#fff"; ctx.fillText(label, lx + LABEL_PAD_X, ly + LABEL_PAD_Y);

          if (rows.length < 5){
            rows.push({
              idx: rows.length+1,
              gender,
              ageGroup: ageGroupOf(det.age),
              zone,
              name: displayName,
              distance: dist ? dist.toFixed(2)+" m" : "-",
            });
          }

          peopleForPost.push({
            gender,
            ageGroup: ageGroupOf(det.age),
            zone,
            name: name || null,
            gid: guestId || null,
            emotion: expr,
          });

          total++; if (zone==="green") green++; else if (zone==="red") red++;
        }

        // -------------- MIC POLICY (with grace; session stays ACTIVE) --------------
        const shouldListen = total > 0 && green === total;
        const nowMs = performance.now();

        if (shouldListen) {
          // We are all-green → refresh timer and ensure mic is ON
          lastAllGreenRef.current = nowMs;

          if (!micOnRef.current) {
            userMicOffRef.current = false;
            startMic().catch(()=>{});
          }
        } else {
          // Not all green → only stop mic if we've been non-green for long enough
          const sinceAllGreen = nowMs - (lastAllGreenRef.current || 0);
          if (micOnRef.current && sinceAllGreen >= MIC_STOP_GRACE_MS) {
            userMicOffRef.current = true;  // behave like user pressed Stop
            stopMic().catch(()=>{});
          }
        }
        // -------------- end MIC POLICY --------------

        const keys = Object.keys(tracks);
        for (const k of keys) {
          const idx = Number(k);
          if (idx >= resized.length) delete tracks[idx];
        }

        while (rows.length < 5) {
          rows.push({ idx: rows.length + 1, gender: "-", ageGroup: "-", zone: "-", name: "-", distance: "-" });
        }
        setTable(prev => {
          const same =
            prev.length === rows.length &&
            prev.every((r,i)=> JSON.stringify(r) === JSON.stringify(rows[i]));
          return same ? prev : rows;
        });
        setTotals(prev => (prev.all===total && prev.green===green && prev.red===red)
          ? prev : { all: total, green, red });

        const count = peopleForPost.length;
        const top5 = rows.slice(0, 5).map(r => ({
          idx: r.idx,
          name: r.name !== "-" ? r.name : null,
          zone: r.zone !== "-" ? r.zone : null,
          gender: r.gender !== "-" ? r.gender : null,
          ageGroup: r.ageGroup !== "-" ? r.ageGroup : null,
          distance: r.distance !== "-" ? r.distance : null,
        }));

        const slots = {
          slot1: rows[0]?.name && rows[0].name !== "-" ? rows[0].name : null,
          slot2: rows[1]?.name && rows[1].name !== "-" ? rows[1].name : null,
          slot3: rows[2]?.name && rows[2].name !== "-" ? rows[2].name : null,
          slot4: rows[3]?.name && rows[3].name !== "-" ? rows[3].name : null,
          slot5: rows[4]?.name && rows[4].name !== "-" ? rows[4].name : null,
        };

        const snapshotPayload = {
          people: peopleForPost,
          count,
          greenCount: green,
          redCount: red,
          top5,
          greenMaxM: greenMaxMRef.current,
          ...slots,
        };

        updateSession(snapshotPayload).catch(() => {});
        detecting = false; // release guard at the end
      })();
    };

    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      video?.removeEventListener("loadedmetadata", resize);
      video?.removeEventListener("resize", resize);
    };
  }, [ready, videoId]);

  useEffect(() => {
    if (!ready) return;
    const CHECK_MS = 1000;
    const MAX_IDLE_MS = 10000;

    const timer = setInterval(() => {
      const ago = performance.now() - lastFrameTsRef.current;

      // If camera truly died (track not live), just stop the mic; try to keep video element intact
      if (!isCamLive() && micOnRef.current) {
        userMicOffRef.current = true;
        stopMic().catch(() => {});
      }

      // If frames stall, DO NOT stop camera; just ensure mic is off.
      if (ago > MAX_IDLE_MS && micOnRef.current) {
        userMicOffRef.current = true;
        stopMic().catch(() => {});
      }
    }, CHECK_MS);

    return () => clearInterval(timer);
  }, [ready]);

  /* ------------------------------ posting ------------------------------ */
  const post = async (which, payload) => {
    const url = N8N[which];
    const nowStr = new Date().toLocaleTimeString();

    // If not in debug mode, try sendBeacon first (no response body available)
    if (!DEBUG_FETCH) {
      try {
        const ok = navigator.sendBeacon(
          url,
          new Blob([JSON.stringify(payload)], { type: "application/json; charset=UTF-8" })
        );
        setPosts(p => ({ ...p, [which]: p[which] + 1 }));
        setLastSent(s => ({ ...s, [which]: nowStr }));
        setLastHttp(h => ({ ...h, [which]: ok ? "sent (beacon)" : "beacon failed" }));
        return null; // no body to parse from beacon
      } catch (e) {
        setLastHttp(h => ({ ...h, [which]: `ERR beacon ${String(e)}` }));
        // fall through to fetch below
      }
    }

    // Fallback or debug: use fetch and parse JSON
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(payload),
        keepalive: true,
      });

      let data = null;
      try { data = await r.json(); } catch { /* ok if empty */ }

      setPosts(p => ({ ...p, [which]: p[which] + 1 }));
      setLastSent(s => ({ ...s, [which]: nowStr }));
      setLastHttp(h => ({ ...h, [which]: r.ok ? "OK (fetch)" : `HTTP ${r.status}` }));

      return data; // <= parsed response JSON (or null if none)
    } catch (e) {
      setLastHttp(h => ({ ...h, [which]: `ERR ${String(e)}` }));
      return null;
    }
  };

  /* ------------------------------ session FSM ------------------------------ */
  async function updateSession(payload) {
    const now = Date.now();
    const any = payload && Array.isArray(payload.people) && payload.people.length > 0;

    // make a compact signature of what we care about
    const sig = any ? JSON.stringify({
      c: payload.count,
      g: payload.greenCount,
      r: payload.redCount,
      s1: payload.top5?.[0]?.name ?? null,
      s2: payload.top5?.[1]?.name ?? null,
      s3: payload.top5?.[2]?.name ?? null,
      s4: payload.top5?.[3]?.name ?? null,
      s5: payload.top5?.[4]?.name ?? null,
      gm: greenMaxMRef.current
    }) : "";

    const changedSinceLast = sig !== lastSnapRef.current;

    // ── SESSION NOT STARTED YET ─────────────────────────────────────────────
    if (!S.current.id) {
      if (any) {
        S.current.seenFrames++;
        if (S.current.seenFrames >= START_FRAMES) {
          S.current.id = uuid();
          S.current.lastFaceTs = now;
          S.current.lastSnapshotTs = 0;
          lastSnapRef.current = "";           // <— reset here
          userMicOffRef.current = false;

          setSessionId(S.current.id);
          setSessionStatus("ACTIVE");

          post("start", { sessionId: S.current.id, ts: now }).catch(() => {});
        }
      } else {
        S.current.seenFrames = 0;
      }
      return;
    }

    // ── SESSION ACTIVE ──────────────────────────────────────────────────────
    if (any) {
      S.current.lastFaceTs = now;
      if (changedSinceLast && (now - S.current.lastSnapshotTs >= SNAPSHOT_EVERY)) {
        S.current.lastSnapshotTs = now;
        post("snapshot", { sessionId: S.current.id, ts: now, ...payload }).catch(()=>{});
        lastSnapRef.current = sig;  // <-- update here
      }
      return;
    }

    // ── NO FACES; CHECK FOR TIMEOUT / END ───────────────────────────────────
    if (now - S.current.lastFaceTs >= END_AFTER_MS) {
      userMicOffRef.current = true;
      if (micOnRef.current) await stopMic();
      // optionally mark session as IDLE without tearing camera down
      if (S.current.id) {
        setSessionStatus("IDLE");
        setSessionId(null);
        S.current = { id:null, seenFrames:0, lastFaceTs:0, lastSnapshotTs:0 };
      }
      return;
    }
  }

  /* ------------------------------ UI ------------------------------ */
  return (
    <main className="app">
      {/* TOP ROW: STATUS (left) + MIC (right) */}
      <div className="grid">
        {/* STATUS PANEL */}
        <div className="panel">
          <div className="statrow">
            <div className="kv"><b>Backend:</b> {backend}</div>
            <div className="kv"><b>Models:</b> {ready ? "loaded" : "loading…"}</div>
            <div className="kv">
              <span className={`dot ${sessionStatus === "ACTIVE" ? "ok" : "warn"}`} />
              <b>Session:</b>&nbsp;{sessionStatus}{sessionId ? ` (${sessionId.slice(0,8)})` : ""}
            </div>
            <div className="kv"><b>Known faces loaded:</b> {knownCount}</div>
            <div className="kv"><b>Posts:</b> start {posts.start} · snap {posts.snapshot} · stop {posts.stop}</div>
          </div>

          {/* under-panel: Last Sent / HTTP / Faces */}
          <div className="subpanel">
            <div className="subline">
              <b>Last Sent:</b>
              <span className="badge">
                <span className="chip">start {lastSent.start}</span>
                <span className="chip">snap {lastSent.snapshot}</span>
                <span className="chip">stop {lastSent.stop}</span>
              </span>
            </div>
            <div className="subline">
              <b>HTTP:</b>
              <span className="badge">
                <span className="chip">start {lastHttp.start || "-"}</span>
                <span className="chip">snap {lastHttp.snapshot || "-"}</span>
                <span className="chip">stop {lastHttp.stop || "-"}</span>
              </span>
              <span style={{opacity:.7}}>
                {DEBUG_FETCH ? " (debug)" : " (beacon)"}
              </span>
            </div>
            <div className="subline">
              <b>Faces:</b>
              <span className="badge">
                <span className="chip">total {totals.all}</span>
                <span className="chip">green {totals.green}</span>
                <span className="chip">red {totals.red}</span>
                <span className="chip">≤ {greenMaxM.toFixed(2)} m</span>
              </span>
            </div>
            {/* Green-zone distance control */}
             <div className="subline" style={{alignItems:'center', gap:12}}>
               <b>Green zone distance:</b>
               <div style={{ display:'flex', alignItems:'center', gap:10, flex:1 }}>
                 <input
                   type="range"
                   min="0.3"
                   max="2.0"
                   step="0.05"
                   value={greenMaxM}
                   onChange={e => setGreenMaxM(Number(e.target.value))}
                   style={{ width:'240px' }}
                   aria-label="Green zone distance in meters"
                   title="Green zone threshold (meters)"
                 />
                 <span className="chip">{greenMaxM.toFixed(2)} m</span>
                 <button className="btn" onClick={()=>setGreenMaxM(DEFAULT_GREEN_MAX_M)} title="Reset">
                 reset
                 </button>
                 <button className="btn" onClick={()=>setGreenMaxM(v=>Math.max(0.3, +(v-0.1).toFixed(2)))} title="-0.1 m">–0.1</button>
                 <button className="btn" onClick={()=>setGreenMaxM(v=>Math.min(2.0, +(v+0.1).toFixed(2)))} title="+0.1 m">+0.1</button>
               </div>
             </div>
             {/* UI, maybe next to "Green zone distance" */}
             <button className="btn" style={{marginLeft:8}}
             onClick={()=>{
             guestSeqRef.current = 1;
             guestMemRef.current = [];
             saveGuestMem({ day: dayKey(), seq: 1, mem: [] });
             }}
             title="Clear in-browser guest memory"
             >
             clear guests
             </button>
          </div>
        </div>

        {/* MIC PANEL */}
        <div className="panel" style={{display:'flex', flexDirection:'column', gap:10}}>
          {/* mic status + buttons */}
          <div className="row" style={{ gap:10 }}>
            <span className={`dot ${micOn ? "ok" : "err"}`} />
            <b>Mic:</b>&nbsp;{micOn ? "listening" : "idle"}

            {/* right side controls */}
            <div style={{ marginLeft:'auto', display:'flex', gap:12, alignItems:'center' }}>
              {/* NEW: Auto-detect toggle (blue when ON) */}
              <ToggleSwitch
                checked={autoDetectOn}
                onChange={setAutoDetectOn}
                label="Auto-calibrate"
              />

              <button
                className="btn"
                data-active={micOn ? "true" : "false"}
                onClick={async ()=>{ userMicOffRef.current = false; await startMic(); }}
                title="Start microphone"
              >
                Start
              </button>

              <button
                className="btn"
                data-active={!micOn ? "true" : "false"}
                onClick={async ()=>{ userMicOffRef.current = true; await stopMic(); }}
                title="Stop microphone"
              >
                Stop
              </button>

              {/* Calibrate: turns blue while pressed */}
              <button
                className="btn"
                data-active={isPressed ? "true" : "false"}
                disabled={!micOn}
                title="Sample 2s room noise and set threshold"
                onClick={autoCalibrate}
                onMouseDown={() => setIsPressed(true)}
                onMouseUp={() => setIsPressed(false)}
                onMouseLeave={() => setIsPressed(false)}
                onTouchStart={() => setIsPressed(true)}
                onTouchEnd={() => setIsPressed(false)}
                onBlur={() => setIsPressed(false)}
              >
                Calibrate
              </button>
            </div>
          </div>

          {/* level meter */}
          <LevelMeter levelDbfs={dbfs} thresholdDbfs={threshold} />

          {/* threshold slider */}
          <div>
            <label className="label">Noise Threshold (dBFS)</label>
            <input
              className="range"
              type="range"
              min="-60"
              max="-20"
              step="1"
              value={threshold}
              onChange={e=>setThreshold(Number(e.target.value))}
            />
          </div>

          {/* Listen / Silence sliders (0–5s) */}
          <div className="row" style={{gap:16}}>
            <div style={{flex:1}}>
              <label className="label">Listen</label>
              <input
                className="range"
                type="range" min="0" max="5000" step="100"
                value={listenMs}
                onChange={e=>setListenMs(Number(e.target.value))}
              />
              <div style={{fontSize:12,opacity:.7,marginTop:4}}>
                {Math.round(listenMs/100)/10}s
              </div>
            </div>
            <div style={{flex:1}}>
              <label className="label">Silence</label>
              <input
                className="range"
                type="range" min="0" max="5000" step="100"
                value={silenceMs}
                onChange={e=>setSilenceMs(Number(e.target.value))}
              />
              <div style={{fontSize:12,opacity:.7,marginTop:4}}>
                {Math.round(silenceMs/100)/10}s
              </div>
            </div>
          </div>

          {/* device selectors (mic + camera) */}
          <div className="row" style={{ gap:8 }}>
            <div style={{flex:1}}>
              <label className="label">Microphone</label>
              <select
                className="select"
                value={audioId}
                onChange={async (e)=>{ 
                  const next = e.target.value; 
                  setAudioId(next); 
                  await startMic(next, { force: true }); 
                }}
              >
                <option value="">(Default)</option>
                {audioDevs.map(d => (
                  <option key={d.deviceId} value={d.deviceId}>{d.label || 'Microphone'}</option>
                ))}
              </select>
            </div>
            <div style={{flex:1}}>
              <label className="label">Camera</label>
              <select
                className="select"
                value={videoId}
                onChange={async (e)=>{ setVideoId(e.target.value); await startCamera(e.target.value); }}
              >
                <option value="">(Default)</option>
                {videoDevs.map(d => (
                  <option key={d.deviceId} value={d.deviceId}>{d.label || 'Camera'}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Optional: show if session active, mic is off */}
      {sessionStatus === "ACTIVE" && !micOn && (
        <div className="panel" style={{ marginTop: 12 }} aria-live="polite">
          <b>Mic permission needed:</b>&nbsp;
          <button className="btn" onClick={()=>startMic()}>
            Enable Microphone
          </button>
        </div>
      )}

      {/* MIDDLE: CAMERA STAGE */}
      <div className="stage">
        <video ref={videoRef} autoPlay muted playsInline />
        <canvas ref={canvasRef} />
      </div>

      {/* BOTTOM: TABLE */}
      <div className="tablewrap panel" style={{padding:12}}>
        <table className="table">
          <thead>
            <tr>
              {["#", "Gender", "AgeGroup", "Zone", "Name", "Distance"].map(h => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.map(r => (
              <tr key={r.idx}>
                <td>{r.idx}</td>
                <td>{r.gender}</td>
                <td>{r.ageGroup}</td>
                <td className={
                  r.zone==="green" ? "zone-green" : r.zone==="red" ? "zone-red" : "zone-unk"
                }>
                  {r.zone}
                </td>
                <td>{r.name}</td>
                <td>{r.distance}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </main>
  );
}