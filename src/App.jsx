// ==== App.jsx — streamlined (Live Preview + Native Audio), neat right sidebar ====
// - TFJS + face-api + webcam detection
// - Socket.IO bridge for Gemini/ElevenLabs audio + text
// - Mic/VAD with auto-calibration + Live AAD knobs in Mic panel
// - Right sidebar: system message, Gemini settings, ElevenLabs settings
// - Status counters + green zone distance + device selectors

import * as React from "react";
import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-wasm";
import { setWasmPaths } from "@tensorflow/tfjs-backend-wasm";
import * as faceapi from "face-api.js";
import io from "socket.io-client";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import "./App.css";

// Add this helper (same-origin by default; allows http(s) and ws(s))
function normalizeServerUrl(u) {
  if (!u) return undefined; // same-origin
  let s = String(u || "").trim();
  if (!s) return undefined;
  if (/^ws(s)?:\/\//i.test(s)) s = s.replace(/^ws/i, "http"); // ws:// → http://, wss:// → https://
  if (!/^https?:\/\//i.test(s)) s = "http://" + s; // allow bare host:port
  return s.replace(/\/+$/, ""); // strip trailing slash
}

/* ====================== CONSTANTS / CONFIG ====================== */
const MODEL_URL = "/models";
const LABELS_URL = "/labels/labels.json";

const uuid = () => (crypto?.randomUUID?.() || Math.random().toString(36).slice(2) + Date.now().toString(36));

// geometry
const FACE_WIDTH_M = 0.15;
let FOCAL_PX = 500; // refined after camera opens

// session heuristics
const DEFAULT_GREEN_MAX_M = 0.8;
const LOOP_STEP_ACTIVE_MS = 120;
const LOOP_STEP_IDLE_MIN_MS = 180;
const LOOP_STEP_IDLE_MAX_MS = 220;

// recognition
const MATCH_THRESHOLD = 0.5;
const MATCH_MARGIN = 0.03;
const STABILIZE_FRAMES = 5;

const MIC_IDLE_MS = 5000;   // mic goes idle after 5s
const CAM_IDLE_MS = 10000;  // cam fully idle after 10s

// drawing
const BOX_SHRINK = 0.7;
const BOX_LINE_WIDTH = 5;
const LABEL_FONT =
  "16px system-ui,-apple-system,Segoe UI,Roboto,'Helvetica Neue',Arial,sans-serif";
const LABEL_PAD_X = 8;
const LABEL_PAD_Y = 6;

// sockets
const SOCKET_URL = undefined; // same-origin
const USE_SOCKET_SERVER = true;

// --- Attention / greeting policy ---
const FACING_YAW_MAX_DEG = 9;   // how “straight on” horizontally
const FACING_PITCH_MAX_DEG = 10; // how “straight on” vertically
const ATTEND_MIN_FRAMES   = 5;   // require 3–5 consecutive frames
const GREET_COOLDOWN_MS   = 35_000;

// Hard cap per identity
const MAX_INVITES_PER_PERSON = 3;

// Optional: if someone disappears for a while, forgive past invites
const NOT_SEEN_RESET_MS = 120_000;   // 2 min of not being seen resets their count

// ===== Hand / Gesture config (tablet-safe) =====
const HANDS_ENABLED = true;

// runtime & cadence
const HANDS_FAST_MS = 66;
const HANDS_IDLE_MS = 180;
const HANDS_CACHE_MS = 1000;
const HANDS_SEND_MS  = 600;

// Game mode cadence (snappier)
const GM_HANDS_FAST_MS = 40;
const GM_HANDS_IDLE_MS = 120;

// Hand constants
const HANDS_MODEL_URL = "/mp/hand_landmarker.task";
const HANDS_MAX_NUM = 2;
const HANDS_IMAGE_SIDE = 256;

// polite “call over” policy
const CALL_OVER_MAX_TRIES = 3;
const CALL_OVER_COOLDOWN_MS = 30_000; // >= 30s between tries

// speaker focus gating
const SPEAKER_STABLE_FRAMES = 3;
const SPEAKER_STABLE_MS = 1200;

// group ask cooldown
const GROUP_ASK_COOLDOWN_MS = 20_000;

/* ====================== SMALL UTILS ====================== */

// Distance estimate from face box width (pixels) via pinhole camera
const estimateDistanceM = (wPx) =>
  (Number.isFinite(wPx) && wPx > 0) ? (FOCAL_PX * FACE_WIDTH_M) / wPx : null;

// Coarse age binning
const ageGroupOf = (age) => {
  if (!Number.isFinite(age)) return "unknown";
  const a = Math.round(age);
  if (a >= 18) return "adult";
  if (a >= 12) return "teen";
  return "child";
};

// Green/Red zone helper
const zoneOf = (d, greenMaxM) =>
  (Number.isFinite(d) && Number.isFinite(greenMaxM) && d <= greenMaxM) ? "green" : "red";

// Best-scoring expression label
const topExpression = (e) => {
  if (!e || typeof e !== "object") return "neutral";
  let bestKey = "neutral", bestVal = -Infinity;
  for (const [k, v] of Object.entries(e)) {
    const val = Number(v) || 0;
    if (val > bestVal) { bestVal = val; bestKey = k; }
  }
  return bestKey;
};

// Add the helper gestureLabelOf
function gestureLabelOf(g) {
  if (!g || !g.type) return null;
  switch (g.type) {
    case "wave": return "wave";
    case "thumbs_up": return "thumbs_up";
    case "peace": return "peace";
    case "raise_hand": return "raise_hand";
    case "on_phone": return "on_phone";
    default: return String(g.type);
  }
}

/* ---- Camera math (pixels <-> angles/positions) ---- */
const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

function focalFromFov(widthPx, fovDeg) {
  // clamp FOV to avoid tan(0)/tan(π)
  const fov = Math.max(1, Math.min(179, Number(fovDeg || 70)));
  const w = Math.max(1, Number(widthPx) || 1);
  return (w / 2) / Math.tan((fov * DEG) / 2);
}

function anglesFromPixel(px, py, fx, fy, cx0, cy0) {
  const x = (px - cx0);
  const y = (py - cy0);
  return {
    yaw:   Math.atan2(x,  Math.max(1e-6, fx)),     // +yaw = right
    pitch: Math.atan2(-y, Math.max(1e-6, fy)),     // +pitch = up
  };
}

function posFromPixel(px, py, fx, fy, cx0, cy0, Z) {
  const x = (px - cx0);
  const y = (py - cy0);
  const z = Number(Z);
  if (!Number.isFinite(z)) return { x: null, y: null, z: null };
  return {
    x: (x / Math.max(1e-6, fx)) * z,
    y: -(y / Math.max(1e-6, fy)) * z,
    z,
  };
}

// Landmarks indices (MediaPipe)
const MP = {
  WRIST: 0,
  THUMB_MCP: 2,
  THUMB_TIP: 4,
  INDEX_MCP: 5,
  INDEX_TIP: 8,
  MIDDLE_MCP: 9,
  MIDDLE_TIP: 12,
  RING_MCP: 13,
  RING_TIP: 16,
  PINKY_MCP: 17,
  PINKY_TIP: 20
};

// simple util
function v2(x, y){ return {x, y}; }
function dist(a, b){ return Math.hypot((a.x-b.x)||0, (a.y-b.y)||0); }

// Keep micro history for wave velocity
const waveHistRef = { t: 0, xs: [] }; // xs: recent x positions (screen-normalized)


// classify a "wave": lateral wrist direction changes within short window
function classifyWave(landmarks, now){
  try {
    const wrist = landmarks[MP.WRIST];
    if (!wrist) return {ok:false};
    // drop tiny/ghost hands
    if (palmSpanLen(landmarks) < 0.020) return { ok: false };
    const x = wrist.x;
    const xs = waveHistRef.xs;

    // keep ~1s of history
    if (now - (waveHistRef.t||0) > 900) xs.length = 0;
    waveHistRef.t = now;
    xs.push(x);
    if (xs.length > 14) xs.shift();
    if (xs.length < 4) return {ok:false};

    // require any back-and-forth with small amplitude
    let flips = 0;
    for (let i=2;i<xs.length;i++){
      const dx1 = xs[i] - xs[i-1];
      const dx0 = xs[i-1] - xs[i-2];
      if (Math.sign(dx1) !== Math.sign(dx0) &&
          Math.abs(dx1) > 0.008 && Math.abs(dx0) > 0.008) {
        flips++;
      }
    }
    const amp = Math.max(...xs) - Math.min(...xs); // 0..1 normalized X span
    const vel = recentLateralMotion();
    // ...inside classifyWave, after `const amp = Math.max(...xs) - Math.min(...xs);`...
    // fast-path A: medium swing with ≥1 flip → lock sooner
    if (xs.length >= 6 && amp > 0.028 && flips >= 1) {
      return { ok: true, type: "wave", score: Math.min(1, 0.50 + Math.min(0.35, amp * 4.0)) };
    }
    // fast-path B: small swing + noticeable lateral velocity (typical subtle wave)
    if (flips >= 2 && amp > 0.018 && vel > 0.048) {
      return { ok: true, type: "wave", score: Math.min(1, 0.44 + Math.min(0.30, amp * 3.2) + Math.min(0.12, (vel - 0.045) * 3.2)) };
    }
    // normal path: 2 flips with modest amplitude
    if (flips >= 2 && amp > 0.012) {
      return { ok:true, type:"wave", score: Math.min(1, 0.28 + 0.18 * flips + Math.min(0.35, amp * 3.8)) };
    }
    return {ok:false};
  } catch { return {ok:false}; }
}

// helper: recent lateral motion magnitude (Σ|Δx| over last frames)
function recentLateralMotion() {
  const xs = waveHistRef.xs || [];
  if (xs.length < 4) return 0;
  let s = 0;
  for (let i = 1; i < xs.length; i++) s += Math.abs(xs[i] - xs[i - 1]);
  return s;
}

// Wave activity summary for anti-wave gating (count flips + amplitude)
function waveActivity() {
  const xs = waveHistRef.xs || [];
  if (xs.length < 3) return { flips: 0, amp: 0 };
  let flips = 0;
  for (let i = 2; i < xs.length; i++) {
    const dx1 = xs[i] - xs[i - 1];
    const dx0 = xs[i - 1] - xs[i - 2];
    if (Math.sign(dx1) !== Math.sign(dx0) && Math.abs(dx1) > 0.008 && Math.abs(dx0) > 0.008) {
      flips++;
    }
  }
  const amp = Math.max(...xs) - Math.min(...xs);
  return { flips, amp };
}

// finger state helpers (normalized by palm width)
function fingerClosed(lm, tipIdx, mcpIdx) {
  try {
    const w = lm[MP.WRIST], tip = lm[tipIdx], mcp = lm[mcpIdx];
    const dTip = Math.hypot(tip.x - w.x, tip.y - w.y);
    const dMcp = Math.hypot(mcp.x - w.x, mcp.y - w.y);
    const span = palmSpanLen(lm);
    const margin = (dTip - dMcp) / Math.max(1e-3, span);
    return { closed: margin < 0.06, margin };
  } catch { return { closed: true, margin: -1 }; }
}

// replace the old thumbs-up with this (keep the same name)
function classifyThumbsUp(landmarks){
  try {
    const thumbTip = landmarks[MP.THUMB_TIP];
    const indexMcp = landmarks[MP.INDEX_MCP];
    const indexTip = landmarks[MP.INDEX_TIP];
    const middleTip = landmarks[MP.MIDDLE_TIP];
    const middleMcp = landmarks[MP.MIDDLE_MCP];
    const ringTip = landmarks[MP.RING_TIP];
    const ringMcp = landmarks[MP.RING_MCP];
    const pinkyTip = landmarks[MP.PINKY_TIP];
    const pinkyMcp = landmarks[MP.PINKY_MCP];
    const wrist = landmarks[MP.WRIST];
    if (!(thumbTip && indexMcp && indexTip && middleTip && middleMcp && ringTip && ringMcp && pinkyTip && pinkyMcp && wrist)) {
      return { ok: false };
    }

    // Anti-wave gate: if hand is moving side-to-side, don't call 👍
    const vel = recentLateralMotion();
    if (vel > 0.08) return { ok: false };

    // Hand axis (wrist → index MCP): want it roughly vertical for 👍
    const vIdx = v2(indexMcp.x - wrist.x, indexMcp.y - wrist.y);
    const axisLen = Math.hypot(vIdx.x, vIdx.y) || 1e-6;
    const axisCosToVertical = Math.abs(vIdx.y) / axisLen; // 1 = vertical, 0 = horizontal

    // Basic pose checks
    const thumbAbove = thumbTip.y < indexMcp.y - 0.012;   // stronger than before
    const open = dist(thumbTip, indexTip) > 0.030;        // small relaxed gap
    const orientedUp = axisCosToVertical > 0.80;
    const bigEnough = palmSpanLen(landmarks) > 0.030;

    // Other fingers mostly closed
    const idxClosed = fingerClosed(landmarks, MP.INDEX_TIP, MP.INDEX_MCP).closed;
    const midClosed = fingerClosed(landmarks, MP.MIDDLE_TIP, MP.MIDDLE_MCP).closed;
    const rngClosed = fingerClosed(landmarks, MP.RING_TIP, MP.RING_MCP).closed;
    const pkyClosed = fingerClosed(landmarks, MP.PINKY_TIP, MP.PINKY_MCP).closed;
    const closedCount = [idxClosed, midClosed, rngClosed, pkyClosed].filter(Boolean).length;
    // Fast path: clear 👍 pose (snappy, anti-false guarded by pose checks)
    const otherClosed = closedCount >= 2;
    if (thumbAbove && open && bigEnough && orientedUp && otherClosed) {
      const openness = Math.max(0, Math.min(1, (dist(thumbTip, indexTip) - 0.03) / 0.12));
      const sFast = Math.min(1, 0.78 + 0.22 * openness);
      return { ok: true, type: "thumbs_up", score: sFast };
    }

    if (thumbAbove && open && bigEnough && orientedUp && closedCount >= 2) {
      // Score: openness + orientation + stillness boost (faster lock when steady)
      const openness = Math.max(0, Math.min(1, (dist(thumbTip, indexTip) - 0.03) / 0.12));
      const orientBoost = Math.min(0.3, Math.max(0, (axisCosToVertical - 0.82) * 1.6));
      const stillBoost = Math.min(0.2, Math.max(0, (0.06 - vel) * 3.0)); // vel small → boost
      const s = Math.max(0, Math.min(1, 0.7 * openness + orientBoost + stillBoost));
      return { ok: true, type: "thumbs_up", score: s };
    }
    return { ok: false };
  } catch { return { ok: false }; }
}

// --- RPS helpers ---
function palmSpanLen(lm) {
  try {
    const a = lm[MP.INDEX_MCP], b = lm[MP.PINKY_MCP];
    return Math.hypot(a.x - b.x, a.y - b.y) || 1e-3;
  } catch { return 1e-3; }
}
function fingerOpen(lm, tipIdx, mcpIdx) {
  try {
    const w = lm[MP.WRIST], tip = lm[tipIdx], mcp = lm[mcpIdx];
    const dTip = Math.hypot(tip.x - w.x, tip.y - w.y);
    const dMcp = Math.hypot(mcp.x - w.x, mcp.y - w.y);
    const span = palmSpanLen(lm);
    const margin = (dTip - dMcp) / Math.max(1e-3, span); // normalize by palm width
    return { open: margin > 0.09, margin }; // ~0.12 tuned empirically
  } catch { return { open: false, margin: -1 }; }
}

// Peace sign (same pose as scissors, but used outside game mode)
function classifyPeace(lm) {
  try {
    const idx = fingerOpen(lm, MP.INDEX_TIP, MP.INDEX_MCP);
    const mid = fingerOpen(lm, MP.MIDDLE_TIP, MP.MIDDLE_MCP);
    const rng = fingerOpen(lm, MP.RING_TIP, MP.RING_MCP);
    const pky = fingerOpen(lm, MP.PINKY_TIP, MP.PINKY_MCP);
    if (idx.open && mid.open && !rng.open && !pky.open) {
      const margin = Math.max(0, idx.margin) + Math.max(0, mid.margin);
      const clamp = Math.max(0, 0.12 - Math.max(0, rng.margin)) + Math.max(0, 0.12 - Math.max(0, pky.margin));
      const score = Math.min(1, 0.50 + 0.35 * margin + 0.25 * clamp);
      return { ok: true, type: "peace", score };
    }
    return { ok: false };
  } catch { return { ok: false }; }
}

// Raise hand: open or flat palm, upright-ish, optionally high in frame.
// Now supports two variants:
//   A) Open palm (>=3 fingers open), upright, modest height
//   B) Flat palm with fingers together (index..pinky tips close), upright
function classifyRaiseHand(lm) {
  try {
    const idx = fingerOpen(lm, MP.INDEX_TIP, MP.INDEX_MCP);
    const mid = fingerOpen(lm, MP.MIDDLE_TIP, MP.MIDDLE_MCP);
    const rng = fingerOpen(lm, MP.RING_TIP, MP.RING_MCP);
    const pky = fingerOpen(lm, MP.PINKY_TIP, MP.PINKY_MCP);
    const opens = [idx, mid, rng, pky].filter(f => f.open).length;

    const wrist = lm[MP.WRIST], iMcp = lm[MP.INDEX_MCP];
    const vx = (iMcp.x - wrist.x), vy = (iMcp.y - wrist.y);
    const vlen = Math.hypot(vx, vy) || 1e-6;
    const cosToVertical = Math.abs(vy) / vlen;

    const tips = [MP.INDEX_TIP, MP.MIDDLE_TIP, MP.RING_TIP, MP.PINKY_TIP].map(i => lm[i]);
    const minY = Math.min(...tips.map(t => t?.y ?? 1));

    const vel = recentLateralMotion();

    const wa = waveActivity();
    const isWaving = (wa.flips >= 3 && wa.amp > 0.030); // require stronger wave to block

    if (!isWaving && opens >= 3 && cosToVertical > 0.52 && palmSpanLen(lm) >= 0.030) {
      const heightBoostFast = Math.max(0, (0.68 - minY) * 0.8);
      const sFast = Math.min(1, 0.60 + 0.20 * Math.min(1, (opens - 2) / 2) + heightBoostFast);
      return { ok: true, type: "raise_hand", score: sFast };
    }

    // Variant A: open palm (stricter)
    const highA = minY <= 0.62; // allow slightly lower in frame
    const passOpenPalm = (opens >= 3) && (cosToVertical > 0.58) && !isWaving && highA && (vel <= 0.060);

    // Variant B: flat palm (fingers together), stricter
    const span = palmSpanLen(lm);
    const tipPairs = [
      [MP.INDEX_TIP, MP.MIDDLE_TIP],
      [MP.MIDDLE_TIP, MP.RING_TIP],
      [MP.RING_TIP, MP.PINKY_TIP],
    ];
    const meanAdj = tipPairs
      .map(([a, b]) => Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y))
      .reduce((s, d) => s + d, 0) / (tipPairs.length || 1);
      const together = (meanAdj / Math.max(1e-3, span)) < 0.20;
      const highB = minY <= 0.68;
      const passFlatPalm = (opens >= 2) && together && (cosToVertical > 0.56) && !isWaving && highB && (vel <= 0.060);

    if (passOpenPalm || passFlatPalm) {
      const openness = Math.max(0, (idx.margin + mid.margin + rng.margin + pky.margin) / 4);
      const orientBoost = Math.max(0, (cosToVertical - 0.65) * 0.9);
      const heightBoost = Math.max(0, (0.60 - minY) * 0.8);
      const togetherBoost = passFlatPalm ? Math.min(0.22, Math.max(0, (0.18 - (meanAdj / Math.max(1e-3, span))) * 2.0)) : 0;
      const score = Math.min(1, 0.34 + 0.26 * openness + orientBoost + heightBoost + togetherBoost);
      return { ok: true, type: "raise_hand", score };
    }
    return { ok: false };
  } catch { return { ok: false }; }
}

/**
 * On phone: wrist or thumb tip near cheek/ear, hand upright-ish, few fingers open,
 * and not actively waving.
 */
function classifyOnPhone(lm, faces, canvasW, canvasH) {
  try {
    if (!lm || lm.length < 21 || !Array.isArray(faces) || !faces.length) return { ok: false };
    const wrist = lm[MP.WRIST], iMcp = lm[MP.INDEX_MCP], thumbTip = lm[MP.THUMB_TIP];
    if (!wrist || !iMcp || !thumbTip) return { ok: false };

    // tiny/ghost hands → skip
    if (palmSpanLen(lm) < 0.020) return { ok: false };

    // Use nearest provided face (we pass the assigned one already)
    const pickNear = (px, py) => {
      let best = null, bestD2 = Infinity;
      for (const f of faces) {
        const fx = (f.cx || 0) / Math.max(1, canvasW);
        const fy = (f.cy || 0) / Math.max(1, canvasH);
        const dx = px - fx, dy = py - fy, d2 = dx*dx + dy*dy;
        if (d2 < bestD2) { bestD2 = d2; best = f; }
      }
      return best;
    };
    const f = pickNear(wrist.x, wrist.y) || faces[0];

    // Face center + half extents (normalized 0..1)
    const fx = (f.cx || 0) / Math.max(1, canvasW);
    const fy = (f.cy || 0) / Math.max(1, canvasH);
    const hw = Math.max(0.02, (f.w || 120) / Math.max(1, canvasW) * 0.5);
    const hh = Math.max(0.03, (f.h || 160) / Math.max(1, canvasH) * 0.5);

    // Side of face (phone side = wrist side)
    const sideSign = wrist.x >= fx ? +1 : -1;

    // Ear anchor (slightly behind temple, a bit above face center)
    const earX = fx + sideSign * hw * 0.78;
    const earY = fy - hh * 0.08;

    // Legacy cheek band (keep as supportive gate)
    const targetBandX = fx + sideSign * hw * 0.92;
    const bandY = (y) => Math.max(0, Math.min(1, 1 - Math.abs(y - fy) / (hh * 0.85)));
    const closeSide = (px, tx) => Math.max(0, Math.min(1, 1 - Math.abs(px - tx) / (hw * 0.95)));

    // Points: wrist, thumb, and phone mid
    const mid = { x: (wrist.x + thumbTip.x) * 0.5, y: (wrist.y + thumbTip.y) * 0.5 };
    const closeW = { side: closeSide(wrist.x, targetBandX),    y: bandY(wrist.y) };
    const closeT = { side: closeSide(thumbTip.x, targetBandX), y: bandY(thumbTip.y) };
    const closeM = { side: closeSide(mid.x, targetBandX),      y: bandY(mid.y) };

    // Ear distance (normalize by face half extents)
    const normDist = (px, py) => {
      const dx = (px - earX) / hw;
      const dy = (py - earY) / hh;
      return Math.hypot(dx, dy);
    };
    const dEarW = normDist(wrist.x, wrist.y);
    const dEarT = normDist(thumbTip.x, thumbTip.y);
    const dEarM = normDist(mid.x,   mid.y);
    const dEarMin = Math.min(dEarW, dEarT, dEarM);
    const earProx = Math.max(0, 1 - Math.min(1.3, dEarMin)); // 0..1

    // Orientation: accept wrist→index OR wrist→thumb as near-vertical
    const vx1 = iMcp.x - wrist.x, vy1 = iMcp.y - wrist.y;
    const vlen1 = Math.hypot(vx1, vy1) || 1e-6;
    const cosVertIdx = Math.abs(vy1) / vlen1;
    const vx2 = thumbTip.x - wrist.x, vy2 = thumbTip.y - wrist.y;
    const vlen2 = Math.hypot(vx2, vy2) || 1e-6;
    const cosVertPhone = Math.abs(vy2) / vlen2;
    const cosToVertical = Math.max(cosVertIdx, cosVertPhone);

    // Few open fingers (holding phone)
    const idx = fingerOpen(lm, MP.INDEX_TIP, MP.INDEX_MCP);
    const midF = fingerOpen(lm, MP.MIDDLE_TIP, MP.MIDDLE_MCP);
    const rng = fingerOpen(lm, MP.RING_TIP, MP.RING_MCP);
    const pky = fingerOpen(lm, MP.PINKY_TIP, MP.PINKY_MCP);
    const opens = [idx, midF, rng, pky].filter(f => f.open).length;
    const fewFingers = opens <= 3;

    // Anti-wave: allow some motion, block strong waving
    const wa = waveActivity();
    const isWaving = (wa.flips >= 2 && wa.amp > 0.025);
    const vel = recentLateralMotion();

    // Gates
    const passBand =
      (closeW.side > 0.08 && closeW.y > 0.08) ||
      (closeT.side > 0.10 && closeT.y > 0.08) ||
      (closeM.side > 0.10 && closeM.y > 0.10);

    // Ear gate: require true near-ear proximity (distance normalized by face size)
    // 0.62 ≈ within ~62% of face half-extent; tweak 0.58–0.68 if needed
    const passEarStrict = dEarMin <= 0.62;

    // Final OK (ear distance is mandatory; band is supportive only)
    const ok =
      passEarStrict &&
      (cosToVertical > 0.45) &&
      fewFingers &&
      !isWaving &&
      (vel <= 0.14);

    if (!ok) return { ok: false };

    // Fast path: very close to ear + upright-ish → snap
    if (dEarMin <= 0.48 && cosToVertical > 0.50 && vel <= 0.12) {
      const sFast = Math.min(1,
        0.78 + 0.16 * earProx +            // boost ear proximity
        0.06 * Math.max(0, (cosToVertical - 0.50) * 2.0)
      );
      return { ok: true, type: "on_phone", score: sFast };
    }

    // Score: emphasize ear proximity; keep cheek band and orientation as helpers
    const closenessSide = Math.max(closeW.side, closeT.side, closeM.side);
    const closenessY = Math.max(closeW.y, closeT.y, closeM.y);
    const multiPointBonus =
      (closeW.side > 0.08 && closeT.side > 0.08 ? 0.08 : 0) +
      (closeM.side > 0.10 && (closeW.side > 0.08 || closeT.side > 0.08) ? 0.06 : 0);

    const score = Math.min(
      1,
      0.30 * closenessSide +
      0.18 * closenessY +
      0.40 * earProx +                       // ← heavier ear weight
      0.08 * Math.max(0, (cosToVertical - 0.50) * 2.0) +
      0.04 * Math.max(0, 0.22 - wa.amp) +    // stillness
      multiPointBonus
    );

    return { ok: true, type: "on_phone", score };
  } catch {
    return { ok: false };
  }
}

function classifyPaper(lm) {
  try {
    const idx = fingerOpen(lm, MP.INDEX_TIP, MP.INDEX_MCP);
    const mid = fingerOpen(lm, MP.MIDDLE_TIP, MP.MIDDLE_MCP);
    const rng = fingerOpen(lm, MP.RING_TIP, MP.RING_MCP);
    const pky = fingerOpen(lm, MP.PINKY_TIP, MP.PINKY_MCP);
    const opens = [idx, mid, rng, pky].filter(f => f.open).length;
    if (opens >= 3) {
      const avgMargin = (idx.margin + mid.margin + rng.margin + pky.margin) / 4;
      const score = Math.min(1, 0.25 * opens + Math.max(0, avgMargin)); // favor 4-finger open + margins
      return { ok: true, type: "paper", score };
    }
    return { ok: false };
  } catch { return { ok: false }; }
}

function classifyRock(lm) {
  try {
    const idx = fingerOpen(lm, MP.INDEX_TIP, MP.INDEX_MCP);
    const mid = fingerOpen(lm, MP.MIDDLE_TIP, MP.MIDDLE_MCP);
    const rng = fingerOpen(lm, MP.RING_TIP, MP.RING_MCP);
    const pky = fingerOpen(lm, MP.PINKY_TIP, MP.PINKY_MCP);
    const opens = [idx, mid, rng, pky].filter(f => f.open).length;
    if (opens <= 1) {
      // stronger if all closed (negative margins)
      const neg = [idx, mid, rng, pky].map(f => Math.max(0, 0.12 - Math.max(0, f.margin)));
      const tight = neg.reduce((a, b) => a + b, 0) / 4;
      const score = Math.min(1, 0.85 - 0.2 * opens + tight);
      return { ok: true, type: "rock", score };
    }
    return { ok: false };
  } catch { return { ok: false }; }
}

function classifyScissors(lm) {
  try {
    const idx = fingerOpen(lm, MP.INDEX_TIP, MP.INDEX_MCP);
    const mid = fingerOpen(lm, MP.MIDDLE_TIP, MP.MIDDLE_MCP);
    const rng = fingerOpen(lm, MP.RING_TIP, MP.RING_MCP);
    const pky = fingerOpen(lm, MP.PINKY_TIP, MP.PINKY_MCP);
    if (idx.open && mid.open && !rng.open && !pky.open) {
      const margin = Math.max(0, idx.margin) + Math.max(0, mid.margin);
      const clamp = Math.max(0, 0.12 - Math.max(0, rng.margin)) + Math.max(0, 0.12 - Math.max(0, pky.margin));
      const score = Math.min(1, 0.45 + 0.35 * margin + 0.25 * clamp);
      return { ok: true, type: "scissors", score };
    }
    return { ok: false };
  } catch { return { ok: false }; }
}

// --- Gesture stabilizer (wave-first, anti-flicker) ---
const WAVE_BOOT_MS = 300;        // after startup/change, allow wave to “claim” quickly
const WAVE_GRACE_MS = 550;       // how long wave is allowed to win ties/near-ties
const CHANGE_COOLDOWN_MS = 520;  // prevent rapid flip-flops after we lock something

// Hoisted: gesture voting config (so pickStableGesture can see them)
const GESTURE_PRIORITY   = ["wave", "raise_hand", "on_phone", "thumbs_up", "peace", "paper", "rock", "scissors"];
const VOTE_WINDOW        = 5;    // keep last ~6 frames
const VOTE_MAX_AGE_MS    = 700;  // ignore old entries
const REQUIRE_CONSISTENT = 2;    // ≥3 agreeing frames
const CLEAR_IF_IDLE_MS   = 450;  // drop stale gesture after this
const MIN_SCORE = {
  wave: 0.40,
  thumbs_up: 0.22,
  peace: 0.46,
  raise_hand: 0.36,
  on_phone: 0.34,
  paper: 0.38,
  rock: 0.45,
  scissors: 0.45,
};

function pickStableGesture(now, win, prevStable) {
  // 1) Fresh window (trim by age and keep only the recent tail)
  const fresh = (win || [])
    .filter(e => e && (now - e.t) <= VOTE_MAX_AGE_MS)
    .slice(-VOTE_WINDOW);

  // Nothing new → keep previous a short while, then clear
  if (!fresh.length) {
    if (prevStable && (now - prevStable.t) < CLEAR_IF_IDLE_MS) return prevStable;
    return null;
  }

  // 2) Aggregate per type with per-type min scores
  const byType = new Map();
  for (const e of fresh) {
    const min = MIN_SCORE[e.type] ?? 0.4;
    if ((e.score ?? 0) < min) continue;
    const rec = byType.get(e.type) || { count: 0, sum: 0, best: 0, firstTs: e.t };
    rec.count += 1;
    rec.sum += (e.score ?? 0);
    rec.best = Math.max(rec.best, e.score ?? 0);
    if (e.t < rec.firstTs) rec.firstTs = e.t;
    byType.set(e.type, rec);
  }

  // Still nothing above thresholds → maybe hold old one briefly
  if (!byType.size) {
    if (prevStable && (now - prevStable.t) < CLEAR_IF_IDLE_MS) return prevStable;
    return null;
  }

  // 3) Pick best: highest count → priority (wave first) → avg score
  let best = null;
  for (const [type, stats] of byType.entries()) {
    const cand = {
      type,
      count: stats.count,
      avg: stats.sum / stats.count,
      pri: GESTURE_PRIORITY.indexOf(type),
      bestScore: stats.best,
      firstTs: stats.firstTs,
    };
    if (
      !best ||
      cand.count > best.count ||
      (cand.count === best.count && cand.pri < best.pri) ||
      (cand.count === best.count && cand.pri === best.pri && cand.avg > best.avg)
    ) {
      best = cand;
    }
  }

  // 4) Wave-first grace: if a wave appeared very recently, let it win near-ties
  const waveStats = byType.get("wave");
  if (waveStats) {
    const waveFirstTs = waveStats.firstTs;
    const waveRecent = (now - waveFirstTs) <= WAVE_GRACE_MS;
    const prevIsWavey = !prevStable || prevStable.type === "wave" || (now - (prevStable?.t || 0)) <= WAVE_BOOT_MS;

    if (waveRecent && prevIsWavey) {
      // If current best isn't wave and wave isn't far behind, prefer wave
      const waveCand = {
        type: "wave",
        count: waveStats.count,
        avg: waveStats.sum / Math.max(1, waveStats.count),
        pri: GESTURE_PRIORITY.indexOf("wave"),
        bestScore: waveStats.best,
        firstTs: waveStats.firstTs,
      };
      const nearTie = (best.type !== "wave") && (waveCand.count >= best.count - 1);
      const notClearlyWorse = (waveCand.avg + 0.05) >= best.avg; // require near score
      if (best.type !== "wave" && nearTie && notClearlyWorse && waveCand.bestScore >= MIN_SCORE.wave) {
        best = waveCand;
      }
    }
  }

  // 5) Require some consistency (≥ N frames or ≥60% of fresh window)
  const bestScoreVal = (byType.get(best.type)?.best ?? 0);
  const strong =
    (bestScoreVal >= 0.68 && best.count >= 2) ||
    (best.type === "thumbs_up" && bestScoreVal >= 0.78) ||
    (best.type === "raise_hand" && bestScoreVal >= 0.68) ||
    (best.type === "peace" && bestScoreVal >= 0.70) ||
    (best.type === "paper" && bestScoreVal >= 0.70) ||
    (best.type === "on_phone" && bestScoreVal >= 0.68);
  const needByType = { wave: 3 }; // wave needs 3 frames; others keep default (2)
  const need = needByType[best.type] || REQUIRE_CONSISTENT;
  const consistent = strong || best.count >= need || best.count >= Math.ceil(0.6 * fresh.length);
  if (!consistent) {
    return (prevStable && (now - prevStable.t) < CLEAR_IF_IDLE_MS) ? prevStable : null;
  }

  // 6) Anti-flicker: if we’d switch types too soon, keep the previous briefly
  if (prevStable && prevStable.type !== best.type) {
    if ((now - prevStable.t) < CHANGE_COOLDOWN_MS) {
      return prevStable;
    }
  }

  // 7) Emit stable using the best observed score for that type
  const bestScore = byType.get(best.type)?.best ?? best.bestScore ?? 0.5;
  return { type: best.type, score: bestScore, t: now };
}

/* ---- Mouth activity (inner mouth aspect ratio, MAR) ---- */
function mouthMAR(landmarks68) {
  try {
    // face-api.js FaceLandmarks68 exposes .positions (and internally ._positions)
    const pts = landmarks68?.positions || landmarks68?._positions || (Array.isArray(landmarks68) ? landmarks68 : null);
    if (!pts) return 0;

    const dist = (a, b) => {
      const pa = pts[a], pb = pts[b];
      return Math.hypot(pa.x - pb.x, pa.y - pb.y);
    };

    // Inner mouth: 60–67. Vertical = avg(61–67, 62–66, 63–65), Horizontal = 60–64
    const V = (dist(61,67) + dist(62,66) + dist(63,65)) / 3;
    const H = dist(60,64) || 1e-6;
    const mar = V / H;

    // Typical closed MAR ≈ 0.25–0.35. Map to 0..1 for UI.
    // Shift + scale, then clamp.
    const norm = Math.max(0, Math.min(1, (mar - 0.30) * 3.0)); // tweak 0.30 & 3.0 to taste
    return norm;
  } catch {
    return 0;
  }
}

/* ---- Box helper ---- */
const shrinkBox = (b, f = BOX_SHRINK) => {
  const w = b.width * f;
  const h = b.height * f;
  return {
    x: b.x + (b.width  - w) / 2,
    y: b.y + (b.height - h) / 2,
    width:  w,
    height: h,
  };
};

// get number (or null) from localStorage safely
function getStoredNumber(key) {
  try {
    const raw = localStorage.getItem(key);
    if (raw == null || raw === "") return null;
    const num = Number(raw);
    return Number.isFinite(num) ? num : null;
  } catch {
    return null;
  }
}

/* ====================== GUEST MEMORY HELPERS ====================== */
function encodeDescFloat32ToU8(descF32) {
  const out = new Uint8Array(descF32.length);
  for (let i = 0; i < descF32.length; i++) {
    const clamped = Math.max(-1, Math.min(1, descF32[i]));
    out[i] = Math.round((clamped + 1) * 127.5);
  }
  return out;
}
function decodeDescU8ToFloat32(u8) {
  const out = new Float32Array(u8.length);
  for (let i = 0; i < u8.length; i++) out[i] = u8[i] / 127.5 - 1;
  return out;
}
function u8ToB64(u8) {
  let bin = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < u8.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, u8.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}
function b64ToU8(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

/* ====================== AUDIO CHUNK PLAYER (server → client) ====================== */
class ChunkAudioPlayer {
  constructor({ onStart, onEnd } = {}) {
    this.queue = [];
    this.playing = false;
    this.audio = new Audio();
    this.audio.autoplay = false;
    this.onStart = onStart;
    this.onEnd = onEnd;

    this.audio.addEventListener("ended", () => this._next());
    this.audio.addEventListener("error", () => this._next());
  }
  enqueue(uint8, mime = "audio/mpeg") {
    if (!(uint8 instanceof Uint8Array)) return;
    this.queue.push({ uint8, mime });
    if (!this.playing) this._next();
  }
  _next() {
    const item = this.queue.shift();
    if (!item) {
      this.playing = false;
      try {
        this.onEnd && this.onEnd();
      } catch {}
      return;
    }
    const blob = new Blob([item.uint8], { type: item.mime });
    const url = URL.createObjectURL(blob);
    this.playing = true;
    try {
      this.onStart && this.onStart();
    } catch {}
    this.audio.src = url;
    this.audio
      .play()
      .catch(() => {})
      .finally(() => setTimeout(() => URL.revokeObjectURL(url), 10_000));
  }
  stop() {
    try {
      this.audio.pause();
    } catch {}
    this.queue = [];
    this.playing = false;
  }
}

  // ---- Standalone: ElevenLabsSettings (top-level, not inside App) ----
  function ElevenLabsSettings() {
    const [apiKey, setApiKey]   = useState(() => localStorage.getItem("ika:11labs:key") || "");
    const [voiceId, setVoiceId] = useState(() => localStorage.getItem("ika:11labs:voiceId") || "");
    const [model, setModel]     = useState(() => localStorage.getItem("ika:11labs:model") || "eleven_turbo_v2_5");

    useEffect(() => { try { localStorage.setItem("ika:11labs:key", apiKey); } catch {} }, [apiKey]);
    useEffect(() => { try { localStorage.setItem("ika:11labs:voiceId", voiceId); } catch {} }, [voiceId]);
    useEffect(() => { try { localStorage.setItem("ika:11labs:model", model); } catch {} }, [model]);

    return (
      <div>
        <label className="label" htmlFor="elevenlabs-api-key-input">API Key</label>
        <input
          id="elevenlabs-api-key-input"
          className="input bigpad"
          type="password"
          placeholder="sk-…"
          value={apiKey}
          onChange={(e) => setApiKey(e.target.value)}
          autoComplete="off"
        />

        <label className="label" htmlFor="elevenlabs-speech-model-select" style={{ marginTop: 8 }}>Speech model</label>
        <select
          id="elevenlabs-speech-model-select"
          className="select big"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          {[
            "eleven_turbo_v2",
            "eleven_multilingual_v2",
            "eleven_turbo_v2_5",
            "eleven_v3_alpha"
          ].map((m) => <option key={m} value={m}>{m}</option>)}
        </select>

        <label className="label" htmlFor="elevenlabs-voice-id-input" style={{ marginTop: 8 }}>Voice ID</label>
        <input
          id="elevenlabs-voice-id-input"
          className="input bigpad"
          placeholder="e.g. Rachel / your-voice-id"
          value={voiceId}
          onChange={(e) => setVoiceId(e.target.value)}
          autoComplete="off"
        />

        <div className="help" style={{ marginTop: 6 }}>
          Saved locally. Uses ElevenLabs realtime speech models only.
        </div>
      </div>
    );
  }

/* ====================== MAIN APP ====================== */
export default function App() {
  /* ---------- Socket + audio playback ---------- */
  const socketRef = useRef(null);
  const ttsPlayerRef = useRef(null);

  // Configurable server URL (empty = same-origin)
  const [serverUrl, setServerUrl] = useState(() => localStorage.getItem("ika:serverUrl") || "");
  const [serverUrlDraft, setServerUrlDraft] = useState(serverUrl);
  const effectiveUrl = useMemo(() => normalizeServerUrl(serverUrl) ?? window.location.origin, [serverUrl]);
  useEffect(() => { setServerUrlDraft(serverUrl); }, [serverUrl]);
  useEffect(() => { try { localStorage.setItem("ika:serverUrl", serverUrl); } catch {} }, [serverUrl]);

  // Pick up ?server=… from query string once
  useEffect(() => {
    try {
      const u = new URLSearchParams(window.location.search).get("server");
      if (u) setServerUrl(u);
    } catch {}
  }, []);

  /* ---------- Global/session UI state ---------- */
  const [sessionStatus, setSessionStatus] = useState("IDLE");
  const [sessionId, setSessionId] = useState(null);
  const [serverInfo, setServerInfo] = useState({ connected:false, model:null, tts:null, boundDeviceId:null, ai_speaking:false });

  const [posts, setPosts] = useState({ start: 0, snapshot: 0, stop: 0 });
  const [lastSent, setLastSent] = useState({ start: "-", snapshot: "-", stop: "-" });
  const [lastHttp, setLastHttp] = useState({ start: "", snapshot: "", stop: "" });

  const [captions, setCaptions] = useState(localStorage.getItem("ika:captions") === "true");
  const [lastText, setLastText] = useState("");

  // Per-identity attention counting & cooldown
  const attentionMapRef = useRef(new Map());
  // Map key: stable id (name || gid)
  // value: { count: number, lastInviteTs: number }

  // ===== HandLandmarker refs =====
  const handLmRef = useRef(null);
  const handsReadyRef = useRef(false);
  const lastHandsRunTsRef = useRef(0);
  const lastLmSeenTsRef = useRef(0);     // added: when we last saw landmarks
  const handsFailRef = useRef(0);        // added: consecutive VIDEO misses

  // downscale buffer for hands
  const handsOffscreenRef = useRef(null);
  const handsCtxRef = useRef(null);

  // last gesture memory
  const stableGestureRef = useRef(null); // { type, score, t }

  // NEW: Per-face gesture windows and stable picks
  const perFaceGestureWinRef = useRef(new Map());   // key -> [{type,score,t}, ...]
  const perFaceStableRef = useRef(new Map());       // key -> {type,score,t}
  const lastGestureSentPerFaceRef = useRef(new Map()); // key -> lastTs

  // NEW: per-face wave histories (used by wave/velocity gates)
  const waveHistByFaceRef = useRef(new Map());

  // --- age/gender stagger + cache ---
  const AGE_SAMPLE_MS = 600;
  const lastAgeSampleRef = useRef(0);
  const ageGenderCacheRef = useRef(new Map()); // key -> { age, gender }

  const [locationLabel, setLocationLabel] = useState(
    localStorage.getItem("ika:locationLabel") || "Jakarta (Bundaran HI)"
  );
  const [weatherLabel, setWeatherLabel] = useState(
    localStorage.getItem("ika:weatherLabel") || "Clear 28°C"
  );
  const [clock, setClock] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // device identity
  const [deviceId] = useState(() => {
    try {
      const k = "ika:deviceId";
      let v = localStorage.getItem(k);
      if (!v) {
        v = uuid();
        localStorage.setItem(k, v);
      }
      return v;
    } catch {
      return uuid();
    }
  });

  // Settings backup helpers (export/import/reset)
  const SETTINGS_KEYS = [
    "ika:serverUrl",
    "ika:systemInstruction","ika:model","ika:voice","ika:langCode","ika:temperature",
    "ika:enableAffective","ika:proactiveAudio","ika:functionCalling","ika:autoFunctionResponse","ika:grounding",
    "ika:ttsProvider","ika:11labs:voiceId","ika:11labs:key","ika:11labs:model",
    "ika:captions","ika:locationLabel","ika:weatherLabel","ika:gemini:key",
    "ika:videoId","ika:audioId",
    "ika:fovHdeg","ika:fovVdeg","ika:panOffsetDeg","ika:tiltOffsetDeg",
    "ika:wNear","ika:wCenter","ika:wMouth",
    "ika:sos","ika:eos","ika:prefixPad","ika:silenceDur",
    "ika:threshold","ika:listenMs","ika:silenceMs","ika:autoDetectOn",
    "ika:greenMaxM", // base key (per-camera variants are also used)
  ];

  const exportSettings = async () => {
    try {
      // Collect all keys with "ika:" prefix (covers new settings automatically)
      const bag = {};
      for (let i = 0; i < localStorage.length; i++) {
        const k = localStorage.key(i);
        if (k && k.startsWith("ika:")) {
          bag[k] = localStorage.getItem(k);
        }
      }
      const json = JSON.stringify(bag, null, 2);
      const blob = new Blob([json], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const ts = new Date();
      const pad = (n) => String(n).padStart(2, "0");
      const filename = `ika-settings-${ts.getFullYear()}${pad(ts.getMonth()+1)}${pad(ts.getDate())}-${pad(ts.getHours())}${pad(ts.getMinutes())}.json`;
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 2000);
    } catch (e) {
      alert("Failed to export settings");
    }
  };

  const importSettings = async () => {
    // Open a file picker and read JSON
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/json";
    input.style.display = "none";
    document.body.appendChild(input);
    input.addEventListener("change", async () => {
      const file = input.files && input.files[0];
      document.body.removeChild(input);
      if (!file) return;
      try {
        const text = await file.text();
        const bag = JSON.parse(text);
        Object.entries(bag).forEach(([k, v]) => {
          try { localStorage.setItem(k, v ?? ""); } catch {}
        });
        window.location.reload();
      } catch (e) {
        alert("Invalid or unreadable settings file");
      }
    }, { once: true });
    input.click();
  };

  const resetSettings = () => {
    if (!confirm("Reset all saved settings?")) return;
    for (const k of SETTINGS_KEYS) {
      try { localStorage.removeItem(k); } catch {}
    }
    window.location.reload();
  };

  // HANDS: detect with VIDEO first, fallback to IMAGE if needed ----
  const detectHandsOnce = useCallback(async (videoEl) => {
    const landmarker = handLmRef.current;
    if (!handsReadyRef.current || !landmarker || !videoEl?.videoWidth) return null;

    const ts = performance.now();

    // 1) Try VIDEO mode
    try {
      const res = landmarker.detectForVideo(videoEl, ts);
      const hands = res?.landmarks || res?.handLandmarks || [];
      if (hands.length) {
        handsFailRef.current = 0;
        lastLmSeenTsRef.current = ts;
        // normalize to [{x,y}...] in 0..1
        return hands.map(h => h.map(pt => ({ x: pt.x, y: pt.y })));
      }
    } catch {}

    // No luck in VIDEO this frame
    handsFailRef.current = (handsFailRef.current || 0) + 1;

    // 2) Occasionally try IMAGE fallback
    if ((handsFailRef.current % 4) !== 0) return null;

    try {
      await landmarker.setOptions?.({ runningMode: "IMAGE", numHands: HANDS_MAX_NUM });

      const c = handsOffscreenRef.current, g = handsCtxRef.current;
      if (!c || !g) return null;
 
      const W = c.width, H = c.height;
      const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
      const scale = Math.min(W / vw, H / vh);
      const dw = Math.round(vw * scale), dh = Math.round(vh * scale);
      const dx = (W - dw) >> 1, dy = (H - dh) >> 1;
      g.clearRect(0, 0, W, H);
      g.drawImage(videoEl, 0, 0, vw, vh, dx, dy, dw, dh);

      const res2 = await landmarker.detect(c);
      const hands2 = res2?.landmarks || res2?.handLandmarks || [];

      await landmarker.setOptions?.({ runningMode: "VIDEO", numHands: HANDS_MAX_NUM });

      if (hands2.length) {
        handsFailRef.current = 0;
        lastLmSeenTsRef.current = performance.now();
        return hands2.map(hand =>
          hand.map(pt => ({
            x: (dx + pt.x * dw) / W,
            y: (dy + pt.y * dh) / H,
          }))
        );
      }
    } catch {
      try { await handLmRef.current?.setOptions?.({ runningMode: "VIDEO", numHands: HANDS_MAX_NUM }); } catch {}
    }

    return null;
  }, []);

  // Gestures on/off (persist to localStorage)
  const [gesturesOn, setGesturesOn] = useState(() => localStorage.getItem("ika:gesturesOn") !== "true");
  const gesturesOnRef = useRef(false);
  useEffect(() => {
    gesturesOnRef.current = gesturesOn;
    try { localStorage.setItem("ika:gesturesOn", String(gesturesOn)); } catch {}
    if (!gesturesOn) {
      // clear per-face gesture state immediately
      perFaceGestureWinRef.current = new Map();
      perFaceStableRef.current = new Map();
      lastGestureSentPerFaceRef.current = new Map();
      stableGestureRef.current = null;
    }
  }, [gesturesOn]);

  // How many people run gesture tracking for (1 or 2)
  const [gestureTargets, setGestureTargets] = useState(() => {
    const v = parseInt(localStorage.getItem("ika:gestureTargets") || "2", 10);
    return v === 1 ? 1 : 2;
  });
  const gestureTargetsRef = useRef(2);
  useEffect(() => {
    gestureTargetsRef.current = gestureTargets;
    try { localStorage.setItem("ika:gestureTargets", String(gestureTargets)); } catch {}
    // Hint the landmarker to track fewer hands when set to 1
    try { handLmRef.current?.setOptions?.({ numHands: gestureTargets }); } catch {}
  }, [gestureTargets]);

  // --- TFJS backend gating (avoid detect while switching) ---
  const backendReadyRef = useRef(Promise.resolve());
  const backendSwitchingRef = useRef(false);
  const backendNameRef = useRef(null);

  // --- Session rotation on crowd change ---
  const groupSigRef = useRef("");           // last stable group signature
  const groupStableSinceRef = useRef(0);    // when current signature first appeared
  const lastRotateRef = useRef(0);          // last time we rotated session
  const SESSION_ROTATE_COOLDOWN_MS = 20_000; // rotate at most every 20s
  const GROUP_STABLE_MS = 1_500;             // need ~1.5s stable group before rotate

  function groupSignature(people) {
    const ids = people
      .map(p => (p.name || p.gid || "").trim())
      .filter(Boolean)
      .sort();
    let s = ids.join("|");
    let h = 0;
    for (let i = 0; i < s.length; i++) { h = ((h << 5) - h) + s.charCodeAt(i); h |= 0; }
    return ids.length ? String(h) : ""; // empty => no crowd
  }

  function maybeRotateSession(sig, people) {
    const now = performance.now();
    const prev = groupSigRef.current;

    if (sig !== prev) {
      groupSigRef.current = sig;
      groupStableSinceRef.current = now;
      return; // start stability window
    }
    if (!sig) return; // nothing in view

    const stableFor = now - (groupStableSinceRef.current || 0);
    const sinceLast = now - (lastRotateRef.current || 0);

    if (stableFor >= GROUP_STABLE_MS && sinceLast >= SESSION_ROTATE_COOLDOWN_MS) {
      const oldId = sessionId || ("web-" + deviceId);
      const newId = uuid();

      try {
        socketRef.current?.emit?.("rotate_session", {
          oldSessionId: oldId,
          newSessionId: newId,
          at: Date.now(),
          people: people.map(p => ({ name: p.name || null, gid: p.gid || null }))
        });
        socketRef.current?.emit?.("close_session", { sessionId: oldId });
      } catch {}

      setSessionId(newId);
      lastRotateRef.current = now;

      // If you want a fresh LLM dialog immediately, you can also:
      // server.createSession({
      //   model: modelQuick, voice: geminiVoiceQuick,
      //   language_code: languageCodeQuick,
      //   system_instruction: systemInstruction,
      //   tts_provider: ttsProviderQuick
      // });
    }
  }

  /* ---------- Socket lifecycle ---------- */
  useEffect(() => {
    if (!USE_SOCKET_SERVER) return;

    const url = normalizeServerUrl(serverUrl || SOCKET_URL);
    const isHttpsPage = window.location.protocol === "https:";
    const isLoopback = /^http:\/\/(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/i.test(url || "");
    // From HTTPS page to http://localhost: use websocket-only to avoid CORS on XHR polling
    const transports = (isHttpsPage && isLoopback) ? ["websocket"] : ["polling", "websocket"];

    const socket = io(url, {
      transports,
      path: "/socket.io",
      autoConnect: true,
      reconnection: true,
      reconnectionDelay: 500,
      reconnectionAttempts: Infinity,
      withCredentials: false,
      timeout: 12000,
      rememberUpgrade: true,   // cache WS upgrade when it succeeds
      forceNew: true
    });
    socketRef.current = socket;
    console.log("[socket] connecting", { url, transports });

    socket.on("connect", () => {
      console.log("[socket] connected", socket.id, "url:", url || "(same-origin)");
      try { socket.emit("hello", { deviceId, sessionId, role: "web" }); } catch {}
      setServerInfo((s) => ({ ...s, connected: true }));
    });
    socket.on("connect_error", (err) => {
      console.warn("[socket] connect_error:", err?.message || err);
    });
    socket.on("reconnect_error", (err) => {
      console.warn("[socket] reconnect_error:", err?.message || err);
    });
    socket.on("disconnect", () => {
      console.log("[socket] disconnected");
      setServerInfo((s) => ({ ...s, connected: false }));
    });
    socket.on("server_status", (m) => setServerInfo((prev) => ({ ...prev, ...m })));
    // Remote control for game mode (ephemeral; no persistence)
    socket.on("set_game_mode", (m) => setGameModeOn(!!m?.on));
    socket.on("start_rps",   () => setGameModeOn(true));
    socket.on("stop_rps",    () => setGameModeOn(false));
    socket.on("session_created", (m) => {
      setSessionStatus("ACTIVE");
      setSessionId((prev) => m?.sessionId || prev || uuid());
      bump("start");
    });
    socket.on("text_response", (t) => {
      const text = typeof t === "string" ? t : t?.text || "";
      if (text) setLastText(text);
    });
    socket.on("audio_chunk_received", (pkt) => {
      try {
        const b64 = pkt?.chunk; if (!b64) return;
        const bin = atob(b64); const u8 = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
        const mime = (pkt?.mime || "").toLowerCase();
    if (mime.includes("mpeg")) {
      ttsPlayerRef.current.enqueue(u8, mime);
    } else {
      // ignore PCM on the website; UE consumes it
    }
      } catch (e) { console.warn("[audio] chunk error:", e); }
    });
    socket.on("server_message", (m) => console.log("[server]", m));
    socket.on("on_connection_failed", () => console.warn("[socket] model connection failed on server"));

    return () => {
      try { socket.disconnect(); } catch {}
      socketRef.current = null;
    };
  }, [serverUrl, deviceId]);
  
  // Auto-reconnect when network comes online or tab becomes visible (no mic/cam impact)
  useEffect(() => {
    const onOnline = () => {
      try { if (socketRef.current && !socketRef.current.connected) socketRef.current.connect(); } catch {}
    };
    const onVisible = () => {
      if (!document.hidden) {
        try { if (socketRef.current && !socketRef.current.connected) socketRef.current.connect(); } catch {}
      }
    };
    window.addEventListener("online", onOnline);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.removeEventListener("online", onOnline);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  const bump = (which) => {
    const nowStr = new Date().toLocaleTimeString();
    setPosts((p) => ({ ...p, [which]: (p[which] || 0) + 1 }));
    setLastSent((s) => ({ ...s, [which]: nowStr }));
    setLastHttp((h) => ({ ...h, [which]: "socket" }));
  };

  useEffect(() => {
    const p = new ChunkAudioPlayer({
      onStart: () => {
        speakingRef.current = true;
        setServerInfo(s => ({ ...s, ai_speaking: true }));
      },
      onEnd: () => {
        speakingRef.current = false;
        setServerInfo(s => ({ ...s, ai_speaking: false }));
      }
    });
    ttsPlayerRef.current = p;
    return () => p.stop();
  }, []);

  /* ---------- Server helpers (emit over socket) ---------- */
  const createServerSession = useCallback(
    (preset) => {
      const s = socketRef.current;
      if (!s) return;

      const payload = {
        model:
          preset?.model ||
          localStorage.getItem("ika:model") ||
          "gemini-2.5-flash-live-preview",

        // system + language + output modalities
        system_instruction:
          preset?.system_instruction ||
          localStorage.getItem("ika:systemInstruction") ||
          "You are a friendly, concise on-site concierge.",
        language_code:
          preset?.language_code || localStorage.getItem("ika:langCode") || "en-US",
        response_modalities: Array.isArray(preset?.response_modalities)
          ? preset.response_modalities
          : ["AUDIO", "TEXT"],

        // voice & style
        voice: preset?.voice || localStorage.getItem("ika:voice") || "Puck",
        temperature: Number.isFinite(preset?.temperature)
          ? preset.temperature
          : Number(localStorage.getItem("ika:temperature") ?? 0.6),

        // Live AAD (from sliders in Mic panel)
        start_of_speech_sensitivity:
          preset?.start_of_speech_sensitivity ??
          getStoredNumber("ika:sos") ??
          0.5,
        end_of_speech_sensitivity:
          preset?.end_of_speech_sensitivity ??
          getStoredNumber("ika:eos") ??
          0.5,
        prefix_padding_ms:
          preset?.prefix_padding_ms ?? getStoredNumber("ika:prefixPad") ?? 150,
        silence_duration_ms:
          preset?.silence_duration_ms ?? getStoredNumber("ika:silenceDur") ?? 800,

        // behaviors
        enable_affective_dialog:
          preset?.enable_affective_dialog ??
          (localStorage.getItem("ika:enableAffective") === "true"),
        proactive_audio:
          preset?.proactive_audio ??
          (localStorage.getItem("ika:proactiveAudio") === "true"),
        function_calling:
          preset?.function_calling ??
          (localStorage.getItem("ika:functionCalling") === "true"),
        auto_function_response:
          preset?.auto_function_response ??
          (localStorage.getItem("ika:autoFunctionResponse") === "true"),
        grounding:
          preset?.grounding ?? (localStorage.getItem("ika:grounding") === "true"),

        // TTS routing
        tts_provider:
          (preset?.tts_provider ||
            localStorage.getItem("ika:ttsProvider") ||
            "gemini")?.toLowerCase(),
        eleven_voice_id:
          preset?.eleven_voice_id || localStorage.getItem("ika:11labs:voiceId"),
        eleven_api_key:
          preset?.eleven_api_key || localStorage.getItem("ika:11labs:key"),
        eleven_model:
          preset?.eleven_model || localStorage.getItem("ika:11labs:model") || "eleven_turbo_v2_5",

        // UI toggles
        captions:
          preset?.captions ?? (localStorage.getItem("ika:captions") === "true"),

        // locale/location hints
        lat: preset?.lat ?? getStoredNumber("ika:lat"),
        lon: preset?.lon ?? getStoredNumber("ika:lon"),
        locale: preset?.locale || localStorage.getItem("ika:locale") || "en-US",

        deviceId,
      };

      s.emit("create_session", {
        ...payload,
        sessionId: (sessionId || ("web-" + deviceId)),
        gemini_api_key: (localStorage.getItem("ika:gemini:key") || undefined),
      });
      bump("start");
      setSessionStatus("ACTIVE");
      setSessionId((id) => id || uuid());
    },
    [deviceId]
  );

  const updateServerSettings = useCallback((fields) => {
    const s = socketRef.current;
    if (!s) return;
    s.emit("update_settings", fields || {});
  }, []);

  const sendTextPrompt = useCallback((text) => {
    const s = socketRef.current;
    if (!s || !text) return;
    s.emit("send_text_prompt", { text });
  }, []);

  const emitCrowdStatus = useCallback(
    (payload) => {
      const s = socketRef.current;
      if (!s) return;
      s.emit("crowd_status", {
        deviceId,
        sessionId: sessionId || ("web-" + deviceId),
        ...payload,
      });
      bump("snapshot");
    },
    [sessionId]
  );

  const server = useMemo(
    () => ({
      createSession: createServerSession,
      updateSettings: updateServerSettings,
      sendText: sendTextPrompt,
      crowdStatus: emitCrowdStatus,
    }),
    [createServerSession, updateServerSettings, sendTextPrompt, emitCrowdStatus]
  );

  /* ====================== MIC / CAMERA / DETECTION ====================== */
  // === Camera alignment (FOV & pan/tilt offsets) ===
  const [fovHdeg, setFovHdeg] = useState(() => Number(localStorage.getItem("ika:fovHdeg") ?? 70));
  // === Calibration & overlay ===
  const [calibDistanceM, setCalibDistanceM] = useState(
    () => Number(localStorage.getItem("ika:calibDistM") ?? 1.0)
  );
  const [showAlign, setShowAlign] = useState(
    localStorage.getItem("ika:showAlign") !== "false"
  );
  const calibMsgRef = useRef("");        // transient overlay message ("3…2…1", "Calibrating…")
  const showAlignRef = useRef(true);
  useEffect(() => { showAlignRef.current = showAlign; }, [showAlign]);
  useEffect(() => {
    try {
      localStorage.setItem("ika:showAlign", String(showAlign));
      localStorage.setItem("ika:calibDistM", String(calibDistanceM));
    } catch {}
  }, [showAlign, calibDistanceM]);

  // small utility
  const sleep = (ms) => new Promise(r => setTimeout(r, ms));

  const [fovVdeg, setFovVdeg] = useState(() => Number(localStorage.getItem("ika:fovVdeg") ?? 43));
  const [panOffsetDeg, setPanOffsetDeg] = useState(() => Number(localStorage.getItem("ika:panOffsetDeg") ?? 0));
  const [tiltOffsetDeg, setTiltOffsetDeg] = useState(() => Number(localStorage.getItem("ika:tiltOffsetDeg") ?? 0));
  useEffect(() => { try {
    localStorage.setItem("ika:fovHdeg", String(fovHdeg));
    localStorage.setItem("ika:fovVdeg", String(fovVdeg));
    localStorage.setItem("ika:panOffsetDeg", String(panOffsetDeg));
    localStorage.setItem("ika:tiltOffsetDeg", String(tiltOffsetDeg));
  } catch {} }, [fovHdeg, fovVdeg, panOffsetDeg, tiltOffsetDeg]);

  const camFxRef = useRef(600);
  const camFyRef = useRef(600);
  const panOffRef  = useRef(0);
  const tiltOffRef = useRef(0);
  useEffect(() => { panOffRef.current = panOffsetDeg; }, [panOffsetDeg]);
  useEffect(() => { tiltOffRef.current = tiltOffsetDeg; }, [tiltOffsetDeg]);

  // === Focus selection weights ===
  const [wNear,   setWNear]   = useState(() => Number(localStorage.getItem("ika:wNear")   ?? 0.45));
  const [wCenter, setWCenter] = useState(() => Number(localStorage.getItem("ika:wCenter") ?? 0.35));
  const [wMouth,  setWMouth]  = useState(() => Number(localStorage.getItem("ika:wMouth")  ?? 0.20));
  useEffect(() => { try {
    localStorage.setItem("ika:wNear",   String(wNear));
    localStorage.setItem("ika:wCenter", String(wCenter));
    localStorage.setItem("ika:wMouth",  String(wMouth));
  } catch {} }, [wNear, wCenter, wMouth]);

  // mouth activity smoothing + click-to-zero memory
  const mouthMapRef = useRef(new Map());
  const trackedFacesRef = useRef([]);
  // All faces (green + red) for hand proximity heuristics
  const allFacesRef = useRef([]);

  const speakingRef = useRef(false);

  // ---- Policy + speaker state ----
  const prevZoneMapRef = useRef(new Map());   // key -> "green" | "red"
  const callOverStateRef = useRef(new Map()); // key -> { tries, last }
  const lastGroupSetRef = useRef(new Set());  // Set(keys) of last frame
  const lastGroupAskTsRef = useRef(0);
  const speakerRef = useRef({ key: null, topKeyPrev: null, framesDominant: 0, topSince: 0 });

  const lastCrowdSendRef = useRef({ t: 0, sig: "" });
  function emitCrowdThrottled(payload) {
    const now = performance.now();
    const MIN_MS = 66; // ~15 Hz
    const state = lastCrowdSendRef.current;
    // compact signature: focus + per-person pose/lip
    const ppl = (payload.people || []).map(p => [
      p.yawDeg, p.pitchDeg, Math.round((p.mouthActivity || 0) * 1000)
    ]);
    const sig = JSON.stringify([payload.focusIndex, ppl]);
    if ((now - state.t) < MIN_MS && sig === state.sig) return;
    try { server.crowdStatus(payload); } catch {}
    lastCrowdSendRef.current = { t: now, sig };
  }

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const onClick = (ev) => {
      const rect = canvas.getBoundingClientRect();
      const px = ev.clientX - rect.left;
      const py = ev.clientY - rect.top;
      const list = trackedFacesRef.current || [];
      if (!list.length) return;
      let best = null, bestD2 = Infinity;
      for (const f of list) {
        const dx = px - f.cx, dy = py - f.cy, d2 = dx*dx + dy*dy;
        if (d2 < bestD2) { bestD2 = d2; best = f; }
      }
      if (best) {
        setPanOffsetDeg(p => p - best.yawDeg);
        setTiltOffsetDeg(t => t - best.pitchDeg);
      }
    };
    canvas.addEventListener("click", onClick);
    return () => canvas.removeEventListener("click", onClick);
  }, []);

  const lastFrameTsRef = useRef(0);
  // Frame cadence (battery saver)
  const loopStepMsRef = useRef(LOOP_STEP_ACTIVE_MS);
  const loopIdleStateRef = useRef(false);

  const micIdleToStandbyTimerRef = useRef(null);

  const MIC_STOP_GRACE_MS = 5_000;
  const lastAllGreenRef = useRef(0);
  const userMicOffRef = useRef(false);

  const [ready, setReady] = useState(false);
  const [backend, setBackend] = useState("cpu");

  // === Green Zone Distance ===
  const [greenMaxM, setGreenMaxM] = useState(DEFAULT_GREEN_MAX_M);
  const greenMaxMRef = useRef(greenMaxM);
  useEffect(() => {
    greenMaxMRef.current = greenMaxM;
  }, [greenMaxM]);

  // === Red Zone Cutoff (ignore faces beyond this) ===
  const DEFAULT_RED_CUTOFF_M = 3.5;
  const [redCutoffM, setRedCutoffM] = useState(() => {
    const raw = localStorage.getItem("ika:redCutoffM");
    const v = raw == null ? DEFAULT_RED_CUTOFF_M : parseFloat(raw);
    return Number.isFinite(v) ? v : DEFAULT_RED_CUTOFF_M;
  });
  useEffect(() => {
    try { localStorage.setItem("ika:redCutoffM", String(redCutoffM)); } catch {}
  }, [redCutoffM]);

  const [videoId, setVideoId] = useState("");
  const [audioId, setAudioId] = useState("");

  const pickInputSize = (w) => (w >= 1920 ? 416 : w >= 1280 ? 320 : 256);
  const tinyOptsRef = useRef(
    new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.4 })
  );
  const triedFallbackRef = useRef(false);

  // auto-calibrate policy
  const MAX_EMPTY_BEFORE_AUTOCAL = 3;
  const FAILED_STARTS_BEFORE_AUTOCAL = 4;
  const AUTO_CAL_MIN_GAP_MS = 30_000;
  const MIN_NONEMPTY_CHARS = 2;

  const [autoDetectOn, setAutoDetectOn] = useState(
    localStorage.getItem("ika:autoDetectOn") !== "false"
  );
  const autoDetectOnRef = useRef(autoDetectOn);
  useEffect(() => {
    autoDetectOnRef.current = autoDetectOn;
    try {
      localStorage.setItem("ika:autoDetectOn", String(autoDetectOn));
    } catch {}
  }, [autoDetectOn]);

  const [emptyTranscripts, setEmptyTranscripts] = useState(0);
  const lastAutoCalRef = useRef(0);
  const [isPressed, setIsPressed] = useState(false);

  // mic/VAD prefs
  const [dbfs, setDbfs] = useState(-60);
  const [threshold, setThreshold] = useState(
    Number(localStorage.getItem("ika:threshold") ?? -45)
  );
  const [listenMs, setListenMs] = useState(
    Number(localStorage.getItem("ika:listenMs") ?? 400)
  );
  const [silenceMs, setSilenceMs] = useState(
    Number(localStorage.getItem("ika:silenceMs") ?? 500)
  );

  // Live AAD knobs (moved here)
  const [sosQuick, setSosQuick] = useState(Number(localStorage.getItem("ika:sos") ?? 0.5));
  const [eosQuick, setEosQuick] = useState(Number(localStorage.getItem("ika:eos") ?? 0.5));
  const [prefixPadQuick, setPrefixPadQuick] = useState(
    Number(localStorage.getItem("ika:prefixPad") ?? 150)
  );
  const [silenceDurQuick, setSilenceDurQuick] = useState(
    Number(localStorage.getItem("ika:silenceDur") ?? 800)
  );

  // mirror into refs for VAD loop
  const thresholdLiveRef = useRef(threshold);
  const listenMsLiveRef = useRef(listenMs);
  const silenceMsLiveRef = useRef(silenceMs);
  useEffect(() => {
    thresholdLiveRef.current = threshold;
    try {
      localStorage.setItem("ika:threshold", String(threshold));
    } catch {}
  }, [threshold]);
  useEffect(() => {
    listenMsLiveRef.current = listenMs;
    try {
      localStorage.setItem("ika:listenMs", String(listenMs));
    } catch {}
  }, [listenMs]);
  useEffect(() => {
    silenceMsLiveRef.current = silenceMs;
    try {
      localStorage.setItem("ika:silenceMs", String(silenceMs));
    } catch {}
  }, [silenceMs]);

  // persist Live AAD too
  useEffect(() => {
    try {
      localStorage.setItem("ika:sos", String(sosQuick));
      localStorage.setItem("ika:eos", String(eosQuick));
      localStorage.setItem("ika:prefixPad", String(prefixPadQuick));
      localStorage.setItem("ika:silenceDur", String(silenceDurQuick));
    } catch {}
  }, [sosQuick, eosQuick, prefixPadQuick, silenceDurQuick]);

  const [micOn, setMicOn] = useState(false);
  const micOnRef = useRef(false);
  useEffect(() => {
    micOnRef.current = micOn;
  }, [micOn]);

  const [audioDevs, setAudioDevs] = useState([]);
  const [videoDevs, setVideoDevs] = useState([]);

  const audioRef = useRef({
    ctx: null,
    stream: null,
    source: null,
    analyser: null,
    raf: 0,
    deviceId: "",
    vad: {
      highSince: 0,
      lowSince: 0,
      recording: false,
      recorder: null,
      chunks: [],
      startTs: 0,
      failedStarts: 0,
    },
  });
  const camRef = useRef({ stream: null });

  // detection bookkeeping
  const [table, setTable] = useState(
    Array.from({ length: 5 }, (_, i) => ({
      idx: i + 1,
      gender: "-",
      ageGroup: "-",
      zone: "-",
      name: "-",
      distance: "-",
    }))
  );
  const [totals, setTotals] = useState({ all: 0, green: 0, red: 0 });
  const recentMapRef = useRef({});
  const S = useRef({ id: null, seenFrames: 0, lastFaceTs: 0, lastSnapshotTs: 0 });

  // labels & matcher
  const faceMatcherRef = useRef(null);
  const [knownCount, setKnownCount] = useState(0);

  /* ---------- Guest memory ---------- */
  const guestSeqRef = useRef(1);
  const guestMemRef = useRef([]);
  const GUEST_TOL = 0.6;
  const guestSavePendingRef = useRef(false);
  const GUEST_STORE_KEY = "ika:guestMem.v1";
  const GUEST_RETENTION_DAYS = 1;
  const dayKey = (d = new Date()) =>
    d.toLocaleDateString("en-CA", {
      timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    });
  const msToNextMidnight = () => {
    const now = new Date();
    const next = new Date(now);
    next.setHours(24, 0, 0, 0);
    return next - now;
  };
  function scheduleGuestSave() {
    if (guestSavePendingRef.current) return;
    guestSavePendingRef.current = true;
    setTimeout(() => {
      saveGuestMem({});
      guestSavePendingRef.current = false;
    }, 750);
  }
  function assignGuestIdFor(descriptor) {
    if (!descriptor || !descriptor.length) {
      const id = `Guest${String(guestSeqRef.current++).padStart(2, "0")}`;
      scheduleGuestSave();
      return id;
    }
    const mem = guestMemRef.current;
    let bestIdx = -1,
      bestDist = 1;
    for (let i = 0; i < mem.length; i++) {
      const d = faceapi.euclideanDistance(descriptor, mem[i].desc);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0 && bestDist <= GUEST_TOL) {
      mem[bestIdx].ts = Date.now();
      scheduleGuestSave();
      return mem[bestIdx].id;
    }
    const id = `Guest${String(guestSeqRef.current++).padStart(2, "0")}`;
    mem.push({ id, ts: Date.now(), desc: Float32Array.from(descriptor) });
    scheduleGuestSave();
    return id;
  }
  function saveGuestMem({
    day = dayKey(),
    seq = guestSeqRef.current,
    mem = guestMemRef.current,
  }) {
    try {
      const list = [...mem].sort((a, b) => (b.ts || 0) - (a.ts || 0));
      const items = list.map((m) => ({
        id: m.id,
        ts: m.ts || Date.now(),
        desc: u8ToB64(encodeDescFloat32ToU8(m.desc)),
      }));
      const payload = { day, seq, items, savedAt: Date.now() };
      localStorage.setItem(GUEST_STORE_KEY, JSON.stringify(payload));
    } catch {}
  }
  function loadGuestMem() {
    try {
      const raw = localStorage.getItem(GUEST_STORE_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch {
      return null;
    }
  }
  function pruneByRetention(data) {
    if (!data) return null;
    if (GUEST_RETENTION_DAYS <= 0) return null;
    if (GUEST_RETENTION_DAYS === 1) {
      if (data.day !== dayKey()) return null;
      return data;
    }
    const cutoff = Date.now() - GUEST_RETENTION_DAYS * 86_400_000;
    data.items = (data.items || []).filter((it) => (it.ts || 0) >= cutoff);
    return data;
  }

  /* ---------- Tiny UI helpers ---------- */
  function ToggleSwitch({ checked, onChange, label }) {
    return (
      <label style={{ display: "inline-flex", alignItems: "center", gap: 8, cursor: "pointer", fontSize: 13, color: "#ebebeb" }}>
        <input
          type="checkbox"
          checked={checked}
          onChange={e => onChange(e.target.checked)}
          style={{ accentColor: "#0ea5e9", width: 18, height: 18 }}
        />
        {label}
      </label>
    );
  }

  function LevelMeter({
    levelDbfs = -60,
    thresholdDbfs = -45,
    bars = 20,
    height = 18,
  }) {
    const norm = (db, min = -60, max = -20) => {
      const x = (db - min) / (max - min);
      return Math.min(1, Math.max(0, x));
    };
    const fill = norm(levelDbfs);
    const tpos = norm(thresholdDbfs);
    return (
      <div
        style={{
          position: "relative",
          background: "#0a0a0a",
          padding: "10px 12px",
          borderRadius: 10,
        }}
      >
        <div style={{ color: "#9ef99f", fontWeight: 600, marginBottom: 6 }}>
          Volume
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${bars}, 1fr)`,
            gap: 4,
          }}
        >
          {Array.from({ length: bars }).map((_, i) => {
            const lit = i / (bars - 1) <= fill;
            return (
              <div
                key={i}
                style={{
                  height,
                  borderRadius: 4,
                  background: lit ? "#22c55e" : "#184a1d",
                  transition: "background 80ms linear",
                }}
              />
            );
          })}
        </div>
        <div
          style={{
            position: "absolute",
            left: `${tpos * 100}%`,
            top: 34,
            transform: "translateX(0%)",
            display: "flex",
            alignItems: "center",
            gap: 6,
            pointerEvents: "none",
          }}
        >
          <div
            style={{
              width: 6,
              height: height + 10,
              background: "#9ef99f",
              borderRadius: 2,
              opacity: 0.9,
            }}
          />
          <div style={{ color: "#9ef99f", fontSize: 12, opacity: 0.9 }}>
            {Math.round(thresholdDbfs)} dB
          </div>
        </div>
      </div>
    );
  }

  /* ---------- Camera sizing + intrinsics (fx/fy) updates ---------- */
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;

    const updateIntrinsicsAndOpts = () => {
      const w = v.videoWidth || 1280;
      const h = v.videoHeight || 720;

      // TinyFaceDetector tuning based on current video width
      tinyOptsRef.current = new faceapi.TinyFaceDetectorOptions({
        inputSize: pickInputSize(w),
        scoreThreshold: 0.4,
      });

      // Update camera intrinsics from current FOVs
      // (fovHdeg / fovVdeg come from the Camera Alignment sliders)
      camFxRef.current = focalFromFov(w, fovHdeg);
      camFyRef.current = focalFromFov(h, fovVdeg);
    };

    // Run on metadata (dimensions become known) and on resize
    v.addEventListener("loadedmetadata", updateIntrinsicsAndOpts);
    v.addEventListener("resize", updateIntrinsicsAndOpts);

    // Run once immediately (in case metadata already loaded)
    updateIntrinsicsAndOpts();

    return () => {
      v.removeEventListener("loadedmetadata", updateIntrinsicsAndOpts);
      v.removeEventListener("resize", updateIntrinsicsAndOpts);
    };
    // Recompute if FOV knobs change, or when camera/ready state changes
  }, [ready, videoId, fovHdeg, fovVdeg]);

  /* ---------- Init: TFJS backend, face models, camera ---------- */
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
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
        let ok = await tryBackend("webgl");
        if (!ok) ok = await tryBackend("wasm");
        if (!ok) await tryBackend("cpu");
        if (!cancelled) setBackend(tf.getBackend());

        // mark TFJS backend as ready for the frame loop
        backendNameRef.current = tf.getBackend();
        setBackend(backendNameRef.current);       // keep your UI/backend label in sync
        backendReadyRef.current = Promise.resolve();

        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
          faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        ]);

          console.log("[models] loaded:", {
          tiny: !!faceapi.nets.tinyFaceDetector?.isLoaded,
          lmk68: !!faceapi.nets.faceLandmark68Net?.isLoaded,
          expr: !!faceapi.nets.faceExpressionNet?.isLoaded,
          ageg: !!faceapi.nets.ageGenderNet?.isLoaded,
          recog: !!faceapi.nets.faceRecognitionNet?.isLoaded,
        });

        await startCamera();

        // warm-up pass
        try {
          const off = document.createElement("canvas");
          off.width = 128;
          off.height = 128;
          await faceapi.detectAllFaces(
            off,
            new faceapi.TinyFaceDetectorOptions({ inputSize: 128, scoreThreshold: 0.4 })
          );
        } catch {}

        if (!cancelled) setReady(true);
      } catch (e) {
        console.error("[init]", e);
        if (!cancelled) setBackend(tf.getBackend?.() || "cpu");
      }
    })();

    return () => {
      cancelled = true;
      userMicOffRef.current = true;
      stopAll({ reason: "unmount" });
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- HandLandmarker (force CPU, IMAGE mode for smoke test) ---
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const fileset = await FilesetResolver.forVisionTasks("/mp"); // <-- local

       // quick preflight to catch path/header issues in Netlify logs
       try {
         const head = await fetch("/mp/hand_landmarker.task", { method: "HEAD", cache: "no-store" });
         console.log("[hands] model reachable:", head.ok, head.status, head.headers.get("content-type"));
       } catch (e) { console.warn("[hands] model HEAD failed:", e); }

       const isSafari = /^((?!chrome|android).)*safari/i.test(navigator.userAgent);
        const lm = await HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: HANDS_MODEL_URL, delegate: isSafari ? "CPU" : "GPU" },
          runningMode: "VIDEO",
          numHands: HANDS_MAX_NUM,
          minHandDetectionConfidence: 0.03,
          minHandPresenceConfidence: 0.03,
          minTrackingConfidence: 0.03,
        });
        if (cancelled) { lm.close?.(); return; }

        handLmRef.current = lm;
        handsReadyRef.current = true;

        const c = document.createElement("canvas");
        c.width = HANDS_IMAGE_SIDE;
        c.height = HANDS_IMAGE_SIDE;
        handsOffscreenRef.current = c;
        handsCtxRef.current = c.getContext("2d", { willReadFrequently: true });

        console.log("[hands] ready (", isSafari ? "CPU" : "GPU", ", VIDEO)");
      } catch (e) {
        console.warn("[hands] init failed:", e);
        handLmRef.current = null;
        handsReadyRef.current = false;
      }
    })();
    return () => {
      cancelled = true;
      try { handLmRef.current?.close?.(); } catch {}
      handLmRef.current = null;
      handsReadyRef.current = false;
    };
  }, []);

  /* ---------- Labels + matcher ---------- */
  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(LABELS_URL, { cache: "no-store" });
        const data = await res.json();

        const entries = Array.isArray(data)
          ? data
          : Object.entries(data).map(([label, descriptors]) => ({
              label,
              descriptors,
            }));

        const labeled = await Promise.all(
          entries.map(async (e) => {
            let descs = [];
            if (Array.isArray(e.descriptors_b64)) {
              descs = e.descriptors_b64.map((b64) =>
                decodeDescU8ToFloat32(b64ToU8(b64))
              );
            } else if (Array.isArray(e.descriptors)) {
              descs = e.descriptors.map((arr) =>
                arr instanceof Float32Array ? arr : new Float32Array(arr)
              );
            }
            descs = descs.filter((d) => d && d.length === 128);
            return new faceapi.LabeledFaceDescriptors(e.label, descs);
          })
        );

        const usable = labeled.filter((l) => l.descriptors?.length);
        const matcher = new faceapi.FaceMatcher(usable, MATCH_THRESHOLD);

        if (!cancelled) {
          faceMatcherRef.current = matcher;
                    setKnownCount(
            usable.reduce((acc, l) => acc + l.descriptors.length, 0)
          );
        }
      } catch (err) {
        console.warn("[labels] failed:", err);
        if (!cancelled) {
          faceMatcherRef.current = null;
          setKnownCount(0);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ready]);

  /* ---------- Preferences & devices ---------- */
  useEffect(() => {
    (async () => {
      try {
        const vId = localStorage.getItem("ika:videoId") || "";
        if (vId) setVideoId(vId);
        const aId = localStorage.getItem("ika:audioId") || "";
        if (aId) setAudioId(aId);
        const gm =
          localStorage.getItem(`ika:greenMaxM:${vId || "default"}`) ??
          localStorage.getItem("ika:greenMaxM");
        if (gm != null) {
          const val = parseFloat(gm);
          if (Number.isFinite(val))
            setGreenMaxM(Math.min(2.0, Math.max(0.3, val)));
        }
      } catch {}
      try {
        const list = await navigator.mediaDevices.enumerateDevices();
        setAudioDevs(list.filter((d) => d.kind === "audioinput"));
        setVideoDevs(list.filter((d) => d.kind === "videoinput"));
      } catch {}
    })();
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem(
        `ika:greenMaxM:${videoId || "default"}`,
        String(greenMaxM)
      );
    } catch {}
  }, [greenMaxM, videoId]);

  // restore guest memory
  useEffect(() => {
    const data = pruneByRetention(loadGuestMem());
    if (data && Array.isArray(data.items)) {
      guestSeqRef.current = Math.max(1, Number(data.seq) || 1);
      guestMemRef.current = data.items.map((it) => ({
        id: it.id,
        ts: it.ts || Date.now(),
        desc: decodeDescU8ToFloat32(b64ToU8(it.desc)),
      }));
    } else {
      guestSeqRef.current = 1;
      guestMemRef.current = [];
      saveGuestMem({});
    }

    if (GUEST_RETENTION_DAYS === 1) {
      const t = setTimeout(() => {
        guestSeqRef.current = 1;
        guestMemRef.current = [];
        saveGuestMem({ day: dayKey(), seq: 1, mem: [] });
      }, msToNextMidnight());
      return () => clearTimeout(t);
    }
  }, []);

  // unlock audio on first gesture (iOS/Safari)
  const [audioUnlocked, setAudioUnlocked] = useState(false);
  useEffect(() => {
    if (audioUnlocked) return;
    const unlock = async () => {
      try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)({
          latencyHint: "interactive",
        });
        await ctx.resume();
        await ctx.close();
        setAudioUnlocked(true);
        window.removeEventListener("touchend", unlock);
        window.removeEventListener("click", unlock);
      } catch {}
    };
    window.addEventListener("touchend", unlock, { once: true });
    window.addEventListener("click", unlock, { once: true });
    return () => {
      window.removeEventListener("touchend", unlock);
      window.removeEventListener("click", unlock);
    };
  }, [audioUnlocked]);

  // load per-camera greenMaxM
  useEffect(() => {
    try {
      const gm = localStorage.getItem(`ika:greenMaxM:${videoId || "default"}`);
      if (gm != null) {
        const v = parseFloat(gm);
        if (Number.isFinite(v)) setGreenMaxM(Math.min(2.0, Math.max(0.3, v)));
      }
    } catch {}
  }, [videoId]);

  // hot-plug devices
  useEffect(() => {
    const onChange = async () => {
      try {
        const list = await navigator.mediaDevices.enumerateDevices();
        setAudioDevs(list.filter((d) => d.kind === "audioinput"));
        setVideoDevs(list.filter((d) => d.kind === "videoinput"));
      } catch {}
    };
    navigator.mediaDevices?.addEventListener?.("devicechange", onChange);
    return () => {
      navigator.mediaDevices?.removeEventListener?.("devicechange", onChange);
    };
  }, []);

  // keyboard nudges for green zone
  useEffect(() => {
    const onKey = (e) => {
      const tag = (document.activeElement?.tagName || "").toUpperCase();
      if (["INPUT", "TEXTAREA", "SELECT"].includes(tag)) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (e.key === "[")
        setGreenMaxM((v) => Math.max(0.3, +(v - 0.05).toFixed(2)));
      else if (e.key === "]")
        setGreenMaxM((v) => Math.min(2.0, +(v + 0.05).toFixed(2)));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // visibility → stop
  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden) {
        userMicOffRef.current = true;
        stopAll({ reason: "visibility" });
      }
    };
    const onPageHide = () => {
      userMicOffRef.current = true;
      stopAll({ reason: "pagehide" });
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

  /* ---------- STT hook (called after utterance upload) ---------- */
  async function handleTranscriptResult(resp) {
    const txt = (resp?.text ?? "").trim();
    if (!txt || txt.length < MIN_NONEMPTY_CHARS) {
      setEmptyTranscripts((c) => {
        const next = c + 1;
        if (autoDetectOn && next >= MAX_EMPTY_BEFORE_AUTOCAL) {
          setTimeout(() => setEmptyTranscripts(0), 0);
          maybeAutoCalibrate();
        }
        return next;
      });
      return;
    }
    setEmptyTranscripts(0);
    try {
      server.sendText(txt);
    } catch {}
  }

  /* ---------- Auto-calibrate ---------- */
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
  async function maybeAutoCalibrate() {
    if (!autoDetectOnRef.current) return;
    if (!micOnRef.current) return;
    const isRecording = !!audioRef.current?.vad?.recording;
    if (isRecording) return;
    const now = Date.now();
    if (now - lastAutoCalRef.current < AUTO_CAL_MIN_GAP_MS) return;
    lastAutoCalRef.current = now;
    setIsPressed(true);
    try {
      await autoCalibrate();
    } finally {
      setTimeout(() => setIsPressed(false), 250);
    }
  }

  /* ---------- Mic control & VAD ---------- */
  async function stopMic() {
  if (!audioRef.current?.stream && !micOn) {
    setMicOn(false);
    setDbfs(-60);

    // schedule standby after MIC_IDLE_MS
    if (micIdleToStandbyTimerRef.current) clearTimeout(micIdleToStandbyTimerRef.current);
    micIdleToStandbyTimerRef.current = setTimeout(() => {
      if (!micOnRef.current && isCamLive()) {
        setSessionStatus("STANDBY");
      }
      micIdleToStandbyTimerRef.current = null;
    }, MIC_IDLE_MS);

    return;
  }
    try {
      if (audioRef.current.raf) cancelAnimationFrame(audioRef.current.raf);
    } catch {}
    try {
      const rec = audioRef.current?.vad?.recorder;
      if (rec && rec.state === "recording") {
        try {
          rec.requestData?.();
        } catch {}
        try {
          rec.stop();
        } catch {}
      }
    } catch {}
    try {
      audioRef.current.stream?.getTracks()?.forEach((t) => t.stop());
    } catch {}
    try {
      await audioRef.current.ctx?.close();
    } catch {}
    try {
      audioRef.current.source?.disconnect?.();
    } catch {}
    try {
      audioRef.current.analyser?.disconnect?.();
    } catch {}

    audioRef.current = {
      ctx: null,
      stream: null,
      source: null,
      analyser: null,
      raf: 0,
      deviceId: "",
      vad: {
        highSince: 0,
        lowSince: 0,
        recording: false,
        recorder: null,
        chunks: [],
        startTs: 0,
        failedStarts: 0,
      },
    };
    setMicOn(false);
    setDbfs(-60);
    
    // schedule standby after MIC_IDLE_MS
    if (micIdleToStandbyTimerRef.current) clearTimeout(micIdleToStandbyTimerRef.current);
    micIdleToStandbyTimerRef.current = setTimeout(() => {
      if (!micOnRef.current && isCamLive()) {
        setSessionStatus("STANDBY");
      }
      micIdleToStandbyTimerRef.current = null;
    }, MIC_IDLE_MS);
  }

  async function startMic(id = audioId, { force = false } = {}) {
    // 👉 if a standby timer was queued by stopMic, cancel it now
    if (micIdleToStandbyTimerRef.current) {
      clearTimeout(micIdleToStandbyTimerRef.current);
      micIdleToStandbyTimerRef.current = null;
    }

    if (audioRef.current?.stream) {
      const same = (id || "") === (audioRef.current.deviceId || "");
      if (!force && same) {
        console.log("[Mic] already running");
        // ensure status reflects live mic
        setSessionStatus(isCamLive() ? "ACTIVE" : "STANDBY");
        return;
      }
      await stopMic();
    }

    // stream
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
        video: false,
      });
    } catch (e) {
      console.error("[Mic] permission/error:", e);
      setMicOn(false);
      return;
    }

    // audio graph
    const ctx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: 48000,
      latencyHint: "interactive",
    });
    try {
      await ctx.resume();
    } catch {}
    const source = ctx.createMediaStreamSource(stream);
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);

    audioRef.current = {
      ctx,
      stream,
      source,
      analyser,
      raf: 0,
      deviceId: id || "",
      vad: {
        highSince: 0,
        lowSince: 0,
        recording: false,
        recorder: null,
        chunks: [],
        startTs: 0,
        failedStarts: 0,
      },
    };
    setMicOn(true);
    setSessionStatus(isCamLive() ? "ACTIVE" : "STANDBY");

    // optional initial auto-cal
    if (autoDetectOnRef.current && !speakingRef.current) {
      setTimeout(async () => {
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

    // choose MediaRecorder MIME
    let mimeType = "";
    if (typeof MediaRecorder !== "undefined") {
      const prefs = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/mp4"
      ];
      mimeType = prefs.find(t => MediaRecorder.isTypeSupported(t)) || "";
    }

    // meter & VAD
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

      smooth = smooth === -60 ? d : EMA * d + (1 - EMA) * smooth;
      setDbfs(smooth);

      const vad = audioRef.current.vad;
      const nowTs = performance.now();

      const thr = thresholdLiveRef.current;
      const minKeepMs = Math.max(400, listenMsLiveRef.current);
      const silenceDur = silenceMsLiveRef.current;

      const edgeLevel = d;
      const isLoud = edgeLevel >= thr;

      // pause VAD when bot is talking
      if (speakingRef.current) {
        if (vad.recording) {
          const rec = vad.recorder;
          if (rec && rec.state === "recording") {
            try {
              rec.requestData?.();
            } catch {}
            try {
              rec.stop();
            } catch {}
          }
          vad.recording = false;
          vad.recorder = null;
          vad.chunks = [];
          vad.startTs = 0;
        }
        audioRef.current.raf = requestAnimationFrame(tick);
        return;
      }

      const HYSTERESIS_DB = 2;
      const QUICK_STOP_DB = 8;
      const QUICK_STOP_MS = 220;
      const riseEdge = edgeLevel >= thr;
      const fallEdge = edgeLevel < thr - HYSTERESIS_DB;
      if (isLoud && !vad.highSince) {
        vad.highSince = nowTs;
        vad.lowSince = 0;
      }

      // START
      if (riseEdge && !vad.recording) {
        try {
          if (typeof MediaRecorder !== "undefined") {
            const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
            vad.recorder = rec;
            vad.chunks = [];
            vad.startTs = nowTs;
            vad.recording = true;
            vad.lowSince = 0;

            rec.ondataavailable = (e) => {
              if (e.data?.size) vad.chunks.push(e.data);
            };
            rec.onstop = async () => {
              try {
                const blob = new Blob(vad.chunks, {
                  type: rec.mimeType || mimeType || "audio/webm",
                });
                const dur = performance.now() - vad.startTs;
                if (dur < minKeepMs) {
                  vad.failedStarts = (vad.failedStarts || 0) + 1;
                  if (
                    autoDetectOnRef.current &&
                    vad.failedStarts >= FAILED_STARTS_BEFORE_AUTOCAL
                  ) {
                    vad.failedStarts = 0;
                    maybeAutoCalibrate();
                  }
                  return;
                }
                vad.failedStarts = 0;

                // TODO: Upload to your STT endpoint; for now simulate
                handleTranscriptResult({ text: "(utterance)" });
              } catch (e) {
                console.error("[STT] onstop error:", e);
              } finally {
                vad.recording = false;
                vad.recorder = null;
                vad.chunks = [];
                vad.startTs = 0;
              }
            };
            try {
              rec.start(250);
            } catch {
              rec.start();
            }
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
          if (lowDur >= silenceDur || (edgeLevel < thr - 8 && lowDur >= 220)) {
            const rec = vad.recorder;
            if (rec && rec.state === "recording") {
              try {
                rec.requestData?.();
              } catch {}
              try {
                rec.stop();
              } catch {}
            }
          }
        } else {
          vad.lowSince = 0;
        }
        const MAX_UTTER_MS = 15_000;
        if (utterMs >= MAX_UTTER_MS) {
          const rec = vad.recorder;
          if (rec && rec.state === "recording") {
            try {
              rec.requestData?.();
            } catch {}
            try {
              rec.stop();
            } catch {}
          }
        }
      } else {
        if (!isLoud) {
          vad.lowSince = vad.lowSince || nowTs;
          vad.highSince = 0;
        } else {
          if (!vad.highSince) vad.highSince = nowTs;
          vad.lowSince = 0;
        }
      }

      audioRef.current.raf = requestAnimationFrame(tick);
    };

    audioRef.current.raf = requestAnimationFrame(tick);
  }

  /* ---------- Camera ---------- */
  async function startCamera(id = videoId) {
    try {
      camRef.current.stream?.getTracks()?.forEach((t) => t.stop());
    } catch {}

    let stream = null;
    if (id) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: { exact: id },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            frameRate: { ideal: 30 },
          },
          audio: false,
        });
      } catch (e) {
        console.warn("[Cam] chosen device failed; fallback:", e.name);
        if (e.name === "OverconstrainedError" || e.name === "NotFoundError") {
          setVideoId("");
          try {
            localStorage.setItem("ika:videoId", "");
          } catch {}
        }
      }
    }

    if (!stream) {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: "user",
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            frameRate: { ideal: 30 },
          },
          audio: false,
        });
      } catch (e) {
        console.error("[Cam] default device failed:", e);
        alert("Camera not available or permission denied.");
        return;
      }
    }

    camRef.current.stream = stream;

    try {
      const onCamGone = () => {
        userMicOffRef.current = true;
        stopAll({ reset: true });
      };
      const vTracks = stream.getVideoTracks();
      const handleMute = () => {
        // Mute can fire briefly on startup; verify after a short delay.
        setTimeout(() => {
          const live = isCamLive();
          const anyMuted = vTracks.some(tr => tr.muted);
          if (!live && anyMuted) onCamGone();
        }, 1200);
      };
      vTracks.forEach((t) => {
        t.addEventListener("ended", onCamGone);
        t.onended = onCamGone;
        t.addEventListener("mute", handleMute);
        t.addEventListener("unmute", () => {});
      });
      if (typeof stream.addEventListener === "function") {
        stream.addEventListener("inactive", onCamGone);
      }
    } catch {}

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.onloadedmetadata = () => {
        const w = videoRef.current.videoWidth || 1280;
        FOCAL_PX = w >= 1920 ? 1350 : 900;
        camFxRef.current = focalFromFov(videoRef.current.videoWidth || 1280,  fovHdeg);
        camFyRef.current = focalFromFov(videoRef.current.videoHeight || 720, fovVdeg);
        tinyOptsRef.current = new faceapi.TinyFaceDetectorOptions({
          inputSize: pickInputSize(w),
          scoreThreshold: 0.4,
        });
      };
    }

    if (videoRef.current) {
      const v = videoRef.current;
      v.srcObject = stream;

      // make sure it’s allowed to autoplay on mobile/Safari
      v.muted = true;
      v.playsInline = true;
      v.autoplay = true;

      // actually start the video
      try { await v.play(); } catch (e) {
        console.warn("[Cam] video.play() blocked until user gesture:", e);
      }

      v.onloadedmetadata = () => {
        const w = v.videoWidth || 1280;
        FOCAL_PX = w >= 1920 ? 1350 : 900;
        camFxRef.current = focalFromFov(v.videoWidth || 1280,  fovHdeg);
        camFyRef.current = focalFromFov(v.videoHeight || 720,  fovVdeg);
        tinyOptsRef.current = new faceapi.TinyFaceDetectorOptions({
          inputSize: pickInputSize(w),
          scoreThreshold: 0.4,
        });
      };
    }

    try {
      const list = await navigator.mediaDevices.enumerateDevices();
      setAudioDevs(list.filter((d) => d.kind === "audioinput"));
      setVideoDevs(list.filter((d) => d.kind === "videoinput"));
    } catch {}
  }

  /* ---------- Calibrate Camera ---------- */
  async function calibrateCameraOneClick() {
    // Assumes one person stands ~center at calibDistanceM
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return;

    calibMsgRef.current = "Stand still… Calibrating";
    const W = canvas.width, H = canvas.height;
    const samples = [];
    const N = 8;

    for (let i = 0; i < N; i++) {
      await sleep(120);
      const dets = await faceapi
        .detectAllFaces(video, tinyOptsRef.current)
        .withFaceLandmarks();

      if (!dets?.length) continue;

      const det = faceapi.resizeResults(dets[0], { width: W, height: H });
      const box = det.detection.box;

      // Estimate focal from known distance & observed face width
      if (!box?.width) continue;
      const fxEst = (box.width * calibDistanceM) / FACE_WIDTH_M;  // pinhole: Z = f*W_real / W_px => f = W_px*Z/W_real

      // Center we measured at (slightly below mid-eye line helps)
      const cx = box.x + box.width * 0.5;
      const cy = box.y + box.height * 0.45;

      const { yaw, pitch } = anglesFromPixel(
        cx, cy,
        camFxRef.current, camFyRef.current,
        W * 0.5, H * 0.5
      );

      samples.push({
        fx: fxEst,
        yawDeg: yaw * RAD,
        pitchDeg: pitch * RAD,
      });
    }
    calibMsgRef.current = "";

    if (!samples.length) return;

    const median = (arr) => {
      const a = [...arr].sort((x, y) => x - y);
      return a[Math.floor(a.length / 2)];
    };

    const fxMed = median(samples.map(s => s.fx));
    const yawMed = median(samples.map(s => s.yawDeg));
    const pitchMed = median(samples.map(s => s.pitchDeg));

    // Update intrinsics + FOV readouts
    camFxRef.current = fxMed;
    const fovH = 2 * Math.atan((W / 2) / fxMed) * RAD;
    setFovHdeg(+fovH.toFixed(1));

    // Approximate fy via square-pixel aspect (good enough for alignment UI)
    const fy = fxMed * (H / W);
    camFyRef.current = fy;
    const fovV = 2 * Math.atan((H / 2) / fy) * RAD;
    setFovVdeg(+fovV.toFixed(1));

    // Zero offsets so centered person yields yaw≈0, pitch≈0
    setPanOffsetDeg(p => p - yawMed);
    setTiltOffsetDeg(t => t - pitchMed);
  }

  async function runCalCountdown() {
    // Cute 3-2-1 banner in the overlay
    for (const n of [3, 2, 1]) {
      calibMsgRef.current = `Calibration in ${n}… Stand on the ${calibDistanceM.toFixed(2)} m mark`;
      await sleep(500);
    }
    calibMsgRef.current = "Calibrating…";
    await calibrateCameraOneClick();
    calibMsgRef.current = "Done!";
    await sleep(600);
    calibMsgRef.current = "";
  }

  // Distance from face width (px) using current intrinsics
  const estimateDistanceMpx = useCallback((wPx) => {
    const fx = camFxRef.current || FOCAL_PX;
    return (Number.isFinite(wPx) && wPx > 0) ? (fx * FACE_WIDTH_M) / wPx : null;
  }, []);

  /* ---------- Frame loop ---------- */
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
    video.addEventListener("resize", resize);
    resize();

    let raf = 0;
    let lastRun = 0;
    let detecting = false;
    let frameCount = 0;

    const loop = async () => {
      raf = requestAnimationFrame(loop);
      const now = performance.now();
      if (now - lastRun < loopStepMsRef.current || !video.videoWidth || video.readyState < 2 || detecting) return;
      frameCount++;
      if (audioRef.current?.vad?.recording && frameCount % 2 === 0) return;
      lastRun = now;

      detecting = true;
      try {
        if (isCamLive()) {
          lastFrameTsRef.current = performance.now();
        } else {
          // Bail early; no point running detections without a live track.
          return;
        }

        // === Guard: don't detect while switching backends ===
        if (backendSwitchingRef.current) {
          return; // skip this frame; will resume when backend is ready
        }
        // Ensure backend is fully ready (awaits if mid-initialization)
        await backendReadyRef.current;

        // ---- choose detection chain; stagger age/gender sampling ----
        const heavyAgeNow = (now - (lastAgeSampleRef.current || 0)) >= AGE_SAMPLE_MS;
        if (heavyAgeNow) lastAgeSampleRef.current = now;

        let dets = [];
        try {
          let chain = faceapi.detectAllFaces(video, tinyOptsRef.current);
          if (faceapi.nets.faceLandmark68Net?.isLoaded)  chain = chain.withFaceLandmarks();
          if (faceapi.nets.faceExpressionNet?.isLoaded) chain = chain.withFaceExpressions();
          if (heavyAgeNow && faceapi.nets.ageGenderNet?.isLoaded) chain = chain.withAgeAndGender();
          if (faceapi.nets.faceRecognitionNet?.isLoaded) chain = chain.withFaceDescriptors();
          dets = await chain;
        } catch (e) {
          console.warn("faceapi detect chain failed:", e?.message || e);
          dets = [];
        }

        // ==== drawing + bookkeeping ====
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        // --- alignment overlay: crosshair and banner ---
        if (showAlignRef.current) {
          const cx0 = canvas.width * 0.5, cy0 = canvas.height * 0.5;

          // crosshair ticks
          ctx.save();
          ctx.strokeStyle = "#0ea5e9";
          ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(cx0 - 22, cy0); ctx.lineTo(cx0 + 22, cy0); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(cx0, cy0 - 22); ctx.lineTo(cx0, cy0 + 22); ctx.stroke();
          ctx.restore();

          // transient banner (countdown / "Calibrating…")
          if (calibMsgRef.current) {
            const msg = calibMsgRef.current;
            ctx.save();
            ctx.font = "bold 18px system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif";
            const tw = ctx.measureText(msg).width + 18;
            const x = Math.max(10, (canvas.width - tw) / 2);
            const y = 10;
            ctx.fillStyle = "rgba(14,165,233,0.18)";
            ctx.fillRect(x, y, tw, 34);
            ctx.strokeStyle = "rgba(14,165,233,0.45)";
            ctx.strokeRect(x, y, tw, 34);
            ctx.fillStyle = "#e6f7ff";
            ctx.textBaseline = "middle";
            ctx.fillText(msg, x + 9, y + 17);
            ctx.restore();
          }
        }
        ctx.font = LABEL_FONT;
        ctx.textBaseline = "top";

        const resized = faceapi
          .resizeResults(dets, { width: canvas.width, height: canvas.height })
          .sort((a, b) => a.detection.box.x - b.detection.box.x);

        const matcher = faceMatcherRef.current;
        const rows = [];
        const peopleForPost = [];
        let total = 0, green = 0, red = 0;

        // build candidates (same as before)
        const cutoff = Number.isFinite(redCutoffM) ? redCutoffM : Infinity;
        const candidates = [];
        for (let i = 0; i < resized.length; i++) {
          const det = resized[i];
          const box = det.detection.box;
          const dist = estimateDistanceMpx(box.width);
          if (dist != null && dist > cutoff) continue; // skip way too far
          const zone = zoneOf(dist, greenMaxMRef.current);
          candidates.push({ i, det, box, dist, zone });
        }

        // Totals for status/mic policy (all visible faces within cutoff)
        total = candidates.length;
        green = candidates.filter(c => c.zone === "green").length;
        red   = total - green;

        // 2) Only TRACK up to 5 people in the GREEN zone, nearest first
        const greenCandidates = candidates
          .filter(c => c.zone === "green" && Number.isFinite(c.dist))
          .sort((a, b) => a.dist - b.dist);
        const tracked = greenCandidates.slice(0, 5);

        if (tracked.length) {
          ctx.save();
          ctx.font = "bold 12px system-ui";
          const msg = `tracked: ${tracked.length}`;
          const w = ctx.measureText(msg).width + 10;
          ctx.fillStyle = "rgba(34,197,94,0.85)";
          ctx.fillRect(10, 10, w, 20);
          ctx.fillStyle = "#fff";
          ctx.fillText(msg, 15, 24);
          ctx.restore();
        }

        // Define per-frame gesture eligibility set (top-2 will be added below)
        const gestureAllowedKeys = new Set();

        // 3) Draw + identify only the tracked subset (stable by name/gid)
        const tracks = recentMapRef.current;
        for (let k = 0; k < tracked.length; k++) {
          const { i, det, box, dist, zone } = tracked[k];

          // --- recognition (fast path + small margin check) ---
          const matcher = faceMatcherRef.current;
          let name = null;
          if (matcher && det.descriptor) {
            const best = matcher.findBestMatch(det.descriptor);
            if (best && best.label !== "unknown" && best.distance <= MATCH_THRESHOLD) {
              name = best.label;
            } else if (
              best &&
              best.label !== "unknown" &&
              best.distance <= MATCH_THRESHOLD + 0.03
            ) {
              // Lightweight margin check vs next-best label
              const bestLabel = best.label;
              const bestDist = best.distance;
              let second = 1;
              for (const ld of matcher.labeledDescriptors) {
                if (ld.label === bestLabel) continue;
                for (const d of ld.descriptors) {
                  const dd = faceapi.euclideanDistance(det.descriptor, d);
                  if (dd < second) second = dd;
                }
              }
              if (second - bestDist >= MATCH_MARGIN) name = bestLabel;
            }
          }

          // --- guest id & display name ---
          let guestId = null;
          if (!name) guestId = assignGuestIdFor(det.descriptor);
          let displayName = name || guestId || "Guest";

          // --- stabilization keyed by stable identity (name or gid), not by index ---
          const stableKey = (name || guestId) ?? `tmp-${i}`;

          // Slot key (stable within this frame order; decouples from identity collisions)
          const slotKey = `slot-${k}`;

          // Mark top-2 by distance as gesture-eligible
          if (gesturesOnRef.current && k < gestureTargetsRef.current) gestureAllowedKeys.add(stableKey);

          const prev = tracks[stableKey];
          if (prev && prev.name !== displayName) {
            if ((prev.count || 0) < STABILIZE_FRAMES) {
              displayName = prev.name;
              prev.count = (prev.count || 0) + 1;
            } else {
              tracks[stableKey] = { name: displayName, count: 0 };
            }
          } else {
            tracks[stableKey] = { name: displayName, count: 0 };
          }

          // pick gender/age with staggered cache
          const cacheGA = ageGenderCacheRef.current.get(stableKey) || {};
          const genderRaw = det.gender ?? cacheGA.gender ?? "";
          const gender = String(genderRaw || "").toLowerCase();
          const ageVal = Number.isFinite(det.age) ? det.age : (Number.isFinite(cacheGA.age) ? cacheGA.age : null);
          if (heavyAgeNow && (Number.isFinite(det.age) || det.gender)) {
            ageGenderCacheRef.current.set(stableKey, { age: det.age, gender: det.gender });
          }
          const expr = topExpression(det.expressions);

          // === angles / position / mouth activity / draw ===
          const dbox = shrinkBox(box);
          const cx = dbox.x + dbox.width * 0.5;
          const cy = dbox.y + dbox.height * 0.45;

          const fx = camFxRef.current, fy = camFyRef.current;
          const cx0 = canvas.width * 0.5, cy0 = canvas.height * 0.5;

          const { yaw, pitch } = anglesFromPixel(cx, cy, fx, fy, cx0, cy0);
          let yawDeg = yaw * RAD + panOffRef.current;
          let pitchDeg = pitch * RAD + tiltOffRef.current;

          const Z = Number.isFinite(dist) ? dist : null;
          const pos = Z != null ? posFromPixel(cx, cy, fx, fy, cx0, cy0, Z) : { x:null, y:null, z:null };

          const normX = Math.min(1, Math.abs((cx - cx0) / (canvas.width * 0.5)));
          const normY = Math.min(1, Math.abs((cy - cy0) / (canvas.height * 0.5)));
          const centerNorm = Math.min(1, Math.hypot(normX, normY));

          // mouth EMA with hold (avoid 0-drops)
          let mouthActivity = 0;
          try {
            const lm = det.landmarks;
            const key = stableKey;
            const rec = mouthMapRef.current.get(key) || { ema: 0.3, t: now };
            const level = mouthMAR(lm);
            if (!Number.isFinite(level) || level <= 0) {
              // hold previous with gentle decay toward neutral 0.3
              rec.ema = 0.98 * rec.ema + 0.02 * 0.3;
            } else {
              rec.ema = rec.ema ? (0.7 * rec.ema + 0.3 * level) : level;
            }
            rec.t = now;
            mouthMapRef.current.set(key, rec);
            mouthActivity = Math.max(0, Math.min(1, rec.ema));
          } catch {
            const rec = mouthMapRef.current.get(stableKey);
            if (rec) mouthActivity = rec.ema; // hold last
          }

          // draw box
          ctx.strokeStyle = "#22c55e";
          ctx.lineWidth = BOX_LINE_WIDTH;
          ctx.strokeRect(dbox.x, dbox.y, dbox.width, dbox.height);

          // Per-face gesture label (no global fallback → true separation)
          const faceStable = perFaceStableRef.current.get(stableKey);
          const freshFaceGesture =
            gestureAllowedKeys.has(stableKey) &&
            faceStable && (now - faceStable.t) <= HANDS_CACHE_MS ? faceStable : null;

          const gestureLbl =
            (zone === "green" && freshFaceGesture) ? gestureLabelOf(freshFaceGesture) : null;

          const ageTxt = Number.isFinite(ageVal) ? Math.max(0, Math.round(ageVal)) : "-";
          const l1 = `${displayName}${gestureLbl ? " • " + gestureLbl : ""} • ${zone} • ${ageTxt} ${gender} • ${expr}`;
          const l2 = `yaw ${yawDeg.toFixed(1)}° · pitch ${pitchDeg.toFixed(1)}° · mouth ${mouthActivity.toFixed(2)}`;

          // ----- LABEL DRAW (fixed: define color; removed duplicate vars/badges) -----
          const color = zone === "green" ? "rgba(34,197,94,0.85)" : "rgba(239,68,68,0.85)";
          const lineH = 18;
          const lines = showAlignRef.current ? 2 : 1;
          const tw = Math.max(ctx.measureText(l1).width, showAlignRef.current ? ctx.measureText(l2).width : 0) + LABEL_PAD_X * 2;
          const th = lineH * lines + LABEL_PAD_Y * 2;

          const lx = Math.max(0, Math.min(dbox.x, canvas.width - tw));
          const ly = Math.max(0, dbox.y - th - 4);

          // pill background
          ctx.fillStyle = color;
          ctx.fillRect(lx, ly, tw, th);

          // text
          ctx.fillStyle = "#fff";
          ctx.fillText(l1, lx + LABEL_PAD_X, ly + LABEL_PAD_Y);
          if (showAlignRef.current) {
            ctx.fillStyle = "#e9ffef";
            ctx.fillText(l2, lx + LABEL_PAD_X, ly + LABEL_PAD_Y + lineH);
          }

          // tiny mouth bar
          if (showAlignRef.current) {
            const barW = 64, barH = 5, gap = 3;
            const bx = lx, by = ly + th + gap;
            ctx.fillStyle = "rgba(255,255,255,0.15)";
            ctx.fillRect(bx, by, barW, barH);
            ctx.fillStyle = "#22c55e";
            ctx.fillRect(bx, by, barW * Math.min(1, Math.max(0, mouthActivity)), barH);
          }

          // Per-face gesture text badge on box (keep only this one)
          if (freshFaceGesture && zone === "green") {
            const gtxt = gestureLabelOf(freshFaceGesture);
            if (gtxt) {
              ctx.save();
              ctx.font = "bold 14px system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif";
              const padX = 6, padY = 4;
              const tw2 = ctx.measureText(gtxt).width + padX * 2;
              const th2 = 18 + padY * 2;
              const gx = Math.max(0, Math.min(dbox.x + dbox.width - tw2 - 4, canvas.width - tw2));
              const gy = Math.max(0, dbox.y + 4);
              ctx.fillStyle = "rgba(34,197,94,0.9)";
              ctx.fillRect(gx, gy, tw2, th2);
              ctx.fillStyle = "#fff";
              ctx.fillText(gtxt, gx + padX, gy + padY);
              ctx.restore();
            }
          }

          // table + server payload
          rows.push({
            idx: rows.length + 1,
            name: displayName,
            gesture: gestureLbl || "-",
            emotion: expr || "-",
            zone,
            ageGroup: ageGroupOf(ageVal),
            gender,
            distance: dist ? dist.toFixed(2) + " m" : "-",
          });

          peopleForPost.push({
            gender,
            ageGroup: ageGroupOf(ageVal),
            zone,
            name: name || null,
            gid: guestId || null,
            emotion: expr,
            yawDeg,
            pitchDeg,
            posCam: pos,
            centerNorm,
            mouthActivity,
            stableKey, // carry stable identity for hand/gesture mapping
            slotKey,   // optional: keep slot for debug/UI
            _cx: cx, _cy: cy,
            _w: dbox.width, _h: dbox.height,
          });
        }

        // remember face centers for click-to-zero + per-face hands mapping
        trackedFacesRef.current = peopleForPost.map((p, idx) => ({
          cx: p._cx, cy: p._cy, w: p._w, h: p._h,
          yawDeg: p.yawDeg, pitchDeg: p.pitchDeg,
          key: p.stableKey,
          name: p.name || null,
          gid: p.gid || null,
          index: idx,
          gestureEligible: gestureAllowedKeys.has(p.stableKey),
          z: p.posCam?.z ?? null, // NEW: carry depth
        }));
        // Also expose ALL faces (GREEN + RED) for hand proximity (on_phone)
        allFacesRef.current = candidates.map(c => {
          const d = shrinkBox(c.box);
          return {
            cx: d.x + d.width * 0.5,
            cy: d.y + d.height * 0.45,
            w: d.width,
            h: d.height,
          };
        });

          // --- ALSO draw non-tracked faces so RED is visible ---
          try {
            // Build a fast lookup of indices we already drew
            const drawn = new Set(tracked.map(t => t.i));

            for (const c of candidates) {
              if (drawn.has(c.i)) continue; // skip tracked (already drawn)

              const { box, dist, zone } = c;
              const dbox = shrinkBox(box);

              // Choose color: RED for red-zone, grey for others we didn't track
              const stroke = zone === "red" ? "#ef4444" : "#999999";
              const fill   = zone === "red" ? "#ef4444" : "#666666";

              // Outline
              ctx.save();
              ctx.strokeStyle = stroke;
              ctx.lineWidth = 3;
              ctx.strokeRect(dbox.x, dbox.y, dbox.width, dbox.height);

              // Minimal label: zone + distance
              const l1 = `${zone} · ${dist ? dist.toFixed(2) + " m" : "-"}`;
              const lineH = 18;
              const tw = ctx.measureText(l1).width + LABEL_PAD_X * 2;
              const th = lineH + LABEL_PAD_Y * 2;

              const lx = Math.max(0, Math.min(dbox.x, canvas.width - tw));
              const ly = Math.max(0, dbox.y - th - 4);

              ctx.fillStyle = fill;
              ctx.fillRect(lx, ly, tw, th);
              ctx.fillStyle = "#ffffff";
              ctx.fillText(l1, lx + LABEL_PAD_X, ly + LABEL_PAD_Y);
              ctx.restore();
            }
          } catch {}

        // prune only faces that are no longer tracked (preserve history for tracked-but-not-eligible)
        {
          const keep = new Set((trackedFacesRef.current || []).map(f => f.key));
          for (const k of Array.from(mouthMapRef.current.keys())) {
            if (!keep.has(k)) mouthMapRef.current.delete(k);
          }
          for (const k of Array.from(perFaceStableRef.current.keys())) {
            if (!keep.has(k)) perFaceStableRef.current.delete(k);
          }
          for (const k of Array.from(waveHistByFaceRef.current.keys())) {
            if (!keep.has(k)) waveHistByFaceRef.current.delete(k);
          }
        }

        // prune unused track slots (use stable keys: name or gid)
        {
          const seen = new Set(peopleForPost.map(p => (p.name || p.gid) ?? ""));
          for (const k of Object.keys(recentMapRef.current)) {
            if (k && !seen.has(k)) delete recentMapRef.current[k];
          }
        }

        // === Focus selection (prefer GREEN, fallback to any tracked) ===
        const pool = peopleForPost.filter(p => p.zone === "green");
        const cand = pool.length ? pool : peopleForPost;

        let focusIndex = (cand.length ? 0 : -1);
        let focusScore = -1, focusMeta = null;

        for (let idx = 0; idx < cand.length; idx++) {
          const p = cand[idx];
          let sNear = 0;
          const z = p?.posCam?.z;
          if (Number.isFinite(z) && z > 0) {
            sNear = Math.max(0, Math.min(1, (2.0 - Math.min(2.0, Math.max(0.3, z))) / (2.0 - 0.3)));
          }
          const sCenter = 1 - Math.max(0, Math.min(1, p.centerNorm ?? 1));
          const sMouth = Math.max(0, Math.min(1, p.mouthActivity ?? 0));
          const score = wNear * sNear + wCenter * sCenter + wMouth * sMouth;
          if (score > focusScore) {
            focusScore = score;
            focusIndex = idx;
            focusMeta = { sNear:+sNear.toFixed(3), sCenter:+sCenter.toFixed(3), sMouth:+sMouth.toFixed(3), score:+score.toFixed(3) };
          }
        }

        // Save focus so other blocks (hands/policy) can include it
        if (focusIndex >= 0) {
          const p = cand[focusIndex];
          focusIndexRef.current = peopleForPost.indexOf(p);
          focusTargetRef.current = { name: p.name || null, gid: p.gid || null };
        } else {
          focusIndexRef.current = -1;
          focusTargetRef.current = null;
        }

        // Mic policy with grace
        const shouldListen = total > 0 && green === total; // unchanged: all visible faces must be green
        const nowMs = performance.now();
        if (shouldListen) {
          lastAllGreenRef.current = nowMs;
          if (!micOnRef.current) {
            userMicOffRef.current = false;
            startMic().catch(() => {});
            setSessionStatus("ACTIVE");
            if (!sessionId) setSessionId(uuid());
          }
        } else {
          const sinceAllGreen = nowMs - (lastAllGreenRef.current || 0);
          if (micOnRef.current && sinceAllGreen >= MIC_STOP_GRACE_MS) {
            userMicOffRef.current = true;
            stopMic().catch(() => {});
          }
        }

        // Always render 5 rows max; pad if fewer tracked
        while (rows.length < 5) {
          rows.push({
            idx: rows.length + 1,
            gender: "-",
            ageGroup: "-",
            zone: "-",
            name: "-",
            gesture: "-",
            emotion: "-",
            distance: "-"
          });
        }

        setTable((prev) => {
          const same =
            prev.length === rows.length &&
            prev.every((r, i) => JSON.stringify(r) === JSON.stringify(rows[i]));
          return same ? prev : rows;
        });
        setTotals((prev) =>
          prev.all === total && prev.green === green && prev.red === red ? prev : { all: total, green, red }
        );

         // Game mode auto-exit on no-face stretch
         if (gameModeRef.current) {
           const now2 = performance.now();
           if (green > 0) lastGameActivityRef.current = now2;
           if (green === 0 && (now2 - (lastGameActivityRef.current || 0)) > GM_NO_FACE_TIMEOUT_MS) {
             setGameModeOn(false);
           }
         }

        // Battery saver: slow the loop when no faces are present
        if (total === 0) {
          if (!loopIdleStateRef.current) {
            loopIdleStateRef.current = true;
            // pick a stable jittered idle step once per idle stretch
            const jitter =
              LOOP_STEP_IDLE_MIN_MS +
              Math.floor(Math.random() * (LOOP_STEP_IDLE_MAX_MS - LOOP_STEP_IDLE_MIN_MS + 1));
            loopStepMsRef.current = jitter;
          }
        } else if (loopIdleStateRef.current) {
          loopIdleStateRef.current = false;
          loopStepMsRef.current = LOOP_STEP_ACTIVE_MS;
        }

        // ---- HANDS: per-person attribution (nearest green face) ----
        const handsEligible = HANDS_ENABLED && handsReadyRef.current && gesturesOnRef.current;
        const gm = !!gameModeRef.current;
        const handsDesiredStep =
          (now - (lastLmSeenTsRef.current || 0) <= 800)
            ? (gm ? GM_HANDS_FAST_MS : HANDS_FAST_MS)
            : (gm ? GM_HANDS_IDLE_MS : HANDS_IDLE_MS);

        if (handsEligible && (now - (lastHandsRunTsRef.current || 0) >= handsDesiredStep)) {
          lastHandsRunTsRef.current = now;
          try {
            const handsList = await detectHandsOnce(video);

            if (handsList && handsList.length) {
              // HUD + wrist dots
              ctx.save();
              ctx.font = "bold 14px system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif";
              const msg = `hands: ${handsList.length}`;
              const w = ctx.measureText(msg).width + 12;
              ctx.fillStyle = "rgba(14,165,233,0.85)";
              ctx.fillRect(10, canvas.height - 62, w, 22);
              ctx.fillStyle = "#fff";
              ctx.fillText(msg, 16, canvas.height - 46);
              ctx.restore();

              for (const lm of handsList) {
                const wrist = lm[0];
                const px = wrist.x * canvas.width;
                const py = wrist.y * canvas.height;
                ctx.beginPath();
                ctx.arc(px, py, 5, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(255,255,0,0.8)";
                ctx.fill();
              }
              const byFace = new Map();

              // helper: anchor near palm base (more stable than wrist alone)
              const handAnchor = (lm) => {
                const w = lm[MP.WRIST], i = lm[MP.INDEX_MCP];
                if (!w || !i) return null;
                return { x: (w.x + i.x) * 0.5, y: (w.y + i.y) * 0.5 };
              };

              // helper: is anchor inside (expanded) face rect
              const wristInFace = (px, py, f) => {
                const left = f.cx - f.w * 0.5;
                const top  = f.cy - f.h * 0.45;
                const right = left + f.w;
                const bottom = top + f.h;
                // allow more room above the face (hands raised), keep sides tighter
                const mx = f.w * 0.09, myUp = f.h * 0.35, myDown = f.h * 0.22;
                return px >= left - mx && px <= right + mx && py >= top - myUp && py <= bottom + myDown;
              };

              // Eligible faces (top-2 closest, marked earlier)
              const facesAll = trackedFacesRef.current || [];
              const faces = facesAll.filter(f => f.gestureEligible);
              if (!faces.length) {
                // No eligible faces this tick → keep per-face state; just update global/fallback below
              } else {
                // 1) Build hand→face candidate pairs (eligible faces only)
                const hands = handsList.map((lm, hi) => {
                  const a = handAnchor(lm);
                  if (!a) return null;
                  const ax = a.x * canvas.width;
                  const ay = a.y * canvas.height;
                  return { lm, hi, ax, ay };
                }).filter(h => {
                  if (!(h && Number.isFinite(h.ax) && Number.isFinite(h.ay))) return false;
                  // reject tiny/tentative hands (ghosts)
                  const span = palmSpanLen(h.lm); // normalized 0..1
                  return span >= 0.020; // ~2% of frame width
                });

                const allPairs = [];
                for (const h of hands) {
                  // primary: anchor inside face window
                  let contenders = faces.filter(f => wristInFace(h.ax, h.ay, f));

                  // fallback: if none hit, accept nearest face if horizontally aligned and vertically near
                  if (!contenders.length) {
                    let best = null, bestDx = Infinity;
                    for (const f of faces) {
                      const dx = Math.abs(h.ax - f.cx);
                      const withinX = dx <= f.w * 0.45;
                      const withinY = h.ay >= (f.cy - f.h * 0.60) && h.ay <= (f.cy + f.h * 0.20);
                      if (withinX && withinY && dx < bestDx) { bestDx = dx; best = f; }
                    }
                    if (best) contenders = [best];
                  }

                  // final fallback: always assign (handles hands far from faces)
                 if (!contenders.length) {
                   if (faces.length === 1) {
                     // single eligible face → give it the hand
                     contenders = [faces[0]];
                   } else if (faces.length === 2) {
                     // 2 faces: split by midline (stable left/right assignment)
                     const [leftF, rightF] = faces[0].cx <= faces[1].cx ? [faces[0], faces[1]] : [faces[1], faces[0]];
                     const midX = (leftF.cx + rightF.cx) * 0.5;
                     contenders = [h.ax <= midX ? leftF : rightF];
                   } else {
                     // 3+ faces: nearest center with small vertical penalty
                     let bestN = null, bestScore = Infinity;
                     for (const f of faces) {
                       const dx = h.ax - f.cx, dy = h.ay - f.cy;
                       const score = dx*dx + (dy*dy) * 0.4;
                       if (score < bestScore) { bestScore = score; bestN = f; }
                     }
                     if (bestN) contenders = [bestN];
                   }
                 }

                  for (const f of contenders) {
                    const dx = h.ax - f.cx, dy = h.ay - f.cy;
                    allPairs.push({ hi: h.hi, lm: h.lm, face: f, d2: dx*dx + dy*dy });
                  }
                }

                // 2) Per-hand filter: keep only best candidate; drop if ambiguous
                const byHand = new Map(); // hi -> sorted pairs
                for (const p of allPairs) {
                  const arr = byHand.get(p.hi) || [];
                  arr.push(p);
                  byHand.set(p.hi, arr);
                }
                const filtered = [];
                for (const [hi, arr] of byHand.entries()) {
                  arr.sort((a,b) => a.d2 - b.d2);
                  const best = arr[0];
                  const second = arr[1];
                  // size-scaled near-tie (don’t drop unless truly ambiguous)
                  const wRef = second ? Math.max(best.face.w || 1, second.face.w || 1) : 1;
                  if (second) {
                    const nearTie = Math.abs(best.d2 - second.d2) <= (wRef * 0.15) * (wRef * 0.15);
                    if (nearTie) {
                      // depth tiebreak: prefer nearer-Z
                      const zBest = Number.isFinite(best.face.z) ? best.face.z : Infinity;
                      const zSecond = Number.isFinite(second.face.z) ? second.face.z : Infinity;
                      if (zSecond < zBest - 0.05) { filtered.push(second); continue; }
                    }
                  }
                  filtered.push(best);
                }

                // 3) Greedy assign without collisions
                filtered.sort((a, b) => a.d2 - b.d2);
                const usedHands = new Set();
                const usedFaces = new Set();
                const assignments = [];
                for (const p of filtered) {
                  if (usedHands.has(p.hi) || usedFaces.has(p.face.key)) continue;
                  assignments.push(p);
                  usedHands.add(p.hi);
                  usedFaces.add(p.face.key);
                }

                // 4) Classify per assigned face (wave history is per-face)
                // Game mode runs 4 classifiers (wave + R/P/S); normal runs 5 (wave + peace + raise_hand + on_phone + thumbs_up)
                for (const { lm, face } of assignments) {
                  let hist = waveHistByFaceRef.current.get(face.key);
                  if (!hist) { hist = { t: 0, xs: [] }; waveHistByFaceRef.current.set(face.key, hist); }
                  const xs = hist.xs;
                  if (now - (hist.t || 0) > 900) xs.length = 0;
                  hist.t = now;

                  waveHistRef.xs = xs;
                  waveHistRef.t = hist.t;

                  const a0 = handAnchor(lm);
                  let allowWave = false;
                  if (a0) {
                    const ax0 = a0.x * canvas.width, ay0 = a0.y * canvas.height;
                    // near this face box or broadly aligned band
                    if (wristInFace(ax0, ay0, face)) {
                      allowWave = true;
                    } else {
                      const dx = Math.abs(ax0 - face.cx);
                      const withinX = dx <= face.w * 0.90;
                      const withinY = ay0 >= (face.cy - face.h * 1.10) && ay0 <= (face.cy + face.h * 0.50);
                      allowWave = withinX && withinY;
                    }
                  }
                  // if far from box but motion clearly wave-y, still allow
                  if (!allowWave) {
                    const wa = waveActivity();
                    if (wa.flips >= 2 && wa.amp > 0.020) allowWave = true;
                  }

                  // debug ear anchor for assigned face `face` (safe: no undefined refs)
                  ctx.save();
                  ctx.fillStyle = "rgba(0,180,255,0.8)";
                  const handX = a0 ? (a0.x * canvas.width) : ((lm[MP.WRIST]?.x || 0) * canvas.width);
                  const sideSignDbg = handX >= face.cx ? +1 : -1;
                  const earX = face.cx + sideSignDbg * (face.w * 0.5) * 0.78;
                  const earY = face.cy - (face.h * 0.5) * 0.08;
                  ctx.beginPath(); ctx.arc(earX, earY, 5, 0, Math.PI * 2); ctx.fill();
                  ctx.restore();

                  const velNow = recentLateralMotion();
                  const wrist = lm[MP.WRIST], iMcp = lm[MP.INDEX_MCP];
                  const vx = (iMcp?.x ?? 0) - (wrist?.x ?? 0), vy = (iMcp?.y ?? 0) - (wrist?.y ?? 0);
                  const axisLen = Math.hypot(vx, vy) || 1e-6;
                  const cosToVertical = Math.abs(vy) / axisLen; // 1 = vertical, 0 = horizontal

                  // Face-relative proximity for “pose” gestures (prevents random pops)
                  let allowNearFace = false, nearX = false, highPalm = false;
                  if (a0) {
                    const ax0 = a0.x * canvas.width, ay0 = a0.y * canvas.height;
                    nearX = Math.abs(ax0 - face.cx) <= face.w * 1.00;
                    // "high palm": above face center by a bit, even if not inside the box
                    highPalm = ay0 <= (face.cy - face.h * 0.05);
                    const highEnough = ay0 <= (face.cy + face.h * 0.25);
                    allowNearFace = (Math.abs(ax0 - face.cx) <= face.w * 0.85) && highEnough;
                  }

                  const cand = [];
                  try {
                    const w = classifyWave(lm, now);
                    // Accept always if wave is strong; else require near-face band
                    if (w.ok && (allowWave || w.score >= 0.62)) {
                      cand.push({ type: "wave", score: w.score });
                    }
                  } catch {}

                  const palm = palmSpanLen(lm);

                  if (gm) {
                    try { const r = classifyRock(lm);     if (r.ok) cand.push({ type: "rock",     score: r.score }); } catch {}
                    try { const s = classifyScissors(lm); if (s.ok) cand.push({ type: "scissors", score: s.score }); } catch {}
                    try {
                      const p = classifyPaper(lm);
                      // paper: open palm; allow near face OR clearly high & aligned; upright-ish; not swinging
                      if (p.ok && (allowNearFace || highPalm || palm >= 0.038) &&
                          velNow <= 0.11 && cosToVertical > 0.50 && palm >= 0.028) {
                        cand.push({ type: "paper", score: p.score });
                      }
                    } catch {}
                  } else {
                    try { const p = classifyPeace(lm); if (p.ok) cand.push({ type: "peace", score: p.score }); } catch {}
                    try {
                      const rh = classifyRaiseHand(lm);
                      if (rh.ok) cand.push({ type: "raise_hand", score: rh.score });
                    } catch {}
                    // phone stays scoped to this face (already done)
                    try {
                      const ph = classifyOnPhone(
                        lm,
                        [{ cx: face.cx, cy: face.cy, w: face.w, h: face.h }],
                        canvas.width,
                        canvas.height
                      );
                    if (ph.ok) cand.push({ type: "on_phone", score: ph.score });
                  } catch {}
                    try { const t = classifyThumbsUp(lm); if (t.ok) cand.push({ type: "thumbs_up", score: t.score }); } catch {}
                  }

                  // If a strong pose is present, drop weaker wave this frame
                  {
                    const poseBest = cand
                      .filter(c => c.type === "thumbs_up" || c.type === "peace" || c.type === "raise_hand" || c.type === "on_phone")
                      .sort((a,b) => b.score - a.score)[0];
                    const waveIdx = cand.findIndex(c => c.type === "wave");
                    if (poseBest && waveIdx >= 0) {
                      const waveScore = cand[waveIdx].score;
                      if (waveScore < poseBest.score + 0.12) {
                        cand.splice(waveIdx, 1);
                      }
                    }
                  }

                  if (!cand.length) continue;
                  const bestFrame = cand.reduce((a, b) => (b.score > a.score ? b : a));

                  const prev = byFace.get(face.key);
                  const adj = gm && bestFrame.type === "wave" ? { type: "paper", score: bestFrame.score }
                            : gm && bestFrame.type === "thumbs_up" ? null
                            : bestFrame;
                  if (adj && (!prev || adj.score > prev.score)) byFace.set(face.key, adj);
                }

                // update per-face windows/stable + emit changes
                const updatedKeys = new Set();
                for (const [key, frame] of byFace.entries()) {
                  const win = perFaceGestureWinRef.current.get(key) || [];
                  win.push({ ...frame, t: now });
                  if (win.length > VOTE_WINDOW * 2) win.splice(0, win.length - VOTE_WINDOW * 2);
                  perFaceGestureWinRef.current.set(key, win);

                  const prevStable = perFaceStableRef.current.get(key) || null;
                  const nextStable = pickStableGesture(now, win, prevStable);
                  if (nextStable) {
                    const changed = !prevStable || prevStable.type !== nextStable.type;
                    perFaceStableRef.current.set(key, nextStable);
                    updatedKeys.add(key);
                    const lastSent = lastGestureSentPerFaceRef.current.get(key) || 0;
                    if (changed && (now - lastSent) >= HANDS_SEND_MS && !speakingRef.current) {
                      const facesMeta = trackedFacesRef.current || [];
                      const meta = facesMeta.find(f => f.key === key) || {};
                      try {
                        socketRef.current?.emit?.(gm ? "game_event" : "gesture_event", gm ? {
                          sessionId: sessionId || ("web-" + deviceId),
                          deviceId,
                          rps: nextStable.type,
                          at: Date.now(),
                          focusIndex: meta.index ?? focusIndexRef.current,
                          focusTarget: { name: meta.name || null, gid: meta.gid || null },
                        } : {
                          sessionId: sessionId || ("web-" + deviceId),
                          deviceId,
                          gesture: { type: nextStable.type, score: nextStable.score },
                          at: Date.now(),
                          focusIndex: meta.index ?? focusIndexRef.current,
                          focusTarget: { name: meta.name || null, gid: meta.gid || null },
                        });
                      } catch {}
                      lastGestureSentPerFaceRef.current.set(key, now);
                    }
                  } else {
                    perFaceStableRef.current.delete(key);
                  }
                }

                // Update global stable gesture for legacy HUD/policy (focus face wins)
                (() => {
                  const faces = (trackedFacesRef.current || []);
                  const eligible = new Set(faces.filter(f => f.gestureEligible).map(f => f.key));
                  const fi = focusIndexRef.current;
                  let chosen = null;
                  if (fi >= 0 && faces[fi] && eligible.has(faces[fi].key)) {
                    chosen = perFaceStableRef.current.get(faces[fi].key) || null;
                  }
                  if (!chosen) {
                    for (const [k, g] of perFaceStableRef.current.entries()) {
                      if (!eligible.has(k)) continue;
                      if ((now - g.t) <= HANDS_CACHE_MS && g.type === "on_phone") { chosen = g; break; }
                    }
                  }
                  if (!chosen) {
                    for (const [k, g] of perFaceStableRef.current.entries()) {
                      if (!eligible.has(k)) continue;
                      if ((now - g.t) <= HANDS_CACHE_MS) { chosen = g; break; }
                    }
                  }
                  stableGestureRef.current = chosen ? { ...chosen, t: now } : null;
                })();
              }
            }

        // AFTER hands: emit snapshot with up-to-date gesture
        {
          const g = gesturesOnRef.current ? stableGestureRef.current : null;
          const fresh = g && (now - g.t) <= HANDS_CACHE_MS ? { type: g.type, score: g.score } : null;
          emitCrowdThrottled({
            deviceId,
            sessionId: sessionId || ("web-" + deviceId),
            timeISO: new Date().toISOString(),
            aiSpeaking: !!serverInfo.ai_speaking,
            backend,
            totals: { all: total, green, red },
            gesture: gesturesOnRef.current ? fresh : null,
            focusIndex: focusIndexRef.current,
            focusTarget: focusTargetRef.current,
            people: peopleForPost.map(p => ({
              name: p.name || null,
              gid: p.gid || null,
              gender: p.gender || null,
              ageGroup: p.ageGroup || null,
              zone: p.zone,
              yawDeg: Number.isFinite(p.yawDeg) ? +p.yawDeg.toFixed(1) : null,
              pitchDeg: Number.isFinite(p.pitchDeg) ? +p.pitchDeg.toFixed(1) : null,
              mouthActivity: +((p.mouthActivity ?? 0).toFixed(3)),
              posCam: p.posCam,
            })),
          });
        }
        } catch (e) {
          // ignore hand pipeline hiccups so the frame loop keeps running
        }

        // ---- Policy: zone transitions -> call-over / greet (candidates include red) ----
        try {
          const matcher = faceMatcherRef.current;

          const allIdentities = candidates.map((c) => {
            const det = c.det;
            // minimal recognition (same logic as tracked, compact)
            let name = null;
            if (matcher && det.descriptor) {
              const best = matcher.findBestMatch(det.descriptor);
              if (best && best.label !== "unknown" && best.distance <= MATCH_THRESHOLD) {
                name = best.label;
              } else if (best && best.label !== "unknown" && best.distance <= MATCH_THRESHOLD + 0.03) {
                const bestLabel = best.label;
                const bestDist = best.distance;
                let second = 1;
                for (const ld of matcher.labeledDescriptors) {
                  if (ld.label === bestLabel) continue;
                  for (const d of ld.descriptors) {
                    const dd = faceapi.euclideanDistance(det.descriptor, d);
                    if (dd < second) second = dd;
                  }
                }
                if (second - bestDist >= MATCH_MARGIN) name = bestLabel;
              }
            }
            let gid = null;
            if (!name) gid = assignGuestIdFor(det.descriptor);
            const key = (name || gid) ?? `tmp-${c.i}`;
            const gender = (det.gender || "").toLowerCase();
            const age = Number(det.age);
            const ageGroup = ageGroupOf(age);
            return { key, name, gid, gender, age, ageGroup, zone: c.zone };
          });

          // group context
          const groupSize = allIdentities.length;
          const hasKid = allIdentities.some(p => p.ageGroup === "child");

          // transitions
          const isOnPhone = (stableGestureRef.current?.type === "on_phone");
          for (const p of allIdentities) {
            const prevZ = prevZoneMapRef.current.get(p.key);
            prevZoneMapRef.current.set(p.key, p.zone);

            // green -> red => polite call-over (max 3, spaced)
            if (!isOnPhone && prevZ === "green" && p.zone === "red") {
              const s = callOverStateRef.current.get(p.key) || { tries: 0, last: 0 };
              if (s.tries < CALL_OVER_MAX_TRIES && (now - (s.last || 0)) >= CALL_OVER_COOLDOWN_MS) {
                s.tries += 1;
                s.last = now;
                callOverStateRef.current.set(p.key, s);
                socketRef.current?.emit?.("policy_event", {
                  deviceId,
                  sessionId: sessionId || ("web-" + deviceId),
                  type: "call_over",
                  attempt: s.tries,
                  target: { name: p.name || null, gid: p.gid || null, gender: p.gender || null },
                  group: { size: groupSize, hasKid },
                  reason: "left_green_zone",
                  at: Date.now(),
                });
              }
            }

            // red/unknown -> green => greet (reset tries)
            if (!isOnPhone && (prevZ === "red" || prevZ == null) && p.zone === "green") {
              callOverStateRef.current.delete(p.key);
              const address =
                p.name ? p.name
                : (groupSize > 1 ? (hasKid ? "family" : "everyone")
                   : (p.gender === "male" ? "sir" : p.gender === "female" ? "ma’am" : "there"));
              socketRef.current?.emit?.("policy_event", {
                deviceId,
                sessionId: sessionId || ("web-" + deviceId),
                type: "greet",
                address,
                target: { name: p.name || null, gid: p.gid || null, gender: p.gender || null },
                group: { size: groupSize, hasKid },
                at: Date.now(),
              });
            }
          }

          // group change detection (>50% different) -> ask softly
          const curSet = new Set(allIdentities.map(p => p.key));
          const prevSet = lastGroupSetRef.current || new Set();
          const inter = new Set([...curSet].filter(k => prevSet.has(k)));
          const overlap = (inter.size / Math.max(1, Math.max(prevSet.size, curSet.size)));
          if (prevSet.size && curSet.size && overlap < 0.5) {
            if ((now - (lastGroupAskTsRef.current || 0)) >= GROUP_ASK_COOLDOWN_MS) {
              lastGroupAskTsRef.current = now;
              socketRef.current?.emit?.("policy_event", {
                deviceId,
                sessionId: sessionId || ("web-" + deviceId),
                type: "ask_group_change",
                prevSize: prevSet.size,
                currSize: curSet.size,
                overlap,
                at: Date.now(),
              });
            }
          }
          // update last group
          lastGroupSetRef.current = curSet;
        } catch {}

        // ---- Speaker focus (1–2s or 3 frames dominance among green tracked) ----
        try {
          const list = peopleForPost || [];
          if (list.length) {
            // Prefer GREEN; else use all
            const greens = list.filter(p => p.zone === "green");
            const pool = greens.length ? greens : list;

            // Pick dominant by mouthActivity
            let topIdx = -1, topScore = -1, topKey = null;
            for (let i = 0; i < pool.length; i++) {
              const s = Number(pool[i].mouthActivity) || 0;
              if (s > topScore) {
                topScore = s;
                topIdx = i;
                topKey = (pool[i].name || pool[i].gid) ?? null;
              }
            }

            const sp = speakerRef.current;
            if (topKey && topKey === sp.topKeyPrev) {
              sp.framesDominant += 1;
            } else {
              sp.topKeyPrev = topKey;
              sp.framesDominant = 1;
              sp.topSince = now;
            }

            const stableByFrames = sp.framesDominant >= SPEAKER_STABLE_FRAMES;
            const stableByTime = (now - (sp.topSince || 0)) >= SPEAKER_STABLE_MS;

            if (topKey && sp.key !== topKey && (stableByFrames || stableByTime)) {
              // Map pool index back to absolute index in peopleForPost
              const absIdx = list.indexOf(pool[topIdx]);
              sp.key = topKey;

              const p = list[absIdx];
              socketRef.current?.emit?.("policy_event", {
                deviceId,
                sessionId: sessionId || ("web-" + deviceId),
                type: "speaker_focus",
                target: {
                  name: p.name || null,
                  gid: p.gid || null,
                  gender: p.gender || null,
                  index: absIdx,
                },
                at: Date.now(),
              });
            }
          }
        } catch (e) {
          console.warn("[speaker] error:", e);
        }
      }
      } catch (e) {
        console.warn("[frame] error:", e);
      } finally {
        detecting = false;
      }
    };

    // start the frame loop
    raf = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf);
      video?.removeEventListener("loadedmetadata", resize);
      video?.removeEventListener("resize", resize);
    };
  }, [ready, videoId, deviceId, server, sessionId]);

  // idle watcher
  useEffect(() => {
    if (!ready) return;
    const CHECK_MS = 1000;
    const timer = setInterval(() => {
      if (!isCamLive()) {
      stopAll({ reset: true });
      return;
    }
      const ago = performance.now() - lastFrameTsRef.current;

      // Mic idle: stop listening
      if (ago > MIC_IDLE_MS && micOnRef.current) {
        userMicOffRef.current = true;
        stopMic().catch(() => {});
      }

      // Camera idle: stop everything, including LLM
      if (ago > CAM_IDLE_MS) {
        stopAll({ reset: true }); // now this is the ONLY place that flips session to IDLE
      }
    }, CHECK_MS);
    return () => clearInterval(timer);
  }, [ready]);

  function isCamLive() {
    const s = camRef.current?.stream;
    if (!s) return false;
    const tracks = s.getVideoTracks?.() || [];
    if (!tracks.length) return false;
    return tracks.some((t) => t.readyState === "live" && t.enabled !== false);
  }

  // hard stop everything
  async function stopAll({ reset = true } = {}) {
    setDbfs(-60);

    // 👉 kill any pending “mic→standby” transition
    if (micIdleToStandbyTimerRef.current) {
      clearTimeout(micIdleToStandbyTimerRef.current);
      micIdleToStandbyTimerRef.current = null;
    }

    try { await stopMic(); } catch {}
    try { camRef.current.stream?.getTracks()?.forEach((t) => t.stop()); } catch {}
    camRef.current.stream = null;

    speakingRef.current = false;
    S.current = { id: null, seenFrames: 0, lastFaceTs: 0, lastSnapshotTs: 0 };

    try { socketRef.current?.emit?.("close_session", { sessionId }); } catch {}
    setSessionId(null);
    setSessionStatus("IDLE");   // full idle immediately
    recentMapRef.current = {};

    if (reset) bump("stop");
  }

  /* ====================== RIGHT-PANEL STATE (Gemini / ElevenLabs) ====================== */
  const MODEL_OPTIONS = [
    {
      value: "gemini-2.5-flash-live-preview",
      label: "Gemini 2.5 Flash Live Preview (realtime)",
      kind: "live",
    },
    {
      value: "gemini-2.5-flash-preview-native-audio",
      label: "Gemini 2.5 Flash Preview Native Audio (dialog)",
      kind: "native",
    },
  ];

  // Live Preview voices (as requested)
  const LIVE_VOICES = ["Puck", "Charon", "Kore", "Fenrir", "Aoede", "Leda", "Orus", "Zephyr"];
  // Native Audio dialog voices (list you gave)
  const NATIVE_VOICES = [
    "Zephyr","Puck","Charon","Kore","Fenrir","Leda","Orus","Aoede","Callirrhoe","Autonoe",
    "Enceladus","Iapetus","Umbriel","Algieba","Despina","Erinome","Algenib","Rasalgethi",
    "Laomedia","Achernar","Alnilam","Schedar","Gacrux","Pulcherrima","Achird","Zubenelgenubi",
    "Vindemiatrix","Sadachbia","Sadaltager"
  ];

  const GEMINI_VOICES = { live: LIVE_VOICES, native: NATIVE_VOICES };

  // Language options (label → code)
  const LANGS = [
    ["English (US)", "en-US"],
    ["English (UK)", "en-GB"],
    ["English (Australia)", "en-AU"],
    ["English (India)", "en-IN"],
    ["German", "de-DE"],
    ["Spanish (US)", "es-US"],
    ["Spanish (Spain)", "es-ES"],
    ["French", "fr-FR"],
    ["French (Canada)", "fr-CA"],
    ["Hindi", "hi-IN"],
    ["Portuguese (Brazil)", "pt-BR"],
    ["Arabic", "ar-SA"],
    ["Indonesian", "id-ID"],
    ["Italian", "it-IT"],
    ["Japanese", "ja-JP"],
    ["Turkish", "tr-TR"],
    ["Vietnamese", "vi-VN"],
    ["Bengali", "bn-BD"],
    ["Gujarati", "gu-IN"],
    ["Kannada", "kn-IN"],
    ["Malayalam", "ml-IN"],
    ["Marathi", "mr-IN"],
    ["Tamil", "ta-IN"],
    ["Telugu", "te-IN"],
    ["Dutch", "nl-NL"],
    ["Korean", "ko-KR"],
    ["Mandarin Chinese", "zh-CN"],
    ["Polish", "pl-PL"],
    ["Russian", "ru-RU"],
    ["Thai", "th-TH"],
  ];

  const [systemInstruction, setSystemInstruction] = useState(
    () => localStorage.getItem("ika:systemInstruction") || "You are a friendly, concise on-site concierge."
  );
  const [modelQuick, setModelQuick] = useState(
    () => localStorage.getItem("ika:model") || "gemini-2.5-flash-live-preview"
  );
  const modelKind = useMemo(
    () => MODEL_OPTIONS.find((m) => m.value === modelQuick)?.kind || "live",
    [modelQuick]
  );
  const voicesForKind = GEMINI_VOICES[modelKind] || LIVE_VOICES;

  const [geminiVoiceQuick, setGeminiVoiceQuick] = useState(
    () => localStorage.getItem("ika:voice") || "Puck"
  );
  const [languageCodeQuick, setLanguageCodeQuick] = useState(
    () => localStorage.getItem("ika:langCode") || "en-US"
  );
  const [temperatureQuick, setTemperatureQuick] = useState(
    () => Number(localStorage.getItem("ika:temperature") ?? 0.6)
  );

  // Behavior toggles
  const [enableAffectiveQuick, setEnableAffectiveQuick] = useState(
    () => localStorage.getItem("ika:enableAffective") === "true"
  );
  const [proactiveAudioQuick, setProactiveAudioQuick] = useState(
    () => localStorage.getItem("ika:proactiveAudio") === "true"
  );
  const [functionCallingQuick, setFunctionCallingQuick] = useState(
    () => localStorage.getItem("ika:functionCalling") === "true"
  );
  const [autoFunctionResponseQuick, setAutoFunctionResponseQuick] = useState(
    () => localStorage.getItem("ika:autoFunctionResponse") === "true"
  );
  const [groundingQuick, setGroundingQuick] = useState(
    () => localStorage.getItem("ika:grounding") === "true"
  );

  const [ttsProviderQuick, setTtsProviderQuick] = useState(
    () => (localStorage.getItem("ika:ttsProvider") || "gemini").toLowerCase()
  );
  const [elevenVoiceIdQuick, setElevenVoiceIdQuick] = useState(
    () => localStorage.getItem("ika:11labs:voiceId") || ""
  );

  // optional UI toggle elsewhere can flip this; default false
  const gameModeRef = useRef(false);
  const [gameModeOn, setGameModeOn] = useState(false);
  useEffect(() => { gameModeRef.current = gameModeOn; }, [gameModeOn]);

  // Game mode idle/visibility timeouts (ephemeral)
  const GM_IDLE_TIMEOUT_MS = 90_000;     // 1.5 min without activity → exit RPS
  const GM_NO_FACE_TIMEOUT_MS = 20_000;  // 20s with no faces → exit RPS
  const lastGameActivityRef = useRef(0);
  useEffect(() => {
    if (gameModeOn) lastGameActivityRef.current = performance.now();
  }, [gameModeOn]);

  // Focus shared across blocks (used by gesture events and crowd payload)
  const focusIndexRef = useRef(-1);
  const focusTargetRef = useRef(null);

  // API keys (stored locally; server may read)
  const [geminiApiKey, setGeminiApiKey] = useState(
    () => localStorage.getItem("ika:gemini:key") || ""
  );

  // persist knobs
  useEffect(() => {
    try {
      localStorage.setItem("ika:systemInstruction", systemInstruction);
      localStorage.setItem("ika:model", modelQuick);
      localStorage.setItem("ika:voice", geminiVoiceQuick);
      localStorage.setItem("ika:langCode", languageCodeQuick);
      localStorage.setItem("ika:temperature", String(temperatureQuick));

      localStorage.setItem("ika:enableAffective", String(enableAffectiveQuick));
      localStorage.setItem("ika:proactiveAudio", String(proactiveAudioQuick));
      localStorage.setItem("ika:functionCalling", String(functionCallingQuick));
      localStorage.setItem(
        "ika:autoFunctionResponse",
        String(autoFunctionResponseQuick)
      );
      localStorage.setItem("ika:grounding", String(groundingQuick));

      localStorage.setItem("ika:ttsProvider", ttsProviderQuick);
      localStorage.setItem("ika:11labs:voiceId", elevenVoiceIdQuick);

      localStorage.setItem("ika:captions", String(captions));
      localStorage.setItem("ika:locationLabel", locationLabel);
      localStorage.setItem("ika:weatherLabel", weatherLabel);

      localStorage.setItem("ika:gemini:key", geminiApiKey);
    } catch {}
  }, [
    systemInstruction,
    modelQuick,
    geminiVoiceQuick,
    languageCodeQuick,
    temperatureQuick,
    enableAffectiveQuick,
    proactiveAudioQuick,
    functionCallingQuick,
    autoFunctionResponseQuick,
    groundingQuick,
    ttsProviderQuick,
    elevenVoiceIdQuick,
    captions,
    locationLabel,
    weatherLabel,
    geminiApiKey,
  ]);

  useEffect(() => {
    if (!voicesForKind.includes(geminiVoiceQuick)) {
      setGeminiVoiceQuick(voicesForKind[0] || "Puck");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modelKind]);

  // apply quick settings → create (or recreate) session
  const onCreateSession = useCallback(() => {
    server.createSession({
      model: modelQuick,
      voice: geminiVoiceQuick,
      language_code: languageCodeQuick,
      system_instruction: systemInstruction,

      // route audio: Gemini (AUDIO+TEXT) vs ElevenLabs (TEXT only)
      tts_provider: ttsProviderQuick, // "gemini" | "elevenlabs"
      response_modalities: (ttsProviderQuick === "elevenlabs" ? ["TEXT"] : ["AUDIO", "TEXT"]),

      mic_input: true,
      enable_affective_dialog: enableAffectiveQuick,
      disable_mic_during_response: false,

      // Live AAD knobs
      start_of_speech_sensitivity: sosQuick,
      end_of_speech_sensitivity: eosQuick,
      prefix_padding_ms: prefixPadQuick,
      silence_duration_ms: silenceDurQuick,

      temperature: temperatureQuick,
      proactive_audio: proactiveAudioQuick,
      transcribe_user_audio: true,
      files_to_upload: null,

      // ✅ ElevenLabs-only fields (used iff tts_provider === "elevenlabs")
      eleven_model: localStorage.getItem("ika:11labs:model") || "eleven_turbo_v2_5",
      eleven_voice_id: localStorage.getItem("ika:11labs:voiceId") || "",
      eleven_api_key: localStorage.getItem("ika:11labs:key") || undefined,
      eleven_output_format: "pcm_24000",
    });
  }, [
    server,
    modelQuick,
    geminiVoiceQuick,
    languageCodeQuick,
    systemInstruction,
    ttsProviderQuick,
    enableAffectiveQuick,
    sosQuick,
    eosQuick,
    prefixPadQuick,
    silenceDurQuick,
    temperatureQuick,
    proactiveAudioQuick
  ]);

  // hot update (also sends current Live AAD values from Mic panel)
  const onHotUpdate = useCallback(() => {
    server.updateSettings({
      temperature: temperatureQuick,
      captions,

      enable_affective_dialog: enableAffectiveQuick,
      proactive_audio: proactiveAudioQuick,
      function_calling: functionCallingQuick,
      auto_function_response: autoFunctionResponseQuick,
      grounding: groundingQuick,

      start_of_speech_sensitivity: sosQuick,
      end_of_speech_sensitivity: eosQuick,
      prefix_padding_ms: prefixPadQuick,
      silence_duration_ms: silenceDurQuick,
    });
  }, [
    server,
    temperatureQuick,
    captions,
    enableAffectiveQuick,
    proactiveAudioQuick,
    functionCallingQuick,
    autoFunctionResponseQuick,
    groundingQuick,
    sosQuick,
    eosQuick,
    prefixPadQuick,
    silenceDurQuick,
  ]);

  /* ====================== UI ====================== */
    return (
    <main className="app">
      <div className="page2">
        {/* ===== LEFT COLUMN ===== */}
        <div className="leftcol">
          {/* Row 1: CAMERA/STATUS | MIC */}
          <div className="left-top2">
            {/* CAMERA / STATUS (compact, grouped) */}
            <div className="panel compact">
              <div className="statgrid">
                {/* ——— Section: Environment ——— */}
                <div className="block">
                  <div className="block-title">Environment</div>
                  <div className="kv"><b>Location:</b> {locationLabel}</div>
                  <div className="kv"><b>Time:</b> {clock.toLocaleTimeString()}</div>
                  <div className="kv"><b>Weather:</b> {weatherLabel}</div>
                  <div className="kv"><b>Backend:</b> {backend}</div>
                  <div className="kv"><b>Models:</b> {ready ? "loaded" : "loading…"}</div>
                </div>

                {/* ——— Section: Live status ——— */}
                <div className="block">
                  <div className="block-title">Live status</div>

                  {/* CAM */}
                  <div className="kv">
                    {(() => {
                      const live = isCamLive();
                      return (
                        <>
                          <span className={`dot ${live ? "ok" : "err"}`} />
                          <b>Cam:</b>&nbsp;{live ? "LIVE" : "IDLE"}
                        </>
                      );
                    })()}
                  </div>

                  {/* Server */}
                  <div className="kv">
                    <span className={`dot ${serverInfo.connected ? "ok" : "err"}`} />
                    <b>Server:</b>&nbsp;{serverInfo.connected ? "connected" : "disconnected"}
                  </div>
                  {serverInfo.model || serverInfo.tts ? (
                    <div className="kv muted small">
                      {serverInfo.model ? <>Model: {serverInfo.model}</> : null}
                      {serverInfo.model && serverInfo.tts ? " · " : null}
                      {serverInfo.tts ? <>TTS: {serverInfo.tts}</> : null}
                    </div>
                  ) : null}

                  {/* Device binding */}

                  {/* Device binding */}
                  <div className="kv">
                    <b>Device:</b>&nbsp;{deviceId.slice(0,8)}…
                    <span className="muted">
                      &nbsp;{serverInfo.boundDeviceId
                        ? `(bound ${String(serverInfo.boundDeviceId).slice(0,8)}…)`
                        : `(not bound)`}
                    </span>
                  </div>
                </div>

                {/* ——— Section: Traffic ——— */}
                <div className="block">
                  <div className="block-title">Traffic</div>

                  <div className="kv chiprow">
                    <b>Last:</b>
                    <span className="chip">start {lastSent.start}</span>
                    <span className="chip">snap {lastSent.snapshot}</span>
                    <span className="chip">stop {lastSent.stop}</span>
                  </div>

                  <div className="kv chiprow">
                    <b>HTTP:</b>
                    <span className="chip">start {lastHttp.start || "-"}</span>
                    <span className="chip">snap {lastHttp.snapshot || "-"}</span>
                    <span className="chip">stop {lastHttp.stop || "-"}</span>
                    <span className="muted">{USE_SOCKET_SERVER ? " (socket)" : ""}</span>
                  </div>

                  <div className="kv chiprow">
                    <b>Faces:</b>
                    <span className="chip">total {totals.all}</span>
                    <span className="chip">green {totals.green}</span>
                    <span className="chip">red {totals.red}</span>
                  </div>

                  <div className="kv">
                    <button
                      className="btn small full"
                      title="Clear in-browser guest memory"
                      onClick={() => {
                        guestSeqRef.current = 1;
                        guestMemRef.current = [];
                        saveGuestMem({ day: dayKey(), seq: 1, mem: [] });
                      }}
                    >
                      clear guests
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* MIC */}
            <div className="panel">
              <div className="row" style={{ gap: 12 }}>
                <span className={`dot ${micOn ? "ok" : "err"}`} />
                <b>Mic:</b>&nbsp;{micOn ? "listening" : "idle"}

                <label className="checkbox" style={{ marginLeft: "auto" }}>
                  <input
                    type="checkbox"
                    checked={autoDetectOn}
                    onChange={(e) => setAutoDetectOn(e.target.checked)}
                  />
                  <span>Auto-calibrate</span>
                </label>

                <button
                  className="btn"
                  data-active={micOn ? "true" : "false"}
                  onClick={async () => { userMicOffRef.current = false; await startMic(); }}
                >
                  Start
                </button>
                <button
                  className="btn"
                  data-active={!micOn ? "true" : "false"}
                  onClick={async () => { userMicOffRef.current = true; await stopMic(); }}
                >
                  Stop
                </button>
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
                >
                  Calibrate
                </button>
              </div>

              <LevelMeter levelDbfs={dbfs} thresholdDbfs={threshold} />

              <div>
                <label className="label">Noise Threshold (dBFS)</label>
                <input
                  className="range"
                  type="range"
                  min="-60"
                  max="-20"
                  step="1"
                  value={threshold}
                  onChange={(e) => setThreshold(Number(e.target.value))}
                />
              </div>

              <div className="row" style={{ gap: 16 }}>
                <div className="flex1">
                  <label className="label">Listen</label>
                  <input
                    className="range"
                    type="range"
                    min="0"
                    max="5000"
                    step="100"
                    value={listenMs}
                    onChange={(e) => setListenMs(Number(e.target.value))}
                  />
                  <div className="help">{(Math.round(listenMs / 100) / 10)}s</div>
                </div>
                <div className="flex1">
                  <label className="label">Silence</label>
                  <input
                    className="range"
                    type="range"
                    min="0"
                    max="5000"
                    step="100"
                    value={silenceMs}
                    onChange={(e) => setSilenceMs(Number(e.target.value))}
                  />
                  <div className="help">{(Math.round(silenceMs / 100) / 10)}s</div>
                </div>
              </div>

              {/* Gemini AAD knobs */}
              <div className="row" style={{ gap: 16 }}>
                <div className="flex1">
                  <label className="label">Start of speech sensitivity</label>
                  <input
                    className="range"
                    type="range"
                    min="0" max="1" step="0.05"
                    value={sosQuick}
                    onChange={(e) => setSosQuick(Number(e.target.value))}
                  />
                  <div className="help">{sosQuick.toFixed(2)}</div>
                </div>
                <div className="flex1">
                  <label className="label">End of speech sensitivity</label>
                  <input
                    className="range"
                    type="range"
                    min="0" max="1" step="0.05"
                    value={eosQuick}
                    onChange={(e) => setEosQuick(Number(e.target.value))}
                  />
                  <div className="help">{eosQuick.toFixed(2)}</div>
                </div>
              </div>

              <div className="row" style={{ gap: 16 }}>
                <div className="flex1">
                  <label className="label">Prefix padding (ms)</label>
                  <input
                    className="range"
                    type="range"
                    min="0" max="400" step="10"
                    value={prefixPadQuick}
                    onChange={(e) => setPrefixPadQuick(Number(e.target.value))}
                  />
                  <div className="help">{prefixPadQuick} ms</div>
                </div>
                <div className="flex1">
                  <label className="label">Silence duration (ms)</label>
                  <input
                    className="range"
                    type="range"
                    min="200" max="2000" step="50"
                    value={silenceDurQuick}
                    onChange={(e) => setSilenceDurQuick(Number(e.target.value))}
                  />
                  <div className="help">{silenceDurQuick} ms</div>
                </div>
              </div>

              {/* Devices */}
              <div className="row" style={{ gap: 12 }}>
                <div className="flex1">
                  <label className="label">Microphone</label>
                  <select
                    className="select big"
                    value={audioId}
                    onChange={async (e) => {
                      const next = e.target.value;
                      setAudioId(next);
                      try { localStorage.setItem("ika:audioId", next); } catch {}
                      await startMic(next, { force: true });
                    }}
                  >
                    <option value="">(Default)</option>
                    {audioDevs.map((d) => (
                      <option key={d.deviceId} value={d.deviceId}>{d.label || "Microphone"}</option>
                    ))}
                  </select>
                </div>
                <div className="flex1">
                  <label className="label">Camera</label>
                  <select
                    className="select big"
                    value={videoId}
                    onChange={async (e) => {
                      const next = e.target.value;
                      setVideoId(next);
                      try { localStorage.setItem("ika:videoId", next); } catch {}
                      await startCamera(next);
                    }}
                  >
                    <option value="">(Default)</option>
                    {videoDevs.map((d) => (
                      <option key={d.deviceId} value={d.deviceId}>{d.label || "Camera"}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          </div>

          {/* Row 2: DISTANCE CONTROLS — two neat panels */}
          <div className="panel">
            {/* GREEN ZONE (interaction range) */}
            <div className="inline-controls">
              <b>Green zone distance</b>
              <input
                className="range"
                type="range"
                min="0.3" max="2.0" step="0.05"
                value={greenMaxM}
                onChange={(e) => setGreenMaxM(Number(e.target.value))}
                aria-label="Green zone distance in meters"
              />
              <span className="chip">{greenMaxM.toFixed(2)} m</span>
              <button className="btn" onClick={() => setGreenMaxM(DEFAULT_GREEN_MAX_M)}>reset</button>
              <button className="btn" onClick={() => setGreenMaxM(v => Math.max(0.3, +(v - 0.1).toFixed(2)))}>–0.1</button>
              <button className="btn" onClick={() => setGreenMaxM(v => Math.min(2.0, +(v + 0.1).toFixed(2)))} >+0.1</button>
            </div>

            {/* RED CUTOFF (ignore beyond) */}
            <div className="inline-controls" style={{ marginTop: 8 }}>
              <b>Red zone distance</b>
              <input
                className="range"
                type="range"
                min="1.0" max="6.0" step="0.1"
                value={redCutoffM}
                onChange={(e) => setRedCutoffM(Number(e.target.value))}
                aria-label="Red zone cutoff in meters"
              />
              <span className="chip">{redCutoffM.toFixed(1)} m</span>
              <button className="btn" onClick={() => setRedCutoffM(DEFAULT_RED_CUTOFF_M)}>reset</button>
              <button className="btn" onClick={() => setRedCutoffM(v => Math.max(1.0, +(v - 0.1).toFixed(1)))} >–0.1</button>
              <button className="btn" onClick={() => setRedCutoffM(v => Math.min(6.0, +(v + 0.1).toFixed(1)))} >+0.1</button>
            </div>
          </div>

          {/* Row 3: CAMERA */}
            <div className="stage">
              <video ref={videoRef} autoPlay muted playsInline />
              <canvas ref={canvasRef} />
            </div>

            {captions && lastText && (
              <div aria-live="polite" className="captions" style={{
                marginTop: 8,
                background: "rgba(0,0,0,0.55)",
                padding: "10px 12px",
                borderRadius: 10,
                lineHeight: 1.35
              }}>
                {lastText}
              </div>
            )}

            <div className="row" style={{ gap: 12, alignItems: "center" }}>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={showAlign}
                  onChange={(e) => setShowAlign(e.target.checked)}
              />
              <span>Show alignment overlay</span>
            </label>

            <label className="checkbox" style={{ marginLeft: 12 }}>
              <input
                type="checkbox"
                checked={gameModeOn}
                onChange={(e) => setGameModeOn(e.target.checked)}
              />
               <span>Game mode (RPS)</span>
             </label>

            <div className="kv" style={{ gap: 6 }}>
              <b>Calib distance:</b>
              <input
                className="input"
                type="number"
                step="0.05"
                min="0.3"
                max="3.0"
                value={calibDistanceM}
                onChange={(e) => setCalibDistanceM(Number(e.target.value))}
                style={{ width: 90 }}
                aria-label="Calibration distance (meters)"
              />
              <span className="muted">m</span>
            </div>

            <button className="btn" onClick={runCalCountdown}>
              Calibrate camera (3-2-1)
            </button>
          </div>

          {/* Camera settings — now directly under the camera */}
          <div className="panel" style={{ marginTop: 10 }}>
            <h3 className="section-title" style={{ marginTop: 0 }}>camera alignment</h3>

            <label className="label">Horizontal FOV (°)</label>
            <input
              className="range"
              type="range"
              min="40" max="110" step="1"
              value={fovHdeg}
              onChange={(e)=>setFovHdeg(Number(e.target.value))}
            />
            <div className="help">{Math.round(fovHdeg)}°</div>

            <label className="label">Vertical FOV (°)</label>
            <input
              className="range"
              type="range"
              min="25" max="90" step="1"
              value={fovVdeg}
              onChange={(e)=>setFovVdeg(Number(e.target.value))}
            />
            <div className="help">{Math.round(fovVdeg)}°</div>

            <div className="row" style={{ gap: 16 }}>
              <div className="flex1">
                <label className="label">Pan offset (°)</label>
                <input
                  className="range"
                  type="range"
                  min="-30" max="30" step="0.5"
                  value={panOffsetDeg}
                  onChange={(e)=>setPanOffsetDeg(Number(e.target.value))}
                />
                <div className="help">{panOffsetDeg.toFixed(1)}°</div>
              </div>
              <div className="flex1">
                <label className="label">Tilt offset (°)</label>
                <input
                  className="range"
                  type="range"
                  min="-30" max="30" step="0.5"
                  value={tiltOffsetDeg}
                  onChange={(e)=>setTiltOffsetDeg(Number(e.target.value))}
                />
                <div className="help">{tiltOffsetDeg.toFixed(1)}°</div>
              </div>
            </div>

            <div className="row" style={{ gap: 8, marginTop: 8 }}>
              <button className="btn" onClick={()=>{ setPanOffsetDeg(0); setTiltOffsetDeg(0); }}>reset offsets</button>
              <span className="help">Tip: click a face on video to auto-zero.</span>
            </div>

            <div className="divider" />

            <h4 className="section-title">focus weights</h4>

            <label className="label">Closeness</label>
            <input
              className="range"
              type="range"
              min="0" max="1" step="0.05"
              value={wNear}
              onChange={(e)=>setWNear(Number(e.target.value))}
            />

            <label className="label">Centeredness</label>
            <input
              className="range"
              type="range"
              min="0" max="1" step="0.05"
              value={wCenter}
              onChange={(e)=>setWCenter(Number(e.target.value))}
            />

            <label className="label">Mouth activity</label>
            <input
              className="range"
              type="range"
              min="0" max="1" step="0.05"
              value={wMouth}
              onChange={(e)=>setWMouth(Number(e.target.value))}
            />
          </div>

          {/* Row 4: GUEST TABLE */}
          <div className="panel tablewrap" style={{ padding: 12 }}>
            <table className="table">
              <thead>
                <tr>{["#", "Name", "Gesture", "Emotion", "Zone", "AgeGroup", "Gender", "Distance"].map(h => <th key={h}>{h}</th>)}</tr>
              </thead>
              <tbody>
                {table.map((r) => (
                  <tr key={r.idx}>
                    <td>{r.idx}</td>
                    <td>{r.name}</td>
                    <td>{r.gesture ?? "-"}</td>
                    <td>{r.emotion ?? "-"}</td>
                    <td className={r.zone === "green" ? "zone-green" : r.zone === "red" ? "zone-red" : "zone-unk"}>{r.zone}</td>
                    <td>{r.ageGroup}</td>
                    <td>{r.gender}</td>
                    <td>{r.distance}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* ===== RIGHT COLUMN ===== */}
        <div className="rightcol" style={{ position: "sticky", top: 12, alignSelf: "start", zIndex: 1000 }}>
          {/* Save/Export Settings */}
        <section className="panel" style={{ pointerEvents: "auto", zIndex: 1001 }}>
          <h3 className="section-title">settings backup</h3>
          <div className="row" style={{ gap: 8 }}>
            <button className="btn" onClick={exportSettings}>Export to file</button>
            <button className="btn" onClick={importSettings}>Import from file</button>
            <button className="btn" onClick={resetSettings}>Reset all</button>
          </div>
          <div className="help" style={{ marginTop: 6 }}>
            Saves to localStorage; import/export helps migrate to another machine.
          </div>
        </section>
          {/* IP Address */}
        <section className="panel" style={{ pointerEvents: "auto", zIndex: 1001 }}>
          <h3 className="section-title">server connection</h3>
          <div className="row" style={{ gap: 8 }}>
            <input
              className="input bigpad"
              placeholder="(same-origin) or http://PC-IP:PORT"
              value={serverUrlDraft}
              onChange={(e) => setServerUrlDraft(e.target.value)}
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
            />
            <button
              className="btn"
              disabled={serverUrlDraft.trim() === (serverUrl || "")}
              onClick={() => setServerUrl(serverUrlDraft.trim())}
            >
              Apply & reconnect
            </button>
            <button
              className="btn"
              onClick={() => { setServerUrl(""); setServerUrlDraft(window.location.origin); }}
            >
              Use same-origin
            </button>
          </div>
          <div className="help" style={{ marginTop: 6 }}>
            Config: {serverUrl && serverUrl.trim() ? serverUrl : "(same-origin)"} · Effective: {effectiveUrl}
            <br />Status: {serverInfo.connected ? "connected" : "disconnected"}
          </div>
        </section>
          {/* System message */}
          <section className="panel system-panel" style={{ pointerEvents: "auto", zIndex: 1001 }}>
            <h3 className="section-title">system message</h3>
            <textarea
              className="input multiline bigpad"
              value={systemInstruction}
              placeholder="Add/override the system instruction…"
              onChange={(e) => setSystemInstruction(e.target.value)}
              autoCorrect="off"
              autoCapitalize="off"
              spellCheck={false}
              id="system-instruction-textarea" // Added ID for accessibility
            />
            <label htmlFor="system-instruction-textarea" className="visually-hidden">System Instruction</label> {/* Added label for accessibility */}
          </section>

          <section className="panel" style={{ pointerEvents: "auto", zIndex: 1001 }}>
            <h3 className="section-title">performance</h3>
            <div className="row" style={{ gap: 12, alignItems: "center" }}>
              <label className="checkbox">
                <input
                  type="checkbox"
                  checked={gesturesOn}
                  onChange={(e) => setGesturesOn(e.target.checked)}
                />
                <span>Gestures</span>
              </label>
              <div className="row" style={{ gap: 8, alignItems: "center", marginLeft: 8 }}>
                <label className="label" style={{ margin: 0 }}>Targets</label>
                <select
                  className="select"
                  value={gestureTargets}
                  onChange={(e) => setGestureTargets(parseInt(e.target.value, 10) === 1 ? 1 : 2)}
                  disabled={!gesturesOn}
                  title="Limit gesture tracking to 1 or 2 people"
                >
                  <option value={1}>1 person</option>
                  <option value={2}>2 people</option>
                </select>
              </div>
            </div>
            <div className="help" style={{ marginTop: 6 }}>
              Turn off on low-power devices (Android/Edge) to improve FPS.
            </div>
          </section>

          {/* Gemini */}
          <section className="panel" style={{ pointerEvents: "auto", zIndex: 1001 }}>
          
            <h3 className="section-title">gemini settings</h3>

            <label className="label" htmlFor="gemini-api-key-input">Gemini API Key</label> {/* Added htmlFor */}
            <input
              className="input bigpad"
              type="password"
              inputMode="text"
              autoComplete="off"
              placeholder="gsk-…"
              value={geminiApiKey}
              onChange={(e) => setGeminiApiKey(e.target.value)}
              id="gemini-api-key-input" // Added ID
            />

            <label className="label" htmlFor="gemini-model-select">Model</label> {/* Added htmlFor */}
            <select
              className="select big"
              value={modelQuick}
              onChange={(e) => setModelQuick(e.target.value)}
              id="gemini-model-select" // Added ID
            >
              {MODEL_OPTIONS.map((m) => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>

            <label className="label" htmlFor="gemini-voice-select">Voice</label> {/* Added htmlFor */}
            <select
              className="select big"
              value={geminiVoiceQuick}
              onChange={(e) => setGeminiVoiceQuick(e.target.value)}
              id="gemini-voice-select" // Added ID
            >
              {(GEMINI_VOICES[modelKind] || []).map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>

            <label className="label" htmlFor="gemini-language-select">Language</label> {/* Added htmlFor */}
            <select
              className="select big"
              value={languageCodeQuick}
              onChange={(e) => setLanguageCodeQuick(e.target.value)}
              id="gemini-language-select" // Added ID
            >
              {LANGS.map(([label, code]) => (
                <option key={code} value={code}>{label}</option>
              ))}
            </select>

            <label className="label" htmlFor="gemini-temperature-range">Temperature</label> {/* Added htmlFor */}
            <input
              className="range"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={temperatureQuick}
              onChange={(e) => setTemperatureQuick(Number(e.target.value))}
              id="gemini-temperature-range" // Added ID
            />
            <div className="help">{temperatureQuick.toFixed(2)}</div>

            <div className="row wrap" style={{ gap: 12, marginTop: 8 }}>
              <label className="checkbox"><input type="checkbox" checked={enableAffectiveQuick} onChange={(e)=>setEnableAffectiveQuick(e.target.checked)} id="checkbox-affective" /><span>Affective dialog</span></label> {/* Added ID */}
              <label className="checkbox"><input type="checkbox" checked={proactiveAudioQuick} onChange={(e)=>setProactiveAudioQuick(e.target.checked)} id="checkbox-proactive" /><span>Proactive dialog</span></label> {/* Added ID */}
              <label className="checkbox"><input type="checkbox" checked={functionCallingQuick} onChange={(e)=>setFunctionCallingQuick(e.target.checked)} id="checkbox-function-calling" /><span>Function calling</span></label> {/* Added ID */}
              <label className="checkbox"><input type="checkbox" checked={autoFunctionResponseQuick} onChange={(e)=>setAutoFunctionResponseQuick(e.target.checked)} id="checkbox-auto-function-response" /><span>Auto function response</span></label> {/* Added ID */}
              <label className="checkbox"><input type="checkbox" checked={groundingQuick} onChange={(e)=>setGroundingQuick(e.target.checked)} id="checkbox-grounding" /><span>Grounding (Search)</span></label> {/* Added ID */}
            </div>

            <div className="row" style={{ gap: 8, marginTop: 10 }}>
              <button className="btn stretch" onClick={onCreateSession}>Apply & (re)start Gemini</button>
              <button className="btn" onClick={onHotUpdate}>Hot update</button>
            </div>

            <div className="divider" />

            <label className="label" htmlFor="tts-provider-select">TTS Provider</label> {/* Added htmlFor */}
            <select
              className="select big"
              value={ttsProviderQuick}
              onChange={(e) => setTtsProviderQuick(e.target.value)}
              id="tts-provider-select" // Added ID
            >
              <option value="gemini">Gemini</option>
              <option value="elevenlabs">ElevenLabs</option>
            </select>
          </section>

          {/* ElevenLabs settings — ALWAYS mounted; visibility via CSS */}
          <section className={`panel elevenlabs-panel ${ttsProviderQuick === "elevenlabs" ? "is-active" : ""}`}>
            <h3 className="section-title">elevenlabs settings</h3>
            <ElevenLabsSettings />
          </section>
        </div>
      </div>
    </main>
  );
}