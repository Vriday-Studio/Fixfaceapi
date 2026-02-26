// ==== App.jsx - streamlined (Live Preview + Native Audio), neat right sidebar ====
// - TFJS + face-api + webcam detection
// - Socket.IO bridge for Gemini/ElevenLabs audio + text
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
import { normalizeServerUrl, uuid } from "./utils/socketUtils";
import {
  ageGroupOf,
  zoneOf,
  topExpression,
  gestureLabelOf,
  focalFromFov,
  anglesFromPixel,
  posFromPixel,
  mouthMAR,
  shrinkBox,
} from "./utils/faceMath";
import {
  MP,
  palmSpanLen,
  recentLateralMotion,
  setWaveHistory,
  waveActivity,
  classifyWave,
  classifyThumbsUp,
  classifyPeace,
  classifyRaiseHand,
  classifyOnPhone,
  classifyPaper,
  classifyRock,
  classifyScissors,
  pickStableGesture,
  VOTE_WINDOW,
} from "./utils/gestureUtils";
import {
  getStoredNumber,
  encodeDescFloat32ToU8,
  decodeDescU8ToFloat32,
  u8ToB64,
  b64ToU8,
} from "./utils/storageUtils";
import ToggleSwitch from "./components/ToggleSwitch";
import PeopleTable from "./components/PeopleTable";
import {
  DEFAULT_GUEST_STORE_KEY,
  dayKey,
  msToNextMidnight,
  loadGuestMem,
  saveGuestMem,
  pruneByRetention,
} from "./utils/guestMemory";
import {
  groupSignature,
  maybeRotateSession,
} from "./utils/sessionRotation";
import { MSG_TYPE, useDirectWebSocket } from "./hooks/useDirectWebSocket";
import { buildVisitContext as buildVisitContextUtil } from "./utils/visitContext";
import {
  exportSettings as exportSettingsUtil,
  importSettings as importSettingsUtil,
  resetSettings as resetSettingsUtil,
} from "./utils/settingsUtils";
import { useCamera } from "./hooks/useCamera";
import { handleZoneTransitions } from "./utils/policyUtils";

/* ====================== CONSTANTS / CONFIG ====================== */
const MODEL_URL = "/models";
const LABELS_URL = "/labels/labels.json";

// geometry
const FACE_WIDTH_M = 0.15;
let FOCAL_PX = 500; // refined after camera opens
const RAD = 180 / Math.PI;

// session heuristics
const DEFAULT_GREEN_MAX_M = 0.8;
const LOOP_STEP_ACTIVE_MS = 120;
const LOOP_STEP_IDLE_MIN_MS = 180;
const LOOP_STEP_IDLE_MAX_MS = 220;

// recognition
const MATCH_THRESHOLD = 0.5;
const MATCH_MARGIN = 0.03;
const STABILIZE_FRAMES = 3;
// Require ~5 stable active frames in green before firing greet
const GREEN_STABLE_MS = STABILIZE_FRAMES * LOOP_STEP_ACTIVE_MS;

const CAM_IDLE_MS = 10000; // cam fully idle after 10s

// drawing
const BOX_LINE_WIDTH = 5;
const LABEL_FONT =
  "16px system-ui,-apple-system,Segoe UI,Roboto,'Helvetica Neue',Arial,sans-serif";
const LABEL_PAD_X = 8;
const LABEL_PAD_Y = 6;

// sockets
const SOCKET_URL = undefined; // same-origin
const USE_SOCKET_BRIDGE = false; // legacy Socket.IO bridge (remote controls / telemetry)
const USE_DIRECT_WEBSOCKET = true; // policy + session pipeline (always on)

// --- Attention / greeting policy ---
// Loosen thresholds so slight head turns don't drop to NONE.
const FACING_YAW_MAX_DEG = 35; // how "straight on" horizontally
const FACING_PITCH_MAX_DEG = 25; // how "straight on" vertically
const ATTEND_MIN_FRAMES = 3; // require 3-5 consecutive frames
const DEBUG_FACE_LOGS = false;
const DEBUG_ZONE_LOGS = false;

const logFace = (...args) => {
  if (DEBUG_FACE_LOGS) console.log(...args);
};
const logZone = (...args) => {
  if (DEBUG_ZONE_LOGS) console.log(...args);
};
// Minimum time between greets for the same identity (per p.key)
const GREET_COOLDOWN_MS = 20_000;

// Hard cap per identity
const MAX_INVITES_PER_PERSON = 3;

// Optional: if someone disappears for a while, forgive past invites
const NOT_SEEN_RESET_MS = 120_000; // 2 min of not being seen resets their count

// ===== Hand / Gesture config (tablet-safe) =====
const HANDS_ENABLED = true;

// runtime & cadence
const HANDS_FAST_MS = 66;
const HANDS_IDLE_MS = 180;
const HANDS_CACHE_MS = 1000;
const HANDS_SEND_MS = 600;

// Game mode cadence (snappier)
const GM_HANDS_FAST_MS = 40;
const GM_HANDS_IDLE_MS = 120;

// Hand constants
const HANDS_MODEL_URL = "/mp/hand_landmarker.task";
const HANDS_MAX_NUM = 2;
const HANDS_IMAGE_SIDE = 256;

// polite "call over" policy
const CALL_OVER_MAX_TRIES = 3;
const CALL_OVER_COOLDOWN_MS = 30_000; // >= 30s between tries

// speaker focus gating
const SPEAKER_STABLE_FRAMES = 3;
const SPEAKER_STABLE_MS = 1200;

// group ask cooldown
const GROUP_ASK_COOLDOWN_MS = 20_000;

// RED zone call-over policy (NONE -> RED transition detection)
const RED_ZONE_STABLE_FRAMES = 5; // Require 5 consecutive RED frames before sending
const RED_ZONE_NONE_RESET_FRAMES = 15; // Require 15 consecutive NONE frames to reset


/* ====================== MAIN APP ====================== */
export default function App() {
  /* ---------- Socket + audio playback ---------- */
  const socketRef = useRef(null);
  const ttsPlayerRef = useRef(null);

  // Configurable server URL (empty = same-origin)
  const [serverUrl, setServerUrl] = useState(
    () => localStorage.getItem("ika:serverUrl") || ""
  );
  const [serverUrlDraft, setServerUrlDraft] = useState(serverUrl);
  const effectiveUrl = useMemo(
    () => normalizeServerUrl(serverUrl) ?? window.location.origin,
    [serverUrl]
  );
  useEffect(() => {
    setServerUrlDraft(serverUrl);
  }, [serverUrl]);
  useEffect(() => {
    try {
      localStorage.setItem("ika:serverUrl", serverUrl);
    } catch { }
  }, [serverUrl]);

  // Pick up ?server=... from query string once
  useEffect(() => {
    try {
      const u = new URLSearchParams(window.location.search).get("server");
      if (u) setServerUrl(u);
    } catch { }
  }, []);

  /* ---------- Global/session UI state ---------- */
  const [sessionStatus, setSessionStatus] = useState("IDLE");
  const [autoSession, setAutoSession] = useState(
    localStorage.getItem("ika:autoSession") !== "false"
  );
  const autoSessionPendingRef = useRef(false);
  useEffect(() => {
    try {
      localStorage.setItem("ika:autoSession", String(autoSession));
    } catch { }
  }, [autoSession]);
  const [machineId, setMachineId] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  const [serverInfo, setServerInfo] = useState({
    connected: false,
    model: null,
    tts: null,
    boundDeviceId: null,
    ai_speaking: false,
  });
  const serverInfoRef = useRef(serverInfo);
  useEffect(() => {
    serverInfoRef.current = serverInfo;
  }, [serverInfo]);
  const [ueConnected, setUeConnected] = useState(false);
  const ueConnectedRef = useRef(false);
  useEffect(() => {
    ueConnectedRef.current = ueConnected;
  }, [ueConnected]);

  const [posts, setPosts] = useState({ start: 0, snapshot: 0, stop: 0 });
  const [lastSent, setLastSent] = useState({
    start: "-",
    snapshot: "-",
    stop: "-",
  });
  const [lastHttp, setLastHttp] = useState({
    start: "",
    snapshot: "",
    stop: "",
  });
  const bump = useCallback((kind) => {
    const now = new Date().toLocaleTimeString();
    setPosts((p) => ({ ...p, [kind]: (p[kind] || 0) + 1 }));
    setLastSent((p) => ({ ...p, [kind]: now }));
  }, []);

  const [captions, setCaptions] = useState(
    localStorage.getItem("ika:captions") === "true"
  );
  const [lastText, setLastText] = useState("");

  // device identity
  const [deviceId, setDeviceId] = useState(() => {
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
  const [deviceIdDraft, setDeviceIdDraft] = useState("");
  useEffect(() => {
    setDeviceIdDraft(deviceId);
    try {
      localStorage.setItem("ika:deviceId", deviceId);
    } catch {}
  }, [deviceId]);

  const { wsIsConnected, sendCommand: sendWsCommand } = useDirectWebSocket({
    serverUrl: effectiveUrl,
    deviceId,
    setServerInfo,
    setSessionStatus,
    setSessionId,
    setMachineId,
    setUeConnected,
    enabled: USE_DIRECT_WEBSOCKET,
  });

  const createServerSession = useCallback(
    (preset = {}) => {
      sendWsCommand(MSG_TYPE.SessionStart, {
        ...preset,
        deviceId,
        sessionId: sessionId || "web-" + deviceId,
      });
      bump("start");
      setSessionStatus("ACTIVE");
      setSessionId((id) => id || uuid());
    },
    [bump, deviceId, sendWsCommand, sessionId]
  );

  const updateServerSettings = useCallback(
    (fields = {}) => {
      socketRef.current?.emit?.("update_settings", fields);
    },
    []
  );

  const sendTextPrompt = useCallback((text) => {
    if (!text) return;
    socketRef.current?.emit?.("send_text_prompt", { text });
  }, []);

  const emitCrowdStatus = useCallback(
    (payload) => {
      socketRef.current?.emit?.("crowd_status", {
        deviceId,
        sessionId: sessionId || "web-" + deviceId,
        ...payload,
      });
      bump("snapshot");
    },
    [bump, deviceId, sessionId]
  );

  const server = useMemo(
    () => ({
      createSession: createServerSession,
      updateSettings: updateServerSettings,
      sendText: sendTextPrompt,
      crowdStatus: emitCrowdStatus,
    }),
    [createServerSession, emitCrowdStatus, sendTextPrompt, updateServerSettings]
  );

  // Per-identity attention counting & cooldown
  const attentionMapRef = useRef(new Map());
  // Map key: stable id (name || gid)
  // value: { count: number, lastInviteTs: number }

  // ===== HandLandmarker refs =====
  const handLmRef = useRef(null);
  const handsReadyRef = useRef(false);
  const lastHandsRunTsRef = useRef(0);
  const lastLmSeenTsRef = useRef(0); // added: when we last saw landmarks
  const handsFailRef = useRef(0); // added: consecutive VIDEO misses
  const lastCrowdStatSentRef = useRef(0); // throttle for SendWebsockCommandToServer

  // downscale buffer for hands
  const handsOffscreenRef = useRef(null);
  const handsCtxRef = useRef(null);

  // last gesture memory
  const stableGestureRef = useRef(null); // { type, score, t }

  // NEW: Per-face gesture windows and stable picks
  const perFaceGestureWinRef = useRef(new Map()); // key -> [{type,score,t}, ...]
  const perFaceStableRef = useRef(new Map()); // key -> {type,score,t}
  const lastGestureSentPerFaceRef = useRef(new Map()); // key -> lastTs

  // NEW: per-face wave histories (used by wave/velocity gates)
  const waveHistByFaceRef = useRef(new Map());

  // NEW: pending greets map (retry when age/gender available)
  const pendingGreetsRef = useRef(new Map()); // Map<personKey, { zone, timestamp }>
  const pushedGreets = useRef(new Set()); // prevent duplicate retry greets

  // --- age/gender stagger + cache ---
  const AGE_SAMPLE_MS = 600;
  const lastAgeSampleRef = useRef(0);
  const ageGenderCacheRef = useRef(new Map()); // key -> { age, gender }
  // key -> { ts: firstGreenTimestamp, greetedOnceInThisGreen: boolean }
  const greenEntryRef = useRef(new Map());

  const [locationLabel, setLocationLabel] = useState(
    localStorage.getItem("ika:locationLabel") || "Galeri Indonesia Kaya"
  );
  const [weatherLabel, setWeatherLabel] = useState(
    localStorage.getItem("ika:weatherLabel") || "Clear 28 degC"
  );
  const [clock, setClock] = useState(() => new Date());
  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);
  const clockRef = useRef(clock);
  useEffect(() => {
    clockRef.current = clock;
  }, [clock]);
  const locationRef = useRef(locationLabel);
  useEffect(() => {
    locationRef.current = locationLabel;
  }, [locationLabel]);
  const weatherRef = useRef(weatherLabel);
  useEffect(() => {
    weatherRef.current = weatherLabel;
  }, [weatherLabel]);
  const totalsRef = useRef({ all: 0, green: 0, red: 0 });

  const buildVisitContext = useCallback(
    (extra = {}) =>
      buildVisitContextUtil({
        clockRef,
        locationRef,
        weatherRef,
        totalsRef,
        extra,
      }),
    [clockRef, locationRef, weatherRef, totalsRef]
  );

  const sendPeopleIntent = useCallback(
    (intent, person, extra = {}) => {
      if (!wsIsConnected.current) {
        return;
      }

      const identityKeyForPayload =
        person.stableKey || person.key || person.gid || person.name || null;

      const payload = {
        intent,
        zone: person.zone,
        name: person.name || null,
        gid: person.gid || null,
        gender: person.gender || null,
        ageGroup: person.ageGroup || null,
        emotion: person.emotion || null,
        yawDeg:
          Number.isFinite(person.yawDeg) && person.yawDeg != null
            ? +Number(person.yawDeg).toFixed(2)
            : null,
        pitchDeg:
          Number.isFinite(person.pitchDeg) && person.pitchDeg != null
            ? +Number(person.pitchDeg).toFixed(2)
            : null,
        mouthActivity:
          Number.isFinite(person.mouthActivity) &&
          person.mouthActivity != null
            ? +Number(person.mouthActivity).toFixed(3)
            : null,
        posCam: person.posCam || null,
        stableKey: identityKeyForPayload,
        slotKey: person.slotKey || null,
        group: extra.group || null,
        reason: extra.reason || null,
        guests: extra.guests || null,
        context: buildVisitContext(extra.context || {}),
      };

      sendWsCommand(MSG_TYPE.PeopleData, payload);
    },
    [buildVisitContext, sendWsCommand, wsIsConnected]
  );

  const lastCrowdSendRef = useRef({ t: 0, sig: "" });
  const lastPeopleSnapshotSentRef = useRef(0);
  function emitCrowdThrottled(payload) {
    const now = performance.now();
    const MIN_MS = 66;
    const state = lastCrowdSendRef.current;
    const ppl = (payload.people || []).map((p) => [
      p.yawDeg,
      p.pitchDeg,
      Math.round((p.mouthActivity || 0) * 1000),
    ]);
    
    const sig = JSON.stringify([payload.focusIndex, ppl]);
    
    if (now - state.t < MIN_MS && sig === state.sig) return;

    try {
      sendWsCommand(MSG_TYPE.CrowdStat, {
        ...payload,
        context: buildVisitContext(),
      });
    } catch {}
    lastCrowdSendRef.current = { t: now, sig };
  }


  function emitCrowdByGid(payload){
    const now = performance.now();
    const MIN_MS = 1000;
    const state = lastCrowdSendRef.current;
    const ppl = (payload.people || []).map((p) => [
      p.gid
    ]);

    if ((now - state.t) <= MIN_MS) 
      return;
    
    const peopleCandidate = payload.people || [];
    if(peopleCandidate.length == 0) return;

    //send all data iterating by gid
    for(const p of peopleCandidate){
      if(p.gid == null) continue;
      try 
      {
        /*
        sendWsCommand(MSG_TYPE.CrowdStat, {
          ...payload,
          context: buildVisitContext(),
        });*/

        const dataToSend = {
          deviceId : payload.deviceId,
          timestamp : payload.timeISO,
          sessionId : payload.sessionId,
          gesture : (p.gesture || null),
          context : buildVisitContext(),
          people: p

        };
        
        sendWsCommand(MSG_TYPE.CrowdStat, dataToSend);
      } 
      catch 
      {
        continue;
      }
    }
    
    //const sig = JSON.stringify([ppl]);
    lastCrowdSendRef.current = { t: now };
    
  }

  // Camera + detection state
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const camRef = useRef({ stream: null });

  const [fovHdeg, setFovHdeg] = useState(() =>
    Number(localStorage.getItem("ika:fovHdeg") ?? 70)
  );
  const [fovVdeg, setFovVdeg] = useState(() =>
    Number(localStorage.getItem("ika:fovVdeg") ?? 43)
  );
  const [panOffsetDeg, setPanOffsetDeg] = useState(() =>
    Number(localStorage.getItem("ika:panOffsetDeg") ?? 0)
  );
  const [tiltOffsetDeg, setTiltOffsetDeg] = useState(() =>
    Number(localStorage.getItem("ika:tiltOffsetDeg") ?? 0)
  );
  useEffect(() => {
    try {
      localStorage.setItem("ika:fovHdeg", String(fovHdeg));
      localStorage.setItem("ika:fovVdeg", String(fovVdeg));
      localStorage.setItem("ika:panOffsetDeg", String(panOffsetDeg));
      localStorage.setItem("ika:tiltOffsetDeg", String(tiltOffsetDeg));
    } catch {}
  }, [fovHdeg, fovVdeg, panOffsetDeg, tiltOffsetDeg]);

  const camFxRef = useRef(600);
  const camFyRef = useRef(600);
  const panOffRef = useRef(0);
  const tiltOffRef = useRef(0);
  useEffect(() => {
    panOffRef.current = panOffsetDeg;
  }, [panOffsetDeg]);
  useEffect(() => {
    tiltOffRef.current = tiltOffsetDeg;
  }, [tiltOffsetDeg]);

  const [showAlign, setShowAlign] = useState(
    localStorage.getItem("ika:showAlign") !== "false"
  );
  const showAlignRef = useRef(true);
  const [calibDistanceM, setCalibDistanceM] = useState(() =>
    Number(localStorage.getItem("ika:calibDistM") ?? 1.0)
  );
  const calibMsgRef = useRef("");
  useEffect(() => {
    showAlignRef.current = showAlign;
    try {
      localStorage.setItem("ika:showAlign", String(showAlign));
      localStorage.setItem("ika:calibDistM", String(calibDistanceM));
    } catch {}
  }, [showAlign, calibDistanceM]);

  const [wNear, setWNear] = useState(() =>
    Number(localStorage.getItem("ika:wNear") ?? 0.45)
  );
  const [wCenter, setWCenter] = useState(() =>
    Number(localStorage.getItem("ika:wCenter") ?? 0.35)
  );
  const [wMouth, setWMouth] = useState(() =>
    Number(localStorage.getItem("ika:wMouth") ?? 0.2)
  );
  useEffect(() => {
    try {
      localStorage.setItem("ika:wNear", String(wNear));
      localStorage.setItem("ika:wCenter", String(wCenter));
      localStorage.setItem("ika:wMouth", String(wMouth));
    } catch {}
  }, [wNear, wCenter, wMouth]);

  const mouthMapRef = useRef(new Map());
  const trackedFacesRef = useRef([]);
  const allFacesRef = useRef([]);
  const speakingRef = useRef(false);

  const prevZoneMapRef = useRef(new Map());
  const callOverStateRef = useRef(new Map());
  const lastGroupSetRef = useRef(new Set());
  const lastGroupAskTsRef = useRef(0);
  const speakerRef = useRef({
    key: null,
    topKeyPrev: null,
    framesDominant: 0,
    topSince: 0,
  });

  const lastFrameTsRef = useRef(0);
  const loopStepMsRef = useRef(LOOP_STEP_ACTIVE_MS);
  const loopIdleStateRef = useRef(false);

  const [ready, setReady] = useState(false);
  const [backend, setBackend] = useState("cpu");

  const [greenMaxM, setGreenMaxM] = useState(DEFAULT_GREEN_MAX_M);
  const greenMaxMRef = useRef(greenMaxM);
  useEffect(() => {
    greenMaxMRef.current = greenMaxM;
  }, [greenMaxM]);

  const DEFAULT_RED_CUTOFF_M = 3.5;
  const [redCutoffM, setRedCutoffM] = useState(() => {
    const raw = localStorage.getItem("ika:redCutoffM");
    const v = raw == null ? DEFAULT_RED_CUTOFF_M : parseFloat(raw);
    return Number.isFinite(v) ? v : DEFAULT_RED_CUTOFF_M;
  });
  useEffect(() => {
    try {
      localStorage.setItem("ika:redCutoffM", String(redCutoffM));
    } catch {}
  }, [redCutoffM]);

  const [videoId, setVideoId] = useState("");
  const [videoDevs, setVideoDevs] = useState([]);

  const pickInputSize = (w) => (w >= 1920 ? 416 : w >= 1280 ? 320 : 256);
  const tinyOptsRef = useRef(
    new faceapi.TinyFaceDetectorOptions({
      inputSize: 416,
      scoreThreshold: 0.4,
    })
  );

  const [table, setTable] = useState(
    Array.from({ length: 5 }, (_, i) => ({
      idx: i + 1,
      name: "-",
      gesture: "-",
      emotion: "-",
      zone: "-",
      ageGroup: "-",
      gender: "-",
      distance: "-",
    }))
  );
  const [totals, setTotals] = useState({ all: 0, green: 0, red: 0 });
  useEffect(() => {
    totalsRef.current = totals;
  }, [totals]);
  const recentMapRef = useRef({});
  const S = useRef({
    id: null,
    seenFrames: 0,
    lastFaceTs: 0,
    lastSnapshotTs: 0,
  });

  const faceMatcherRef = useRef(null);
  const [knownCount, setKnownCount] = useState(0);

  const guestSeqRef = useRef(1);
  const guestMemRef = useRef([]);
  const GUEST_TOL = 0.6;
  const guestSavePendingRef = useRef(false);
  const GUEST_STORE_KEY = DEFAULT_GUEST_STORE_KEY;
  const GUEST_RETENTION_DAYS = 1;
  const saveGuestMemSafe = useCallback(
    ({ day = dayKey(), seq = guestSeqRef.current, mem = guestMemRef.current } = {}) =>
      saveGuestMem({
        day,
        seq,
        mem,
        storeKey: GUEST_STORE_KEY,
        encodeDescriptor: (desc) => u8ToB64(encodeDescFloat32ToU8(desc)),
      }),
    [GUEST_STORE_KEY]
  );

  const setFocalPx = useCallback((value) => {
    FOCAL_PX = value;
  }, []);

  const {
    startCamera,
    runCalCountdown,
    isCamLive,
  } = useCamera({
    videoRef,
    canvasRef,
    camRef,
    tinyOptsRef,
    pickInputSize,
    setVideoDevs,
    setVideoId,
    fovHdeg,
    fovVdeg,
    camFxRef,
    camFyRef,
    setFocalPx,
    setFovHdeg,
    setFovVdeg,
    setPanOffsetDeg,
    setTiltOffsetDeg,
    calibDistanceM,
    calibMsgRef,
    faceapi,
    focalFromFov,
    anglesFromPixel,
    faceWidthM: FACE_WIDTH_M,
    rad: RAD,
    onCameraGone: (opts) => stopAll(opts),
  });

  const exportSettings = () => exportSettingsUtil("ika:");

  const importSettings = () => importSettingsUtil("ika:");

  const applyDeviceId = useCallback(() => {
    const next = (deviceIdDraft || "").trim();
    if (!next || next === deviceId) return;
    setDeviceId(next);
  }, [deviceIdDraft, deviceId]);

  const randomizeDeviceId = useCallback(() => {
    setDeviceId(uuid());
  }, []);

  const resetSettings = () => resetSettingsUtil("ika:");

  // HANDS: detect with VIDEO first, fallback to IMAGE if needed ----
  const detectHandsOnce = useCallback(async (videoEl) => {
    const landmarker = handLmRef.current;
    if (!handsReadyRef.current || !landmarker || !videoEl?.videoWidth)
      return null;

    const ts = performance.now();

    // 1) Try VIDEO mode
    try {
      const res = landmarker.detectForVideo(videoEl, ts);
      const hands = res?.landmarks || res?.handLandmarks || [];
      if (hands.length) {
        handsFailRef.current = 0;
        lastLmSeenTsRef.current = ts;
        // normalize to [{x,y}...] in 0..1
        return hands.map((h) => h.map((pt) => ({ x: pt.x, y: pt.y })));
      }
    } catch { }

    // No luck in VIDEO this frame
    handsFailRef.current = (handsFailRef.current || 0) + 1;

    // 2) Occasionally try IMAGE fallback
    if (handsFailRef.current % 4 !== 0) return null;

    try {
      await landmarker.setOptions?.({
        runningMode: "IMAGE",
        numHands: gestureTargetsRef.current,
      });

      const c = handsOffscreenRef.current,
        g = handsCtxRef.current;
      if (!c || !g) return null;

      const W = c.width,
        H = c.height;
      const vw = videoEl.videoWidth,
        vh = videoEl.videoHeight;
      const scale = Math.min(W / vw, H / vh);
      const dw = Math.round(vw * scale),
        dh = Math.round(vh * scale);
      const dx = (W - dw) >> 1,
        dy = (H - dh) >> 1;
      g.clearRect(0, 0, W, H);
      g.drawImage(videoEl, 0, 0, vw, vh, dx, dy, dw, dh);

      const res2 = await landmarker.detect(c);
      const hands2 = res2?.landmarks || res2?.handLandmarks || [];

      await landmarker.setOptions?.({
        runningMode: "VIDEO",
        numHands: gestureTargetsRef.current,
      });

      if (hands2.length) {
        handsFailRef.current = 0;
        lastLmSeenTsRef.current = performance.now();
        return hands2.map((hand) =>
          hand.map((pt) => ({
            x: (dx + pt.x * dw) / W,
            y: (dy + pt.y * dh) / H,
          }))
        );
      }
    } catch {
      try {
        await handLmRef.current?.setOptions?.({
          runningMode: "VIDEO",
          numHands: HANDS_MAX_NUM,
        });
      } catch { }
    }

    return null;
  }, []);

  // Gestures on/off (persist to localStorage)
  const [gesturesOn, setGesturesOn] = useState(
    () => localStorage.getItem("ika:gesturesOn") === "true"
  );
  const gesturesOnRef = useRef(false);
  useEffect(() => {
    gesturesOnRef.current = gesturesOn;
    try {
      localStorage.setItem("ika:gesturesOn", String(gesturesOn));
    } catch { }
    if (!gesturesOn) {
      // clear per-face gesture state immediately
      perFaceGestureWinRef.current = new Map();
      perFaceStableRef.current = new Map();
      lastGestureSentPerFaceRef.current = new Map();
      stableGestureRef.current = null;
    }
  }, [gesturesOn]);

  // Keep camera alive when tab is in background (Windows fix)
  const [keepBgOn, setKeepBgOn] = useState(
    () => localStorage.getItem("ika:keepBgOn") !== "false"
  );
  const keepBgOnRef = useRef(true);
  useEffect(() => {
    keepBgOnRef.current = keepBgOn;
    try {
      localStorage.setItem("ika:keepBgOn", String(keepBgOn));
    } catch { }
  }, [keepBgOn]);

  // How many people run gesture tracking for (1 or 2)
  const [gestureTargets, setGestureTargets] = useState(() => {
    const v = parseInt(localStorage.getItem("ika:gestureTargets") || "2", 10);
    return v === 1 ? 1 : 2;
  });
  const gestureTargetsRef = useRef(2);
  useEffect(() => {
    gestureTargetsRef.current = gestureTargets;
    try {
      localStorage.setItem("ika:gestureTargets", String(gestureTargets));
    } catch { }
    // Hint the landmarker to track fewer hands when set to 1
    try {
      handLmRef.current?.setOptions?.({ numHands: gestureTargets });
    } catch { }
  }, [gestureTargets]);

  // --- TFJS backend gating (avoid detect while switching) ---
  const backendReadyRef = useRef(Promise.resolve());
  const backendSwitchingRef = useRef(false);
  const backendNameRef = useRef(null);

  // --- Session rotation on crowd change ---
  const groupSigRef = useRef(""); // last stable group signature
  const groupStableSinceRef = useRef(0); // when current signature first appeared
  const lastRotateRef = useRef(0); // last time we rotated session
  const SESSION_ROTATE_COOLDOWN_MS = 20_000; // rotate at most every 20s
  const GROUP_STABLE_MS = 1_500; // need ~1.5s stable group before rotate


  function scheduleGuestSave() {
    if (guestSavePendingRef.current) return;
    guestSavePendingRef.current = true;
    setTimeout(() => {
      saveGuestMemSafe();
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
        setBackend(backendNameRef.current); // keep your UI/backend label in sync
        backendReadyRef.current = Promise.resolve();

        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
        ]);

        console.log("[models] loaded:", {
          tiny: !!faceapi.nets.tinyFaceDetector?.isLoaded,
          lmk68: !!faceapi.nets.faceLandmark68Net?.isLoaded,
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
            new faceapi.TinyFaceDetectorOptions({
              inputSize: 128,
              scoreThreshold: 0.4,
            })
          );
        } catch { }

        if (!cancelled) setReady(true);
      } catch (e) {
        console.error("[init]", e);
        if (!cancelled) setBackend(tf.getBackend?.() || "cpu");
      }
    })();

    return () => {
      cancelled = true;
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
          const head = await fetch("/mp/hand_landmarker.task", {
            method: "HEAD",
            cache: "no-store",
          });
          console.log(
            "[hands] model reachable:",
            head.ok,
            head.status,
            head.headers.get("content-type")
          );
        } catch (e) {
          console.warn("[hands] model HEAD failed:", e);
        }

        const isSafari = /^((?!chrome|android).)*safari/i.test(
          navigator.userAgent
        );
        const lm = await HandLandmarker.createFromOptions(fileset, {
          baseOptions: {
            modelAssetPath: HANDS_MODEL_URL,
            delegate: isSafari ? "CPU" : "GPU",
          },
          runningMode: "VIDEO",
          numHands: HANDS_MAX_NUM,
          minHandDetectionConfidence: 0.03,
          minHandPresenceConfidence: 0.03,
          minTrackingConfidence: 0.03,
        });
        if (cancelled) {
          lm.close?.();
          return;
        }

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
      try {
        handLmRef.current?.close?.();
      } catch { }
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
        const gm =
          localStorage.getItem(`ika:greenMaxM:${vId || "default"}`) ??
          localStorage.getItem("ika:greenMaxM");
        if (gm != null) {
          const val = parseFloat(gm);
          if (Number.isFinite(val))
            setGreenMaxM(Math.min(2.0, Math.max(0.3, val)));
        }
      } catch { }
      try {
        const list = await navigator.mediaDevices.enumerateDevices();
        setVideoDevs(list.filter((d) => d.kind === "videoinput"));
      } catch { }
    })();
  }, []);
  useEffect(() => {
    try {
      localStorage.setItem(
        `ika:greenMaxM:${videoId || "default"}`,
        String(greenMaxM)
      );
    } catch { }
  }, [greenMaxM, videoId]);

  // restore guest memory
  useEffect(() => {
    const data = pruneByRetention(
      loadGuestMem(GUEST_STORE_KEY),
      GUEST_RETENTION_DAYS,
      dayKey
    );
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
      saveGuestMemSafe();
    }

    if (GUEST_RETENTION_DAYS === 1) {
      const t = setTimeout(() => {
        guestSeqRef.current = 1;
        guestMemRef.current = [];
        saveGuestMemSafe({ day: dayKey(), seq: 1, mem: [] });
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
      } catch { }
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
    } catch { }
  }, [videoId]);

  // hot-plug devices
  useEffect(() => {
    const onChange = async () => {
      try {
        const list = await navigator.mediaDevices.enumerateDevices();
        setVideoDevs(list.filter((d) => d.kind === "videoinput"));
      } catch { }
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

  // visibility -> stop
  useEffect(() => {
    const onVisibility = () => {
      if (document.hidden && !keepBgOnRef.current) {
        stopAll({ reason: "visibility" });
      }
    };
    const onPageHide = () => {
      // Page is actually leaving (navigate/close) -> always stop
      stopAll({ reason: "pagehide" });
    };
    const onBeforeUnload = () => {
      /* ...existing... */
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

  // Distance from face width (px) using current intrinsics
  const estimateDistanceMpx = useCallback((wPx) => {
    const fx = camFxRef.current || FOCAL_PX;
    return Number.isFinite(wPx) && wPx > 0 ? (fx * FACE_WIDTH_M) / wPx : null;
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
      if (
        now - lastRun < loopStepMsRef.current ||
        !video.videoWidth ||
        video.readyState < 2 ||
        detecting
      )
        return;
      frameCount++;
      lastRun = now;

      detecting = true;
      let frameCandidates = [];
      let guestSnapshots = [];
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
        const heavyAgeNow =
          now - (lastAgeSampleRef.current || 0) >= AGE_SAMPLE_MS;
        if (heavyAgeNow) lastAgeSampleRef.current = now;

        //#region detection chain
        let dets = [];
        try {
          let chain = faceapi.detectAllFaces(video, tinyOptsRef.current);
          if (faceapi.nets.faceLandmark68Net?.isLoaded)
            chain = chain.withFaceLandmarks();
          if (heavyAgeNow && faceapi.nets.ageGenderNet?.isLoaded)
            chain = chain.withAgeAndGender();
          if (faceapi.nets.faceRecognitionNet?.isLoaded)
            chain = chain.withFaceDescriptors();
          dets = await chain;
        } catch (e) {
          console.warn("faceapi detect chain failed:", e?.message || e);
          dets = [];
        }

        //#endregion

        logFace(`[DEBUG Face Detection] Detected ${dets.length} faces, wsConnected=${wsIsConnected.current}`);

        //  FIX: Send PeopleData with zone="none" when no faces detected
        if (dets.length === 0 && wsIsConnected.current) {
          logFace(`[DEBUG Face Detection] No faces detected - sending zone="none"`);
          sendWsCommand(MSG_TYPE.PeopleData, {
            intent: "none",
            zone: "none",
            guests: [],
            context: buildVisitContext(),
          });

          // Reset local zone/greet tracking state when camera sees nobody.
          try {
            prevZoneMapRef.current.clear();
            greenEntryRef.current.clear();
            callOverStateRef.current.clear();
            pendingGreetsRef.current.clear();
            logZone("[DEBUG Zone Transitions] Cleared zone/greet state (no faces present)");
          } catch {}
        }

        // ==== drawing + bookkeeping ====
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        // --- alignment overlay: crosshair and banner ---
        if (showAlignRef.current) {
          const cx0 = canvas.width * 0.5,
            cy0 = canvas.height * 0.5;

          // crosshair ticks
          ctx.save();
          ctx.strokeStyle = "#0ea5e9";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(cx0 - 22, cy0);
          ctx.lineTo(cx0 + 22, cy0);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(cx0, cy0 - 22);
          ctx.lineTo(cx0, cy0 + 22);
          ctx.stroke();
          ctx.restore();

          // transient banner (countdown / "Calibrating...")
          if (calibMsgRef.current) {
            const msg = calibMsgRef.current;
            ctx.save();
            ctx.font =
              "bold 18px system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif";
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
        let total = 0,
          green = 0,
          red = 0;

        // build candidates (same as before)
        const cutoff = Number.isFinite(redCutoffM) ? redCutoffM : Infinity;
        const candidates = [];
        for (let i = 0; i < resized.length; i++) {
          const det = resized[i];
          const box = det.detection.box;
          const dist = estimateDistanceMpx(box.width);
          if (dist != null && dist > cutoff) continue; // skip way too far

          // Calculate face center X position for center-frame filtering
          const faceCenterX = box.x + box.width / 2;
          const frameWidth = canvas.width;
          const leftBoundary = frameWidth * (1 / 3 - 1 / 8); // match widened green zone (~0.208w)
          const rightBoundary = frameWidth * (2 / 3 + 1 / 8); // match widened green zone (~0.792w)
          const isInCenter = faceCenterX >= leftBoundary && faceCenterX <= rightBoundary;
          const zone = zoneOf(dist, greenMaxMRef.current, faceCenterX, frameWidth);

          logFace(`[DEBUG Face Detection] Face ${i}: dist=${dist?.toFixed(2)}m, centerX=${faceCenterX.toFixed(0)}/${frameWidth} (L:${leftBoundary.toFixed(0)} R:${rightBoundary.toFixed(0)}), inCenter=${isInCenter}, zone=${zone}, greenMax=${greenMaxMRef.current}`);
          candidates.push({ i, det, box, dist, zone });
        }
        frameCandidates = candidates;

        // Totals for status policy (all visible faces within cutoff)
        total = candidates.length;
        green = candidates.filter((c) => c.zone === "green").length;
        red = total - green;

        // === RED ZONE CALL-OVER DETECTION (NONE -> RED) ===
        // This runs every frame to detect NONE->RED transitions with stabilization
        try {
          const hasRedOnly = red > 0 && green === 0; // People in RED zone, none in GREEN
          const hasNone = total === 0; // Nobody detected at all

          if (hasNone) {
            // Increment NONE counter, reset RED counter
            noneZoneCounterRef.current += 1;
            redZoneCounterRef.current = 0; // STRICT: Any flicker resets RED counter
            if (noneZoneCounterRef.current <= 3 || noneZoneCounterRef.current === RED_ZONE_NONE_RESET_FRAMES) {
              logZone(`[RED Zone] NONE frame ${noneZoneCounterRef.current}/${RED_ZONE_NONE_RESET_FRAMES} (total=${total})`);
            }

            // After 15 consecutive NONE frames, allow RED zone to trigger again
            if (noneZoneCounterRef.current >= RED_ZONE_NONE_RESET_FRAMES) {
              if (redZoneTriggeredRef.current) {
                logZone(`[RED Zone] Reset after ${RED_ZONE_NONE_RESET_FRAMES} NONE frames`);
                redZoneTriggeredRef.current = false;
              }
            }
          } else if (hasRedOnly) {
            // Increment RED counter, reset NONE counter
            redZoneCounterRef.current += 1;
            noneZoneCounterRef.current = 0; // STRICT: Any flicker resets NONE counter
            logZone(`[RED Zone] Frame ${redZoneCounterRef.current}/${RED_ZONE_STABLE_FRAMES} (red=${red}, green=${green})`);

            // Send PeopleData on EVERY RED frame (server ZoneMonitor needs continuous frames)
            if (wsIsConnected.current) {
              const redPayload = {
                intent: "none", // No specific intent, just zone data
                zone: "red",
                guests: Array.from({ length: red }, (_, i) => ({
                  gid: `RedGuest${i + 1}`,
                  name: null,
                  gender: null,
                  ageGroup: null,
                  zone: "red"
                })),
                context: buildVisitContext()
              };

              // Log only on first send and every 5th frame
              if (redZoneCounterRef.current === 1 || redZoneCounterRef.current % 5 === 0) {
                logZone(`[RED Zone] Sending PeopleData frame ${redZoneCounterRef.current}:`, redPayload);
              }
              sendWsCommand(MSG_TYPE.PeopleData, redPayload);

              // Mark as triggered after first stable detection (for any future logic)
              if (redZoneCounterRef.current >= RED_ZONE_STABLE_FRAMES && !redZoneTriggeredRef.current) {
                logZone(`[RED Zone] OK STABLE: ${redZoneCounterRef.current} consecutive RED frames detected`);
                redZoneTriggeredRef.current = true;
              }
            }
          } else {
            // Mixed GREEN + RED, or just GREEN - reset both counters
            redZoneCounterRef.current = 0;
            noneZoneCounterRef.current = 0;
          }
        } catch (err) {
          console.error("[RED Zone] Error in detection logic:", err);
        }

        // 2) Only TRACK up to 5 people in the GREEN zone, nearest first
        const greenCandidates = candidates
          .filter((c) => c.zone === "green" && Number.isFinite(c.dist))
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
            if (
              best &&
              best.label !== "unknown" &&
              best.distance <= MATCH_THRESHOLD
            ) {
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
          if (gesturesOnRef.current && k < gestureTargetsRef.current)
            gestureAllowedKeys.add(stableKey);

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
          const ageVal = Number.isFinite(det.age)
            ? det.age
            : Number.isFinite(cacheGA.age)
              ? cacheGA.age
              : null;
          if (heavyAgeNow && (Number.isFinite(det.age) || det.gender)) {
            ageGenderCacheRef.current.set(stableKey, {
              age: det.age,
              gender: det.gender,
            });
          }
          const expr = topExpression(det.expressions);

          // === angles / position / mouth activity / draw ===
          const dbox = shrinkBox(box);
          const cx = dbox.x + dbox.width * 0.5;
          const cy = dbox.y + dbox.height * 0.45;

          const fx = camFxRef.current,
            fy = camFyRef.current;
          const cx0 = canvas.width * 0.5,
            cy0 = canvas.height * 0.5;

          const { yaw, pitch } = anglesFromPixel(cx, cy, fx, fy, cx0, cy0);
          let yawDeg = yaw * RAD + panOffRef.current;
          let pitchDeg = pitch * RAD + tiltOffRef.current;

          const Z = Number.isFinite(dist) ? dist : null;
          const pos =
            Z != null
              ? posFromPixel(cx, cy, fx, fy, cx0, cy0, Z)
              : { x: null, y: null, z: null };

          const normX = Math.min(
            1,
            Math.abs((cx - cx0) / (canvas.width * 0.5))
          );
          const normY = Math.min(
            1,
            Math.abs((cy - cy0) / (canvas.height * 0.5))
          );
          const centerNorm = Math.min(1, Math.hypot(normX, normY));

          // mouth EMA with hold (avoid 0-drops)
          let mouthActivity = 0;
          const lmBox = det.detection?.box;

          try {
            const lm = det.landmarks;
            const key = stableKey;
            const rec = mouthMapRef.current.get(key) || { ema: 0.3, t: now };
            const level = mouthMAR(lm);
            if (!Number.isFinite(level) || level <= 0) {
              // hold previous with gentle decay toward neutral 0.3
              rec.ema = 0.98 * rec.ema + 0.02 * 0.3;
            } else {
              rec.ema = rec.ema ? 0.7 * rec.ema + 0.3 * level : level;
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

          // Per-face gesture label (no global fallback -> true separation)
          const faceStable = perFaceStableRef.current.get(stableKey);
          const freshFaceGesture =
            gestureAllowedKeys.has(stableKey) &&
              faceStable &&
              now - faceStable.t <= HANDS_CACHE_MS
              ? faceStable
              : null;

          const gestureLbl =
            zone === "green" && freshFaceGesture
              ? gestureLabelOf(freshFaceGesture)
              : null;

          const ageTxt = Number.isFinite(ageVal)
            ? Math.max(0, Math.round(ageVal))
            : "-";
          const l1 = `${displayName}${gestureLbl ? "  |  " + gestureLbl : ""
            }  |  ${zone}  |  ${ageTxt} ${gender}  |  ${expr}`;
          const l2 = `yaw ${yawDeg.toFixed(1)} deg | pitch ${pitchDeg.toFixed(
            1
          )} deg | mouth ${mouthActivity.toFixed(2)} | landmarks :${lmBox.x}`;

          // ----- LABEL DRAW (fixed: define color; removed duplicate vars/badges) -----
          const color =
            zone === "green" ? "rgba(34,197,94,0.85)" : "rgba(239,68,68,0.85)";
          const lineH = 18;
          const lines = showAlignRef.current ? 2 : 1;
          const tw =
            Math.max(
              ctx.measureText(l1).width,
              showAlignRef.current ? ctx.measureText(l2).width : 0
            ) +
            LABEL_PAD_X * 2;
          const th = lineH * lines + LABEL_PAD_Y * 2;

          //general location of the face
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
            const barW = 64,
              barH = 5,
              gap = 3;
            const bx = lx,
              by = ly + th + gap;
            ctx.fillStyle = "rgba(255,255,255,0.15)";
            ctx.fillRect(bx, by, barW, barH);
            ctx.fillStyle = "#22c55e";
            ctx.fillRect(
              bx,
              by,
              barW * Math.min(1, Math.max(0, mouthActivity)),
              barH
            );
          }

          // Per-face gesture text badge on box (keep only this one)
          if (freshFaceGesture && zone === "green") {
            const gtxt = gestureLabelOf(freshFaceGesture);
            if (gtxt) {
              ctx.save();
              ctx.font =
                "bold 14px system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif";
              const padX = 6,
                padY = 4;
              const tw2 = ctx.measureText(gtxt).width + padX * 2;
              const th2 = 18 + padY * 2;
              const gx = Math.max(
                0,
                Math.min(dbox.x + dbox.width - tw2 - 4, canvas.width - tw2)
              );
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

          //gathering all green zone 
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
            slotKey, // optional: keep slot for debug/UI
            _cx: cx,
            _cy: cy,
            _w: dbox.width,
            _h: dbox.height,
            _can_w : canvas.width,
            _can_h : canvas.height,
          });

          console.log(`canvas width :${canvas.width} | height :${canvas.height}`);
        }

        // remember face centers for click-to-zero + per-face hands mapping
        trackedFacesRef.current = peopleForPost.map((p, idx) => ({
          cx: p._cx,
          cy: p._cy,
          w: p._w,
          h: p._h,
          yawDeg: p.yawDeg,
          pitchDeg: p.pitchDeg,
          key: p.stableKey,
          name: p.name || null,
          gid: p.gid || null,
          index: idx,
          gestureEligible: gestureAllowedKeys.has(p.stableKey),
          z: p.posCam?.z ?? null, // NEW: carry depth
        }));

        
        guestSnapshots = peopleForPost.map((p) => ({
          name: p.name || null,
          zone: p.zone,
          gender: p.gender || null,
          ageGroup: p.ageGroup || null,
        }));

        // Also expose ALL faces (GREEN + RED) for hand proximity (on_phone)
        allFacesRef.current = candidates.map((c) => 
        {
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
          const drawn = new Set(tracked.map((t) => t.i));

          for (const c of candidates) {
            if (drawn.has(c.i)) continue; // skip tracked (already drawn)

            const { box, dist, zone } = c;
            const dbox = shrinkBox(box);

            // Choose color: RED for red-zone, grey for others we didn't track
            const stroke = zone === "red" ? "#ef4444" : "#999999";
            const fill = zone === "red" ? "#ef4444" : "#666666";

            // Outline
            ctx.save();
            ctx.strokeStyle = stroke;
            ctx.lineWidth = 3;
            ctx.strokeRect(dbox.x, dbox.y, dbox.width, dbox.height);

            // Minimal label: zone + distance
            const l1 = `${zone} | ${dist ? dist.toFixed(2) + " m" : "-"}`;
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

        // Draw center-frame guide lines (middle third for GREEN zone)
        try {
          const leftBoundary = canvas.width * (1 / 3 - 1 / 8); // match widened green zone (~0.208w)
          const rightBoundary = canvas.width * (2 / 3 + 1 / 8); // match widened green zone (~0.792w)

          ctx.save();
          ctx.strokeStyle = "rgba(34, 197, 94, 0.5)"; // Semi-transparent green
          ctx.lineWidth = 2;
          ctx.setLineDash([10, 10]); // Dashed line

          // Left boundary
          ctx.beginPath();
          ctx.moveTo(leftBoundary, 0);
          ctx.lineTo(leftBoundary, canvas.height);
          ctx.stroke();

          // Right boundary
          ctx.beginPath();
          ctx.moveTo(rightBoundary, 0);
          ctx.lineTo(rightBoundary, canvas.height);
          ctx.stroke();

          ctx.restore();
        } catch {}

        // prune only faces that are no longer tracked (preserve history for tracked-but-not-eligible)
        {
          const keep = new Set(
            (trackedFacesRef.current || []).map((f) => f.key)
          );
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
          const seen = new Set(
            peopleForPost.map((p) => (p.name || p.gid) ?? "")
          );
          for (const k of Object.keys(recentMapRef.current)) {
            if (k && !seen.has(k)) delete recentMapRef.current[k];
          }
        }

        // === Focus selection (prefer GREEN, fallback to any tracked) ===
        const pool = peopleForPost.filter((p) => p.zone === "green");
        const cand = pool.length ? pool : peopleForPost;

        let focusIndex = cand.length ? 0 : -1;
        let focusScore = -1,
          focusMeta = null;

        for (let idx = 0; idx < cand.length; idx++) {
          const p = cand[idx];
          let sNear = 0;
          const z = p?.posCam?.z;
          if (Number.isFinite(z) && z > 0) {
            sNear = Math.max(
              0,
              Math.min(1, (2.0 - Math.min(2.0, Math.max(0.3, z))) / (2.0 - 0.3))
            );
          }
          const sCenter = 1 - Math.max(0, Math.min(1, p.centerNorm ?? 1));
          const sMouth = Math.max(0, Math.min(1, p.mouthActivity ?? 0));
          const score = wNear * sNear + wCenter * sCenter + wMouth * sMouth;
          if (score > focusScore) {
            focusScore = score;
            focusIndex = idx;
            focusMeta = {
              sNear: +sNear.toFixed(3),
              sCenter: +sCenter.toFixed(3),
              sMouth: +sMouth.toFixed(3),
              score: +score.toFixed(3),
            };
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

        if(cand.length > 0){
          if (wsIsConnected.current) {
            const ts = performance.now();
            if (ts - (lastPeopleSnapshotSentRef.current || 0) >= 200) {
              const guests = cand.map((p) => ({
                gid: p.gid || null,
                name: p.name || null,
                gender: p.gender || null,
                ageGroup: p.ageGroup || null,
                zone: "green",
              }));

              sendWsCommand(MSG_TYPE.PeopleData, {
                intent: "none",
                zone: "green",
                guests,
                context: buildVisitContext(),
              });

              lastPeopleSnapshotSentRef.current = ts;
            }
          }

          //emit all green face here
          console.log(`[EMIT] Crowd data: ${JSON.stringify(cand)}`);
          /*
          emitCrowdThrottled({
                  deviceId,
                  sessionId: sessionId || "web-" + deviceId,
                  timeISO: new Date().toISOString(),
                  backend,
                  totals: green,
                  gesture: gesturesOnRef.current ? fresh : null,
                  focusIndex: focusIndexRef.current,
                  focusTarget: focusTargetRef.current,
                  people: cand
                });*/

          emitCrowdByGid({deviceId,
                  sessionId: sessionId || "web-" + deviceId,
                  timeISO: new Date().toISOString(),
                  backend,
                  totals: green,
                  gesture: gesturesOnRef.current ? fresh : null,
                  focusIndex: focusIndexRef.current,
                  focusTarget: focusTargetRef.current,
                  people: cand
                });
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
            distance: "-",
          });
        }

        setTable((prev) => {
          const same =
            prev.length === rows.length &&
            prev.every((r, i) => JSON.stringify(r) === JSON.stringify(rows[i]));
          return same ? prev : rows;
        });
        setTotals((prev) =>
          prev.all === total && prev.green === green && prev.red === red
            ? prev
            : { all: total, green, red }
        );

        // Game mode auto-exit on no-face stretch
        if (gameModeRef.current) {
          const now2 = performance.now();
          if (green > 0) lastGameActivityRef.current = now2;
          if (
            green === 0 &&
            now2 - (lastGameActivityRef.current || 0) > GM_NO_FACE_TIMEOUT_MS
          ) {
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
              Math.floor(
                Math.random() *
                  (LOOP_STEP_IDLE_MAX_MS - LOOP_STEP_IDLE_MIN_MS + 1)
              );
            loopStepMsRef.current = jitter;
          }
        } else if (loopIdleStateRef.current) {
          loopIdleStateRef.current = false;
          loopStepMsRef.current = LOOP_STEP_ACTIVE_MS;
        }

        // ---- HANDS: per-person attribution (nearest green face) ----
        const handsEligible =
          HANDS_ENABLED && handsReadyRef.current && gesturesOnRef.current;
        const gm = !!gameModeRef.current;
        const handsDesiredStep =
          now - (lastLmSeenTsRef.current || 0) <= 800
            ? gm
              ? GM_HANDS_FAST_MS
              : HANDS_FAST_MS
            : gm
            ? GM_HANDS_IDLE_MS
            : HANDS_IDLE_MS;

        //test apakah tangan terdeteksi, kmd kirim data ke server, hanya untuk gesture
        if (handsEligible && now - (lastHandsRunTsRef.current || 0) >= handsDesiredStep) 
        {
          lastHandsRunTsRef.current = now;
          try {
            const handsList = await detectHandsOnce(video);

            if (handsList && handsList.length) {
              // HUD + wrist dots
              ctx.save();
              ctx.font =
                "bold 14px system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif";
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
                const w = lm[MP.WRIST],
                  i = lm[MP.INDEX_MCP];
                if (!w || !i) return null;
                return { x: (w.x + i.x) * 0.5, y: (w.y + i.y) * 0.5 };
              };

              // helper: is anchor inside (expanded) face rect
              const wristInFace = (px, py, f) => {
                const left = f.cx - f.w * 0.5;
                const top = f.cy - f.h * 0.45;
                const right = left + f.w;
                const bottom = top + f.h;
                // allow more room above the face (hands raised), keep sides tighter
                const mx = f.w * 0.09,
                  myUp = f.h * 0.35,
                  myDown = f.h * 0.22;
                return (
                  px >= left - mx &&
                  px <= right + mx &&
                  py >= top - myUp &&
                  py <= bottom + myDown
                );
              };

              // Eligible faces (top-2 closest, marked earlier)
              const facesAll = trackedFacesRef.current || [];
              const faces = facesAll.filter((f) => f.gestureEligible);
              if (!faces.length) {
                // No eligible faces this tick -> keep per-face state; just update global/fallback below
              } else {
                // 1) Build hand->face candidate pairs (eligible faces only)
                const hands = handsList
                  .map((lm, hi) => {
                    const a = handAnchor(lm);
                    if (!a) return null;
                    const ax = a.x * canvas.width;
                    const ay = a.y * canvas.height;
                    return { lm, hi, ax, ay };
                  })
                  .filter((h) => {
                    if (!(h && Number.isFinite(h.ax) && Number.isFinite(h.ay)))
                      return false;
                    // reject tiny/tentative hands (ghosts)
                    const span = palmSpanLen(h.lm); // normalized 0..1
                    return span >= 0.02; // ~2% of frame width
                  });

                const allPairs = [];
                for (const h of hands) {
                  // primary: anchor inside face window
                  let contenders = faces.filter((f) =>
                    wristInFace(h.ax, h.ay, f)
                  );

                  // fallback: if none hit, accept nearest face if horizontally aligned and vertically near
                  if (!contenders.length) {
                    let best = null,
                      bestDx = Infinity;
                    for (const f of faces) {
                      const dx = Math.abs(h.ax - f.cx);
                      const withinX = dx <= f.w * 0.45;
                      const withinY =
                        h.ay >= f.cy - f.h * 0.6 && h.ay <= f.cy + f.h * 0.2;
                      if (withinX && withinY && dx < bestDx) {
                        bestDx = dx;
                        best = f;
                      }
                    }
                    if (best) contenders = [best];
                  }

                  // final fallback: always assign (handles hands far from faces)
                  if (!contenders.length) {
                    if (faces.length === 1) {
                      // single eligible face -> give it the hand
                      contenders = [faces[0]];
                    } else if (faces.length === 2) {
                      // 2 faces: split by midline (stable left/right assignment)
                      const [leftF, rightF] =
                        faces[0].cx <= faces[1].cx
                          ? [faces[0], faces[1]]
                          : [faces[1], faces[0]];
                      const midX = (leftF.cx + rightF.cx) * 0.5;
                      contenders = [h.ax <= midX ? leftF : rightF];
                    } else {
                      // 3+ faces: nearest center with small vertical penalty
                      let bestN = null,
                        bestScore = Infinity;
                      for (const f of faces) {
                        const dx = h.ax - f.cx,
                          dy = h.ay - f.cy;
                        const score = dx * dx + dy * dy * 0.4;
                        if (score < bestScore) {
                          bestScore = score;
                          bestN = f;
                        }
                      }
                      if (bestN) contenders = [bestN];
                    }
                  }

                  for (const f of contenders) {
                    const dx = h.ax - f.cx,
                      dy = h.ay - f.cy;
                    allPairs.push({
                      hi: h.hi,
                      lm: h.lm,
                      face: f,
                      d2: dx * dx + dy * dy,
                    });
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
                  arr.sort((a, b) => a.d2 - b.d2);
                  const best = arr[0];
                  const second = arr[1];
                  // size-scaled near-tie (don't drop unless truly ambiguous)
                  const wRef = second
                    ? Math.max(best.face.w || 1, second.face.w || 1)
                    : 1;
                  if (second) {
                    const nearTie =
                      Math.abs(best.d2 - second.d2) <=
                      wRef * 0.15 * (wRef * 0.15);
                    if (nearTie) {
                      // depth tiebreak: prefer nearer-Z
                      const zBest = Number.isFinite(best.face.z)
                        ? best.face.z
                        : Infinity;
                      const zSecond = Number.isFinite(second.face.z)
                        ? second.face.z
                        : Infinity;
                      if (zSecond < zBest - 0.05) {
                        filtered.push(second);
                        continue;
                      }
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
                  if (usedHands.has(p.hi) || usedFaces.has(p.face.key))
                    continue;
                  assignments.push(p);
                  usedHands.add(p.hi);
                  usedFaces.add(p.face.key);
                }

                // 4) Classify per assigned face (wave history is per-face)
                // Game mode runs 4 classifiers (wave + R/P/S); normal runs 5 (wave + peace + raise_hand + on_phone + thumbs_up)
                for (const { lm, face } of assignments) {
                  let hist = waveHistByFaceRef.current.get(face.key);
                  if (!hist) {
                    hist = { t: 0, xs: [] };
                    waveHistByFaceRef.current.set(face.key, hist);
                  }
                  const xs = hist.xs;
                  if (now - (hist.t || 0) > 900) xs.length = 0;
                  hist.t = now;

                  setWaveHistory(xs, hist.t);

                  const a0 = handAnchor(lm);
                  let allowWave = false;
                  if (a0) {
                    const ax0 = a0.x * canvas.width,
                      ay0 = a0.y * canvas.height;
                    // near this face box or broadly aligned band
                    if (wristInFace(ax0, ay0, face)) {
                      allowWave = true;
                    } else {
                      const dx = Math.abs(ax0 - face.cx);
                      const withinX = dx <= face.w * 0.9;
                      const withinY =
                        ay0 >= face.cy - face.h * 1.1 &&
                        ay0 <= face.cy + face.h * 0.5;
                      allowWave = withinX && withinY;
                    }
                  }
                  // if far from box but motion clearly wave-y, still allow
                  if (!allowWave) {
                    const wa = waveActivity();
                    if (wa.flips >= 2 && wa.amp > 0.02) allowWave = true;
                  }

                  // debug ear anchor for assigned face `face` (safe: no undefined refs)
                  ctx.save();
                  ctx.fillStyle = "rgba(0,180,255,0.8)";
                  const handX = a0
                    ? a0.x * canvas.width
                    : (lm[MP.WRIST]?.x || 0) * canvas.width;
                  const sideSignDbg = handX >= face.cx ? +1 : -1;
                  const earX = face.cx + sideSignDbg * (face.w * 0.5) * 0.78;
                  const earY = face.cy - face.h * 0.5 * 0.08;
                  ctx.beginPath();
                  ctx.arc(earX, earY, 5, 0, Math.PI * 2);
                  ctx.fill();
                  ctx.restore();

                  const velNow = recentLateralMotion();
                  const wrist = lm[MP.WRIST],
                    iMcp = lm[MP.INDEX_MCP];
                  const vx = (iMcp?.x ?? 0) - (wrist?.x ?? 0),
                    vy = (iMcp?.y ?? 0) - (wrist?.y ?? 0);
                  const axisLen = Math.hypot(vx, vy) || 1e-6;
                  const cosToVertical = Math.abs(vy) / axisLen; // 1 = vertical, 0 = horizontal

                  // Face-relative proximity for "pose" gestures (prevents random pops)
                  let allowNearFace = false,
                    nearX = false,
                    highPalm = false;
                  if (a0) {
                    const ax0 = a0.x * canvas.width,
                      ay0 = a0.y * canvas.height;
                    nearX = Math.abs(ax0 - face.cx) <= face.w * 1.0;
                    // "high palm": above face center by a bit, even if not inside the box
                    highPalm = ay0 <= face.cy - face.h * 0.05;
                    const highEnough = ay0 <= face.cy + face.h * 0.25;
                    allowNearFace =
                      Math.abs(ax0 - face.cx) <= face.w * 0.85 && highEnough;
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
                    try {
                      const r = classifyRock(lm);
                      if (r.ok) cand.push({ type: "rock", score: r.score });
                    } catch {}
                    try {
                      const s = classifyScissors(lm);
                      if (s.ok) cand.push({ type: "scissors", score: s.score });
                    } catch {}
                    try {
                      const p = classifyPaper(lm);
                      // paper: open palm; allow near face OR clearly high & aligned; upright-ish; not swinging
                      if (
                        p.ok &&
                        (allowNearFace || highPalm || palm >= 0.038) &&
                        velNow <= 0.11 &&
                        cosToVertical > 0.5 &&
                        palm >= 0.028
                      ) {
                        cand.push({ type: "paper", score: p.score });
                      }
                    } catch {}
                  } else {
                    try {
                      const p = classifyPeace(lm);
                      if (p.ok) cand.push({ type: "peace", score: p.score });
                    } catch {}
                    try {
                      const rh = classifyRaiseHand(lm);
                      if (rh.ok)
                        cand.push({ type: "raise_hand", score: rh.score });
                    } catch {}
                    // phone stays scoped to this face (already done)
                    try {
                      const ph = classifyOnPhone(
                        lm,
                        [{ cx: face.cx, cy: face.cy, w: face.w, h: face.h }],
                        canvas.width,
                        canvas.height
                      );
                      if (ph.ok)
                        cand.push({ type: "on_phone", score: ph.score });
                    } catch {}
                    try {
                      const t = classifyThumbsUp(lm);
                      if (t.ok)
                        cand.push({ type: "thumbs_up", score: t.score });
                    } catch {}
                  }

                  // If a strong pose is present, drop weaker wave this frame
                  {
                    const poseBest = cand
                      .filter(
                        (c) =>
                          c.type === "thumbs_up" ||
                          c.type === "peace" ||
                          c.type === "raise_hand" ||
                          c.type === "on_phone"
                      )
                      .sort((a, b) => b.score - a.score)[0];
                    const waveIdx = cand.findIndex((c) => c.type === "wave");
                    if (poseBest && waveIdx >= 0) {
                      const waveScore = cand[waveIdx].score;
                      if (waveScore < poseBest.score + 0.12) {
                        cand.splice(waveIdx, 1);
                      }
                    }
                  }

                  if (!cand.length) continue;
                  const bestFrame = cand.reduce((a, b) =>
                    b.score > a.score ? b : a
                  );

                  const prev = byFace.get(face.key);
                  const adj =
                    gm && bestFrame.type === "wave"
                      ? { type: "paper", score: bestFrame.score }
                      : gm && bestFrame.type === "thumbs_up"
                      ? null
                      : bestFrame;
                  if (adj && (!prev || adj.score > prev.score))
                    byFace.set(face.key, adj);
                }

                // update per-face windows/stable + emit changes
                const updatedKeys = new Set();
                for (const [key, frame] of byFace.entries()) {
                  const win = perFaceGestureWinRef.current.get(key) || [];
                  win.push({ ...frame, t: now });
                  if (win.length > VOTE_WINDOW * 2)
                    win.splice(0, win.length - VOTE_WINDOW * 2);
                  perFaceGestureWinRef.current.set(key, win);

                  const prevStable = perFaceStableRef.current.get(key) || null;
                  const nextStable = pickStableGesture(now, win, prevStable);
                  if (nextStable) {
                    const changed =
                      !prevStable || prevStable.type !== nextStable.type;
                    perFaceStableRef.current.set(key, nextStable);
                    updatedKeys.add(key);
                    const lastSent =
                      lastGestureSentPerFaceRef.current.get(key) || 0;
                    if (
                      changed &&
                      now - lastSent >= HANDS_SEND_MS &&
                      !speakingRef.current
                    ) {
                      const facesMeta = trackedFacesRef.current || [];
                      const meta = facesMeta.find((f) => f.key === key) || {};
                      try {
                        socketRef.current?.emit?.(
                          gm ? "game_event" : "gesture_event",
                          gm
                            ? {
                                sessionId: sessionId || "web-" + deviceId,
                                deviceId,
                                rps: nextStable.type,
                                at: Date.now(),
                                focusIndex: meta.index ?? focusIndexRef.current,
                                focusTarget: {
                                  name: meta.name || null,
                                  gid: meta.gid || null,
                                },
                              }
                            : {
                                sessionId: sessionId || "web-" + deviceId,
                                deviceId,
                                gesture: {
                                  type: nextStable.type,
                                  score: nextStable.score,
                                },
                                at: Date.now(),
                                focusIndex: meta.index ?? focusIndexRef.current,
                                focusTarget: {
                                  name: meta.name || null,
                                  gid: meta.gid || null,
                                },
                              }
                        );
                      } catch {}
                      lastGestureSentPerFaceRef.current.set(key, now);
                    }
                  } else {
                    perFaceStableRef.current.delete(key);
                  }
                }

                // Update global stable gesture for legacy HUD/policy (focus face wins)
                (() => {
                  const faces = trackedFacesRef.current || [];
                  const eligible = new Set(
                    faces.filter((f) => f.gestureEligible).map((f) => f.key)
                  );
                  const fi = focusIndexRef.current;
                  let chosen = null;
                  if (fi >= 0 && faces[fi] && eligible.has(faces[fi].key)) {
                    chosen =
                      perFaceStableRef.current.get(faces[fi].key) || null;
                  }
                  if (!chosen) {
                    for (const [k, g] of perFaceStableRef.current.entries()) {
                      if (!eligible.has(k)) continue;
                      if (
                        now - g.t <= HANDS_CACHE_MS &&
                        g.type === "on_phone"
                      ) {
                        chosen = g;
                        break;
                      }
                    }
                  }
                  if (!chosen) {
                    for (const [k, g] of perFaceStableRef.current.entries()) {
                      if (!eligible.has(k)) continue;
                      if (now - g.t <= HANDS_CACHE_MS) {
                        chosen = g;
                        break;
                      }
                    }
                  }
                  stableGestureRef.current = chosen
                    ? { ...chosen, t: now }
                    : null;
                })();
              }
            }

            // AFTER hands: emit snapshot with up-to-date gesture
            const g = gesturesOnRef.current ? stableGestureRef.current : null;
              const fresh =
                g && now - g.t <= HANDS_CACHE_MS
                  ? { type: g.type, score: g.score }
                  : null;

              console.log("emiting crowds");
              //emit crowd snapshot
              emitCrowdThrottled({
                deviceId,
                sessionId: sessionId || "web-" + deviceId,
                timeISO: new Date().toISOString(),
                aiSpeaking: !!serverInfo.ai_speaking,
                backend,
                totals: { all: total, green, red },
                gesture: gesturesOnRef.current ? fresh : null,
                focusIndex: focusIndexRef.current,
                focusTarget: focusTargetRef.current,
                people: peopleForPost.map((p) => ({
                  name: p.name || null,
                  gid: p.gid || null,
                  gender: p.gender || null,
                  ageGroup: p.ageGroup || null,
                  zone: p.zone,
                  yawDeg: Number.isFinite(p.yawDeg)
                    ? +p.yawDeg.toFixed(1)
                    : null,
                  pitchDeg: Number.isFinite(p.pitchDeg)
                    ? +p.pitchDeg.toFixed(1)
                    : null,
                  mouthActivity: +(p.mouthActivity ?? 0).toFixed(3),
                  posCam: p.posCam,
                })),
              });
          } catch (e) {
            // ignore hand pipeline hiccups so the frame loop keeps running
          }
        } // OK properly close inner try before new catch
        // handled above; removed extraneous catch
        // ---- Policy: zone transitions -> call-over / greet (candidates include red) ----
        try {
          handleZoneTransitions({
            allIdentities,
            now,
            logZone,
            stableGestureRef,
            prevZoneMapRef,
            greenEntryRef,
            callOverStateRef,
            pendingGreetsRef,
            pushedGreets,
            ageGenderCacheRef,
            greetInviteRef,
            lastGroupSetRef,
            lastGroupAskTsRef,
            sendPeopleIntent,
            groupInfo,
            guestSnapshots,
            CALL_OVER_MAX_TRIES,
            CALL_OVER_COOLDOWN_MS,
            GREEN_STABLE_MS,
            NOT_SEEN_RESET_MS,
            GROUP_ASK_COOLDOWN_MS,
            socketRef,
            deviceId,
            sessionId,
            ageGroupOf,
            groupSignature,
            maybeRotateSession,
            rotateConfig: {
              refs: { groupSigRef, groupStableSinceRef, lastRotateRef },
              setSessionId,
              uuid,
              stableMs: GROUP_STABLE_MS,
              cooldownMs: SESSION_ROTATE_COOLDOWN_MS,
            },
          });
        } catch (e) {
          console.warn("[policy] zone transition error:", e);
        }

        // ---- Speaker focus (1-2s or 3 frames dominance among green tracked) ----
          try {
            const list = peopleForPost || [];
            if (list.length) {
              // Prefer GREEN; else use all
              const greens = list.filter((p) => p.zone === "green");
              const pool = greens.length ? greens : list;

              // Pick dominant by mouthActivity
              let topIdx = -1,
                topScore = -1,
                topKey = null;
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
              const stableByTime =
                now - (sp.topSince || 0) >= SPEAKER_STABLE_MS;

              if (
                topKey &&
                sp.key !== topKey &&
                (stableByFrames || stableByTime)
              ) {
                // Map pool index back to absolute index in peopleForPost
                const absIdx = list.indexOf(pool[topIdx]);
                sp.key = topKey;

                const p = list[absIdx];
                socketRef.current?.emit?.("policy_event", {
                  deviceId,
                  sessionId: sessionId || "web-" + deviceId,
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
        } catch (e) {
          console.warn("[speaker] error:", e);
        }
      finally {
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
  }, [ready, videoId, deviceId, sessionId]);

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

      // When tab is hidden and keepBgOn is true, skip idle enforcement
      if (document.hidden && keepBgOnRef.current) return;
      // Camera idle: stop everything unless background mode is enabled
      if (keepBgOnRef.current) return;
      if (ago > CAM_IDLE_MS) {
        stopAll({ reset: true });
      }
    }, CHECK_MS);
    return () => clearInterval(timer);
  }, [ready]);

  // hard stop everything
  async function stopAll({ reset = true } = {}) {
    try {
      camRef.current.stream?.getTracks()?.forEach((t) => t.stop());
    } catch { }
    camRef.current.stream = null;

    speakingRef.current = false;
    S.current = { id: null, seenFrames: 0, lastFaceTs: 0, lastSnapshotTs: 0 };

    try {
      sendWsCommand(MSG_TYPE.SessionEnd, { sessionId });
    } catch { }
    setSessionId(null);
    setSessionStatus("IDLE");
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
  const LIVE_VOICES = [
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Aoede",
    "Leda",
    "Orus",
    "Zephyr",
  ];
  // Native Audio dialog voices (list you gave)
  const NATIVE_VOICES = [
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
  ];

  const GEMINI_VOICES = { live: LIVE_VOICES, native: NATIVE_VOICES };

  // Language options (label -> code)
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
    () =>
      localStorage.getItem("ika:systemInstruction") ||
      "You are a friendly, concise on-site concierge."
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
  const [temperatureQuick, setTemperatureQuick] = useState(() =>
    Number(localStorage.getItem("ika:temperature") ?? 0.6)
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

  const [ttsProviderQuick, setTtsProviderQuick] = useState(() =>
    (localStorage.getItem("ika:ttsProvider") || "gemini").toLowerCase()
  );
  const [elevenVoiceIdQuick, setElevenVoiceIdQuick] = useState(
    () => localStorage.getItem("ika:11labs:voiceId") || ""
  );

  // optional UI toggle elsewhere can flip this; default false
  const gameModeRef = useRef(false);
  const [gameModeOn, setGameModeOn] = useState(false);
  useEffect(() => {
    gameModeRef.current = gameModeOn;
  }, [gameModeOn]);

  // Game mode idle/visibility timeouts (ephemeral)
  const GM_IDLE_TIMEOUT_MS = 90_000; // 1.5 min without activity -> exit RPS
  const GM_NO_FACE_TIMEOUT_MS = 20_000; // 20s with no faces -> exit RPS
  const lastGameActivityRef = useRef(0);
  useEffect(() => {
    if (gameModeOn) lastGameActivityRef.current = performance.now();
  }, [gameModeOn]);

  // Focus shared across blocks (used by gesture events and crowd payload)
  const focusIndexRef = useRef(-1);
  const focusTargetRef = useRef(null);

  // RED zone stabilization tracking
  const redZoneCounterRef = useRef(0); // Counts consecutive RED frames
  const noneZoneCounterRef = useRef(0); // Counts consecutive NONE frames
  const redZoneTriggeredRef = useRef(false); // Prevents re-triggering until reset

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
    } catch { }
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

  // apply quick settings -> create (or recreate) session
  const onCreateSession = useCallback(() => {
    server.createSession({
      model: modelQuick,
      voice: geminiVoiceQuick,
      language_code: languageCodeQuick,
      system_instruction: systemInstruction,

      // route audio: Gemini (AUDIO+TEXT) vs ElevenLabs (TEXT only)
      tts_provider: ttsProviderQuick, // "gemini" | "elevenlabs"
      response_modalities:
        ttsProviderQuick === "elevenlabs" ? ["TEXT"] : ["AUDIO", "TEXT"],

      enable_affective_dialog: enableAffectiveQuick,

      temperature: temperatureQuick,
      proactive_audio: proactiveAudioQuick,
      transcribe_user_audio: true,
      files_to_upload: null,

      // OK ElevenLabs-only fields (used iff tts_provider === "elevenlabs")
      eleven_model:
        localStorage.getItem("ika:11labs:model") || "eleven_turbo_v2_5",
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
    temperatureQuick,
    proactiveAudioQuick,
  ]);

  // hot update
  const handleStartSession = useCallback(() => {
    try {
      onCreateSession();
    } catch (err) {
      console.warn("[session] start failed:", err);
    }
  }, [onCreateSession]);

  const handleStopSession = useCallback(() => {
    try {
      sendWsCommand(MSG_TYPE.SessionEnd, {
        sessionId: sessionId || "web-" + deviceId,
      });
      setSessionStatus("IDLE");
      setSessionId(null);
    } catch (err) {
      console.warn("[session] stop failed:", err);
    }
  }, [sessionId, deviceId]);

  useEffect(() => {
    if (sessionStatus === "ACTIVE" || sessionStatus === "IDLE") {
      autoSessionPendingRef.current = false;
    }
  }, [sessionStatus]);

  useEffect(() => {
    if (!autoSession) return;

    const serverUp = !!serverInfo.connected;
    const ueUp = !!ueConnected;

    if (serverUp && ueUp) {
      if (sessionStatus !== "ACTIVE" && !autoSessionPendingRef.current) {
        autoSessionPendingRef.current = true;
        handleStartSession();
      }
    } else {
      autoSessionPendingRef.current = false;
      // Keep session alive; do not auto-send SessionEnd on transient disconnects.
      // Camera/zone policy on server controls mic open/close safely.
    }
  }, [
    autoSession,
    serverInfo.connected,
    ueConnected,
    sessionStatus,
    handleStartSession,
  ]);

  const onHotUpdate = useCallback(() => {
    server.updateSettings({
      temperature: temperatureQuick,
      captions,

      enable_affective_dialog: enableAffectiveQuick,
      proactive_audio: proactiveAudioQuick,
      function_calling: functionCallingQuick,
      auto_function_response: autoFunctionResponseQuick,
      grounding: groundingQuick,
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
  ]);

  /* ====================== UI ====================== */
  return (
    <main className="app">
      <div
        className="app-body"
        style={{
          display: "flex",
          justifyContent: "center",
          width: "100%",
        }}
      >
        <div
          className="main-column"
          style={{ margin: "0 auto", width: "100%", maxWidth: 1280 }}
        >
          {/* Camera window pinned to top */}
          <div className="stage">
            <video ref={videoRef} autoPlay muted playsInline />
            <canvas ref={canvasRef} />
          </div>

          {/* Row 1: CAMERA/STATUS */}
          <div className="left-top2">
            {/* CAMERA / STATUS (compact, grouped) */}
            <div className="panel compact">
              <div className="statgrid">
                {/* --- Section: Environment --- */}
                <div className="block">
                  <div className="block-title">Environment</div>
                  <div className="kv">
                    <b>Location:</b> {locationLabel}
                  </div>
                  <div className="kv">
                    <b>Time:</b> {clock.toLocaleTimeString()}
                  </div>
                  <div className="kv">
                    <b>Weather:</b> {weatherLabel}
                  </div>
                  <div className="kv">
                    <b>Backend:</b> {backend}
                  </div>
                  <div className="kv">
                    <b>Models:</b> {ready ? "loaded" : "loading..."}
                  </div>
                </div>

                {/* --- Section: Live status --- */}
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
                    <span
                      className={`dot ${serverInfo.connected ? "ok" : "err"}`}
                    />
                    <b>Server:</b>&nbsp;
                    {serverInfo.connected ? "connected" : "disconnected"}
                  </div>
                  <div className="kv">
                    <span className={`dot ${ueConnected ? "ok" : "err"}`} />
                    <b>UE link:</b>&nbsp;
                    {ueConnected ? "connected" : "waiting"}
                  </div>
                  {serverInfo.model || serverInfo.tts ? (
                    <div className="kv muted small">
                      {serverInfo.model ? <>Model: {serverInfo.model}</> : null}
                      {serverInfo.model && serverInfo.tts ? " | " : null}
                      {serverInfo.tts ? <>TTS: {serverInfo.tts}</> : null}
                    </div>
                  ) : null}

                  {/* Device binding */}
                  <div className="kv">
                    <b>Device:</b>&nbsp;{deviceId.slice(0, 8)}...
                    <span className="muted">
                      &nbsp;
                      {serverInfo.boundDeviceId
                        ? `(bound ${String(serverInfo.boundDeviceId).slice(
                            0,
                            8
                          )}...)`
                        : `(not bound)`}
                    </span>
                  </div>
                </div>

                {/* --- Section: Traffic --- */}
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
                    <span className="chip">
                      snap {lastHttp.snapshot || "-"}
                    </span>
                    <span className="chip">stop {lastHttp.stop || "-"}</span>
                    <span className="muted">
                      {USE_SOCKET_BRIDGE ? " (socket bridge)" : ""}
                    </span>
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
                        saveGuestMemSafe({ day: dayKey(), seq: 1, mem: [] });
                      }}
                    >
                      clear guests
                    </button>
                  </div>
                </div>
              </div>
            </div>
            <div
              className="left-top-side"
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 12,
                minWidth: 300,
                flex: "0 0 auto",
              }}
            >
              {/* Save/Export Settings */}
              <section className="panel">
                <h3 className="section-title">settings backup</h3>
                <div className="row" style={{ gap: 8 }}>
                  <button className="btn" onClick={exportSettings}>
                    Export to file
                  </button>
                  <button className="btn" onClick={importSettings}>
                    Import from file
                  </button>
                  <button className="btn" onClick={resetSettings}>
                    Reset all
                  </button>
                </div>
    
              </section>

              {/* Server connection */}
              <section className="panel">
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
                    onClick={() => {
                      setServerUrl("");
                      setServerUrlDraft(window.location.origin);
                    }}
                  >
                    Use same-origin
                  </button>
                </div>
                <div className="help" style={{ marginTop: 6 }}>
                  Config:{" "}
                  {serverUrl && serverUrl.trim() ? serverUrl : "(same-origin)"} |
                  Effective: {effectiveUrl}
                  <br />
                  Status: {serverInfo.connected ? "connected" : "disconnected"}
                </div>
                <div
                  className="row"
                  style={{ gap: 8, marginTop: 12, alignItems: "center" }}
                >
                  <input
                    className="input bigpad"
                    placeholder="Device / session ID (match UE)"
                    value={deviceIdDraft}
                    onChange={(e) => setDeviceIdDraft(e.target.value)}
                    autoCorrect="off"
                    autoCapitalize="off"
                    spellCheck={false}
                  />
                  <button
                    className="btn"
                    disabled={
                      !deviceIdDraft.trim() ||
                      deviceIdDraft.trim() === deviceId
                    }
                    onClick={applyDeviceId}
                  >
                    Apply device ID
                  </button>
                  <button className="btn" onClick={randomizeDeviceId}>
                    New ID
                  </button>
                </div>
                <div className="help" style={{ marginTop: 6 }}>
                  Stored as <code>ika:deviceId</code>. Use the same value as the
                  UE client to share a single session.
                </div>
              </section>

              {/* Session control */}
              <section className="panel">
                <h3 className="section-title">session control</h3>
                <div className="row" style={{ gap: 8 }}>
                  <button
                    className="btn"
                    onClick={handleStartSession}
                    disabled={sessionStatus === "ACTIVE"}
                  >
                    Start session
                  </button>
                  <button
                    className="btn"
                    onClick={handleStopSession}
                    disabled={sessionStatus !== "ACTIVE"}
                  >
                    Stop session
                  </button>
                </div>
                <div
                  className="row"
                  style={{ gap: 8, marginTop: 12, alignItems: "center" }}
                >
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={autoSession}
                      onChange={(e) => setAutoSession(e.target.checked)}
                    />
                    <span>Auto-manage when server + UE are online</span>
                  </label>
                </div>
                <div className="help" style={{ marginTop: 6 }}>
                  Status: {sessionStatus}
                  <br />
                  {autoSession
                    ? "Auto-starts when both links are up and stops if either drops."
                    : "Auto session is off; use the buttons above."}
                </div>
              </section>

              {/* Performance */}
              <section className="panel">
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
                  <div
                    className="row"
                    style={{ gap: 8, alignItems: "center", marginLeft: 8 }}
                  >
                    <label className="label" style={{ margin: 0 }}>
                      Targets
                    </label>
                    <select
                      className="select"
                      value={gestureTargets}
                      onChange={(e) =>
                        setGestureTargets(
                          parseInt(e.target.value, 10) === 1 ? 1 : 2
                        )
                      }
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
                <div
                  className="row"
                  style={{ gap: 12, alignItems: "center", marginTop: 8 }}
                >
                  <label className="checkbox">
                    <input
                      type="checkbox"
                      checked={keepBgOn}
                      onChange={(e) => setKeepBgOn(e.target.checked)}
                    />
                    <span>Keep running in background</span>
                  </label>
                </div>
              </section>
            </div>
          </div>

          {/* Devices */}
          <div className="panel">
            <label className="label">Camera</label>
            <select
              className="select big"
              value={videoId}
              onChange={async (e) => {
                const next = e.target.value;
                setVideoId(next);
                try {
                  localStorage.setItem("ika:videoId", next);
                } catch {}
                await startCamera(next);
              }}
            >
              <option value="">(Default)</option>
              {videoDevs.map((d) => (
                <option key={d.deviceId} value={d.deviceId}>
                  {d.label || "Camera"}
                </option>
              ))}
            </select>
            <div
              className="row"
              style={{ gap: 12, marginTop: 12, flexWrap: "wrap" }}
            >
              <button
                className="btn"
                onClick={() => {
                  startCamera().catch((err) =>
                    console.warn("[Cam] restart failed", err)
                  );
                }}
              >
                Restart camera
              </button>
              <button
                className="btn"
                onClick={() => {
                  stopAll({ reset: true }).catch((err) =>
                    console.warn("[Cam] stop failed", err)
                  );
                }}
              >
                Stop camera
              </button>
            </div>
            <div className="help" style={{ marginTop: 6 }}>
              Use restart after plugging in a new webcam or if autoplay was
              blocked by the browser.
            </div>
          </div>

          {/* Row 2: DISTANCE CONTROLS - two neat panels */}
          <div className="panel">
            {/* GREEN ZONE (interaction range) */}
            <div className="inline-controls">
              <b>Green zone distance</b>
              <input
                className="range"
                type="range"
                min="0.3"
                max="2.0"
                step="0.05"
                value={greenMaxM}
                onChange={(e) => setGreenMaxM(Number(e.target.value))}
                aria-label="Green zone distance in meters"
              />
              <span className="chip">{greenMaxM.toFixed(2)} m</span>
              <button
                className="btn"
                onClick={() => setGreenMaxM(DEFAULT_GREEN_MAX_M)}
              >
                reset
              </button>
              <button
                className="btn"
                onClick={() =>
                  setGreenMaxM((v) => Math.max(0.3, +(v - 0.1).toFixed(2)))
                }
              >
                -0.1
              </button>
              <button
                className="btn"
                onClick={() =>
                  setGreenMaxM((v) => Math.min(2.0, +(v + 0.1).toFixed(2)))
                }
              >
                +0.1
              </button>
            </div>

            {/* RED CUTOFF (ignore beyond) */}
            <div className="inline-controls" style={{ marginTop: 8 }}>
              <b>Red zone distance</b>
              <input
                className="range"
                type="range"
                min="1.0"
                max="4.0"
                step="0.1"
                value={redCutoffM}
                onChange={(e) => setRedCutoffM(Number(e.target.value))}
                aria-label="Red zone cutoff in meters"
              />
              <span className="chip">{redCutoffM.toFixed(1)} m</span>
              <button
                className="btn"
                onClick={() => setRedCutoffM(DEFAULT_RED_CUTOFF_M)}
              >
                reset
              </button>
              <button
                className="btn"
                onClick={() =>
                  setRedCutoffM((v) => Math.max(1.0, +(v - 0.1).toFixed(1)))
                }
              >
                -0.1
              </button>
              <button
                className="btn"
                onClick={() =>
                  setRedCutoffM((v) => Math.min(4.0, +(v + 0.1).toFixed(1)))
                }
              >
                +0.1
              </button>
            </div>
          </div>

          {captions && lastText && (
            <div
              aria-live="polite"
              className="captions"
              style={{
                marginTop: 8,
                background: "rgba(0,0,0,0.55)",
                padding: "10px 12px",
                borderRadius: 10,
                lineHeight: 1.35,
              }}
            >
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

          {/* Camera settings - now directly under the camera */}
          <div className="panel" style={{ marginTop: 10 }}>
            <h3 className="section-title" style={{ marginTop: 0 }}>
              camera alignment
            </h3>

            <label className="label">Horizontal FOV ( deg)</label>
            <input
              className="range"
              type="range"
              min="40"
              max="110"
              step="1"
              value={fovHdeg}
              onChange={(e) => setFovHdeg(Number(e.target.value))}
            />
            <div className="help">{Math.round(fovHdeg)} deg</div>

            <label className="label">Vertical FOV ( deg)</label>
            <input
              className="range"
              type="range"
              min="25"
              max="90"
              step="1"
              value={fovVdeg}
              onChange={(e) => setFovVdeg(Number(e.target.value))}
            />
            <div className="help">{Math.round(fovVdeg)} deg</div>

            <div className="row" style={{ gap: 16 }}>
              <div className="flex1">
                <label className="label">Pan offset ( deg)</label>
                <input
                  className="range"
                  type="range"
                  min="-30"
                  max="30"
                  step="0.5"
                  value={panOffsetDeg}
                  onChange={(e) => setPanOffsetDeg(Number(e.target.value))}
                />
                <div className="help">{panOffsetDeg.toFixed(1)} deg</div>
              </div>
              <div className="flex1">
                <label className="label">Tilt offset ( deg)</label>
                <input
                  className="range"
                  type="range"
                  min="-30"
                  max="30"
                  step="0.5"
                  value={tiltOffsetDeg}
                  onChange={(e) => setTiltOffsetDeg(Number(e.target.value))}
                />
                <div className="help">{tiltOffsetDeg.toFixed(1)} deg</div>
              </div>
            </div>

            <div className="row" style={{ gap: 8, marginTop: 8 }}>
              <button
                className="btn"
                onClick={() => {
                  setPanOffsetDeg(0);
                  setTiltOffsetDeg(0);
                }}
              >
                reset offsets
              </button>
              <span className="help">
                Tip: click a face on video to auto-zero.
              </span>
            </div>

            <div className="divider" />

            <h4 className="section-title">focus weights</h4>

            <label className="label">Closeness</label>
            <input
              className="range"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={wNear}
              onChange={(e) => setWNear(Number(e.target.value))}
            />

            <label className="label">Centeredness</label>
            <input
              className="range"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={wCenter}
              onChange={(e) => setWCenter(Number(e.target.value))}
            />

            <label className="label">Mouth activity</label>
            <input
              className="range"
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={wMouth}
              onChange={(e) => setWMouth(Number(e.target.value))}
            />
          </div>

          {/* Row 4: GUEST TABLE */}
          <div className="panel tablewrap" style={{ padding: 12 }}>
            <PeopleTable table={table} />
          </div>
        </div>
      </div>
    </main>
  );
}

















