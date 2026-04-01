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
  getStoredNumber,
  encodeDescFloat32ToU8,
  decodeDescU8ToFloat32,
  u8ToB64,
  b64ToU8,
} from "./utils/storageUtils";
import StatusOverviewPanel from "./components/StatusOverviewPanel";
import CameraStagePanel from "./components/CameraStagePanel";
import ControlSidebarPanel from "./components/ControlSidebarPanel";
import CameraControlsPanel from "./components/CameraControlsPanel";
import GuestTablePanel from "./components/GuestTablePanel";
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
import { useCamera } from "./hooks/useCamera";
import { useControlActions } from "./hooks/useControlActions";
import { useSessionPolicy } from "./hooks/useSessionPolicy";
import { useServerBridge } from "./hooks/useServerBridge";
import { useTelemetryEmitters } from "./hooks/useTelemetryEmitters";
import { useFrameOutput } from "./hooks/useFrameOutput";
import { usePresencePolicy } from "./hooks/usePresencePolicy";
import { useGesturePipeline } from "./hooks/useGesturePipeline";
import { useFaceCandidates } from "./hooks/useFaceCandidates";
import { useFocusSelection } from "./hooks/useFocusSelection";

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
const STABILIZE_FRAMES = 5;
// Require ~5 stable active frames in green before firing greet
const GREEN_STABLE_MS = 240;

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
const ATTEND_MIN_FRAMES = 5; // require 3-5 consecutive frames
const DEBUG_FACE_LOGS = false;
const DEBUG_ZONE_LOGS = false;
const DEBUG_PERF_LOGS = false;
const WEATHER_POLL_MS = 10 * 60 * 1000;
const GIK_COORDS = { latitude: -6.1944, longitude: 106.8219 };

function describeWeatherCode(code) {
  switch (code) {
    case 0:
      return "Clear";
    case 1:
    case 2:
      return "Partly cloudy";
    case 3:
      return "Overcast";
    case 45:
    case 48:
      return "Fog";
    case 51:
    case 53:
    case 55:
    case 56:
    case 57:
      return "Drizzle";
    case 61:
    case 63:
    case 65:
    case 66:
    case 67:
    case 80:
    case 81:
    case 82:
      return "Rain";
    case 71:
    case 73:
    case 75:
    case 77:
    case 85:
    case 86:
      return "Snow";
    case 95:
    case 96:
    case 99:
      return "Thunderstorm";
    default:
      return "Weather";
  }
}

function buildWeatherLabel(current) {
  const temp = Number(current?.temperature_2m);
  const precipitation = Number(current?.precipitation ?? 0);
  const rain = Number(current?.rain ?? 0);
  const showers = Number(current?.showers ?? 0);
  const rawCode = Number(current?.weather_code);
  const hasPrecipitation =
    Number.isFinite(precipitation) && precipitation > 0.1 ||
    Number.isFinite(rain) && rain > 0 ||
    Number.isFinite(showers) && showers > 0;
  const condition = hasPrecipitation
    ? "Rain"
    : describeWeatherCode(Number.isFinite(rawCode) ? rawCode : -1);
  const roundedTemp = Number.isFinite(temp) ? Math.round(temp) : null;
  return roundedTemp == null ? condition : `${condition} ${roundedTemp} degC`;
}

const logFace = (...args) => {
  if (DEBUG_FACE_LOGS) console.log(...args);
};
const logZone = (...args) => {
  if (DEBUG_ZONE_LOGS) console.log(...args);
};
const logPerf = (...args) => {
  if (DEBUG_PERF_LOGS) console.log(...args);
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
const HANDS_NO_TARGET_MS = 350;
const HANDS_CACHE_MS = 1000;
const HANDS_SEND_MS = 600;

// Game mode cadence (snappier)
const GM_HANDS_FAST_MS = 40;
const GM_HANDS_IDLE_MS = 120;
const GM_HANDS_NO_TARGET_MS = 220;

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

  const { server, onCreateSession, onHotUpdate } = useServerBridge({
    socketRef,
    sendWsCommand,
    deviceId,
    sessionId,
    bump,
    setSessionStatus,
    setSessionId,
    modelQuick,
    geminiVoiceQuick,
    languageCodeQuick,
    systemInstruction,
    ttsProviderQuick,
    enableAffectiveQuick,
    temperatureQuick,
    proactiveAudioQuick,
    captions,
    functionCallingQuick,
    autoFunctionResponseQuick,
    groundingQuick,
  });

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

  useEffect(() => {
    let isDisposed = false;

    const updateWeather = async () => {
      try {
        const params = new URLSearchParams({
          latitude: String(GIK_COORDS.latitude),
          longitude: String(GIK_COORDS.longitude),
          current: [
            "temperature_2m",
            "weather_code",
            "precipitation",
            "rain",
            "showers",
          ].join(","),
          timezone: "Asia/Jakarta",
          forecast_days: "1",
        });
        const res = await fetch(
          `https://api.open-meteo.com/v1/forecast?${params.toString()}`,
          { cache: "no-store" }
        );
        if (!res.ok) throw new Error(`weather http ${res.status}`);
        const data = await res.json();
        const nextLabel = buildWeatherLabel(data?.current);
        if (!isDisposed && nextLabel) setWeatherLabel(nextLabel);
      } catch (err) {
        console.warn("weather update failed", err);
      }
    };

    updateWeather();
    const timer = setInterval(updateWeather, WEATHER_POLL_MS);
    return () => {
      isDisposed = true;
      clearInterval(timer);
    };
  }, []);
  const totalsRef = useRef({ all: 0, green: 0, red: 0 });

  const {
    buildVisitContext,
    sendPeopleIntent,
    emitCrowdThrottled,
    emitCrowdByGid,
    sendGreenSnapshot,
  } = useTelemetryEmitters({
    clockRef,
    locationRef,
    weatherRef,
    totalsRef,
    sendWsCommand,
    wsIsConnected,
  });

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
  const lastGlobalCallOverTsRef = useRef(0);
  const greetInviteRef = useRef(new Map());
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

  const preloadFaceMatcher = useCallback(async () => {
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
    const count = usable.reduce((acc, l) => acc + l.descriptors.length, 0);

    return { matcher, count };
  }, []);

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

  const {
    exportSettings,
    importSettings,
    resetSettings,
    applyDeviceId,
    randomizeDeviceId,
    handleClearGuests,
    handleVideoChange,
    handleRestartCamera,
    handleStopCamera,
  } = useControlActions({
    deviceIdDraft,
    deviceId,
    setDeviceId,
    guestSeqRef,
    guestMemRef,
    saveGuestMemSafe,
    setVideoId,
    startCamera,
    stopAll,
  });

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

        const [_, preloadedMatcher] = await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
          faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
          preloadFaceMatcher().catch((err) => {
            console.warn("[labels] preload failed:", err);
            return null;
          }),
        ]);

        if (!cancelled && preloadedMatcher) {
          faceMatcherRef.current = preloadedMatcher.matcher;
          setKnownCount(preloadedMatcher.count);
          console.log("[labels] preloaded matcher:", preloadedMatcher.count);
        }

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
  }, [preloadFaceMatcher]);

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
    if (faceMatcherRef.current) return;
    let cancelled = false;
    (async () => {
      try {
        const { matcher, count } = await preloadFaceMatcher();

        if (!cancelled) {
          faceMatcherRef.current = matcher;
          setKnownCount(count);
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
  }, [preloadFaceMatcher, ready]);

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
      const gm =
        localStorage.getItem(`ika:greenMaxM:${videoId || "default"}`) ??
        localStorage.getItem("ika:greenMaxM");
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
      const perf = { frameStart: now, detectMs: 0, buildMs: 0, drawMs: 0, handsMs: 0, policyMs: 0 };
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
        const detectStart = performance.now();
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
        perf.detectMs = performance.now() - detectStart;

        logFace(`[DEBUG Face Detection] Detected ${dets.length} faces, wsConnected=${wsIsConnected.current}`);

        //  FIX: Send PeopleData with zone="none" when no faces detected
        if (dets.length === 0 && wsIsConnected.current) {
          handleNoFaceDetected();
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

        const buildStart = performance.now();
        const {
          rows,
          peopleForPost,
          gestureAllowedKeys,
          tracked,
          candidates,
          total,
          green,
          red,
          guestSnapshots: frameGuestSnapshots,
        } = buildFaceFrameData({
          resized,
          canvas,
          ctx,
          now,
          redCutoffM,
          showAlign: showAlignRef.current,
        });
        frameCandidates = candidates;
        guestSnapshots = frameGuestSnapshots;
        perf.buildMs = performance.now() - buildStart;

        // === RED ZONE CALL-OVER DETECTION (NONE -> RED) ===
        // This runs every frame to detect NONE->RED transitions with stabilization
        const drawStart = performance.now();
        processRedZoneState({ total, green, red });

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

        perf.drawMs = performance.now() - drawStart;
        // prune only faces that are no longer tracked (preserve history for tracked-but-not-eligible)
        {
          const keep = new Set(
            (trackedFacesRef.current || []).map((f) => f.key)
          );
          for (const k of Array.from(mouthMapRef.current.keys())) {
            if (!keep.has(k)) mouthMapRef.current.delete(k);
          }
          pruneGestureState(keep);
        }

        // prune unused track slots (use stable keys: name or gid)
        {
          const seen = new Set();
          for (const p of peopleForPost) {
            seen.add((p.name || p.gid) ?? "");
          }
          for (const k of Object.keys(recentMapRef.current)) {
            if (k && !seen.has(k)) delete recentMapRef.current[k];
          }
        }

        const { cand, focusIndex } = selectFocus(peopleForPost);

        // Save focus so other blocks (hands/policy) can include it
        updateFocusTarget({ focusIndex, cand, peopleForPost });
        const fresh = getFreshGesture(now);

        commitFrameOutput({
          cand,
          peopleForPost,
          rows,
          total,
          green,
          red,
          fresh,
        });

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
        const hasGestureTargets = trackedFacesRef.current.some(
          (f) => f.gestureEligible
        );
        const handsDesiredStep =
          !hasGestureTargets
            ? gm
              ? GM_HANDS_NO_TARGET_MS
              : HANDS_NO_TARGET_MS
            : now - (lastLmSeenTsRef.current || 0) <= 800
            ? gm
              ? GM_HANDS_FAST_MS
              : HANDS_FAST_MS
            : gm
            ? GM_HANDS_IDLE_MS
            : HANDS_IDLE_MS;

        //test apakah tangan terdeteksi, kmd kirim data ke server, hanya untuk gesture
        if (handsEligible && now - (lastHandsRunTsRef.current || 0) >= handsDesiredStep) {
          lastHandsRunTsRef.current = now;
          const handsStart = performance.now();
          try {
            const handsList = await detectHandsOnce(video);
            const fresh = processHandsFrame({
              handsList,
              now,
              ctx,
              canvas,
            });

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
              peopleSource: peopleForPost,
            });
          } catch (e) {
            // ignore hand pipeline hiccups so the frame loop keeps running
          }
          perf.handsMs = performance.now() - handsStart;
        }
        // handled above; removed extraneous catch
        // ---- Policy: zone transitions -> call-over / greet (candidates include red) ----
        try {
          const policyStart = performance.now();
          const matcher = faceMatcherRef.current;
          const allIdentities = frameCandidates.map((c) => {
            const det = c.det;
            let name = null;
            if (matcher && det.descriptor) {
              const best = matcher.findBestMatch(det.descriptor);
              if (best && best.label !== "unknown" && best.distance <= MATCH_THRESHOLD) {
                name = best.label;
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
          dispatchZoneTransitions({
            allIdentities,
            now,
            groupInfo,
            guestSnapshots,
          });
          perf.policyMs = performance.now() - policyStart;
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
        if (DEBUG_PERF_LOGS && frameCount % 15 === 0) {
          const totalMs = performance.now() - perf.frameStart;
          logPerf(`[perf] frame=${frameCount} total=${totalMs.toFixed(1)}ms detect=${perf.detectMs.toFixed(1)} build=${perf.buildMs.toFixed(1)} draw=${perf.drawMs.toFixed(1)} hands=${perf.handsMs.toFixed(1)} policy=${perf.policyMs.toFixed(1)} faces=${dets.length}`);
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

  const { handleStartSession, handleStopSession } = useSessionPolicy({
    autoSession,
    serverConnected: !!serverInfo.connected,
    ueConnected: !!ueConnected,
    sessionStatus,
    sessionId,
    deviceId,
    onCreateSession,
    sendWsCommand,
    setSessionStatus,
    setSessionId,
  });
  const { commitFrameOutput, updateFocusTarget } = useFrameOutput({
    focusIndexRef,
    focusTargetRef,
    sendGreenSnapshot,
    emitCrowdByGid,
    setTable,
    setTotals,
    gesturesOnRef,
    deviceId,
    sessionId,
    backend,
  });
  const {
    stableGestureRef,
    perFaceStableRef,
    processHandsFrame,
    getFreshGesture,
    pruneGestureState,
  } = useGesturePipeline({
    gesturesOn,
    gesturesOnRef,
    socketRef,
    deviceId,
    sessionId,
    speakingRef,
    focusIndexRef,
    focusTargetRef,
    trackedFacesRef,
    gameModeRef,
    handsCacheMs: HANDS_CACHE_MS,
    handsSendMs: HANDS_SEND_MS,
  });
  const {
    handleNoFaceDetected,
    processRedZoneState,
    dispatchZoneTransitions,
  } = usePresencePolicy({
    wsIsConnected,
    sendWsCommand,
    buildVisitContext,
    logFace,
    logZone,
    prevZoneMapRef,
    greenEntryRef,
    callOverStateRef,
    lastGlobalCallOverTsRef,
    pendingGreetsRef,
    redZoneCounterRef,
    noneZoneCounterRef,
    redZoneTriggeredRef,
    stableGestureRef,
    pushedGreets,
    ageGenderCacheRef,
    greetInviteRef,
    lastGroupSetRef,
    lastGroupAskTsRef,
    sendPeopleIntent,
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
    callOverMaxTries: CALL_OVER_MAX_TRIES,
    callOverCooldownMs: CALL_OVER_COOLDOWN_MS,
    greenStableMs: GREEN_STABLE_MS,
    notSeenResetMs: NOT_SEEN_RESET_MS,
    groupAskCooldownMs: GROUP_ASK_COOLDOWN_MS,
    redZoneNoneResetFrames: RED_ZONE_NONE_RESET_FRAMES,
    redZoneStableFrames: RED_ZONE_STABLE_FRAMES,
  });
  const { buildFaceFrameData } = useFaceCandidates({
    faceMatcherRef,
    matchThreshold: MATCH_THRESHOLD,
    matchMargin: MATCH_MARGIN,
    stabilizeFrames: STABILIZE_FRAMES,
    handsCacheMs: HANDS_CACHE_MS,
    rad: RAD,
    boxLineWidth: BOX_LINE_WIDTH,
    labelPadX: LABEL_PAD_X,
    labelPadY: LABEL_PAD_Y,
    camFxRef,
    camFyRef,
    panOffRef,
    tiltOffRef,
    greenMaxMRef,
    recentMapRef,
    ageGenderCacheRef,
    mouthMapRef,
    gesturesOnRef,
    gestureTargetsRef,
    perFaceStableRef,
    trackedFacesRef,
    allFacesRef,
    estimateDistanceMpx,
    assignGuestIdFor,
    ageGroupOf,
    topExpression,
    anglesFromPixel,
    posFromPixel,
    mouthMAR,
    shrinkBox,
    gestureLabelOf,
    faceapi,
    zoneOf,
  });
  const { selectFocus } = useFocusSelection({
    wNear,
    wCenter,
    wMouth,
  });

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
          <CameraStagePanel
            videoRef={videoRef}
            canvasRef={canvasRef}
            captions={captions}
            lastText={lastText}
          />

          {/* Row 1: CAMERA/STATUS */}
          <div className="left-top2">
            <StatusOverviewPanel
              locationLabel={locationLabel}
              clock={clock}
              weatherLabel={weatherLabel}
              backend={backend}
              ready={ready}
              camLive={isCamLive()}
              serverInfo={serverInfo}
              ueConnected={ueConnected}
              deviceId={deviceId}
              lastSent={lastSent}
              lastHttp={lastHttp}
              totals={totals}
              useSocketBridge={USE_SOCKET_BRIDGE}
              onClearGuests={handleClearGuests}
            />
            <ControlSidebarPanel
              exportSettings={exportSettings}
              importSettings={importSettings}
              resetSettings={resetSettings}
              serverUrlDraft={serverUrlDraft}
              setServerUrlDraft={setServerUrlDraft}
              serverUrl={serverUrl}
              setServerUrl={setServerUrl}
              effectiveUrl={effectiveUrl}
              serverInfo={serverInfo}
              deviceIdDraft={deviceIdDraft}
              setDeviceIdDraft={setDeviceIdDraft}
              deviceId={deviceId}
              applyDeviceId={applyDeviceId}
              randomizeDeviceId={randomizeDeviceId}
              handleStartSession={handleStartSession}
              handleStopSession={handleStopSession}
              sessionStatus={sessionStatus}
              autoSession={autoSession}
              setAutoSession={setAutoSession}
              gesturesOn={gesturesOn}
              setGesturesOn={setGesturesOn}
              gestureTargets={gestureTargets}
              setGestureTargets={setGestureTargets}
              keepBgOn={keepBgOn}
              setKeepBgOn={setKeepBgOn}
            />
          </div>

          <CameraControlsPanel
            videoId={videoId}
            videoDevs={videoDevs}
            onVideoChange={handleVideoChange}
            onRestartCamera={handleRestartCamera}
            onStopCamera={handleStopCamera}
            greenMaxM={greenMaxM}
            setGreenMaxM={setGreenMaxM}
            defaultGreenMaxM={DEFAULT_GREEN_MAX_M}
            redCutoffM={redCutoffM}
            setRedCutoffM={setRedCutoffM}
            defaultRedCutoffM={DEFAULT_RED_CUTOFF_M}
            showAlign={showAlign}
            setShowAlign={setShowAlign}
            gameModeOn={gameModeOn}
            setGameModeOn={setGameModeOn}
            calibDistanceM={calibDistanceM}
            setCalibDistanceM={setCalibDistanceM}
            runCalCountdown={runCalCountdown}
            fovHdeg={fovHdeg}
            setFovHdeg={setFovHdeg}
            fovVdeg={fovVdeg}
            setFovVdeg={setFovVdeg}
            panOffsetDeg={panOffsetDeg}
            setPanOffsetDeg={setPanOffsetDeg}
            tiltOffsetDeg={tiltOffsetDeg}
            setTiltOffsetDeg={setTiltOffsetDeg}
            wNear={wNear}
            setWNear={setWNear}
            wCenter={wCenter}
            setWCenter={setWCenter}
            wMouth={wMouth}
            setWMouth={setWMouth}
          />

          <GuestTablePanel table={table} />
        </div>
      </div>
    </main>
  );
}























