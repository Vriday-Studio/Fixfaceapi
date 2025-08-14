import { useEffect, useRef } from "react";
import * as faceapi from "face-api.js";
import "./App.css";

/* ---------- CONFIG ---------- */
const FACE_WIDTH_M = 0.15;              // avg face width (meters)
const FOCAL_PX     = 500;               // tune if distance looks off
const GREEN_M      = 0.8;               // <= 0.8 m = GREEN (greet), > 0.8 m = RED (call over)
const SCORE_TH     = 0.3;               // detector sensitivity (lower = more sensitive)
const INPUT_STEPS  = [320, 416, 512, 608]; // adaptive sizes to catch small/far faces
const MATCH_THRESHOLD_PX = 60;          // tracking radius
const TRACK_TIMEOUT_MS   = 1500;        // drop after unseen (new ID on re-entry)
// const N8N_WEBHOOK_URL = "https://your-n8n/webhook-id"; // optional

/* ---------- helpers ---------- */
const center = (b) => ({ cx: b.x + b.width / 2, cy: b.y + b.height / 2 });
const euclid  = (a, b) => Math.hypot(a.cx - b.cx, a.cy - b.cy);
const topExpression = (exp = {}) =>
  Object.entries(exp)
    .map(([expression, probability]) => ({ expression, probability }))
    .sort((a, b) => b.probability - a.probability)[0] || { expression: "neutral", probability: 0 };

/* ---------- DPR-aware canvas prep ---------- */
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

export default function App() {
  const videoRef  = useRef(null);
  const canvasRef = useRef(null);
  const rafRef    = useRef(null);

  // stable tracking state
  const tracksRef = useRef([]); // [{id, detection, age, gender, expressions, cx, cy, lastSeen}]
  const nextIdRef = useRef(1);
  const cpuSetRef = useRef(false);
  const lastGoodInputRef = useRef(INPUT_STEPS[0]); // remember last size that saw faces

  useEffect(() => {
    (async () => {
      // 1) start camera
      const s = await navigator.mediaDevices.getUserMedia({ video: true });
      const v = videoRef.current;
      v.srcObject = s;
      await new Promise(res => { if (v.readyState >= 2) res(); else v.onloadedmetadata = () => res(); });
      await v.play().catch(()=>{});
      // 2) load models
      await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
        faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
        faceapi.nets.faceExpressionNet.loadFromUri("/models"),
        faceapi.nets.ageGenderNet.loadFromUri("/models"),
      ]);
      // 3) CPU backend (CSP-safe)
      if (!cpuSetRef.current) {
        await faceapi.tf.setBackend("cpu");
        await faceapi.tf.ready();
        cpuSetRef.current = true;
      }
      // 4) start loop
      startLoop();
    })();

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      const s = videoRef.current?.srcObject;
      if (s) s.getTracks().forEach(t => t.stop());
    };
  }, []);

  function startLoop() {
    const tick = async () => {
      const detections = await detectAdaptive();
      const people = updateTracks(detections);
      drawOverlayAndTable(people);
      // await sendToN8N(people); // uncomment + set URL if you want to send
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
  }

  async function detectOnce(inputSize) {
    const v = videoRef.current;
    if (!v) return [];
    const opts = new faceapi.TinyFaceDetectorOptions({
      inputSize,
      scoreThreshold: SCORE_TH,
    });
    const det = await faceapi
      .detectAllFaces(v, opts)
      .withFaceLandmarks()
      .withFaceExpressions()
      .withAgeAndGender();

    const displaySize = prepCanvasForDisplay(v, canvasRef.current);
    return faceapi.resizeResults(det, displaySize);
  }

  // try lastGoodInput first, then fall back to larger sizes to catch far/small faces
  async function detectAdaptive() {
    const sizes = [lastGoodInputRef.current, ...INPUT_STEPS.filter(s => s !== lastGoodInputRef.current)];
    for (const sz of sizes) {
      const result = await detectOnce(sz);
      if (result.length > 0) {
        lastGoodInputRef.current = sz;
        return result;
      }
    }
    // none found; keep lastGoodInput as-is
    return [];
  }

  function updateTracks(resized) {
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
        });
      }
    });

    // drop stale tracks
    const kept = next.filter(p => now - p.lastSeen < TRACK_TIMEOUT_MS);
    kept.sort((a, b) => a.id - b.id);
    tracksRef.current = kept;
    return kept;
  }

  function drawOverlayAndTable(people) {
    const v  = videoRef.current;
    const displayW = v?.clientWidth  || 640;
    const displayH = v?.clientHeight || 480;

    const c   = canvasRef.current;
    const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, displayW, displayH);

    // boxes + labels
    people.forEach(p => {
      const b = p.detection.box;
      const distanceM = (FOCAL_PX * FACE_WIDTH_M) / b.width;
      const zone = distanceM <= GREEN_M ? "green" : "red";
      const expr = topExpression(p.expressions);

      const drawBox = new faceapi.draw.DrawBox(b, {
        label: `P${p.id} • ${zone} • ${Math.round(p.age)} ${p.gender} • ${expr.expression}`,
        boxColor: zone === "green" ? "#17c964" : "#f31260",
        lineWidth: 2,
      });
      drawBox.draw(c);
    });

    // count
    const countEl = document.getElementById("peopleCount");
    if (countEl) countEl.innerText = `People on screen: ${people.length}`;

    // table
    const body = document.getElementById("dataBodyindex");
    if (body) body.innerHTML = "";
    people.forEach(p => {
      const b = p.detection.box;
      const distanceM = (FOCAL_PX * FACE_WIDTH_M) / b.width;
      const zone = distanceM <= GREEN_M ? "green" : "red";
      const expr = topExpression(p.expressions);

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="border:1px solid #fff;padding:6px;text-align:center;">P${p.id}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${p.gender}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${Math.round(p.age)}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${expr.expression} (${expr.probability.toFixed(2)})</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${distanceM.toFixed(2)}</td>
        <td style="border:1px solid #fff;padding:6px;text-align:center;">${zone}</td>
      `;
      body.appendChild(tr);
    });
  }

  // OPTIONAL: send to n8n (uncomment N8N_WEBHOOK_URL above to enable)
  // async function sendToN8N(people) {
  //   if (!N8N_WEBHOOK_URL) return;
  //   const payload = {
  //     timestamp: new Date().toISOString(),
  //     peopleCount: people.length,
  //     anyInGreen: people.some(p => (FOCAL_PX * FACE_WIDTH_M) / p.detection.box.width <= GREEN_M),
  //     people: people.map(p => {
  //       const b = p.detection.box;
  //       const distanceM = (FOCAL_PX * FACE_WIDTH_M) / b.width;
  //       const expr = topExpression(p.expressions);
  //       return {
  //         id: p.id,
  //         age: Math.round(p.age),
  //         gender: p.gender,
  //         emotion: expr.expression,
  //         emotionProb: Number(expr.probability.toFixed(3)),
  //         distanceM: Number(distanceM.toFixed(2)),
  //         zone: distanceM <= GREEN_M ? "green" : "red",
  //         box: { x: Math.round(b.x), y: Math.round(b.y), w: Math.round(b.width), h: Math.round(b.height) }
  //       };
  //     })
  //   };
  //   try {
  //     await fetch(N8N_WEBHOOK_URL, {
  //       method: "POST",
  //       headers: { "Content-Type": "application/json" },
  //       body: JSON.stringify(payload)
  //     });
  //   } catch (e) {
  //     console.warn("n8n webhook failed:", e);
  //   }
  // }

  return (
    <div style={{ padding: 12 }}>
      <h3>Face Tracker (near/far)</h3>

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
              <th style={{ border: "1px solid #fff", padding: 6 }}>Person</th>
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