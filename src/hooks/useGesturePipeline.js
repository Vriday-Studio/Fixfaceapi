import { useCallback, useEffect, useRef } from "react";
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
} from "../utils/gestureUtils";

export function useGesturePipeline({
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
  handsCacheMs,
  handsSendMs,
}) {
  const stableGestureRef = useRef(null);
  const perFaceGestureWinRef = useRef(new Map());
  const perFaceStableRef = useRef(new Map());
  const lastGestureSentPerFaceRef = useRef(new Map());
  const waveHistByFaceRef = useRef(new Map());

  useEffect(() => {
    gesturesOnRef.current = gesturesOn;
    try {
      localStorage.setItem("ika:gesturesOn", String(gesturesOn));
    } catch {}
    if (!gesturesOn) {
      perFaceGestureWinRef.current = new Map();
      perFaceStableRef.current = new Map();
      lastGestureSentPerFaceRef.current = new Map();
      stableGestureRef.current = null;
    }
  }, [gesturesOn, gesturesOnRef]);

  const pruneGestureState = useCallback((keepKeys) => {
    for (const k of Array.from(perFaceStableRef.current.keys())) {
      if (!keepKeys.has(k)) perFaceStableRef.current.delete(k);
    }
    for (const k of Array.from(waveHistByFaceRef.current.keys())) {
      if (!keepKeys.has(k)) waveHistByFaceRef.current.delete(k);
    }
  }, []);

  const updateGlobalStableGesture = useCallback(
    (now) => {
      const faces = trackedFacesRef.current || [];
      const eligible = new Set(
        faces.filter((f) => f.gestureEligible).map((f) => f.key)
      );
      const fi = focusIndexRef.current;
      let chosen = null;
      if (fi >= 0 && faces[fi] && eligible.has(faces[fi].key)) {
        chosen = perFaceStableRef.current.get(faces[fi].key) || null;
      }
      if (!chosen) {
        for (const [k, g] of perFaceStableRef.current.entries()) {
          if (!eligible.has(k)) continue;
          if (now - g.t <= handsCacheMs && g.type === "on_phone") {
            chosen = g;
            break;
          }
        }
      }
      if (!chosen) {
        for (const [k, g] of perFaceStableRef.current.entries()) {
          if (!eligible.has(k)) continue;
          if (now - g.t <= handsCacheMs) {
            chosen = g;
            break;
          }
        }
      }
      stableGestureRef.current = chosen ? { ...chosen, t: now } : null;
    },
    [focusIndexRef, handsCacheMs, trackedFacesRef]
  );

  const getFreshGesture = useCallback(
    (now) => {
      const g = gesturesOnRef.current ? stableGestureRef.current : null;
      return g && now - g.t <= handsCacheMs
        ? { type: g.type, score: g.score }
        : null;
    },
    [gesturesOnRef, handsCacheMs]
  );

  const processHandsFrame = useCallback(
    ({ handsList, now, ctx, canvas }) => {
      if (handsList && handsList.length) {
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

        const handAnchor = (lm) => {
          const w = lm[MP.WRIST];
          const i = lm[MP.INDEX_MCP];
          if (!w || !i) return null;
          return { x: (w.x + i.x) * 0.5, y: (w.y + i.y) * 0.5 };
        };

        const wristInFace = (px, py, f) => {
          const left = f.cx - f.w * 0.5;
          const top = f.cy - f.h * 0.45;
          const right = left + f.w;
          const bottom = top + f.h;
          const mx = f.w * 0.09;
          const myUp = f.h * 0.35;
          const myDown = f.h * 0.22;
          return (
            px >= left - mx &&
            px <= right + mx &&
            py >= top - myUp &&
            py <= bottom + myDown
          );
        };

        const facesAll = trackedFacesRef.current || [];
        const faces = facesAll.filter((f) => f.gestureEligible);
        if (faces.length) {
          const hands = handsList
            .map((lm, hi) => {
              const a = handAnchor(lm);
              if (!a) return null;
              const ax = a.x * canvas.width;
              const ay = a.y * canvas.height;
              return { lm, hi, ax, ay };
            })
            .filter((h) => {
              if (!(h && Number.isFinite(h.ax) && Number.isFinite(h.ay))) {
                return false;
              }
              const span = palmSpanLen(h.lm);
              return span >= 0.02;
            });

          const allPairs = [];
          for (const h of hands) {
            let contenders = faces.filter((f) => wristInFace(h.ax, h.ay, f));

            if (!contenders.length) {
              let best = null;
              let bestDx = Infinity;
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

            if (!contenders.length) {
              if (faces.length === 1) {
                contenders = [faces[0]];
              } else if (faces.length === 2) {
                const [leftF, rightF] =
                  faces[0].cx <= faces[1].cx
                    ? [faces[0], faces[1]]
                    : [faces[1], faces[0]];
                const midX = (leftF.cx + rightF.cx) * 0.5;
                contenders = [h.ax <= midX ? leftF : rightF];
              } else {
                let bestN = null;
                let bestScore = Infinity;
                for (const f of faces) {
                  const dx = h.ax - f.cx;
                  const dy = h.ay - f.cy;
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
              const dx = h.ax - f.cx;
              const dy = h.ay - f.cy;
              allPairs.push({
                hi: h.hi,
                lm: h.lm,
                face: f,
                d2: dx * dx + dy * dy,
              });
            }
          }

          const byHand = new Map();
          for (const p of allPairs) {
            const arr = byHand.get(p.hi) || [];
            arr.push(p);
            byHand.set(p.hi, arr);
          }
          const filtered = [];
          for (const arr of byHand.values()) {
            arr.sort((a, b) => a.d2 - b.d2);
            const best = arr[0];
            const second = arr[1];
            const wRef = second
              ? Math.max(best.face.w || 1, second.face.w || 1)
              : 1;
            if (second) {
              const nearTie =
                Math.abs(best.d2 - second.d2) <= wRef * 0.15 * (wRef * 0.15);
              if (nearTie) {
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
              const ax0 = a0.x * canvas.width;
              const ay0 = a0.y * canvas.height;
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
            if (!allowWave) {
              const wa = waveActivity();
              if (wa.flips >= 2 && wa.amp > 0.02) allowWave = true;
            }

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
            const wrist = lm[MP.WRIST];
            const iMcp = lm[MP.INDEX_MCP];
            const vx = (iMcp?.x ?? 0) - (wrist?.x ?? 0);
            const vy = (iMcp?.y ?? 0) - (wrist?.y ?? 0);
            const axisLen = Math.hypot(vx, vy) || 1e-6;
            const cosToVertical = Math.abs(vy) / axisLen;

            let allowNearFace = false;
            let highPalm = false;
            if (a0) {
              const ax0 = a0.x * canvas.width;
              const ay0 = a0.y * canvas.height;
              highPalm = ay0 <= face.cy - face.h * 0.05;
              const highEnough = ay0 <= face.cy + face.h * 0.25;
              allowNearFace =
                Math.abs(ax0 - face.cx) <= face.w * 0.85 && highEnough;
            }

            const cand = [];
            try {
              const w = classifyWave(lm, now);
              if (w.ok && (allowWave || w.score >= 0.62)) {
                cand.push({ type: "wave", score: w.score });
              }
            } catch {}

            const palm = palmSpanLen(lm);
            const gm = !!gameModeRef.current;

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
                if (rh.ok) cand.push({ type: "raise_hand", score: rh.score });
              } catch {}
              try {
                const ph = classifyOnPhone(
                  lm,
                  [{ cx: face.cx, cy: face.cy, w: face.w, h: face.h }],
                  canvas.width,
                  canvas.height
                );
                if (ph.ok) cand.push({ type: "on_phone", score: ph.score });
              } catch {}
              try {
                const t = classifyThumbsUp(lm);
                if (t.ok) cand.push({ type: "thumbs_up", score: t.score });
              } catch {}
            }

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
            if (adj && (!prev || adj.score > prev.score)) {
              byFace.set(face.key, adj);
            }
          }

          for (const [key, frame] of byFace.entries()) {
            const win = perFaceGestureWinRef.current.get(key) || [];
            win.push({ ...frame, t: now });
            if (win.length > VOTE_WINDOW * 2) {
              win.splice(0, win.length - VOTE_WINDOW * 2);
            }
            perFaceGestureWinRef.current.set(key, win);

            const prevStable = perFaceStableRef.current.get(key) || null;
            const nextStable = pickStableGesture(now, win, prevStable);
            if (nextStable) {
              const changed =
                !prevStable || prevStable.type !== nextStable.type;
              perFaceStableRef.current.set(key, nextStable);
              const lastSent =
                lastGestureSentPerFaceRef.current.get(key) || 0;
              if (
                changed &&
                now - lastSent >= handsSendMs &&
                !speakingRef.current
              ) {
                const facesMeta = trackedFacesRef.current || [];
                const meta = facesMeta.find((f) => f.key === key) || {};
                try {
                  socketRef.current?.emit?.(
                    gameModeRef.current ? "game_event" : "gesture_event",
                    gameModeRef.current
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

          updateGlobalStableGesture(now);
        }
      }

      return getFreshGesture(now);
    },
    [
      deviceId,
      focusIndexRef,
      gameModeRef,
      getFreshGesture,
      handsCacheMs,
      handsSendMs,
      sessionId,
      socketRef,
      speakingRef,
      trackedFacesRef,
      updateGlobalStableGesture,
    ]
  );

  return {
    stableGestureRef,
    perFaceStableRef,
    waveHistByFaceRef,
    processHandsFrame,
    getFreshGesture,
    pruneGestureState,
  };
}
