export const MP = {
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
  PINKY_TIP: 20,
};

const waveHistRef = { t: 0, xs: [] };

function v2(x, y) {
  return { x, y };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x || 0, a.y - b.y || 0);
}

export function recentLateralMotion() {
  const xs = waveHistRef.xs || [];
  if (xs.length < 4) return 0;
  let s = 0;
  for (let i = 1; i < xs.length; i++) s += Math.abs(xs[i] - xs[i - 1]);
  return s;
}

export function setWaveHistory(xs, t) {
  waveHistRef.xs = xs || [];
  waveHistRef.t = t || 0;
}

export function waveActivity() {
  const xs = waveHistRef.xs || [];
  if (xs.length < 3) return { flips: 0, amp: 0 };
  let flips = 0;
  for (let i = 2; i < xs.length; i++) {
    const dx1 = xs[i] - xs[i - 1];
    const dx0 = xs[i - 1] - xs[i - 2];
    if (
      Math.sign(dx1) !== Math.sign(dx0) &&
      Math.abs(dx1) > 0.008 &&
      Math.abs(dx0) > 0.008
    ) {
      flips++;
    }
  }
  const amp = Math.max(...xs) - Math.min(...xs);
  return { flips, amp };
}

export function palmSpanLen(lm) {
  try {
    const a = lm[MP.INDEX_MCP],
      b = lm[MP.PINKY_MCP];
    return Math.hypot(a.x - b.x, a.y - b.y) || 1e-3;
  } catch {
    return 1e-3;
  }
}

function fingerClosed(lm, tipIdx, mcpIdx) {
  try {
    const w = lm[MP.WRIST],
      tip = lm[tipIdx],
      mcp = lm[mcpIdx];
    const dTip = Math.hypot(tip.x - w.x, tip.y - w.y);
    const dMcp = Math.hypot(mcp.x - w.x, mcp.y - w.y);
    const span = palmSpanLen(lm);
    const margin = (dTip - dMcp) / Math.max(1e-3, span);
    return { closed: margin < 0.06, margin };
  } catch {
    return { closed: true, margin: -1 };
  }
}

function fingerOpen(lm, tipIdx, mcpIdx) {
  try {
    const w = lm[MP.WRIST],
      tip = lm[tipIdx],
      mcp = lm[mcpIdx];
    const dTip = Math.hypot(tip.x - w.x, tip.y - w.y);
    const dMcp = Math.hypot(mcp.x - w.x, mcp.y - w.y);
    const span = palmSpanLen(lm);
    const margin = (dTip - dMcp) / Math.max(1e-3, span);
    return { open: margin > 0.09, margin };
  } catch {
    return { open: false, margin: -1 };
  }
}

export function classifyWave(landmarks, now) {
  try {
    const wrist = landmarks[MP.WRIST];
    if (!wrist) return { ok: false };
    if (palmSpanLen(landmarks) < 0.02) return { ok: false };
    const x = wrist.x;
    const xs = waveHistRef.xs;

    if (now - (waveHistRef.t || 0) > 900) xs.length = 0;
    waveHistRef.t = now;
    xs.push(x);
    if (xs.length > 14) xs.shift();
    if (xs.length < 4) return { ok: false };

    let flips = 0;
    for (let i = 2; i < xs.length; i++) {
      const dx1 = xs[i] - xs[i - 1];
      const dx0 = xs[i - 1] - xs[i - 2];
      if (
        Math.sign(dx1) !== Math.sign(dx0) &&
        Math.abs(dx1) > 0.008 &&
        Math.abs(dx0) > 0.008
      ) {
        flips++;
      }
    }
    const amp = Math.max(...xs) - Math.min(...xs);
    const vel = recentLateralMotion();

    if (xs.length >= 6 && amp > 0.028 && flips >= 1) {
      return {
        ok: true,
        type: "wave",
        score: Math.min(1, 0.5 + Math.min(0.35, amp * 4.0)),
      };
    }
    if (flips >= 2 && amp > 0.018 && vel > 0.048) {
      return {
        ok: true,
        type: "wave",
        score: Math.min(
          1,
          0.44 + Math.min(0.3, amp * 3.2) + Math.min(0.12, (vel - 0.045) * 3.2)
        ),
      };
    }
    if (flips >= 2 && amp > 0.012) {
      return {
        ok: true,
        type: "wave",
        score: Math.min(1, 0.28 + 0.18 * flips + Math.min(0.35, amp * 3.8)),
      };
    }
    return { ok: false };
  } catch {
    return { ok: false };
  }
}

export function classifyThumbsUp(landmarks) {
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
    if (
      !(
        thumbTip &&
        indexMcp &&
        indexTip &&
        middleTip &&
        middleMcp &&
        ringTip &&
        ringMcp &&
        pinkyTip &&
        pinkyMcp &&
        wrist
      )
    ) {
      return { ok: false };
    }

    const vel = recentLateralMotion();
    if (vel > 0.08) return { ok: false };

    const vIdx = v2(indexMcp.x - wrist.x, indexMcp.y - wrist.y);
    const axisLen = Math.hypot(vIdx.x, vIdx.y) || 1e-6;
    const axisCosToVertical = Math.abs(vIdx.y) / axisLen;

    const thumbAbove = thumbTip.y < indexMcp.y - 0.012;
    const open = dist(thumbTip, indexTip) > 0.03;
    const orientedUp = axisCosToVertical > 0.8;
    const bigEnough = palmSpanLen(landmarks) > 0.03;

    const idxClosed = fingerClosed(
      landmarks,
      MP.INDEX_TIP,
      MP.INDEX_MCP
    ).closed;
    const midClosed = fingerClosed(
      landmarks,
      MP.MIDDLE_TIP,
      MP.MIDDLE_MCP
    ).closed;
    const rngClosed = fingerClosed(landmarks, MP.RING_TIP, MP.RING_MCP).closed;
    const pkyClosed = fingerClosed(
      landmarks,
      MP.PINKY_TIP,
      MP.PINKY_MCP
    ).closed;
    const closedCount = [idxClosed, midClosed, rngClosed, pkyClosed].filter(
      Boolean
    ).length;

    const otherClosed = closedCount >= 2;
    if (thumbAbove && open && bigEnough && orientedUp && otherClosed) {
      const openness = Math.max(
        0,
        Math.min(1, (dist(thumbTip, indexTip) - 0.03) / 0.12)
      );
      const sFast = Math.min(1, 0.78 + 0.22 * openness);
      return { ok: true, type: "thumbs_up", score: sFast };
    }

    if (thumbAbove && open && bigEnough && orientedUp && closedCount >= 2) {
      const openness = Math.max(
        0,
        Math.min(1, (dist(thumbTip, indexTip) - 0.03) / 0.12)
      );
      const orientBoost = Math.min(
        0.3,
        Math.max(0, (axisCosToVertical - 0.82) * 1.6)
      );
      const stillBoost = Math.min(0.2, Math.max(0, (0.06 - vel) * 3.0));
      const s = Math.max(
        0,
        Math.min(1, 0.7 * openness + orientBoost + stillBoost)
      );
      return { ok: true, type: "thumbs_up", score: s };
    }
    return { ok: false };
  } catch {
    return { ok: false };
  }
}

export function classifyPeace(lm) {
  try {
    const idx = fingerOpen(lm, MP.INDEX_TIP, MP.INDEX_MCP);
    const mid = fingerOpen(lm, MP.MIDDLE_TIP, MP.MIDDLE_MCP);
    const rng = fingerOpen(lm, MP.RING_TIP, MP.RING_MCP);
    const pky = fingerOpen(lm, MP.PINKY_TIP, MP.PINKY_MCP);
    if (idx.open && mid.open && !rng.open && !pky.open) {
      const margin = Math.max(0, idx.margin) + Math.max(0, mid.margin);
      const clamp =
        Math.max(0, 0.12 - Math.max(0, rng.margin)) +
        Math.max(0, 0.12 - Math.max(0, pky.margin));
      const score = Math.min(1, 0.5 + 0.35 * margin + 0.25 * clamp);
      return { ok: true, type: "peace", score };
    }
    return { ok: false };
  } catch {
    return { ok: false };
  }
}

export function classifyRaiseHand(lm) {
  try {
    const idx = fingerOpen(lm, MP.INDEX_TIP, MP.INDEX_MCP);
    const mid = fingerOpen(lm, MP.MIDDLE_TIP, MP.MIDDLE_MCP);
    const rng = fingerOpen(lm, MP.RING_TIP, MP.RING_MCP);
    const pky = fingerOpen(lm, MP.PINKY_TIP, MP.PINKY_MCP);
    const opens = [idx, mid, rng, pky].filter((f) => f.open).length;

    const wrist = lm[MP.WRIST],
      iMcp = lm[MP.INDEX_MCP];
    const vx = iMcp.x - wrist.x,
      vy = iMcp.y - wrist.y;
    const vlen = Math.hypot(vx, vy) || 1e-6;
    const cosToVertical = Math.abs(vy) / vlen;

    const tips = [MP.INDEX_TIP, MP.MIDDLE_TIP, MP.RING_TIP, MP.PINKY_TIP].map(
      (i) => lm[i]
    );
    const minY = Math.min(...tips.map((t) => t?.y ?? 1));

    const vel = recentLateralMotion();

    const wa = waveActivity();
    const isWaving = wa.flips >= 3 && wa.amp > 0.03;

    if (!isWaving && opens >= 3 && cosToVertical > 0.52 && palmSpanLen(lm) >= 0.03) {
      const heightBoostFast = Math.max(0, (0.68 - minY) * 0.8);
      const sFast = Math.min(
        1,
        0.6 + 0.2 * Math.min(1, (opens - 2) / 2) + heightBoostFast
      );
      return { ok: true, type: "raise_hand", score: sFast };
    }

    const highA = minY <= 0.62;
    const passOpenPalm =
      opens >= 3 && cosToVertical > 0.58 && !isWaving && highA && vel <= 0.06;

    const span = palmSpanLen(lm);
    const tipPairs = [
      [MP.INDEX_TIP, MP.MIDDLE_TIP],
      [MP.MIDDLE_TIP, MP.RING_TIP],
      [MP.RING_TIP, MP.PINKY_TIP],
    ];
    const meanAdj =
      tipPairs
        .map(([a, b]) => Math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y))
        .reduce((s, d) => s + d, 0) / (tipPairs.length || 1);
    const together = meanAdj / Math.max(1e-3, span) < 0.2;
    const highB = minY <= 0.68;
    const passFlatPalm =
      opens >= 2 &&
      together &&
      cosToVertical > 0.56 &&
      !isWaving &&
      highB &&
      vel <= 0.06;

    if (passOpenPalm || passFlatPalm) {
      const openness = Math.max(
        0,
        (idx.margin + mid.margin + rng.margin + pky.margin) / 4
      );
      const orientBoost = Math.max(0, (cosToVertical - 0.65) * 0.9);
      const heightBoost = Math.max(0, (0.6 - minY) * 0.8);
      const togetherBoost = passFlatPalm
        ? Math.min(
            0.22,
            Math.max(0, (0.18 - meanAdj / Math.max(1e-3, span)) * 2.0)
          )
        : 0;
      const score = Math.min(
        1,
        0.34 + 0.26 * openness + orientBoost + heightBoost + togetherBoost
      );
      return { ok: true, type: "raise_hand", score };
    }
    return { ok: false };
  } catch {
    return { ok: false };
  }
}

export function classifyOnPhone(lm, faces, canvasW, canvasH) {
  try {
    if (!lm || lm.length < 21 || !Array.isArray(faces) || !faces.length)
      return { ok: false };
    const wrist = lm[MP.WRIST],
      iMcp = lm[MP.INDEX_MCP],
      thumbTip = lm[MP.THUMB_TIP];
    if (!wrist || !iMcp || !thumbTip) return { ok: false };

    if (palmSpanLen(lm) < 0.02) return { ok: false };

    const pickNear = (px, py) => {
      let best = null,
        bestD2 = Infinity;
      for (const f of faces) {
        const fx = (f.cx || 0) / Math.max(1, canvasW);
        const fy = (f.cy || 0) / Math.max(1, canvasH);
        const dx = px - fx,
          dy = py - fy,
          d2 = dx * dx + dy * dy;
        if (d2 < bestD2) {
          bestD2 = d2;
          best = f;
        }
      }
      return best;
    };
    const f = pickNear(wrist.x, wrist.y) || faces[0];

    const fx = (f.cx || 0) / Math.max(1, canvasW);
    const fy = (f.cy || 0) / Math.max(1, canvasH);
    const hw = Math.max(0.02, ((f.w || 120) / Math.max(1, canvasW)) * 0.5);
    const hh = Math.max(0.03, ((f.h || 160) / Math.max(1, canvasH)) * 0.5);

    const sideSign = wrist.x >= fx ? +1 : -1;

    const earX = fx + sideSign * hw * 0.78;
    const earY = fy - hh * 0.08;

    const targetBandX = fx + sideSign * hw * 0.92;
    const bandY = (y) =>
      Math.max(0, Math.min(1, 1 - Math.abs(y - fy) / (hh * 0.85)));
    const closeSide = (px, tx) =>
      Math.max(0, Math.min(1, 1 - Math.abs(px - tx) / (hw * 0.95)));

    const mid = {
      x: (wrist.x + thumbTip.x) * 0.5,
      y: (wrist.y + thumbTip.y) * 0.5,
    };
    const closeW = { side: closeSide(wrist.x, targetBandX), y: bandY(wrist.y) };
    const closeT = {
      side: closeSide(thumbTip.x, targetBandX),
      y: bandY(thumbTip.y),
    };
    const closeM = { side: closeSide(mid.x, targetBandX), y: bandY(mid.y) };

    const normDist = (px, py) => {
      const dx = (px - earX) / hw;
      const dy = (py - earY) / hh;
      return Math.hypot(dx, dy);
    };
    const dEarW = normDist(wrist.x, wrist.y);
    const dEarT = normDist(thumbTip.x, thumbTip.y);
    const dEarM = normDist(mid.x, mid.y);
    const dEarMin = Math.min(dEarW, dEarT, dEarM);
    const earProx = Math.max(0, 1 - Math.min(1.3, dEarMin));

    const vx1 = iMcp.x - wrist.x,
      vy1 = iMcp.y - wrist.y;
    const vlen1 = Math.hypot(vx1, vy1) || 1e-6;
    const cosVertIdx = Math.abs(vy1) / vlen1;
    const vx2 = thumbTip.x - wrist.x,
      vy2 = thumbTip.y - wrist.y;
    const vlen2 = Math.hypot(vx2, vy2) || 1e-6;
    const cosVertPhone = Math.abs(vy2) / vlen2;
    const cosToVertical = Math.max(cosVertIdx, cosVertPhone);

    const idx = fingerOpen(lm, MP.INDEX_TIP, MP.INDEX_MCP);
    const midF = fingerOpen(lm, MP.MIDDLE_TIP, MP.MIDDLE_MCP);
    const rng = fingerOpen(lm, MP.RING_TIP, MP.RING_MCP);
    const pky = fingerOpen(lm, MP.PINKY_TIP, MP.PINKY_MCP);
    const opens = [idx, midF, rng, pky].filter((f) => f.open).length;
    const fewFingers = opens <= 3;

    const wa = waveActivity();
    const isWaving = wa.flips >= 2 && wa.amp > 0.025;
    const vel = recentLateralMotion();

    const passBand =
      (closeW.side > 0.08 && closeW.y > 0.08) ||
      (closeT.side > 0.1 && closeT.y > 0.08) ||
      (closeM.side > 0.1 && closeM.y > 0.1);

    const passEarStrict = dEarMin <= 0.62;

    const ok =
      passEarStrict &&
      cosToVertical > 0.45 &&
      fewFingers &&
      !isWaving &&
      vel <= 0.14;

    if (!ok) return { ok: false };

    if (dEarMin <= 0.48 && cosToVertical > 0.5 && vel <= 0.12) {
      const sFast = Math.min(
        1,
        0.78 +
          0.16 * earProx +
          0.06 * Math.max(0, (cosToVertical - 0.5) * 2.0)
      );
      return { ok: true, type: "on_phone", score: sFast };
    }

    const closenessSide = Math.max(closeW.side, closeT.side, closeM.side);
    const closenessY = Math.max(closeW.y, closeT.y, closeM.y);
    const multiPointBonus =
      (closeW.side > 0.08 && closeT.side > 0.08 ? 0.08 : 0) +
      (closeM.side > 0.1 && (closeW.side > 0.08 || closeT.side > 0.08)
        ? 0.06
        : 0);

    const score = Math.min(
      1,
      0.3 * closenessSide +
        0.18 * closenessY +
        0.4 * earProx +
        0.08 * Math.max(0, (cosToVertical - 0.5) * 2.0) +
        0.04 * Math.max(0, 0.22 - wa.amp) +
        multiPointBonus
    );

    return { ok: true, type: "on_phone", score };
  } catch {
    return { ok: false };
  }
}

export function classifyPaper(lm) {
  try {
    const idx = fingerOpen(lm, MP.INDEX_TIP, MP.INDEX_MCP);
    const mid = fingerOpen(lm, MP.MIDDLE_TIP, MP.MIDDLE_MCP);
    const rng = fingerOpen(lm, MP.RING_TIP, MP.RING_MCP);
    const pky = fingerOpen(lm, MP.PINKY_TIP, MP.PINKY_MCP);
    const opens = [idx, mid, rng, pky].filter((f) => f.open).length;
    if (opens >= 3) {
      const avgMargin = (idx.margin + mid.margin + rng.margin + pky.margin) / 4;
      const score = Math.min(1, 0.25 * opens + Math.max(0, avgMargin));
      return { ok: true, type: "paper", score };
    }
    return { ok: false };
  } catch {
    return { ok: false };
  }
}

export function classifyRock(lm) {
  try {
    const idx = fingerOpen(lm, MP.INDEX_TIP, MP.INDEX_MCP);
    const mid = fingerOpen(lm, MP.MIDDLE_TIP, MP.MIDDLE_MCP);
    const rng = fingerOpen(lm, MP.RING_TIP, MP.RING_MCP);
    const pky = fingerOpen(lm, MP.PINKY_TIP, MP.PINKY_MCP);
    const opens = [idx, mid, rng, pky].filter((f) => f.open).length;
    if (opens <= 1) {
      const neg = [idx, mid, rng, pky].map((f) =>
        Math.max(0, 0.12 - Math.max(0, f.margin))
      );
      const tight = neg.reduce((a, b) => a + b, 0) / 4;
      const score = Math.min(1, 0.85 - 0.2 * opens + tight);
      return { ok: true, type: "rock", score };
    }
    return { ok: false };
  } catch {
    return { ok: false };
  }
}

export function classifyScissors(lm) {
  try {
    const idx = fingerOpen(lm, MP.INDEX_TIP, MP.INDEX_MCP);
    const mid = fingerOpen(lm, MP.MIDDLE_TIP, MP.MIDDLE_MCP);
    const rng = fingerOpen(lm, MP.RING_TIP, MP.RING_MCP);
    const pky = fingerOpen(lm, MP.PINKY_TIP, MP.PINKY_MCP);
    if (idx.open && mid.open && !rng.open && !pky.open) {
      const margin = Math.max(0, idx.margin) + Math.max(0, mid.margin);
      const clamp =
        Math.max(0, 0.12 - Math.max(0, rng.margin)) +
        Math.max(0, 0.12 - Math.max(0, pky.margin));
      const score = Math.min(1, 0.45 + 0.35 * margin + 0.25 * clamp);
      return { ok: true, type: "scissors", score };
    }
    return { ok: false };
  } catch {
    return { ok: false };
  }
}

const WAVE_BOOT_MS = 300;
const WAVE_GRACE_MS = 550;
const CHANGE_COOLDOWN_MS = 520;

const GESTURE_PRIORITY = [
  "wave",
  "raise_hand",
  "on_phone",
  "thumbs_up",
  "peace",
  "paper",
  "rock",
  "scissors",
];
export const VOTE_WINDOW = 5;
const VOTE_MAX_AGE_MS = 700;
const REQUIRE_CONSISTENT = 2;
const CLEAR_IF_IDLE_MS = 450;
const MIN_SCORE = {
  wave: 0.4,
  thumbs_up: 0.22,
  peace: 0.46,
  raise_hand: 0.36,
  on_phone: 0.34,
  paper: 0.38,
  rock: 0.45,
  scissors: 0.45,
};

export function pickStableGesture(now, win, prevStable) {
  const fresh = (win || [])
    .filter((e) => e && now - e.t <= VOTE_MAX_AGE_MS)
    .slice(-VOTE_WINDOW);

  if (!fresh.length) {
    if (prevStable && now - prevStable.t < CLEAR_IF_IDLE_MS) return prevStable;
    return null;
  }

  const byType = new Map();
  for (const e of fresh) {
    const min = MIN_SCORE[e.type] ?? 0.4;
    if ((e.score ?? 0) < min) continue;
    const rec = byType.get(e.type) || {
      count: 0,
      sum: 0,
      best: 0,
      firstTs: e.t,
    };
    rec.count += 1;
    rec.sum += e.score ?? 0;
    rec.best = Math.max(rec.best, e.score ?? 0);
    if (e.t < rec.firstTs) rec.firstTs = e.t;
    byType.set(e.type, rec);
  }

  if (!byType.size) {
    if (prevStable && now - prevStable.t < CLEAR_IF_IDLE_MS) return prevStable;
    return null;
  }

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

  if (!best) return prevStable;

  const waveFirstTs = best.firstTs;
  const waveRecent = now - waveFirstTs <= WAVE_GRACE_MS;
  const recentChange =
    now - (prevStable?.t || 0) <= WAVE_BOOT_MS;
  if (waveRecent && recentChange) {
    const waveCand = {
      type: "wave",
      count: byType.get("wave")?.count || 0,
      avg: byType.get("wave")?.sum / (byType.get("wave")?.count || 1),
      pri: GESTURE_PRIORITY.indexOf("wave"),
      bestScore: byType.get("wave")?.best || 0,
      firstTs: byType.get("wave")?.firstTs || 0,
    };
    if (waveCand.bestScore >= MIN_SCORE.wave) {
      best = waveCand;
    }
  }

  const needByType = { wave: 3 };
  const need = needByType[best.type] || REQUIRE_CONSISTENT;
  if (best.count < need) {
    return prevStable && now - prevStable.t < CLEAR_IF_IDLE_MS
      ? prevStable
      : null;
  }

  if (prevStable && prevStable.type !== best.type) {
    if (now - prevStable.t < CHANGE_COOLDOWN_MS) {
      return prevStable;
    }
  }

  return { type: best.type, t: now, score: best.bestScore };
}
