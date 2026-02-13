export const estimateDistanceM = (wPx, focalPx, faceWidthM = 0.15) =>
  Number.isFinite(wPx) && wPx > 0
    ? (focalPx * faceWidthM) / wPx
    : null;

export const ageGroupOf = (age) => {
  if (!Number.isFinite(age)) return "unknown";
  const a = Math.round(age);
  if (a >= 18) return "adult";
  if (a >= 12) return "teen";
  return "child";
};

export const zoneOf = (d, greenMaxM, faceCenterX, frameWidth) => {
  if (!Number.isFinite(d) || !Number.isFinite(greenMaxM) || d > greenMaxM) {
    return "red";
  }

  if (
    Number.isFinite(faceCenterX) &&
    Number.isFinite(frameWidth) &&
    frameWidth > 0
  ) {
    const leftBoundary = frameWidth * (1 / 3 - 1 / 8);
    const rightBoundary = frameWidth * (2 / 3 + 1 / 8);
    const isInCenter =
      faceCenterX >= leftBoundary && faceCenterX <= rightBoundary;

    return isInCenter ? "green" : "red";
  }

  return "green";
};

export const topExpression = (e) => {
  if (!e || typeof e !== "object") return "neutral";
  let bestKey = "neutral",
    bestVal = -Infinity;
  for (const [k, v] of Object.entries(e)) {
    const val = Number(v) || 0;
    if (val > bestVal) {
      bestVal = val;
      bestKey = k;
    }
  }
  return bestKey;
};

export function gestureLabelOf(g) {
  if (!g || !g.type) return null;
  switch (g.type) {
    case "wave":
      return "wave";
    case "thumbs_up":
      return "thumbs_up";
    case "peace":
      return "peace";
    case "raise_hand":
      return "raise_hand";
    case "on_phone":
      return "on_phone";
    default:
      return String(g.type);
  }
}

const DEG = Math.PI / 180;

export function focalFromFov(widthPx, fovDeg) {
  const fov = Math.max(1, Math.min(179, Number(fovDeg || 70)));
  const w = Math.max(1, Number(widthPx) || 1);
  return w / 2 / Math.tan((fov * DEG) / 2);
}

export function anglesFromPixel(px, py, fx, fy, cx0, cy0) {
  const x = px - cx0;
  const y = py - cy0;
  return {
    yaw: Math.atan2(x, Math.max(1e-6, fx)),
    pitch: Math.atan2(-y, Math.max(1e-6, fy)),
  };
}

export function posFromPixel(px, py, fx, fy, cx0, cy0, Z) {
  const x = px - cx0;
  const y = py - cy0;
  const z = Number(Z);
  if (!Number.isFinite(z)) return { x: null, y: null, z: null };
  return {
    x: (x / Math.max(1e-6, fx)) * z,
    y: -(y / Math.max(1e-6, fy)) * z,
    z,
  };
}

export function mouthMAR(landmarks68) {
  try {
    const p = landmarks68?.positions || landmarks68?._positions;
    if (!p || p.length < 68) return 0;
    const dist = (a, b) => {
      const dx = p[a].x - p[b].x;
      const dy = p[a].y - p[b].y;
      return Math.hypot(dx, dy);
    };
    const V = (dist(61, 67) + dist(62, 66) + dist(63, 65)) / 3;
    const H = dist(60, 64) || 1;
    const mar = V / H;
    return Math.max(0, Math.min(1, mar));
  } catch {
    return 0;
  }
}

export const shrinkBox = (b, f = 0.7) => {
  const w = b.width * f;
  const h = b.height * f;
  const x = b.x + (b.width - w) / 2;
  const y = b.y + (b.height - h) / 2;
  return { x, y, width: w, height: h };
};
