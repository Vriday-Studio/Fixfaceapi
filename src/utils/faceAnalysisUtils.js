// ====================== FACE ANALYSIS UTILITIES ====================== 
// Mouth activity, pose estimation, and facial analysis functions

/* ====================== CONSTANTS ====================== */

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

/* ====================== CAMERA MATH UTILITIES ====================== */

// Calculate focal length from field of view
export function focalFromFov(widthPx, fovDeg) {
  const fov = Math.max(1, Math.min(179, Number(fovDeg || 70)));
  const w = Math.max(1, Number(widthPx) || 1);
  return w / 2 / Math.tan((fov * DEG) / 2);
}

// Calculate angles from pixel coordinates
export function anglesFromPixel(px, py, fx, fy, cx0, cy0) {
  const x = px - cx0;
  const y = py - cy0;
  return {
    yaw: Math.atan2(x, Math.max(1e-6, fx)), // +yaw = right
    pitch: Math.atan2(-y, Math.max(1e-6, fy)), // +pitch = up
  };
}

// Calculate 3D position from pixel coordinates and depth
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

/* ====================== MOUTH ACTIVITY ANALYSIS ====================== */

// Calculate mouth aspect ratio (MAR) from 68-point face landmarks
export function mouthMAR(landmarks68) {
  try {
    // Extract landmark positions
    const pts = landmarks68?.positions || 
                landmarks68?._positions || 
                (Array.isArray(landmarks68) ? landmarks68 : null);
    if (!pts) return 0;

    const dist = (a, b) => {
      const pa = pts[a];
      const pb = pts[b];
      return Math.hypot(pa.x - pb.x, pa.y - pb.y);
    };

    // Inner mouth landmarks: 60–67
    // Vertical = average of 3 vertical distances
    // Horizontal = distance between mouth corners
    const V = (dist(61, 67) + dist(62, 66) + dist(63, 65)) / 3;
    const H = dist(60, 64) || 1e-6;
    const mar = V / H;

    // Normalize to 0-1 range
    // Typical closed mouth MAR ≈ 0.25–0.35
    // Shift baseline to 0.3 and scale by 3.0
    const norm = Math.max(0, Math.min(1, (mar - 0.3) * 3.0));
    return norm;
  } catch {
    return 0;
  }
}

// Update mouth activity with exponential moving average
export function updateMouthActivity(landmarks, stableKey, mouthMap, now) {
  let mouthActivity = 0;
  
  try {
    const key = stableKey;
    const rec = mouthMap.get(key) || { ema: 0.3, t: now };
    const level = mouthMAR(landmarks);
    
    if (!Number.isFinite(level) || level <= 0) {
      // Hold previous with gentle decay toward neutral 0.3
      rec.ema = 0.98 * rec.ema + 0.02 * 0.3;
    } else {
      // Exponential moving average
      rec.ema = rec.ema ? 0.7 * rec.ema + 0.3 * level : level;
    }
    
    rec.t = now;
    mouthMap.set(key, rec);
    mouthActivity = Math.max(0, Math.min(1, rec.ema));
  } catch {
    // Use cached value if available
    const rec = mouthMap.get(stableKey);
    if (rec) mouthActivity = rec.ema;
  }
  
  return mouthActivity;
}

/* ====================== POSE ESTIMATION ====================== */

// Calculate face pose (yaw/pitch) from face center and camera parameters
export function calculateFacePose(box, camFx, camFy, canvas, panOff = 0, tiltOff = 0) {
  const dbox = shrinkBox(box);
  const cx = dbox.x + dbox.width * 0.5;
  const cy = dbox.y + dbox.height * 0.45; // Slightly above center for better face centering
  
  const cx0 = canvas.width * 0.5;
  const cy0 = canvas.height * 0.5;
  
  const { yaw, pitch } = anglesFromPixel(cx, cy, camFx, camFy, cx0, cy0);
  const yawDeg = yaw * RAD + panOff;
  const pitchDeg = pitch * RAD + tiltOff;
  
  return { yawDeg, pitchDeg, cx, cy };
}

// Calculate center normalization (distance from canvas center)
export function calculateCenterNorm(cx, cy, canvas) {
  const cx0 = canvas.width * 0.5;
  const cy0 = canvas.height * 0.5;
  
  const normX = Math.min(1, Math.abs((cx - cx0) / (canvas.width * 0.5)));
  const normY = Math.min(1, Math.abs((cy - cy0) / (canvas.height * 0.5)));
  const centerNorm = Math.min(1, Math.hypot(normX, normY));
  
  return centerNorm;
}

// Helper function to shrink bounding box (shared with drawing utils)
function shrinkBox(box, factor = 0.7) {
  const w = box.width * factor;
  const h = box.height * factor;
  return {
    x: box.x + (box.width - w) / 2,
    y: box.y + (box.height - h) / 2,
    width: w,
    height: h,
  };
}

/* ====================== FACE ANALYSIS ====================== */

// Analyze single face detection and extract all relevant data
export function analyzeFaceDetection(detection, options = {}) {
  const {
    stableKey,
    mouthMap,
    camFx,
    camFy,
    canvas,
    panOff = 0,
    tiltOff = 0,
    now = performance.now()
  } = options;
  
  const { box, landmarks, expressions, age, gender } = detection;
  
  // Calculate pose
  const { yawDeg, pitchDeg, cx, cy } = calculateFacePose(box, camFx, camFy, canvas, panOff, tiltOff);
  
  // Calculate center normalization
  const centerNorm = calculateCenterNorm(cx, cy, canvas);
  
  // Update mouth activity
  const mouthActivity = stableKey && mouthMap ? 
    updateMouthActivity(landmarks, stableKey, mouthMap, now) : 0;
  
  // Get expression
  const expr = topExpression(expressions);
  
  // Format age and gender
  const ageVal = Number.isFinite(age) ? age : null;
  const genderStr = String(gender || "").toLowerCase();
  
  return {
    yawDeg,
    pitchDeg,
    centerNorm,
    mouthActivity,
    expr,
    ageVal,
    gender: genderStr,
    cx,
    cy
  };
}

// Best-scoring expression helper
function topExpression(expressions) {
  if (!expressions || typeof expressions !== "object") return "neutral";
  
  let bestKey = "neutral";
  let bestVal = -Infinity;
  
  for (const [k, v] of Object.entries(expressions)) {
    const val = Number(v) || 0;
    if (val > bestVal) {
      bestVal = val;
      bestKey = k;
    }
  }
  
  return bestKey;
}

/* ====================== FACE ATTENTION DETECTION ====================== */

// Check if face is looking at camera (within angle thresholds)
export function isFacingCamera(yawDeg, pitchDeg, maxYaw = 9, maxPitch = 10) {
  return Math.abs(yawDeg) <= maxYaw && Math.abs(pitchDeg) <= maxPitch;
}

// Calculate attention score based on pose and position
export function calculateAttentionScore(yawDeg, pitchDeg, centerNorm, mouthActivity) {
  // Facing score (0-1, higher when looking at camera)
  const facingScore = Math.max(0, 1 - (Math.abs(yawDeg) / 15 + Math.abs(pitchDeg) / 15) / 2);
  
  // Center score (0-1, higher when near center)
  const centerScore = 1 - Math.min(1, centerNorm);
  
  // Activity score (mouth movement)
  const activityScore = Math.min(1, mouthActivity);
  
  // Weighted combination
  const attention = 0.4 * facingScore + 0.3 * centerScore + 0.3 * activityScore;
  
  return {
    total: Math.max(0, Math.min(1, attention)),
    facing: facingScore,
    center: centerScore,
    activity: activityScore
  };
}

/* ====================== AGE/GENDER CACHING ====================== */

// Manage age/gender cache with staggered updates
export function updateAgeGenderCache(detection, stableKey, cache, shouldUpdate = true) {
  const cacheEntry = cache.get(stableKey) || {};
  
  // Get values with cache fallback
  const genderRaw = detection.gender ?? cacheEntry.gender ?? "";
  const gender = String(genderRaw || "").toLowerCase();
  const ageVal = Number.isFinite(detection.age) ? 
    detection.age : 
    (Number.isFinite(cacheEntry.age) ? cacheEntry.age : null);
  
  // Update cache if requested and we have new data
  if (shouldUpdate && (Number.isFinite(detection.age) || detection.gender)) {
    cache.set(stableKey, {
      age: detection.age,
      gender: detection.gender,
    });
  }
  
  return { gender, ageVal };
}