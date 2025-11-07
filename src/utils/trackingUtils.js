// ====================== TRACKING UTILITIES ====================== 
// Face tracking, recognition, and zone classification utilities

import * as faceapi from "face-api.js";

/* ====================== CONSTANTS ====================== */
// geometry
export const FACE_WIDTH_M = 0.15;
export let FOCAL_PX = 500; // refined after camera opens

// recognition
export const MATCH_THRESHOLD = 0.5;
export const MATCH_MARGIN = 0.03;
export const STABILIZE_FRAMES = 5;

// zones
export const DEFAULT_GREEN_MAX_M = 0.8;
export const DEFAULT_RED_CUTOFF_M = 3.5;

/* ====================== UTILITY FUNCTIONS ====================== */

// Age classification helper
export const ageGroup = (a) => {
  if (!Number.isFinite(a)) return "unknown";
  if (a >= 60) return "senior";
  if (a >= 40) return "adult";
  if (a >= 18) return "young_adult";
  if (a >= 12) return "teen";
  return "child";
};

// Green/Red zone classification
export const zoneOf = (d, greenMaxM) =>
  Number.isFinite(d) && Number.isFinite(greenMaxM) && d <= greenMaxM
    ? "green"
    : "red";

// Best-scoring expression label
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

/* ====================== DISTANCE ESTIMATION ====================== */

// Update focal length (called when camera opens)
export function updateFocalLength(newFocal) {
  FOCAL_PX = newFocal;
}

// Estimate distance based on face box width in pixels
export function estimateDistanceMpx(boxWidthPx) {
  const w = Number(boxWidthPx);
  if (!Number.isFinite(w) || w <= 0) return null;
  const dist = (FACE_WIDTH_M * FOCAL_PX) / w;
  return Number.isFinite(dist) && dist > 0 ? dist : null;
}

/* ====================== FACE RECOGNITION ====================== */

// Find best match for a face descriptor
export function findBestFaceMatch(descriptor, matcher) {
  if (!matcher || !descriptor) return null;
  
  try {
    const best = matcher.findBestMatch(descriptor);
    if (!best || best.label === "unknown") return null;
    
    // Primary threshold check
    if (best.distance <= MATCH_THRESHOLD) {
      return { name: best.label, confidence: 1 - best.distance };
    }
    
    // Margin check for borderline cases
    if (best.distance <= MATCH_THRESHOLD + 0.03) {
      const bestLabel = best.label;
      const bestDist = best.distance;
      let secondBest = 1;
      
      // Find second-best distance
      for (const ld of matcher.labeledDescriptors) {
        if (ld.label === bestLabel) continue;
        for (const d of ld.descriptors) {
          const dd = faceapi.euclideanDistance(descriptor, d);
          if (dd < secondBest) secondBest = dd;
        }
      }
      
      // Check margin
      if (secondBest - bestDist >= MATCH_MARGIN) {
        return { name: bestLabel, confidence: 1 - bestDist };
      }
    }
    
    return null;
  } catch (error) {
    console.warn("Face matching error:", error);
    return null;
  }
}

/* ====================== GUEST ID ASSIGNMENT ====================== */

// Guest ID counter
let guestCounter = 1;
const guestMemory = []; // { desc, gid, lastSeen }
const GUEST_MATCH_THRESHOLD = 0.58;
const GUEST_FORGET_MS = 300_000; // 5 minutes

export function assignGuestIdFor(descriptor) {
  if (!descriptor) return `G${guestCounter++}`;
  
  const now = performance.now();
  
  // Clean old entries
  const cutoff = now - GUEST_FORGET_MS;
  for (let i = guestMemory.length - 1; i >= 0; i--) {
    if (guestMemory[i].lastSeen < cutoff) {
      guestMemory.splice(i, 1);
    }
  }
  
  // Find existing match
  for (const mem of guestMemory) {
    try {
      const d = faceapi.euclideanDistance(descriptor, mem.desc);
      if (d <= GUEST_MATCH_THRESHOLD) {
        mem.lastSeen = now;
        return mem.gid;
      }
    } catch {
      // Skip on error
    }
  }
  
  // Create new guest ID
  const newGid = `G${guestCounter++}`;
  guestMemory.push({ desc: descriptor, gid: newGid, lastSeen: now });
  return newGid;
}

/* ====================== FACE PROCESSING ====================== */

// Process face detections and filter by distance
export function processFaceDetections(detections, canvas, redCutoffM, greenMaxM) {
  const resized = faceapi
    .resizeResults(detections, { width: canvas.width, height: canvas.height })
    .sort((a, b) => a.detection.box.x - b.detection.box.x);

  const cutoff = Number.isFinite(redCutoffM) ? redCutoffM : Infinity;
  const candidates = [];
  
  for (let i = 0; i < resized.length; i++) {
    const det = resized[i];
    const box = det.detection.box;
    const dist = estimateDistanceMpx(box.width);
    
    // Skip faces that are too far
    if (dist != null && dist > cutoff) continue;
    
    const zone = zoneOf(dist, greenMaxM);
    candidates.push({ i, det, box, dist, zone });
  }
  
  return candidates;
}

// Filter and sort green zone candidates for tracking
export function getTrackingCandidates(candidates, maxTrack = 5) {
  const greenCandidates = candidates
    .filter((c) => c.zone === "green" && Number.isFinite(c.dist))
    .sort((a, b) => a.dist - b.dist);
    
  return greenCandidates.slice(0, maxTrack);
}

// Calculate zone totals
export function calculateZoneTotals(candidates) {
  const total = candidates.length;
  const green = candidates.filter((c) => c.zone === "green").length;
  const red = total - green;
  
  return { total, green, red };
}

/* ====================== FACE STABILIZATION ====================== */

// Stabilize face identity over frames
export function stabilizeFaceIdentity(name, guestId, stableKey, tracks) {
  let displayName = name || guestId || "Guest";
  
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
  
  return displayName;
}

/* ====================== CACHE MANAGEMENT ====================== */

// Clean up tracking caches
export function cleanupTrackingCaches(trackedFaces, mouthMap, perFaceStable, waveHistByFace, recentMap, peopleForPost) {
  // Prune faces that are no longer tracked
  const trackedKeys = new Set((trackedFaces || []).map((f) => f.key));
  
  // Clean mouth activity cache
  for (const k of Array.from(mouthMap.keys())) {
    if (!trackedKeys.has(k)) mouthMap.delete(k);
  }
  
  // Clean gesture cache
  for (const k of Array.from(perFaceStable.keys())) {
    if (!trackedKeys.has(k)) perFaceStable.delete(k);
  }
  
  // Clean wave history
  for (const k of Array.from(waveHistByFace.keys())) {
    if (!trackedKeys.has(k)) waveHistByFace.delete(k);
  }
  
  // Clean recent tracking map
  const seenKeys = new Set(peopleForPost.map((p) => (p.name || p.gid) ?? ""));
  for (const k of Object.keys(recentMap)) {
    if (k && !seenKeys.has(k)) delete recentMap[k];
  }
}