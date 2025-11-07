// ====================== FACE TRACKING MODULE ====================== 
// Main face tracking orchestration that combines detection, analysis, and drawing

import * as faceapi from "face-api.js";
import {
  processFaceDetections,
  getTrackingCandidates,
  calculateZoneTotals,
  findBestFaceMatch,
  assignGuestIdFor,
  stabilizeFaceIdentity,
  cleanupTrackingCaches,
  ageGroup,
  estimateDistanceMpx,
  updateFocalLength
} from './trackingUtils';

import {
  drawVideoBackground,
  drawFaceDetection,
  drawUIOverlays,
  drawAlignmentOverlay
} from './drawingUtils';

import {
  analyzeFaceDetection,
  updateAgeGenderCache,
  calculateAttentionScore,
  isFacingCamera
} from './faceAnalysisUtils';

/* ====================== MAIN TRACKING CLASS ====================== */

export class FaceTracker {
  constructor(options = {}) {
    // Configuration
    this.config = {
      maxTracked: 5,
      ageGenderSampleMs: 1000,
      handsEnabled: true,
      handsCacheMs: 1000,
      gestureTargets: 2,
      ...options
    };
    
    // State
    this.trackedFaces = [];
    this.mouthMap = new Map();
    this.perFaceStable = new Map();
    this.waveHistByFace = new Map();
    this.recentMap = {};
    this.ageGenderCache = new Map();
    this.lastAgeSample = 0;
    
    // Camera parameters
    this.camFx = 500;
    this.camFy = 500;
    this.panOff = 0;
    this.tiltOff = 0;
    
    // Matcher for face recognition
    this.faceMatcher = null;
  }
  
  // Initialize face matcher with labeled descriptors
  setFaceMatcher(matcher) {
    this.faceMatcher = matcher;
  }
  
  // Update camera parameters
  updateCameraParams(fx, fy, panOff = 0, tiltOff = 0) {
    this.camFx = fx;
    this.camFy = fy;
    this.panOff = panOff;
    this.tiltOff = tiltOff;
    updateFocalLength(fx);
  }
  
  // Main tracking function
  async trackFaces(detections, canvas, ctx, options = {}) {
    const {
      redCutoffM = 3.5,
      greenMaxM = 0.8,
      showAlignment = false,
      calibrationMessage = null,
      gesturesOn = false,
      gestureAllowedKeys = new Set(),
      handsCount = 0,
      handsList = []
    } = options;
    
    const now = performance.now();
    
    // Clear canvas and draw video background
    const video = options.video;
    if (video) {
      drawVideoBackground(ctx, video, canvas);
    }
    
    // Draw alignment overlay if needed
    if (showAlignment) {
      drawAlignmentOverlay(ctx, canvas, calibrationMessage);
    }
    
    // Process face detections
    const candidates = processFaceDetections(detections, canvas, redCutoffM, greenMaxM);
    const tracked = getTrackingCandidates(candidates, this.config.maxTracked);
    const { total, green, red } = calculateZoneTotals(candidates);
    
    // Determine if we should update age/gender
    const shouldUpdateAge = now - this.lastAgeSample >= this.config.ageGenderSampleMs;
    if (shouldUpdateAge) this.lastAgeSample = now;
    
    // Process tracked faces
    const peopleForPost = [];
    const rows = [];
    
    // Define gesture eligibility (top N faces by proximity)
    const gestureEligible = new Set();
    if (gesturesOn) {
      for (let i = 0; i < Math.min(tracked.length, this.config.gestureTargets); i++) {
        const stableKey = this.getStableKey(tracked[i], i);
        gestureEligible.add(stableKey);
      }
    }
    
    // Process each tracked face
    for (let k = 0; k < tracked.length; k++) {
      const { det, box, dist, zone } = tracked[k];
      
      // Face recognition
      const matchResult = findBestFaceMatch(det.descriptor, this.faceMatcher);
      const name = matchResult?.name || null;
      
      // Guest ID assignment
      const guestId = name ? null : assignGuestIdFor(det.descriptor);
      const stableKey = (name || guestId) ?? `tmp-${k}`;
      
      // Stabilize identity
      const displayName = stabilizeFaceIdentity(name, guestId, stableKey, this.recentMap);
      
      // Analyze face
      const analysis = analyzeFaceDetection(det, {
        stableKey,
        mouthMap: this.mouthMap,
        camFx: this.camFx,
        camFy: this.camFy,
        canvas,
        panOff: this.panOff,
        tiltOff: this.tiltOff,
        now
      });
      
      // Update age/gender cache
      const { gender, ageVal } = updateAgeGenderCache(
        det, 
        stableKey, 
        this.ageGenderCache, 
        shouldUpdateAge
      );
      
      // Get gesture information
      const isGestureEligible = gestureEligible.has(stableKey);
      const faceStable = this.perFaceStable.get(stableKey);
      const freshGesture = isGestureEligible && 
        faceStable && 
        (now - faceStable.t <= this.config.handsCacheMs) ? faceStable : null;
      
      const gestureLbl = zone === "green" && freshGesture ? 
        this.getGestureLabel(freshGesture) : null;
      const gestureText = freshGesture ? this.getGestureLabel(freshGesture) : null;
      
      // Calculate 3D position
      const pos = analysis.cx && analysis.cy && dist ? 
        this.calculate3DPosition(analysis.cx, analysis.cy, dist) : 
        { x: null, y: null, z: null };
      
      // Create person data for API
      const personData = {
        idx: k + 1,
        name: displayName,
        gid: guestId,
        zone,
        dist: dist != null ? +dist.toFixed(2) : null,
        posCam: pos,
        gender,
        ageGroup: ageGroup(ageVal),
        expr: analysis.expr,
        yaw: +analysis.yawDeg.toFixed(1),
        pitch: +analysis.pitchDeg.toFixed(1),
        centerNorm: +analysis.centerNorm.toFixed(3),
        mouthActivity: +analysis.mouthActivity.toFixed(3),
        gesture: gestureText,
        confidence: matchResult?.confidence || 0
      };
      
      peopleForPost.push(personData);
      
      // Create table row for UI
      const ageTxt = Number.isFinite(ageVal) ? Math.max(0, Math.round(ageVal)) : "-";
      rows.push({
        idx: k + 1,
        gender,
        ageGroup: ageGroup(ageVal),
        zone,
        name: displayName,
        gesture: gestureText || "-",
        emotion: analysis.expr,
        distance: dist != null ? dist.toFixed(2) + "m" : "-"
      });
      
      // Draw face visualization
      const faceData = {
        box,
        zone,
        displayName,
        gestureLbl,
        ageTxt,
        gender,
        expr: analysis.expr,
        yawDeg: analysis.yawDeg,
        pitchDeg: analysis.pitchDeg,
        mouthActivity: analysis.mouthActivity,
        gestureText: zone === "green" ? gestureText : null,
        showAlignment
      };
      
      drawFaceDetection(ctx, canvas, faceData);
    }
    
    // Clean up caches
    cleanupTrackingCaches(
      this.trackedFaces,
      this.mouthMap,
      this.perFaceStable,
      this.waveHistByFace,
      this.recentMap,
      peopleForPost
    );
    
    // Draw UI overlays
    drawUIOverlays(ctx, canvas, {
      trackedCount: tracked.length,
      handsCount,
      handsList,
      showAlignment,
      calibrationMessage
    });
    
    // Pad table rows to 5
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
    
    return {
      peopleForPost,
      rows,
      totals: { total, green, red },
      tracked: tracked.length
    };
  }
  
  // Helper methods
  getStableKey(candidate, index) {
    const { det } = candidate;
    const matchResult = findBestFaceMatch(det.descriptor, this.faceMatcher);
    const name = matchResult?.name || null;
    const guestId = name ? null : assignGuestIdFor(det.descriptor);
    return (name || guestId) ?? `tmp-${index}`;
  }
  
  getGestureLabel(gestureData) {
    if (!gestureData || !gestureData.type) return null;
    switch (gestureData.type) {
      case "wave": return "wave";
      case "thumbs_up": return "thumbs_up";
      case "peace": return "peace";
      case "raise_hand": return "raise_hand";
      case "on_phone": return "on_phone";
      default: return String(gestureData.type);
    }
  }
  
  calculate3DPosition(cx, cy, Z) {
    const x = cx - (this.canvas?.width * 0.5 || 0);
    const y = cy - (this.canvas?.height * 0.5 || 0);
    const z = Number(Z);
    
    if (!Number.isFinite(z)) return { x: null, y: null, z: null };
    
    return {
      x: (x / Math.max(1e-6, this.camFx)) * z,
      y: -(y / Math.max(1e-6, this.camFy)) * z,
      z
    };
  }
  
  // Update gesture data for a face
  updateGestureData(stableKey, gestureData, timestamp = performance.now()) {
    this.perFaceStable.set(stableKey, {
      ...gestureData,
      t: timestamp
    });
  }
  
  // Get current tracking statistics
  getTrackingStats() {
    return {
      trackedFaces: this.trackedFaces.length,
      mouthMapSize: this.mouthMap.size,
      gestureMapSize: this.perFaceStable.size,
      ageGenderCacheSize: this.ageGenderCache.size,
      recentMapSize: Object.keys(this.recentMap).length
    };
  }
  
  // Clear all tracking data
  clearTracking() {
    this.trackedFaces = [];
    this.mouthMap.clear();
    this.perFaceStable.clear();
    this.waveHistByFace.clear();
    this.recentMap = {};
    this.ageGenderCache.clear();
    this.lastAgeSample = 0;
  }
}

/* ====================== FACTORY FUNCTION ====================== */

// Create and configure a new face tracker instance
export function createFaceTracker(options = {}) {
  return new FaceTracker(options);
}