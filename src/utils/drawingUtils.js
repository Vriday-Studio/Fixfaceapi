// ====================== DRAWING UTILITIES ====================== 
// Canvas drawing functions for face detection visualization

/* ====================== CONSTANTS ====================== */

// drawing
export const BOX_SHRINK = 0.7;
export const BOX_LINE_WIDTH = 5;
export const LABEL_FONT = "16px system-ui,-apple-system,Segoe UI,Roboto,'Helvetica Neue',Arial,sans-serif";
export const LABEL_PAD_X = 8;
export const LABEL_PAD_Y = 6;

// colors
export const COLORS = {
  GREEN_BOX: "#22c55e",
  RED_BOX: "#ef4444",
  GREEN_LABEL: "rgba(34,197,94,0.85)",
  RED_LABEL: "rgba(239,68,68,0.85)",
  WHITE_TEXT: "#fff",
  LIGHT_GREEN_TEXT: "#e9ffef",
  CROSSHAIR: "#0ea5e9",
  BANNER_BG: "rgba(14,165,233,0.18)",
  BANNER_BORDER: "rgba(14,165,233,0.45)",
  BANNER_TEXT: "#e6f7ff",
  MOUTH_BAR_BG: "rgba(255,255,255,0.15)",
  MOUTH_BAR_FILL: "#22c55e",
  GESTURE_BADGE: "rgba(34,197,94,0.9)",
  TRACKED_COUNTER: "rgba(34,197,94,0.85)",
  HAND_COUNTER: "rgba(14,165,233,0.85)",
  WRIST_DOT: "rgba(255,255,0,0.8)"
};

/* ====================== HELPER FUNCTIONS ====================== */

// Shrink bounding box for better visual appearance
export function shrinkBox(box, factor = BOX_SHRINK) {
  const w = box.width * factor;
  const h = box.height * factor;
  return {
    x: box.x + (box.width - w) / 2,
    y: box.y + (box.height - h) / 2,
    width: w,
    height: h,
  };
}

// Get zone-appropriate colors
export function getZoneColors(zone) {
  const isGreen = zone === "green";
  return {
    boxColor: isGreen ? COLORS.GREEN_BOX : COLORS.RED_BOX,
    labelColor: isGreen ? COLORS.GREEN_LABEL : COLORS.RED_LABEL
  };
}

/* ====================== DRAWING FUNCTIONS ====================== */

// Draw video frame as background
export function drawVideoBackground(ctx, video, canvas) {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
}

// Draw alignment crosshair overlay
export function drawAlignmentCrosshair(ctx, canvas) {
  const cx0 = canvas.width * 0.5;
  const cy0 = canvas.height * 0.5;

  ctx.save();
  ctx.strokeStyle = COLORS.CROSSHAIR;
  ctx.lineWidth = 1.5;
  
  // Horizontal line
  ctx.beginPath();
  ctx.moveTo(cx0 - 22, cy0);
  ctx.lineTo(cx0 + 22, cy0);
  ctx.stroke();
  
  // Vertical line
  ctx.beginPath();
  ctx.moveTo(cx0, cy0 - 22);
  ctx.lineTo(cx0, cy0 + 22);
  ctx.stroke();
  
  ctx.restore();
}

// Draw calibration banner
export function drawCalibrationBanner(ctx, canvas, message) {
  if (!message) return;
  
  ctx.save();
  ctx.font = "bold 18px system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif";
  const tw = ctx.measureText(message).width + 18;
  const x = Math.max(10, (canvas.width - tw) / 2);
  const y = 10;
  
  ctx.fillStyle = COLORS.BANNER_BG;
  ctx.fillRect(x, y, tw, 34);
  ctx.strokeStyle = COLORS.BANNER_BORDER;
  ctx.strokeRect(x, y, tw, 34);
  ctx.fillStyle = COLORS.BANNER_TEXT;
  ctx.textBaseline = "middle";
  ctx.fillText(message, x + 9, y + 17);
  ctx.restore();
}

// Draw face bounding box
export function drawFaceBox(ctx, box, zone) {
  const { boxColor } = getZoneColors(zone);
  const dbox = shrinkBox(box);
  
  ctx.strokeStyle = boxColor;
  ctx.lineWidth = BOX_LINE_WIDTH;
  ctx.strokeRect(dbox.x, dbox.y, dbox.width, dbox.height);
  
  return dbox; // Return for label positioning
}

// Draw face information label
export function drawFaceLabel(ctx, canvas, dbox, labelData, showAlignment = false) {
  const { displayName, gestureLbl, zone, ageTxt, gender, expr, yawDeg, pitchDeg, mouthActivity } = labelData;
  const { labelColor } = getZoneColors(zone);
  
  // Prepare label text
  const l1 = `${displayName}${gestureLbl ? " • " + gestureLbl : ""} • ${zone} • ${ageTxt} ${gender} • ${expr}`;
  const l2 = `yaw ${yawDeg.toFixed(1)}° · pitch ${pitchDeg.toFixed(1)}° · mouth ${mouthActivity.toFixed(2)}`;
  
  // Calculate dimensions
  const lineH = 18;
  const lines = showAlignment ? 2 : 1;
  const tw = Math.max(
    ctx.measureText(l1).width,
    showAlignment ? ctx.measureText(l2).width : 0
  ) + LABEL_PAD_X * 2;
  const th = lineH * lines + LABEL_PAD_Y * 2;
  
  // Position label
  const lx = Math.max(0, Math.min(dbox.x, canvas.width - tw));
  const ly = Math.max(0, dbox.y - th - 4);
  
  // Draw background
  ctx.fillStyle = labelColor;
  ctx.fillRect(lx, ly, tw, th);
  
  // Draw text
  ctx.fillStyle = COLORS.WHITE_TEXT;
  ctx.fillText(l1, lx + LABEL_PAD_X, ly + LABEL_PAD_Y);
  
  if (showAlignment) {
    ctx.fillStyle = COLORS.LIGHT_GREEN_TEXT;
    ctx.fillText(l2, lx + LABEL_PAD_X, ly + LABEL_PAD_Y + lineH);
  }
  
  return { lx, ly, tw, th };
}

// Draw mouth activity bar
export function drawMouthActivityBar(ctx, labelRect, mouthActivity, showAlignment = false) {
  if (!showAlignment) return;
  
  const { lx, ly, tw, th } = labelRect;
  const barW = 64;
  const barH = 5;
  const gap = 3;
  const bx = lx;
  const by = ly + th + gap;
  
  // Background bar
  ctx.fillStyle = COLORS.MOUTH_BAR_BG;
  ctx.fillRect(bx, by, barW, barH);
  
  // Activity fill
  ctx.fillStyle = COLORS.MOUTH_BAR_FILL;
  const fillWidth = barW * Math.min(1, Math.max(0, mouthActivity));
  ctx.fillRect(bx, by, fillWidth, barH);
}

// Draw gesture badge
export function drawGestureBadge(ctx, canvas, dbox, gestureText) {
  if (!gestureText) return;
  
  ctx.save();
  ctx.font = "bold 14px system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif";
  const padX = 6;
  const padY = 4;
  const tw = ctx.measureText(gestureText).width + padX * 2;
  const th = 18 + padY * 2;
  
  const gx = Math.max(0, Math.min(dbox.x + dbox.width - tw - 4, canvas.width - tw));
  const gy = Math.max(0, dbox.y + 4);
  
  ctx.fillStyle = COLORS.GESTURE_BADGE;
  ctx.fillRect(gx, gy, tw, th);
  ctx.fillStyle = COLORS.WHITE_TEXT;
  ctx.fillText(gestureText, gx + padX, gy + padY + 9);
  ctx.restore();
}

// Draw tracked faces counter
export function drawTrackedCounter(ctx, count) {
  if (count <= 0) return;
  
  ctx.save();
  ctx.font = "bold 12px system-ui";
  const msg = `tracked: ${count}`;
  const w = ctx.measureText(msg).width + 10;
  
  ctx.fillStyle = COLORS.TRACKED_COUNTER;
  ctx.fillRect(10, 10, w, 20);
  ctx.fillStyle = COLORS.WHITE_TEXT;
  ctx.fillText(msg, 15, 24);
  ctx.restore();
}

// Draw hands counter
export function drawHandsCounter(ctx, canvas, count) {
  if (count <= 0) return;
  
  ctx.save();
  ctx.font = "bold 14px system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif";
  const msg = `hands: ${count}`;
  const w = ctx.measureText(msg).width + 12;
  
  ctx.fillStyle = COLORS.HAND_COUNTER;
  ctx.fillRect(10, canvas.height - 62, w, 22);
  ctx.fillStyle = COLORS.WHITE_TEXT;
  ctx.fillText(msg, 16, canvas.height - 46);
  ctx.restore();
}

// Draw hand wrist dots
export function drawHandWristDots(ctx, canvas, handsList) {
  if (!handsList || handsList.length === 0) return;
  
  for (const landmarks of handsList) {
    const wrist = landmarks[0]; // WRIST is index 0
    if (!wrist) continue;
    
    const px = wrist.x * canvas.width;
    const py = wrist.y * canvas.height;
    
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.WRIST_DOT;
    ctx.fill();
  }
}

/* ====================== COMPOSITE DRAWING FUNCTIONS ====================== */

// Draw complete face visualization
export function drawFaceDetection(ctx, canvas, faceData, options = {}) {
  const {
    box,
    zone,
    displayName,
    gestureLbl,
    ageTxt,
    gender,
    expr,
    yawDeg,
    pitchDeg,
    mouthActivity,
    gestureText,
    showAlignment = false
  } = faceData;
  
  // Set up canvas context
  ctx.font = LABEL_FONT;
  ctx.textBaseline = "top";
  
  // Draw bounding box
  const dbox = drawFaceBox(ctx, box, zone);
  
  // Prepare label data
  const labelData = {
    displayName,
    gestureLbl,
    zone,
    ageTxt,
    gender,
    expr,
    yawDeg,
    pitchDeg,
    mouthActivity
  };
  
  // Draw label
  const labelRect = drawFaceLabel(ctx, canvas, dbox, labelData, showAlignment);
  
  // Draw mouth activity bar
  drawMouthActivityBar(ctx, labelRect, mouthActivity, showAlignment);
  
  // Draw gesture badge if in green zone
  if (zone === "green" && gestureText) {
    drawGestureBadge(ctx, canvas, dbox, gestureText);
  }
}

// Draw alignment overlay (crosshair + banner)
export function drawAlignmentOverlay(ctx, canvas, calibrationMessage = null) {
  drawAlignmentCrosshair(ctx, canvas);
  if (calibrationMessage) {
    drawCalibrationBanner(ctx, canvas, calibrationMessage);
  }
}

// Draw all UI overlays
export function drawUIOverlays(ctx, canvas, overlayData) {
  const {
    trackedCount = 0,
    handsCount = 0,
    handsList = [],
    showAlignment = false,
    calibrationMessage = null
  } = overlayData;
  
  // Draw counters
  if (trackedCount > 0) {
    drawTrackedCounter(ctx, trackedCount);
  }
  
  if (handsCount > 0) {
    drawHandsCounter(ctx, canvas, handsCount);
    drawHandWristDots(ctx, canvas, handsList);
  }
  
  // Draw alignment overlay
  if (showAlignment) {
    drawAlignmentOverlay(ctx, canvas, calibrationMessage);
  }
}