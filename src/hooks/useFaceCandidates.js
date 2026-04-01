import { useCallback } from "react";

export function useFaceCandidates({
  faceMatcherRef,
  matchThreshold,
  matchMargin,
  stabilizeFrames,
  handsCacheMs,
  rad,
  boxLineWidth,
  labelPadX,
  labelPadY,
  camFxRef,
  camFyRef,
  panOffRef,
  tiltOffRef,
  zoneModeRef,
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
}) {
  const buildFaceFrameData = useCallback(
    ({
      resized,
      canvas,
      ctx,
      now,
      redCutoffM,
      showAlign,
    }) => {
      const rows = [];
      const peopleForPost = [];
      const gestureAllowedKeys = new Set();
      const tracks = recentMapRef.current;

      const cutoff = Number.isFinite(redCutoffM) ? redCutoffM : Infinity;
      const candidates = [];
      for (let i = 0; i < resized.length; i++) {
        const det = resized[i];
        const box = det.detection.box;
        const dist = estimateDistanceMpx(box.width);
        if (dist != null && dist > cutoff) continue;
        const faceCenterX = box.x + box.width / 2;
        const frameWidth = canvas.width;
        const zone = zoneOf(
          dist,
          greenMaxMRef.current,
          faceCenterX,
          frameWidth,
          zoneModeRef.current
        );
        candidates.push({ i, det, box, dist, zone });
      }

      const total = candidates.length;
      const green = candidates.filter((c) => c.zone === "green").length;
      const red = total - green;

      const greenCandidates = candidates
        .filter((c) => c.zone === "green" && Number.isFinite(c.dist))
        .sort((a, b) => a.dist - b.dist);
      const tracked = greenCandidates.slice(0, 5);

      for (let k = 0; k < tracked.length; k++) {
        const { i, det, box, dist, zone } = tracked[k];

        // --- recognition (fast path + small margin check) ---
        const faceMatcher = faceMatcherRef.current;
        let name = null;
        if (faceMatcher && det.descriptor) {
          const best = faceMatcher.findBestMatch(det.descriptor);
          if (
            best &&
            best.label !== "unknown" &&
            best.distance <= matchThreshold
          ) {
            name = best.label;
          } else if (
            best &&
            best.label !== "unknown" &&
            best.distance <= matchThreshold + 0.03
          ) {
            const bestLabel = best.label;
            const bestDist = best.distance;
            let second = 1;
            for (const ld of faceMatcher.labeledDescriptors) {
              if (ld.label === bestLabel) continue;
              for (const d of ld.descriptors) {
                const dd = faceapi.euclideanDistance(det.descriptor, d);
                if (dd < second) second = dd;
              }
            }
            if (second - bestDist >= matchMargin) name = bestLabel;
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
        if (gesturesOnRef.current && k < gestureTargetsRef.current) {
          gestureAllowedKeys.add(stableKey);
        }

        const prev = tracks[stableKey];
        if (prev && prev.name !== displayName) {
          if ((prev.count || 0) < stabilizeFrames) {
            displayName = prev.name;
            prev.count = (prev.count || 0) + 1;
          } else {
            tracks[stableKey] = { name: displayName, count: 0 };
          }
        } else {
          tracks[stableKey] = { name: displayName, count: 0 };
        }

        const cacheGA = ageGenderCacheRef.current.get(stableKey) || {};
        const genderRaw = det.gender ?? cacheGA.gender ?? "";
        const gender = String(genderRaw || "").toLowerCase();
        const ageVal = Number.isFinite(det.age)
          ? det.age
          : Number.isFinite(cacheGA.age)
            ? cacheGA.age
            : null;
        if (Number.isFinite(det.age) || det.gender) {
          ageGenderCacheRef.current.set(stableKey, {
            age: det.age,
            gender: det.gender,
          });
        }
        const expr = topExpression(det.expressions);

        const dbox = shrinkBox(box);
        const cx = dbox.x + dbox.width * 0.5;
        const cy = dbox.y + dbox.height * 0.45;

        const fx = camFxRef.current;
        const fy = camFyRef.current;
        const cx0 = canvas.width * 0.5;
        const cy0 = canvas.height * 0.5;

        const { yaw, pitch } = anglesFromPixel(cx, cy, fx, fy, cx0, cy0);
        const yawDeg = yaw * rad + panOffRef.current;
        const pitchDeg = pitch * rad + tiltOffRef.current;

        const Z = Number.isFinite(dist) ? dist : null;
        const pos =
          Z != null
            ? posFromPixel(cx, cy, fx, fy, cx0, cy0, Z)
            : { x: null, y: null, z: null };

        const normX = Math.min(1, Math.abs((cx - cx0) / (canvas.width * 0.5)));
        const normY = Math.min(1, Math.abs((cy - cy0) / (canvas.height * 0.5)));
        const centerNorm = Math.min(1, Math.hypot(normX, normY));

        let mouthActivity = 0;
        const lmBox = det.detection?.box;
        try {
          const lm = det.landmarks;
          const rec = mouthMapRef.current.get(stableKey) || { ema: 0.3, t: now };
          const level = mouthMAR(lm);
          if (!Number.isFinite(level) || level <= 0) {
            rec.ema = 0.98 * rec.ema + 0.02 * 0.3;
          } else {
            rec.ema = rec.ema ? 0.7 * rec.ema + 0.3 * level : level;
          }
          rec.t = now;
          mouthMapRef.current.set(stableKey, rec);
          mouthActivity = Math.max(0, Math.min(1, rec.ema));
        } catch {
          const rec = mouthMapRef.current.get(stableKey);
          if (rec) mouthActivity = rec.ema;
        }

        ctx.strokeStyle = "#22c55e";
        ctx.lineWidth = boxLineWidth;
        ctx.strokeRect(dbox.x, dbox.y, dbox.width, dbox.height);

        const faceStable = perFaceStableRef.current.get(stableKey);
        const freshFaceGesture =
          gestureAllowedKeys.has(stableKey) &&
          faceStable &&
          now - faceStable.t <= handsCacheMs
            ? faceStable
            : null;
        const gestureLbl =
          zone === "green" && freshFaceGesture
            ? gestureLabelOf(freshFaceGesture)
            : null;

        const ageTxt = Number.isFinite(ageVal)
          ? Math.max(0, Math.round(ageVal))
          : "-";
        const l1 = `${displayName}${gestureLbl ? "  |  " + gestureLbl : ""}  |  ${zone}  |  ${ageTxt} ${gender}  |  ${expr}`;
        const l2 = `yaw ${yawDeg.toFixed(1)} deg | pitch ${pitchDeg.toFixed(1)} deg | mouth ${mouthActivity.toFixed(2)} | landmarks :${lmBox.x}`;
        const color =
          zone === "green" ? "rgba(34,197,94,0.85)" : "rgba(239,68,68,0.85)";
        const lineH = 18;
        const lines = showAlign ? 2 : 1;
        const tw =
          Math.max(
            ctx.measureText(l1).width,
            showAlign ? ctx.measureText(l2).width : 0
          ) +
          labelPadX * 2;
        const th = lineH * lines + labelPadY * 2;
        const lx = Math.max(0, Math.min(dbox.x, canvas.width - tw));
        const ly = Math.max(0, dbox.y - th - 4);

        ctx.fillStyle = color;
        ctx.fillRect(lx, ly, tw, th);
        ctx.fillStyle = "#fff";
        ctx.fillText(l1, lx + labelPadX, ly + labelPadY);
        if (showAlign) {
          ctx.fillStyle = "#e9ffef";
          ctx.fillText(l2, lx + labelPadX, ly + labelPadY + lineH);
        }

        if (showAlign) {
          const barW = 64;
          const barH = 5;
          const gap = 3;
          const bx = lx;
          const by = ly + th + gap;
          ctx.fillStyle = "rgba(255,255,255,0.15)";
          ctx.fillRect(bx, by, barW, barH);
          ctx.fillStyle = "#22c55e";
          ctx.fillRect(bx, by, barW * Math.min(1, Math.max(0, mouthActivity)), barH);
        }

        if (freshFaceGesture && zone === "green") {
          const gtxt = gestureLabelOf(freshFaceGesture);
          if (gtxt) {
            ctx.save();
            ctx.font =
              "bold 14px system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif";
            const padX = 6;
            const padY = 4;
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
          stableKey,
          slotKey,
          _cx: cx,
          _cy: cy,
          _w: dbox.width,
          _h: dbox.height,
          _can_w: canvas.width,
          _can_h: canvas.height,
        });
      }

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
        z: p.posCam?.z ?? null,
      }));

      const guestSnapshots = peopleForPost.map((p) => ({
        name: p.name || null,
        zone: p.zone,
        gender: p.gender || null,
        ageGroup: p.ageGroup || null,
      }));

      allFacesRef.current = candidates.map((c) => {
        const d = shrinkBox(c.box);
        return {
          cx: d.x + d.width * 0.5,
          cy: d.y + d.height * 0.45,
          w: d.width,
          h: d.height,
        };
      });

      return {
        rows,
        peopleForPost,
        gestureAllowedKeys,
        tracked,
        candidates,
        total,
        green,
        red,
        guestSnapshots,
      };
    },
    [
      ageGenderCacheRef,
      ageGroupOf,
      allFacesRef,
      anglesFromPixel,
      assignGuestIdFor,
      boxLineWidth,
      camFxRef,
      camFyRef,
      estimateDistanceMpx,
      faceMatcherRef,
      faceapi,
      gestureLabelOf,
      gestureTargetsRef,
      gesturesOnRef,
      greenMaxMRef,
      zoneModeRef,
      handsCacheMs,
      labelPadX,
      labelPadY,
      matchMargin,
      matchThreshold,
      mouthMapRef,
      mouthMAR,
      panOffRef,
      perFaceStableRef,
      posFromPixel,
      rad,
      recentMapRef,
      shrinkBox,
      stabilizeFrames,
      tiltOffRef,
      topExpression,
      trackedFacesRef,
      zoneOf,
    ]
  );

  return { buildFaceFrameData };
}



