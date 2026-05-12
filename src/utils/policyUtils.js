const GREET_COOLDOWN_MS = 20_000;

export function handleZoneTransitions({
  allIdentities,
  now,
  logZone,
  stableGestureRef,
  prevZoneMapRef,
  greenEntryRef,
  callOverStateRef,
  lastGlobalCallOverTsRef,
  pendingGreetsRef,
  pushedGreets,
  ageGenderCacheRef,
  greetInviteRef,
  lastGroupSetRef,
  lastGroupAskTsRef,
  sendPeopleIntent,
  groupInfo,
  guestSnapshots,
  CALL_OVER_MAX_TRIES,
  CALL_OVER_COOLDOWN_MS,
  GREEN_STABLE_MS,
  NOT_SEEN_RESET_MS,
  GROUP_ASK_COOLDOWN_MS,
  socketRef,
  deviceId,
  sessionId,
  ageGroupOf,
  groupSignature,
  maybeRotateSession,
  rotateConfig,
}) {
  const log = logZone || (() => {});

  const currentKeys = new Set(allIdentities.map((p) => p.key));
  const greenPeople = allIdentities.filter((p) => p.zone === "green");
  const primaryGreenKey = greenPeople[0]?.key || null;
  for (const [key] of prevZoneMapRef.current.entries()) {
    if (!currentKeys.has(key)) {
      const seenMeta = greetInviteRef.current.get(key);
      const lastSeenAt = seenMeta?.lastSeen || 0;
      if (!lastSeenAt || now - lastSeenAt > 35000) {
        prevZoneMapRef.current.delete(key);
        greenEntryRef.current.delete(key);
        callOverStateRef.current.delete(key);
        pendingGreetsRef.current.delete(key);
        log(`[DEBUG Zone Transitions] Cleared stale zone state for ${key}`);
      }
    }
  }

  const isOnPhone = stableGestureRef.current?.type === "on_phone";
  log(
    `[DEBUG Zone Transitions] Processing ${allIdentities.length} identities, isOnPhone=${isOnPhone}`
  );

  for (const p of allIdentities) {
    try {
      const rec =
        greetInviteRef.current.get(p.key) || {
          count: 0,
          last: 0,
          lastSeen: 0,
        };
      rec.lastSeen = now;
      greetInviteRef.current.set(p.key, rec);
    } catch {}

    const prevZ = prevZoneMapRef.current.get(p.key);
    const currentZ = p.zone;

    log(
      `[DEBUG Zone Transitions] Person ${p.key}: prevZone=${prevZ} -> currentZone=${currentZ}`
    );

    if (currentZ === "green") {
      if (primaryGreenKey && p.key !== primaryGreenKey) {
        prevZoneMapRef.current.set(p.key, currentZ);
        log(
          `[DEBUG Zone Transitions] Person ${p.key}: non-primary green ignored (primary=${primaryGreenKey})`
        );
        continue;
      }

      // reset call_over tries when person enters green
      if (prevZ !== "green") {
        callOverStateRef.current.delete(p.key);
      }

      let entry = greenEntryRef.current.get(p.key);
      if (!entry) {
        entry = { ts: now, lastGreetTs: 0 };
        greenEntryRef.current.set(p.key, entry);
        log(
          `[DEBUG Zone Transitions] Mark green entry for ${p.key} at ${now.toFixed(
            0
          )}ms`
        );
      }

      const inGreenMs = now - (entry.ts || 0);
      const isStablyInGreen = inGreenMs >= GREEN_STABLE_MS;

      const everyoneReady =
        greenPeople.length > 0 &&
        greenPeople.every((q) => {
          const hasId = !!(q.name || q.gid || q.key);
          return hasId;
        });

      prevZoneMapRef.current.set(p.key, currentZ);

      const greetCooldownReady = now - (entry.lastGreetTs || 0) >= GREET_COOLDOWN_MS;
      if (
        !isStablyInGreen ||
        !everyoneReady ||
        !greetCooldownReady ||
        isOnPhone
      ) {
        log("[DEBUG Zone Transitions] Skipping greet", {
          isStablyInGreen,
          everyoneReady,
          greetCooldownReady,
          isOnPhone,
        });
        continue;
      }

      log(
        `[DEBUG Zone Transitions] TRIGGER: stable green (${inGreenMs.toFixed(
          0
        )}ms) for ${p.key} (all green faces ready)`
      );

      const cacheGA = ageGenderCacheRef.current.get(p.key) || {};
      const effectiveGender = (p.gender || cacheGA.gender || "").toLowerCase();
      const effectiveAgeGroup = p.ageGroup || ageGroupOf(cacheGA.age);

      callOverStateRef.current.delete(p.key);

      const address = p.name
        ? p.name
        : groupInfo.size > 1
        ? groupInfo.hasKid
          ? "family"
          : "everyone"
        : p.gender === "male"
        ? "sir"
        : p.gender === "female"
        ? "ma'am"
        : "there";

      log("[DEBUG Zone Transitions] Sending greet intent for", p.key);
      const greetSent = sendPeopleIntent(
        "greet",
        { ...p, gender: effectiveGender, ageGroup: effectiveAgeGroup },
        {
          group: { ...groupInfo, address },
          guests: guestSnapshots,
        }
      );

      if (greetSent) {
        entry.lastGreetTs = now;
        greenEntryRef.current.set(p.key, entry);
        pendingGreetsRef.current.delete(p.key);
      } else {
        log(
          `[DEBUG Zone Transitions] Suppressed greet for ${p.key}: speech lock active`
        );
      }
    } else {
      prevZoneMapRef.current.set(p.key, currentZ);

      if (prevZ === "green") {
        log(
          `[DEBUG Zone Transitions] Person ${p.key}: leaving green -> ${currentZ}`
        );
        greenEntryRef.current.delete(p.key);

        if (currentZ === "none") {
          sendPeopleIntent("none", { ...p, zone: "none" }, {
            group: groupInfo,
            reason: "left_green_zone",
            guests: guestSnapshots,
          });
        }
      }

      if (currentZ === "red" && !isOnPhone) {
        const s = callOverStateRef.current.get(p.key) || { tries: 0, last: 0 };
        const globalReady =
          now - (lastGlobalCallOverTsRef.current || 0) >= CALL_OVER_COOLDOWN_MS;
        if (
          s.tries < CALL_OVER_MAX_TRIES &&
          now - (s.last || 0) >= CALL_OVER_COOLDOWN_MS &&
          globalReady
        ) {
          log(
            `[DEBUG Zone Transitions] TRIGGER: call_over for ${p.key} (zone=red, prevZ=${prevZ})`
          );
          const callOverSent = sendPeopleIntent("call_over", p, {
            group: groupInfo,
            reason: prevZ === "green" ? "left_green_zone" : "in_red_zone",
            guests: guestSnapshots,
            context: { attempt: s.tries + 1 },
          });
          if (callOverSent) {
            s.tries += 1;
            s.last = now;
            callOverStateRef.current.set(p.key, s);
            lastGlobalCallOverTsRef.current = now;
          } else {
            log(
              `[DEBUG Zone Transitions] Suppressed call_over for ${p.key}: speech lock active`
            );
          }
        } else if (!globalReady) {
          log(
            `[DEBUG Zone Transitions] Suppressed call_over for ${p.key}: global cooldown active`
          );
        }
      } else {
        log(
          `[DEBUG Zone Transitions] Person ${p.key}: ${prevZ || "none"} -> ${currentZ} (no action)`
        );
      }
    }
  }

  const curSet = new Set(allIdentities.map((p) => p.key));
  const prevSet = lastGroupSetRef.current || new Set();
  const inter = new Set([...curSet].filter((k) => prevSet.has(k)));
  const overlap = inter.size / Math.max(1, Math.max(prevSet.size, curSet.size));
  if (prevSet.size && curSet.size && overlap < 0.5) {
    if (now - (lastGroupAskTsRef.current || 0) >= GROUP_ASK_COOLDOWN_MS) {
      lastGroupAskTsRef.current = now;
      socketRef.current?.emit?.("policy_event", {
        deviceId,
        sessionId: sessionId || "web-" + deviceId,
        type: "ask_group_change",
        prevSize: prevSet.size,
        currSize: curSet.size,
        overlap,
        at: Date.now(),
      });
    }
  }

  try {
    const sig = groupSignature(allIdentities);
    maybeRotateSession({
      sig,
      people: allIdentities,
      now,
      refs: rotateConfig.refs,
      sessionId,
      deviceId,
      socket: socketRef.current,
      setSessionId: rotateConfig.setSessionId,
      uuid: rotateConfig.uuid,
      stableMs: rotateConfig.stableMs,
      cooldownMs: rotateConfig.cooldownMs,
    });
  } catch {}

  try {
    for (const [k, rec] of Array.from(greetInviteRef.current.entries())) {
      if (!rec || typeof rec !== "object") {
        greetInviteRef.current.delete(k);
        continue;
      }
      const lastSeen = rec.lastSeen || 0;
      if (now - lastSeen > NOT_SEEN_RESET_MS) {
        greetInviteRef.current.delete(k);
      }
    }
  } catch {}

  try {
    for (const k of Array.from(pendingGreetsRef.current.keys())) {
      if (!curSet.has(k)) {
        pendingGreetsRef.current.delete(k);
      }
    }
  } catch {}

  lastGroupSetRef.current = curSet;
}

