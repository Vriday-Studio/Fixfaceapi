import { useCallback, useRef } from "react";
import { MSG_TYPE } from "./useDirectWebSocket";
import { buildVisitContext as buildVisitContextUtil } from "../utils/visitContext";

const PENDING_SPEECH_LOCK_MS = 4000;

export function useTelemetryEmitters({
  clockRef,
  locationRef,
  weatherRef,
  totalsRef,
  sendWsCommand,
  wsIsConnected,
  socketRef,
  deviceId,
  sessionId,
  policySpeechLockRef,
  speakingRef,
}) {
  const lastCrowdSendRef = useRef({ t: 0, sig: "" });
  const lastPeopleSnapshotSentRef = useRef(0);

  const buildVisitContext = useCallback(
    (extra = {}) =>
      buildVisitContextUtil({
        clockRef,
        locationRef,
        weatherRef,
        totalsRef,
        extra,
      }),
    [clockRef, locationRef, weatherRef, totalsRef]
  );

  const sendPeopleIntent = useCallback(
    (intent, person, extra = {}) => {
      const isSpokenIntent = intent === "greet" || intent === "call_over";
      const now = Date.now();
      const activeSpeechLock =
        !!policySpeechLockRef?.current?.active &&
        now - (policySpeechLockRef?.current?.since || 0) < PENDING_SPEECH_LOCK_MS;

      if (isSpokenIntent && (speakingRef?.current || activeSpeechLock)) {
        return false;
      }

      if (isSpokenIntent && policySpeechLockRef?.current) {
        policySpeechLockRef.current = { active: true, since: now, intent };
      }

      const identityKeyForPayload =
        person.stableKey || person.key || person.gid || person.name || null;

      const payload = {
        intent,
        zone: person.zone,
        name: person.name || null,
        gid: person.gid || null,
        gender: person.gender || null,
        ageGroup: person.ageGroup || null,
        emotion: person.emotion || null,
        yawDeg:
          Number.isFinite(person.yawDeg) && person.yawDeg != null
            ? +Number(person.yawDeg).toFixed(2)
            : null,
        pitchDeg:
          Number.isFinite(person.pitchDeg) && person.pitchDeg != null
            ? +Number(person.pitchDeg).toFixed(2)
            : null,
        mouthActivity:
          Number.isFinite(person.mouthActivity) &&
          person.mouthActivity != null
            ? +Number(person.mouthActivity).toFixed(3)
            : null,
        posCam: person.posCam || null,
        stableKey: identityKeyForPayload,
        slotKey: person.slotKey || null,
        group: extra.group || null,
        reason: extra.reason || null,
        guests: extra.guests || null,
        context: buildVisitContext(extra.context || {}),
      };

      if (wsIsConnected.current) {
        sendWsCommand(MSG_TYPE.PeopleData, payload);
      }

      const socket = socketRef?.current;
      if (!socket?.emit) return true;

      if (intent === "greet") {
        socket.emit("policy_event", {
          deviceId,
          sessionId: sessionId || "web-" + deviceId,
          type: "greet",
          address: extra?.group?.address || null,
          target: {
            name: person.name || null,
            gid: person.gid || null,
            gender: person.gender || null,
          },
          group: extra?.group
            ? {
                size: extra.group.size ?? null,
                hasKid: !!extra.group.hasKid,
              }
            : null,
          at: Date.now(),
        });
        return true;
      }

      if (intent === "call_over") {
        socket.emit("policy_event", {
          deviceId,
          sessionId: sessionId || "web-" + deviceId,
          type: "call_over",
          attempt: extra?.context?.attempt ?? null,
          target: {
            name: person.name || null,
            gid: person.gid || null,
            gender: person.gender || null,
          },
          group: extra?.group
            ? {
                size: extra.group.size ?? null,
                hasKid: !!extra.group.hasKid,
              }
            : null,
          reason: extra?.reason || null,
          at: Date.now(),
        });
      }

      return true;
    },
    [
      buildVisitContext,
      deviceId,
      policySpeechLockRef,
      sendWsCommand,
      sessionId,
      speakingRef,
      socketRef,
      wsIsConnected,
    ]
  );

  const emitCrowdThrottled = useCallback(
    (payload) => {
      const now = performance.now();
      const minMs = 66;
      const state = lastCrowdSendRef.current;
      const sourcePeople = payload.peopleSource || payload.people || [];
      let sig = `${payload.focusIndex ?? -1}|${sourcePeople.length}`;
      for (let i = 0; i < sourcePeople.length; i++) {
        const p = sourcePeople[i];
        const yaw = Number.isFinite(p.yawDeg) ? Math.round(p.yawDeg * 10) : -9999;
        const pitch = Number.isFinite(p.pitchDeg)
          ? Math.round(p.pitchDeg * 10)
          : -9999;
        const mouth = Math.round((p.mouthActivity || 0) * 1000);
        sig += `|${yaw},${pitch},${mouth}`;
      }
      if (now - state.t < minMs && sig === state.sig) return;

      const people =
        payload.people ||
        sourcePeople.map((p) => ({
          name: p.name || null,
          gid: p.gid || null,
          gender: p.gender || null,
          ageGroup: p.ageGroup || null,
          zone: p.zone,
          yawDeg: Number.isFinite(p.yawDeg) ? +p.yawDeg.toFixed(1) : null,
          pitchDeg: Number.isFinite(p.pitchDeg) ? +p.pitchDeg.toFixed(1) : null,
          mouthActivity: +(p.mouthActivity ?? 0).toFixed(3),
          posCam: p.posCam,
        }));

      try {
        const { peopleSource, ...rest } = payload;
        void peopleSource;
        sendWsCommand(MSG_TYPE.CrowdStat, {
          ...rest,
          people,
          context: buildVisitContext(),
        });
      } catch {}

      lastCrowdSendRef.current = { t: now, sig };
    },
    [buildVisitContext, sendWsCommand]
  );

  const emitCrowdByGid = useCallback(
    (payload) => {
      const now = performance.now();
      const minMs = 1000;
      const state = lastCrowdSendRef.current;

      if (now - state.t <= minMs) return;

      const peopleCandidate = payload.people || [];
      //this to notify backend that no people detected, so that it can reset the crowd status and not keep showing stale data
      if (peopleCandidate.length === 0){
         const dataToSend = {
                    deviceId: payload.deviceId,
                    timestamp: payload.timeISO,
                    sessionId: payload.sessionId,
                    gesture: null,
                    context: buildVisitContext(),
                    people: null,
                  };

          sendWsCommand(MSG_TYPE.CrowdStat, dataToSend);
      } 
      else
      {
        for (const p of peopleCandidate) {
                if (p.gid == null) continue;
                try {
                  const dataToSend = {
                    deviceId: payload.deviceId,
                    timestamp: payload.timeISO,
                    sessionId: payload.sessionId,
                    gesture: p.gesture || null,
                    context: buildVisitContext(),
                    people: p,
                  };

                  sendWsCommand(MSG_TYPE.CrowdStat, dataToSend);
                } catch {
                  continue;
                }
              }
      };

      lastCrowdSendRef.current = { t: now };
    },
    [buildVisitContext, sendWsCommand]
  );

  const sendGreenSnapshot = useCallback(
    (cand) => {
      if (!wsIsConnected.current) return;

      const ts = performance.now();
      if (ts - (lastPeopleSnapshotSentRef.current || 0) < 200) return;

      const guests = cand.map((p) => ({
        gid: p.gid || null,
        name: p.name || null,
        gender: p.gender || null,
        ageGroup: p.ageGroup || null,
        zone: "green",
      }));

      sendWsCommand(MSG_TYPE.PeopleData, {
        intent: "none",
        zone: "green",
        guests,
        context: buildVisitContext(),
      });

      lastPeopleSnapshotSentRef.current = ts;
    },
    [buildVisitContext, sendWsCommand, wsIsConnected]
  );

  return {
    buildVisitContext,
    sendPeopleIntent,
    emitCrowdThrottled,
    emitCrowdByGid,
    sendGreenSnapshot,
  };
}
