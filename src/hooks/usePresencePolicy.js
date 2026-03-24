import { useCallback } from "react";
import { MSG_TYPE } from "./useDirectWebSocket";
import { handleZoneTransitions } from "../utils/policyUtils";

export function usePresencePolicy({
  wsIsConnected,
  sendWsCommand,
  buildVisitContext,
  logFace,
  logZone,
  prevZoneMapRef,
  greenEntryRef,
  callOverStateRef,
  pendingGreetsRef,
  redZoneCounterRef,
  noneZoneCounterRef,
  redZoneTriggeredRef,
  stableGestureRef,
  pushedGreets,
  ageGenderCacheRef,
  greetInviteRef,
  lastGroupSetRef,
  lastGroupAskTsRef,
  sendPeopleIntent,
  socketRef,
  deviceId,
  sessionId,
  ageGroupOf,
  groupSignature,
  maybeRotateSession,
  rotateConfig,
  callOverMaxTries,
  callOverCooldownMs,
  greenStableMs,
  notSeenResetMs,
  groupAskCooldownMs,
  redZoneNoneResetFrames,
  redZoneStableFrames,
}) {
  const handleNoFaceDetected = useCallback(() => {
    if (!wsIsConnected.current) return;

    logFace(`[DEBUG Face Detection] No faces detected - sending zone="none"`);
    sendWsCommand(MSG_TYPE.PeopleData, {
      intent: "none",
      zone: "none",
      guests: [],
      context: buildVisitContext(),
    });

    // Keep local interaction state during short camera absences.
    // Server-side none-zone timeout decides when the conversation truly ends.
  }, [
    buildVisitContext,
    logFace,
    sendWsCommand,
    wsIsConnected,
  ]);

  const processRedZoneState = useCallback(
    ({ total, green, red }) => {
      try {
        const hasRedOnly = red > 0 && green === 0;
        const hasNone = total === 0;

        if (hasNone) {
          noneZoneCounterRef.current += 1;
          redZoneCounterRef.current = 0;
          if (
            noneZoneCounterRef.current <= 3 ||
            noneZoneCounterRef.current === redZoneNoneResetFrames
          ) {
            logZone(
              `[RED Zone] NONE frame ${noneZoneCounterRef.current}/${redZoneNoneResetFrames} (total=${total})`
            );
          }

          if (
            noneZoneCounterRef.current >= redZoneNoneResetFrames &&
            redZoneTriggeredRef.current
          ) {
            logZone(
              `[RED Zone] Reset after ${redZoneNoneResetFrames} NONE frames`
            );
            redZoneTriggeredRef.current = false;
          }
          return;
        }

        if (hasRedOnly) {
          redZoneCounterRef.current += 1;
          noneZoneCounterRef.current = 0;
          logZone(
            `[RED Zone] Frame ${redZoneCounterRef.current}/${redZoneStableFrames} (red=${red}, green=${green})`
          );

          if (wsIsConnected.current) {
            const redPayload = {
              intent: "none",
              zone: "red",
              guests: Array.from({ length: red }, (_, i) => ({
                gid: `RedGuest${i + 1}`,
                name: null,
                gender: null,
                ageGroup: null,
                zone: "red",
              })),
              context: buildVisitContext(),
            };

            if (
              redZoneCounterRef.current === 1 ||
              redZoneCounterRef.current % 5 === 0
            ) {
              logZone(
                `[RED Zone] Sending PeopleData frame ${redZoneCounterRef.current}:`,
                redPayload
              );
            }
            sendWsCommand(MSG_TYPE.PeopleData, redPayload);

            if (
              redZoneCounterRef.current >= redZoneStableFrames &&
              !redZoneTriggeredRef.current
            ) {
              logZone(
                `[RED Zone] OK STABLE: ${redZoneCounterRef.current} consecutive RED frames detected`
              );
              redZoneTriggeredRef.current = true;
            }
          }
          return;
        }

        redZoneCounterRef.current = 0;
        noneZoneCounterRef.current = 0;
      } catch (err) {
        console.error("[RED Zone] Error in detection logic:", err);
      }
    },
    [
      buildVisitContext,
      logZone,
      noneZoneCounterRef,
      redZoneCounterRef,
      redZoneNoneResetFrames,
      redZoneStableFrames,
      redZoneTriggeredRef,
      sendWsCommand,
      wsIsConnected,
    ]
  );

  const dispatchZoneTransitions = useCallback(
    ({ allIdentities, now, groupInfo, guestSnapshots }) => {
      handleZoneTransitions({
        allIdentities,
        now,
        logZone,
        stableGestureRef,
        prevZoneMapRef,
        greenEntryRef,
        callOverStateRef,
        pendingGreetsRef,
        pushedGreets,
        ageGenderCacheRef,
        greetInviteRef,
        lastGroupSetRef,
        lastGroupAskTsRef,
        sendPeopleIntent,
        groupInfo,
        guestSnapshots,
        CALL_OVER_MAX_TRIES: callOverMaxTries,
        CALL_OVER_COOLDOWN_MS: callOverCooldownMs,
        GREEN_STABLE_MS: greenStableMs,
        NOT_SEEN_RESET_MS: notSeenResetMs,
        GROUP_ASK_COOLDOWN_MS: groupAskCooldownMs,
        socketRef,
        deviceId,
        sessionId,
        ageGroupOf,
        groupSignature,
        maybeRotateSession,
        rotateConfig,
      });
    },
    [
      ageGenderCacheRef,
      ageGroupOf,
      callOverCooldownMs,
      callOverMaxTries,
      callOverStateRef,
      deviceId,
      greenEntryRef,
      greenStableMs,
      greetInviteRef,
      groupAskCooldownMs,
      groupSignature,
      lastGroupAskTsRef,
      lastGroupSetRef,
      logZone,
      maybeRotateSession,
      notSeenResetMs,
      pendingGreetsRef,
      prevZoneMapRef,
      pushedGreets,
      rotateConfig,
      sendPeopleIntent,
      sessionId,
      socketRef,
      stableGestureRef,
    ]
  );

  return {
    handleNoFaceDetected,
    processRedZoneState,
    dispatchZoneTransitions,
  };
}
