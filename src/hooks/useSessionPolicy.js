import { useCallback, useEffect, useRef } from "react";
import { MSG_TYPE } from "./useDirectWebSocket";

export function useSessionPolicy({
  autoSession,
  serverConnected,
  ueConnected,
  sessionStatus,
  sessionId,
  deviceId,
  onCreateSession,
  sendWsCommand,
  setSessionStatus,
  setSessionId,
}) {
  const autoSessionPendingRef = useRef(false);

  const handleStartSession = useCallback(() => {
    try {
      onCreateSession();
    } catch (err) {
      console.warn("[session] start failed:", err);
    }
  }, [onCreateSession]);

  const handleStopSession = useCallback(() => {
    try {
      sendWsCommand(MSG_TYPE.SessionEnd, {
        sessionId: sessionId || "web-" + deviceId,
      });
      setSessionStatus("IDLE");
      setSessionId(null);
    } catch (err) {
      console.warn("[session] stop failed:", err);
    }
  }, [deviceId, sendWsCommand, sessionId, setSessionId, setSessionStatus]);

  useEffect(() => {
    if (sessionStatus === "ACTIVE" || sessionStatus === "IDLE") {
      autoSessionPendingRef.current = false;
    }
  }, [sessionStatus]);

  useEffect(() => {
    if (!autoSession) return;

    if (serverConnected && ueConnected) {
      if (sessionStatus !== "ACTIVE" && !autoSessionPendingRef.current) {
        autoSessionPendingRef.current = true;
        handleStartSession();
      }
      return;
    }

    autoSessionPendingRef.current = false;
    // Keep session alive; do not auto-send SessionEnd on transient disconnects.
    // Camera/zone policy on server controls mic open/close safely.
  }, [
    autoSession,
    serverConnected,
    ueConnected,
    sessionStatus,
    handleStartSession,
  ]);

  return {
    handleStartSession,
    handleStopSession,
  };
}
