import { useCallback, useEffect, useRef } from "react";
import { normalizeServerUrl } from "../utils/socketUtils";

export const MSG_TYPE = {
  Unknown: 0,
  Heartbeat: 1,
  PeopleData: 2,
  AudioData: 3,
  SessionStart: 4,
  SessionEnd: 5,
  Standby: 6,
  VocStart: 7,
  VocEnd: 8,
  CrowdStat: 9,
  MicData: 10,
};

const CONNECTION_POOL_MS = 5_000;

export function useDirectWebSocket({
  serverUrl,
  deviceId,
  setServerInfo,
  setSessionStatus,
  setSessionId,
  setMachineId,
  setUeConnected,
  enabled = true,
}) {
  const wsSocket = useRef(null);
  const wsIsConnected = useRef(false);

  const connectWebSocket = useCallback(() => {
    if (!enabled) return;

    const baseUrl = normalizeServerUrl(serverUrl);
    if (!baseUrl) return;

    const wsUrl = baseUrl.endsWith("/ws") ? baseUrl : `${baseUrl}/ws`;
    if (
      wsSocket.current &&
      (wsSocket.current.readyState === WebSocket.OPEN ||
        wsSocket.current.readyState === WebSocket.CONNECTING)
    ) {
      return;
    }

    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      wsSocket.current = ws;
      wsIsConnected.current = true;
      setServerInfo?.((s) => ({ ...s, connected: true }));
    };

    ws.onmessage = (evt) => {
      try {
        const jsonObj = JSON.parse(evt.data);
        if (jsonObj?.ClassName === "SessionData") {
          setSessionStatus?.("ACTIVE");
          if (jsonObj?.MachineId) {
            setMachineId?.(jsonObj.MachineId);
            setSessionId?.(jsonObj.MachineId);
          }
        }
        if (jsonObj?.ClassName === "SessionPresence") {
          const hasUnreal =
            !!jsonObj?.HasUnreal || Number(jsonObj?.UnrealCount || 0) > 0;
          setUeConnected?.(hasUnreal);
        }
      } catch {}
    };

    ws.onerror = () => {
      wsIsConnected.current = false;
      setUeConnected?.(false);
    };

    ws.onclose = () => {
      wsIsConnected.current = false;
      setServerInfo?.((s) => ({ ...s, connected: false }));
      setUeConnected?.(false);
    };
  }, [
    deviceId,
    enabled,
    serverUrl,
    setMachineId,
    setServerInfo,
    setSessionId,
    setSessionStatus,
    setUeConnected,
  ]);

  const sendCommand = useCallback(
    (messageType, inputData = null) => {
      if (!enabled) return;

      if (
        !wsSocket.current ||
        wsSocket.current.readyState !== WebSocket.OPEN
      ) {
        connectWebSocket();
      }

      if (
        !wsSocket.current ||
        wsSocket.current.readyState !== WebSocket.OPEN
      ) {
        return;
      }

      const customSendData = {
        MachineId: deviceId,
        Platform: "web",
        Custom: inputData ?? {},
      };

      wsSocket.current.send(
        JSON.stringify({
          MessageType: messageType,
          TimeStamp: new Date().toISOString(),
          Data: customSendData,
        })
      );
    },
    [connectWebSocket, deviceId, enabled]
  );

  useEffect(() => {
    if (!enabled) return undefined;
    connectWebSocket();
    return () => {
      try {
        wsSocket.current?.close();
      } catch {}
      wsSocket.current = null;
      wsIsConnected.current = false;
    };
  }, [connectWebSocket, enabled, serverUrl]);

  useEffect(() => {
    if (!enabled) return undefined;
    const intervalId = setInterval(() => {
      sendCommand(MSG_TYPE.Heartbeat);
    }, CONNECTION_POOL_MS);
    return () => clearInterval(intervalId);
  }, [enabled, sendCommand]);

  return {
    wsIsConnected,
    sendCommand,
  };
}
