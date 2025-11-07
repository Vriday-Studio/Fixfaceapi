


// socketBridge.js: Reusable Socket.IO bridge for React apps
// Usage: import { createSocketBridge } from './socketBridge';

import { useRef, useEffect, useCallback } from "react";
import { io } from "socket.io-client";

// Helper to normalize server URL (copy from App.jsx if needed)
function normalizeServerUrl(u) {
  if (!u) return undefined;
  let s = String(u || "").trim();
  if (!s) return undefined;
  if (/^ws(s)?:\/\//i.test(s)) s = s.replace(/^ws/, "http");
  if (!/^https?:\/\//i.test(s)) s = "http://" + s;
  return s.replace(/\/+$/, "");
}

/**
 * Creates a socket bridge and returns helpers for emitting events.
 * @param {Object} params
 * @param {string} params.serverUrl - The server URL (empty for same-origin)
 * @param {string} params.deviceId - Device ID
 * @param {string} params.sessionId - Session ID
 * @param {function} params.bump - Function to call on certain events
 * @param {function} params.setSessionStatus - Setter for session status
 * @param {function} params.setSessionId - Setter for session ID
 * @param {function} params.uuid - UUID generator
 * @param {boolean} [params.USE_SOCKET_SERVER=true] - Whether to use socket server
 * @returns {object} { socketRef, createServerSession, updateServerSettings, sendTextPrompt, emitCrowdStatus }
 */
export function createSocketBridge({
  serverUrl,
  deviceId,
  sessionId,
  bump = () => {},
  setSessionStatus = () => {},
  setSessionId = () => {},
  uuid = () => Date.now().toString(36),
  USE_SOCKET_SERVER = true,
}) {
  const socketRef = useRef(null);

  // Socket lifecycle
  useEffect(() => {
    if (!USE_SOCKET_SERVER) return;
    const url = normalizeServerUrl(serverUrl);
    const isHttpsPage = window.location.protocol === "https:";
    const isLoopback =
      /^http:\/\/(localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$/i.test(url || "");
    const transports =
      isHttpsPage && isLoopback ? ["websocket"] : ["polling", "websocket"];
    const socket = io(url, {
      transports,
      path: "/socket.io",
      autoConnect: true,
      reconnection: true,
      reconnectionDelay: 500,
      reconnectionAttempts: Infinity,
      withCredentials: false,
      timeout: 12000,
      rememberUpgrade: true,
      forceNew: true,
    });
    socketRef.current = socket;
    // TODO: Add socket event handlers as needed
    return () => {
      try { socket.disconnect(); } catch {}
      socketRef.current = null;
    };
  }, [serverUrl, deviceId]);

  // Auto-reconnect on Network/Visibility
  useEffect(() => {
    const onOnline = () => {
      try {
        if (socketRef.current && !socketRef.current.connected)
          socketRef.current.connect();
      } catch {}
    };
    const onVisible = () => {
      if (!document.hidden) {
        try {
          if (socketRef.current && !socketRef.current.connected)
            socketRef.current.connect();
        } catch {}
      }
    };
    window.addEventListener("online", onOnline);
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.removeEventListener("online", onOnline);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, []);

  // Server helpers (emit over socket)
  const createServerSession = useCallback(
    (preset) => {
      const s = socketRef.current;
      if (!s) return;
      s.emit("create_session", { ...preset });
      bump("start");
      setSessionStatus("ACTIVE");
      setSessionId((id) => id || uuid());
    },
    [deviceId, bump, setSessionStatus, setSessionId, uuid]
  );
  const updateServerSettings = useCallback((fields) => {
    const s = socketRef.current;
    if (!s) return;
    s.emit("update_settings", fields || {});
  }, []);
  const sendTextPrompt = useCallback((text) => {
    const s = socketRef.current;
    if (!s || !text) return;
    s.emit("send_text_prompt", { text });
  }, []);
  const emitCrowdStatus = useCallback(
    (payload) => {
      const s = socketRef.current;
      if (!s) return;
      s.emit("crowd_status", {
        deviceId,
        sessionId: sessionId || "web-" + deviceId,
        ...payload,
      });
      bump("snapshot");
    },
    [sessionId, deviceId, bump]
  );

  return {
    socketRef,
    createServerSession,
    updateServerSettings,
    sendTextPrompt,
    emitCrowdStatus,
  };
}