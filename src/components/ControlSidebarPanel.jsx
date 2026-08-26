import React from "react";

export default function ControlSidebarPanel({
  exportSettings,
  importSettings,
  resetSettings,
  serverUrlDraft,
  setServerUrlDraft,
  serverUrl,
  setServerUrl,
  effectiveUrl,
  serverInfo,
  glchatApiKey,
  setGlchatApiKey,
  hasGlchatApiKey,
  glchatSlug,
  setGlchatSlug,
  deviceIdDraft,
  setDeviceIdDraft,
  deviceId,
  applyDeviceId,
  randomizeDeviceId,
  handleStartSession,
  handleStopSession,
  sessionStatus,
  autoSession,
  setAutoSession,
  gesturesOn,
  setGesturesOn,
  gestureTargets,
  setGestureTargets,
  keepBgOn,
  setKeepBgOn,
}) {
  return (
    <div
      className="left-top-side"
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 12,
        minWidth: 300,
        flex: "0 0 auto",
      }}
    >
      <section className="panel">
        <h3 className="section-title">settings backup</h3>
        <div className="row" style={{ gap: 8 }}>
          <button className="btn" onClick={exportSettings}>
            Export to file
          </button>
          <button className="btn" onClick={importSettings}>
            Import from file
          </button>
          <button className="btn" onClick={resetSettings}>
            Reset all
          </button>
        </div>
      </section>

      <section className="panel">
        <h3 className="section-title">server connection</h3>
        <div className="row" style={{ gap: 8 }}>
          <input
            className="input bigpad"
            placeholder="(same-origin) or http://PC-IP:PORT"
            value={serverUrlDraft}
            onChange={(e) => setServerUrlDraft(e.target.value)}
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck={false}
          />
          <button
            className="btn"
            disabled={serverUrlDraft.trim() === (serverUrl || "")}
            onClick={() => setServerUrl(serverUrlDraft.trim())}
          >
            Apply & reconnect
          </button>
          <button
            className="btn"
            onClick={() => {
              setServerUrl("");
              setServerUrlDraft(window.location.origin);
            }}
          >
            Use same-origin
          </button>
        </div>
        <div className="help" style={{ marginTop: 6 }}>
          Config: {serverUrl && serverUrl.trim() ? serverUrl : "(same-origin)"} |
          Effective: {effectiveUrl}
          <br />
          Status: {serverInfo.connected ? "connected" : "disconnected"}
        </div>
        <div
          className="row"
          style={{ gap: 8, marginTop: 12, alignItems: "center" }}
        >
          <input
            className="input bigpad"
            placeholder="GLChat slug"
            value={glchatSlug}
            onChange={(e) => setGlchatSlug(e.target.value)}
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck={false}
            style={{ maxWidth: 140 }}
          />
          <input
            className="input bigpad"
            type="password"
            placeholder="GLChat API key (X-API-Key)"
            value={glchatApiKey}
            onChange={(e) => setGlchatApiKey(e.target.value)}
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck={false}
          />
          <button
            className="btn"
            disabled={!glchatApiKey}
            onClick={() => setGlchatApiKey("")}
          >
            Clear key
          </button>
        </div>
        <div className="help" style={{ marginTop: 6 }}>
          GLChat key: {hasGlchatApiKey ? "configured" : "not configured"}.
          Slug/key storage: <code>ika:glchat:slug</code> and{" "}
          <code>ika:glchat:key</code>.
        </div>
        <div
          className="row"
          style={{ gap: 8, marginTop: 12, alignItems: "center" }}
        >
          <input
            className="input bigpad"
            placeholder="Device / session ID (match UE)"
            value={deviceIdDraft}
            onChange={(e) => setDeviceIdDraft(e.target.value)}
            autoCorrect="off"
            autoCapitalize="off"
            spellCheck={false}
          />
          <button
            className="btn"
            disabled={!deviceIdDraft.trim() || deviceIdDraft.trim() === deviceId}
            onClick={applyDeviceId}
          >
            Apply device ID
          </button>
          <button className="btn" onClick={randomizeDeviceId}>
            New ID
          </button>
        </div>
        <div className="help" style={{ marginTop: 6 }}>
          Stored as <code>ika:deviceId</code>. Use the same value as the UE
          client to share a single session.
        </div>
      </section>

      <section className="panel">
        <h3 className="section-title">session control</h3>
        <div className="row" style={{ gap: 8 }}>
          <button
            className="btn"
            onClick={handleStartSession}
            disabled={sessionStatus === "ACTIVE"}
          >
            Start session
          </button>
          <button
            className="btn"
            onClick={handleStopSession}
            disabled={sessionStatus !== "ACTIVE"}
          >
            Stop session
          </button>
        </div>
        <div
          className="row"
          style={{ gap: 8, marginTop: 12, alignItems: "center" }}
        >
          <label className="checkbox">
            <input
              type="checkbox"
              checked={autoSession}
              onChange={(e) => setAutoSession(e.target.checked)}
            />
            <span>Auto-manage when server + UE are online</span>
          </label>
        </div>
        <div className="help" style={{ marginTop: 6 }}>
          Status: {sessionStatus}
          <br />
          {autoSession
            ? "Auto-starts when both links are up and stops if either drops."
            : "Auto session is off; use the buttons above."}
        </div>
      </section>

      <section className="panel">
        <h3 className="section-title">performance</h3>
        <div className="row" style={{ gap: 12, alignItems: "center" }}>
          <label className="checkbox">
            <input
              type="checkbox"
              checked={gesturesOn}
              onChange={(e) => setGesturesOn(e.target.checked)}
            />
            <span>Gestures</span>
          </label>
          <div
            className="row"
            style={{ gap: 8, alignItems: "center", marginLeft: 8 }}
          >
            <label className="label" style={{ margin: 0 }}>
              Targets
            </label>
            <select
              className="select"
              value={gestureTargets}
              onChange={(e) =>
                setGestureTargets(parseInt(e.target.value, 10) === 1 ? 1 : 2)
              }
              disabled={!gesturesOn}
              title="Limit gesture tracking to 1 or 2 people"
            >
              <option value={1}>1 person</option>
              <option value={2}>2 people</option>
            </select>
          </div>
        </div>
        <div className="help" style={{ marginTop: 6 }}>
          Turn off on low-power devices (Android/Edge) to improve FPS.
        </div>
        <div
          className="row"
          style={{ gap: 12, alignItems: "center", marginTop: 8 }}
        >
          <label className="checkbox">
            <input
              type="checkbox"
              checked={keepBgOn}
              onChange={(e) => setKeepBgOn(e.target.checked)}
            />
            <span>Keep running in background</span>
          </label>
        </div>
      </section>
    </div>
  );
}
