import React from "react";

export default function StatusOverviewPanel({
  locationLabel,
  clock,
  weatherLabel,
  backend,
  ready,
  camLive,
  serverInfo,
  ueConnected,
  deviceId,
  lastSent,
  lastHttp,
  totals,
  useSocketBridge,
  onClearGuests,
}) {
  return (
    <div className="panel compact">
      <div className="statgrid">
        <div className="block">
          <div className="block-title">Environment</div>
          <div className="kv">
            <b>Location:</b> {locationLabel}
          </div>
          <div className="kv">
            <b>Time:</b> {clock.toLocaleTimeString()}
          </div>
          <div className="kv">
            <b>Weather:</b> {weatherLabel}
          </div>
          <div className="kv">
            <b>Backend:</b> {backend}
          </div>
          <div className="kv">
            <b>Models:</b> {ready ? "loaded" : "loading..."}
          </div>
        </div>

        <div className="block">
          <div className="block-title">Live status</div>

          <div className="kv">
            <span className={`dot ${camLive ? "ok" : "err"}`} />
            <b>Cam:</b>&nbsp;{camLive ? "LIVE" : "IDLE"}
          </div>

          <div className="kv">
            <span className={`dot ${serverInfo.connected ? "ok" : "err"}`} />
            <b>Server:</b>&nbsp;
            {serverInfo.connected ? "connected" : "disconnected"}
          </div>

          <div className="kv">
            <span className={`dot ${ueConnected ? "ok" : "err"}`} />
            <b>UE link:</b>&nbsp;
            {ueConnected ? "connected" : "waiting"}
          </div>

          {serverInfo.model || serverInfo.tts ? (
            <div className="kv muted small">
              {serverInfo.model ? <>Model: {serverInfo.model}</> : null}
              {serverInfo.model && serverInfo.tts ? " | " : null}
              {serverInfo.tts ? <>TTS: {serverInfo.tts}</> : null}
            </div>
          ) : null}

          <div className="kv">
            <b>Device:</b>&nbsp;{deviceId.slice(0, 8)}...
            <span className="muted">
              &nbsp;
              {serverInfo.boundDeviceId
                ? `(bound ${String(serverInfo.boundDeviceId).slice(0, 8)}...)`
                : "(not bound)"}
            </span>
          </div>
        </div>

        <div className="block">
          <div className="block-title">Traffic</div>

          <div className="kv chiprow">
            <b>Last:</b>
            <span className="chip">start {lastSent.start}</span>
            <span className="chip">snap {lastSent.snapshot}</span>
            <span className="chip">stop {lastSent.stop}</span>
          </div>

          <div className="kv chiprow">
            <b>HTTP:</b>
            <span className="chip">start {lastHttp.start || "-"}</span>
            <span className="chip">snap {lastHttp.snapshot || "-"}</span>
            <span className="chip">stop {lastHttp.stop || "-"}</span>
            <span className="muted">
              {useSocketBridge ? " (socket bridge)" : ""}
            </span>
          </div>

          <div className="kv chiprow">
            <b>Faces:</b>
            <span className="chip">total {totals.all}</span>
            <span className="chip">green {totals.green}</span>
            <span className="chip">red {totals.red}</span>
          </div>

          <div className="kv">
            <button
              className="btn small full"
              title="Clear in-browser guest memory"
              onClick={onClearGuests}
            >
              clear guests
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
