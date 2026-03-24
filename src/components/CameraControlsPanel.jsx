import React from "react";

export default function CameraControlsPanel({
  videoId,
  videoDevs,
  onVideoChange,
  onRestartCamera,
  onStopCamera,
  greenMaxM,
  setGreenMaxM,
  defaultGreenMaxM,
  redCutoffM,
  setRedCutoffM,
  defaultRedCutoffM,
  showAlign,
  setShowAlign,
  gameModeOn,
  setGameModeOn,
  calibDistanceM,
  setCalibDistanceM,
  runCalCountdown,
  fovHdeg,
  setFovHdeg,
  fovVdeg,
  setFovVdeg,
  panOffsetDeg,
  setPanOffsetDeg,
  tiltOffsetDeg,
  setTiltOffsetDeg,
  wNear,
  setWNear,
  wCenter,
  setWCenter,
  wMouth,
  setWMouth,
}) {
  return (
    <>
      <div className="panel">
        <label className="label">Camera</label>
        <select
          className="select big"
          value={videoId}
          onChange={onVideoChange}
        >
          <option value="">(Default)</option>
          {videoDevs.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label || "Camera"}
            </option>
          ))}
        </select>
        <div className="row" style={{ gap: 12, marginTop: 12, flexWrap: "wrap" }}>
          <button className="btn" onClick={onRestartCamera}>
            Restart camera
          </button>
          <button className="btn" onClick={onStopCamera}>
            Stop camera
          </button>
        </div>
        <div className="help" style={{ marginTop: 6 }}>
          Use restart after plugging in a new webcam or if autoplay was blocked
          by the browser.
        </div>
      </div>

      <div className="panel">
        <div className="inline-controls">
          <b>Green zone distance</b>
          <input
            className="range"
            type="range"
            min="0.3"
            max="2.0"
            step="0.05"
            value={greenMaxM}
            onChange={(e) => setGreenMaxM(Number(e.target.value))}
            aria-label="Green zone distance in meters"
          />
          <span className="chip">{greenMaxM.toFixed(2)} m</span>
          <button className="btn" onClick={() => setGreenMaxM(defaultGreenMaxM)}>
            reset
          </button>
          <button
            className="btn"
            onClick={() => setGreenMaxM((v) => Math.max(0.3, +(v - 0.1).toFixed(2)))}
          >
            -0.1
          </button>
          <button
            className="btn"
            onClick={() => setGreenMaxM((v) => Math.min(2.0, +(v + 0.1).toFixed(2)))}
          >
            +0.1
          </button>
        </div>

        <div className="inline-controls" style={{ marginTop: 8 }}>
          <b>Red zone distance</b>
          <input
            className="range"
            type="range"
            min="1.0"
            max="4.0"
            step="0.1"
            value={redCutoffM}
            onChange={(e) => setRedCutoffM(Number(e.target.value))}
            aria-label="Red zone cutoff in meters"
          />
          <span className="chip">{redCutoffM.toFixed(1)} m</span>
          <button className="btn" onClick={() => setRedCutoffM(defaultRedCutoffM)}>
            reset
          </button>
          <button
            className="btn"
            onClick={() => setRedCutoffM((v) => Math.max(1.0, +(v - 0.1).toFixed(1)))}
          >
            -0.1
          </button>
          <button
            className="btn"
            onClick={() => setRedCutoffM((v) => Math.min(4.0, +(v + 0.1).toFixed(1)))}
          >
            +0.1
          </button>
        </div>
      </div>

      <div className="row" style={{ gap: 12, alignItems: "center" }}>
        <label className="checkbox">
          <input
            type="checkbox"
            checked={showAlign}
            onChange={(e) => setShowAlign(e.target.checked)}
          />
          <span>Show alignment overlay</span>
        </label>

        <label className="checkbox" style={{ marginLeft: 12 }}>
          <input
            type="checkbox"
            checked={gameModeOn}
            onChange={(e) => setGameModeOn(e.target.checked)}
          />
          <span>Game mode (RPS)</span>
        </label>

        <div className="kv" style={{ gap: 6 }}>
          <b>Calib distance:</b>
          <input
            className="input"
            type="number"
            step="0.05"
            min="0.3"
            max="3.0"
            value={calibDistanceM}
            onChange={(e) => setCalibDistanceM(Number(e.target.value))}
            style={{ width: 90 }}
            aria-label="Calibration distance (meters)"
          />
          <span className="muted">m</span>
        </div>

        <button className="btn" onClick={runCalCountdown}>
          Calibrate camera (3-2-1)
        </button>
      </div>

      <div className="panel" style={{ marginTop: 10 }}>
        <h3 className="section-title" style={{ marginTop: 0 }}>
          camera alignment
        </h3>

        <label className="label">Horizontal FOV ( deg)</label>
        <input
          className="range"
          type="range"
          min="40"
          max="110"
          step="1"
          value={fovHdeg}
          onChange={(e) => setFovHdeg(Number(e.target.value))}
        />
        <div className="help">{Math.round(fovHdeg)} deg</div>

        <label className="label">Vertical FOV ( deg)</label>
        <input
          className="range"
          type="range"
          min="25"
          max="90"
          step="1"
          value={fovVdeg}
          onChange={(e) => setFovVdeg(Number(e.target.value))}
        />
        <div className="help">{Math.round(fovVdeg)} deg</div>

        <div className="row" style={{ gap: 16 }}>
          <div className="flex1">
            <label className="label">Pan offset ( deg)</label>
            <input
              className="range"
              type="range"
              min="-30"
              max="30"
              step="0.5"
              value={panOffsetDeg}
              onChange={(e) => setPanOffsetDeg(Number(e.target.value))}
            />
            <div className="help">{panOffsetDeg.toFixed(1)} deg</div>
          </div>
          <div className="flex1">
            <label className="label">Tilt offset ( deg)</label>
            <input
              className="range"
              type="range"
              min="-30"
              max="30"
              step="0.5"
              value={tiltOffsetDeg}
              onChange={(e) => setTiltOffsetDeg(Number(e.target.value))}
            />
            <div className="help">{tiltOffsetDeg.toFixed(1)} deg</div>
          </div>
        </div>

        <div className="row" style={{ gap: 8, marginTop: 8 }}>
          <button
            className="btn"
            onClick={() => {
              setPanOffsetDeg(0);
              setTiltOffsetDeg(0);
            }}
          >
            reset offsets
          </button>
          <span className="help">Tip: click a face on video to auto-zero.</span>
        </div>

        <div className="divider" />

        <h4 className="section-title">focus weights</h4>

        <label className="label">Closeness</label>
        <input
          className="range"
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={wNear}
          onChange={(e) => setWNear(Number(e.target.value))}
        />

        <label className="label">Centeredness</label>
        <input
          className="range"
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={wCenter}
          onChange={(e) => setWCenter(Number(e.target.value))}
        />

        <label className="label">Mouth activity</label>
        <input
          className="range"
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={wMouth}
          onChange={(e) => setWMouth(Number(e.target.value))}
        />
      </div>
    </>
  );
}
