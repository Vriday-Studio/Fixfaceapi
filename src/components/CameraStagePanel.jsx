import React from "react";

export default function CameraStagePanel({
  videoRef,
  canvasRef,
  captions,
  lastText,
}) {
  return (
    <>
      <div className="stage">
        <video ref={videoRef} autoPlay muted playsInline />
        <canvas ref={canvasRef} />
      </div>

      {captions && lastText ? (
        <div
          aria-live="polite"
          className="captions"
          style={{
            marginTop: 8,
            background: "rgba(0,0,0,0.55)",
            padding: "10px 12px",
            borderRadius: 10,
            lineHeight: 1.35,
          }}
        >
          {lastText}
        </div>
      ) : null}
    </>
  );
}
