import { useCallback } from "react";

export function useCamera({
  videoRef,
  canvasRef,
  camRef,
  tinyOptsRef,
  pickInputSize,
  setVideoDevs,
  setVideoId,
  fovHdeg,
  fovVdeg,
  camFxRef,
  camFyRef,
  setFocalPx,
  setFovHdeg,
  setFovVdeg,
  setPanOffsetDeg,
  setTiltOffsetDeg,
  calibDistanceM,
  calibMsgRef,
  sleep,
  faceapi,
  focalFromFov,
  anglesFromPixel,
  faceWidthM,
  rad,
  onCameraGone,
}) {
  const isCamLive = useCallback(() => {
    const s = camRef.current?.stream;
    if (!s) return false;
    const tracks = s.getVideoTracks?.() || [];
    if (!tracks.length) return false;
    return tracks.some((t) => t.readyState === "live" && t.enabled !== false);
  }, [camRef]);

  const startCamera = useCallback(
    async (id) => {
      try {
        camRef.current.stream?.getTracks()?.forEach((t) => t.stop());
      } catch {}

      let stream = null;
      if (id) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              deviceId: { exact: id },
              width: { ideal: 1920 },
              height: { ideal: 1080 },
              frameRate: { ideal: 30 },
            },
            audio: false,
          });
        } catch (e) {
          console.warn("[Cam] chosen device failed; fallback:", e.name);
          if (e.name === "OverconstrainedError" || e.name === "NotFoundError") {
            setVideoId("");
            try {
              localStorage.setItem("ika:videoId", "");
            } catch {}
          }
        }
      }

      if (!stream) {
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            video: {
              facingMode: "user",
              width: { ideal: 1920 },
              height: { ideal: 1080 },
              frameRate: { ideal: 30 },
            },
            audio: false,
          });
        } catch (e) {
          console.error("[Cam] default device failed:", e);
          alert("Camera not available or permission denied.");
          return;
        }
      }

      camRef.current.stream = stream;

      try {
        const onCamGone = () => onCameraGone?.({ reset: true });
        const vTracks = stream.getVideoTracks();
        const handleMute = () => {
          setTimeout(() => {
            const live = isCamLive();
            const anyMuted = vTracks.some((tr) => tr.muted);
            if (!live && anyMuted) onCamGone();
          }, 1200);
        };
        vTracks.forEach((t) => {
          t.addEventListener("ended", onCamGone);
          t.onended = onCamGone;
          t.addEventListener("mute", handleMute);
          t.addEventListener("unmute", () => {});
        });
        if (typeof stream.addEventListener === "function") {
          stream.addEventListener("inactive", onCamGone);
        }
      } catch {}

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          const w = videoRef.current.videoWidth || 1280;
          setFocalPx?.(w >= 1920 ? 1350 : 900);
          camFxRef.current = focalFromFov(
            videoRef.current.videoWidth || 1280,
            fovHdeg
          );
          camFyRef.current = focalFromFov(
            videoRef.current.videoHeight || 720,
            fovVdeg
          );
          tinyOptsRef.current = new faceapi.TinyFaceDetectorOptions({
            inputSize: pickInputSize(w),
            scoreThreshold: 0.4,
          });
        };
      }

      if (videoRef.current) {
        const v = videoRef.current;
        v.srcObject = stream;
        v.muted = true;
        v.playsInline = true;
        v.autoplay = true;

        try {
          await v.play();
        } catch (e) {
          console.warn("[Cam] video.play() blocked until user gesture:", e);
        }

        v.onloadedmetadata = () => {
          const w = v.videoWidth || 1280;
          setFocalPx?.(w >= 1920 ? 1350 : 900);
          camFxRef.current = focalFromFov(v.videoWidth || 1280, fovHdeg);
          camFyRef.current = focalFromFov(v.videoHeight || 720, fovVdeg);
          tinyOptsRef.current = new faceapi.TinyFaceDetectorOptions({
            inputSize: pickInputSize(w),
            scoreThreshold: 0.4,
          });
        };
      }

      try {
        const list = await navigator.mediaDevices.enumerateDevices();
        setVideoDevs(list.filter((d) => d.kind === "videoinput"));
      } catch {}
    },
    [
      camRef,
      camFxRef,
      camFyRef,
      faceapi,
      focalFromFov,
      fovHdeg,
      fovVdeg,
      isCamLive,
      onCameraGone,
      pickInputSize,
      setVideoDevs,
      setVideoId,
      tinyOptsRef,
      videoRef,
    ]
  );

  const delay = useCallback(
    (ms) => (sleep ? sleep(ms) : new Promise((r) => setTimeout(r, ms))),
    [sleep]
  );

  const calibrateCameraOneClick = useCallback(async () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return;

    calibMsgRef.current = "Stand still... Calibrating";
    const W = canvas.width,
      H = canvas.height;
    const samples = [];
    const N = 8;

    for (let i = 0; i < N; i++) {
      await delay(120);
      const dets = await faceapi
        .detectAllFaces(video, tinyOptsRef.current)
        .withFaceLandmarks();

      if (!dets?.length) continue;

      const det = faceapi.resizeResults(dets[0], { width: W, height: H });
      const box = det.detection.box;

      if (!box?.width) continue;
      const fxEst = (box.width * calibDistanceM) / faceWidthM;

      const cx = box.x + box.width * 0.5;
      const cy = box.y + box.height * 0.45;

      const { yaw, pitch } = anglesFromPixel(
        cx,
        cy,
        camFxRef.current,
        camFyRef.current,
        W * 0.5,
        H * 0.5
      );

      samples.push({
        fx: fxEst,
        yawDeg: yaw * rad,
        pitchDeg: pitch * rad,
      });
    }
    calibMsgRef.current = "";

    if (!samples.length) return;

    const median = (arr) => {
      const a = [...arr].sort((x, y) => x - y);
      return a[Math.floor(a.length / 2)];
    };

    const fxMed = median(samples.map((s) => s.fx));
    const yawMed = median(samples.map((s) => s.yawDeg));
    const pitchMed = median(samples.map((s) => s.pitchDeg));

    camFxRef.current = fxMed;
    const fovH = 2 * Math.atan(W / 2 / fxMed) * rad;
    setFovHdeg(+fovH.toFixed(1));

    const fy = fxMed * (H / W);
    camFyRef.current = fy;
    const fovV = 2 * Math.atan(H / 2 / fy) * rad;
    setFovVdeg(+fovV.toFixed(1));

    setPanOffsetDeg((p) => p - yawMed);
    setTiltOffsetDeg((t) => t - pitchMed);
  }, [
    anglesFromPixel,
    calibDistanceM,
    calibMsgRef,
    camFxRef,
    camFyRef,
    canvasRef,
    faceapi,
    faceWidthM,
    rad,
    setFovHdeg,
    setFovVdeg,
    setPanOffsetDeg,
    setTiltOffsetDeg,
    delay,
    tinyOptsRef,
    videoRef,
  ]);

  const runCalCountdown = useCallback(async () => {
    for (const n of [3, 2, 1]) {
      calibMsgRef.current = `Calibration in ${n}... Stand on the ${calibDistanceM.toFixed(
        2
      )} m mark`;
      await delay(500);
    }
    calibMsgRef.current = "Calibrating...";
    await calibrateCameraOneClick();
    calibMsgRef.current = "Done!";
    await delay(600);
    calibMsgRef.current = "";
  }, [calibDistanceM, calibMsgRef, calibrateCameraOneClick, delay]);

  return {
    startCamera,
    calibrateCameraOneClick,
    runCalCountdown,
    isCamLive,
  };
}
