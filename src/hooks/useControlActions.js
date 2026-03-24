import { useCallback } from "react";
import { uuid } from "../utils/socketUtils";
import {
  exportSettings as exportSettingsUtil,
  importSettings as importSettingsUtil,
  resetSettings as resetSettingsUtil,
} from "../utils/settingsUtils";
import { dayKey } from "../utils/guestMemory";

export function useControlActions({
  deviceIdDraft,
  deviceId,
  setDeviceId,
  guestSeqRef,
  guestMemRef,
  saveGuestMemSafe,
  setVideoId,
  startCamera,
  stopAll,
}) {
  const exportSettings = () => exportSettingsUtil("ika:");
  const importSettings = () => importSettingsUtil("ika:");
  const resetSettings = () => resetSettingsUtil("ika:");

  const applyDeviceId = useCallback(() => {
    const next = (deviceIdDraft || "").trim();
    if (!next || next === deviceId) return;
    setDeviceId(next);
  }, [deviceIdDraft, deviceId, setDeviceId]);

  const randomizeDeviceId = useCallback(() => {
    setDeviceId(uuid());
  }, [setDeviceId]);

  const handleClearGuests = useCallback(() => {
    guestSeqRef.current = 1;
    guestMemRef.current = [];
    saveGuestMemSafe({ day: dayKey(), seq: 1, mem: [] });
  }, [guestMemRef, guestSeqRef, saveGuestMemSafe]);

  const handleVideoChange = useCallback(
    async (e) => {
      const next = e.target.value;
      setVideoId(next);
      try {
        localStorage.setItem("ika:videoId", next);
      } catch {}
      await startCamera(next);
    },
    [setVideoId, startCamera]
  );

  const handleRestartCamera = useCallback(() => {
    startCamera().catch((err) => console.warn("[Cam] restart failed", err));
  }, [startCamera]);

  const handleStopCamera = useCallback(() => {
    stopAll({ reset: true }).catch((err) => console.warn("[Cam] stop failed", err));
  }, [stopAll]);

  return {
    exportSettings,
    importSettings,
    resetSettings,
    applyDeviceId,
    randomizeDeviceId,
    handleClearGuests,
    handleVideoChange,
    handleRestartCamera,
    handleStopCamera,
  };
}
