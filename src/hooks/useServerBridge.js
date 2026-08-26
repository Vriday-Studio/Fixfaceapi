import { useCallback, useMemo } from "react";
import { uuid } from "../utils/socketUtils";
import { buildGlchatAuthPayload } from "../utils/glchatAuth";
import { MSG_TYPE } from "./useDirectWebSocket";

function buildSpeechSafeInstruction(baseInstruction) {
  const base = (baseInstruction || "").trim();
  const speechRules = [
    "You are speaking out loud to visitors.",
    "Do not say raw URLs, links, email addresses, markdown, headings, or bullet symbols.",
    "If a source includes a link, summarize what it is for instead of reading it aloud.",
    "For reservations or web pages, say that the reservation link or page can be shown on screen.",
    "Use short, natural spoken sentences suitable for TTS.",
  ].join(" ");

  return base ? `${base}\n\n${speechRules}` : speechRules;
}

export function useServerBridge({
  socketRef,
  sendWsCommand,
  deviceId,
  sessionId,
  bump,
  setSessionStatus,
  setSessionId,
  modelQuick,
  geminiVoiceQuick,
  languageCodeQuick,
  systemInstruction,
  ttsProviderQuick,
  enableAffectiveQuick,
  temperatureQuick,
  proactiveAudioQuick,
  captions,
  functionCallingQuick,
  autoFunctionResponseQuick,
  groundingQuick,
  glchatApiKey,
  glchatSlug,
}) {
  const glchatAuthPayload = useMemo(
    () => buildGlchatAuthPayload(glchatApiKey, glchatSlug),
    [glchatApiKey, glchatSlug]
  );

  const createServerSession = useCallback(
    (preset = {}) => {
      sendWsCommand(MSG_TYPE.SessionStart, {
        ...preset,
        ...glchatAuthPayload,
        deviceId,
        sessionId: sessionId || "web-" + deviceId,
      });
      bump("start");
      setSessionStatus("ACTIVE");
      setSessionId((id) => id || uuid());
    },
    [
      bump,
      deviceId,
      glchatAuthPayload,
      sendWsCommand,
      sessionId,
      setSessionId,
      setSessionStatus,
    ]
  );

  const updateServerSettings = useCallback((fields = {}) => {
    socketRef.current?.emit?.("update_settings", {
      ...fields,
      ...glchatAuthPayload,
    });
  }, [glchatAuthPayload, socketRef]);

  const sendTextPrompt = useCallback((text) => {
    if (!text) return;
    socketRef.current?.emit?.("send_text_prompt", {
      text,
      ...glchatAuthPayload,
    });
  }, [glchatAuthPayload, socketRef]);

  const emitCrowdStatus = useCallback(
    (payload) => {
      socketRef.current?.emit?.("crowd_status", {
        deviceId,
        sessionId: sessionId || "web-" + deviceId,
        ...payload,
      });
      bump("snapshot");
    },
    [bump, deviceId, sessionId, socketRef]
  );

  const server = useMemo(
    () => ({
      createSession: createServerSession,
      updateSettings: updateServerSettings,
      sendText: sendTextPrompt,
      crowdStatus: emitCrowdStatus,
    }),
    [createServerSession, emitCrowdStatus, sendTextPrompt, updateServerSettings]
  );

  const onCreateSession = useCallback(() => {
    server.createSession({
      model: modelQuick,
      voice: geminiVoiceQuick,
      language_code: languageCodeQuick,
      system_instruction: buildSpeechSafeInstruction(systemInstruction),
      tts_provider: ttsProviderQuick,
      response_modalities:
        ttsProviderQuick === "elevenlabs" ? ["TEXT"] : ["AUDIO", "TEXT"],
      enable_affective_dialog: enableAffectiveQuick,
      temperature: temperatureQuick,
      proactive_audio: proactiveAudioQuick,
      transcribe_user_audio: true,
      files_to_upload: null,
      eleven_model:
        localStorage.getItem("ika:11labs:model") || "eleven_turbo_v2_5",
      eleven_voice_id: localStorage.getItem("ika:11labs:voiceId") || "",
      eleven_api_key: localStorage.getItem("ika:11labs:key") || undefined,
      eleven_output_format: "pcm_24000",
    });
  }, [
    server,
    modelQuick,
    geminiVoiceQuick,
    languageCodeQuick,
    systemInstruction,
    ttsProviderQuick,
    enableAffectiveQuick,
    temperatureQuick,
    proactiveAudioQuick,
  ]);

  const onHotUpdate = useCallback(() => {
    server.updateSettings({
      temperature: temperatureQuick,
      captions,
      enable_affective_dialog: enableAffectiveQuick,
      proactive_audio: proactiveAudioQuick,
      function_calling: functionCallingQuick,
      auto_function_response: autoFunctionResponseQuick,
      grounding: groundingQuick,
    });
  }, [
    server,
    temperatureQuick,
    captions,
    enableAffectiveQuick,
    proactiveAudioQuick,
    functionCallingQuick,
    autoFunctionResponseQuick,
    groundingQuick,
  ]);

  return {
    server,
    onCreateSession,
    onHotUpdate,
  };
}
