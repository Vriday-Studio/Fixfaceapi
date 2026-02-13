export function groupSignature(people) {
  const ids = people
    .map((p) => (p.name || p.gid || "").trim())
    .filter(Boolean)
    .sort();
  let s = ids.join("|");
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h << 5) - h + s.charCodeAt(i);
    h |= 0;
  }
  return ids.length ? String(h) : "";
}

export function maybeRotateSession({
  sig,
  people,
  now,
  refs,
  sessionId,
  deviceId,
  socket,
  setSessionId,
  uuid,
  stableMs,
  cooldownMs,
}) {
  const prev = refs.groupSigRef.current;

  if (sig !== prev) {
    refs.groupSigRef.current = sig;
    refs.groupStableSinceRef.current = now;
    return;
  }
  if (!sig) return;

  const stableFor = now - (refs.groupStableSinceRef.current || 0);
  const sinceLast = now - (refs.lastRotateRef.current || 0);

  if (stableFor >= stableMs && sinceLast >= cooldownMs) {
    const oldId = sessionId || "web-" + deviceId;
    const newId = uuid();

    try {
      socket?.emit?.("rotate_session", {
        oldSessionId: oldId,
        newSessionId: newId,
        at: Date.now(),
        people: people.map((p) => ({
          name: p.name || null,
          gid: p.gid || null,
        })),
      });
      socket?.emit?.("close_session", { sessionId: oldId });
    } catch {}

    setSessionId(newId);
    refs.lastRotateRef.current = now;
  }
}
