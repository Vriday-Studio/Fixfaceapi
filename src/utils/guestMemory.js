export const DEFAULT_GUEST_STORE_KEY = "ika:guestMem.v1";

export const dayKey = (d = new Date()) =>
  d.toLocaleDateString("en-CA", {
    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
  });

export const msToNextMidnight = () => {
  const now = new Date();
  const next = new Date(now);
  next.setHours(24, 0, 0, 0);
  return next - now;
};

export function loadGuestMem(storeKey = DEFAULT_GUEST_STORE_KEY) {
  try {
    const raw = localStorage.getItem(storeKey);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export function saveGuestMem({
  day = dayKey(),
  seq,
  mem,
  storeKey = DEFAULT_GUEST_STORE_KEY,
  encodeDescriptor,
}) {
  try {
    const list = [...mem].sort((a, b) => (b.ts || 0) - (a.ts || 0));
    const items = list.map((m) => ({
      id: m.id,
      ts: m.ts || Date.now(),
      desc: encodeDescriptor(m.desc),
    }));
    const payload = { day, seq, items, savedAt: Date.now() };
    localStorage.setItem(storeKey, JSON.stringify(payload));
  } catch {}
}

export function pruneByRetention(
  data,
  retentionDays,
  dayKeyFn = dayKey
) {
  if (!data) return null;
  if (retentionDays <= 0) return null;
  if (retentionDays === 1) {
    if (data.day !== dayKeyFn()) return null;
    return data;
  }
  const cutoff = Date.now() - retentionDays * 86_400_000;
  data.items = (data.items || []).filter((it) => (it.ts || 0) >= cutoff);
  return data;
}
