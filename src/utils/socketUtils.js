export function normalizeServerUrl(u) {
  if (!u) return undefined;
  let s = String(u || "").trim();
  if (!s) return undefined;
  if (/^ws(s)?:\/\//i.test(s)) s = s.replace(/^ws/i, "http");
  if (!/^https?:\/\//i.test(s)) s = "http://" + s;
  return s.replace(/\/+$/, "");
}

export function uuid() {
  return (
    crypto?.randomUUID?.() ||
    Math.random().toString(36).slice(2) + Date.now().toString(36)
  );
}
