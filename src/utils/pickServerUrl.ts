export async function pickServerUrl(preferred?: string): Promise<string | null> {
  const saved = (preferred || localStorage.getItem("serverUrl") || "").trim();
  const fromEnv = (import.meta as any).env?.VITE_SERVER_URL as string | undefined;

  const candidates = [
    saved || undefined,
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://[::1]:3000",
    fromEnv || undefined
  ].filter(Boolean) as string[];

  for (const base of candidates) {
    const url = base.replace(/\/$/, "");
    try {
      const r = await fetch(`${url}/health`, { method: "GET", cache: "no-store", mode: "cors" });
      if (r.ok) return url;
    } catch {}
  }
  return saved || null;
}

// Optional: derive ws URL for Socket.IO
export function httpToWs(base: string): string {
  if (!base) return "";
  return base.replace(/^http:/i, "ws:").replace(/^https:/i, "wss:");
}