export function exportSettings(prefix = "ika:") {
  try {
    const bag = {};
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith(prefix)) {
        bag[k] = localStorage.getItem(k);
      }
    }
    const json = JSON.stringify(bag, null, 2);
    const blob = new Blob([json], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const ts = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const filename = `${prefix.replace(":", "")}-settings-${ts.getFullYear()}${pad(
      ts.getMonth() + 1
    )}${pad(ts.getDate())}-${pad(ts.getHours())}${pad(ts.getMinutes())}.json`;
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  } catch {
    alert("Failed to export settings");
  }
}

export function importSettings(prefix = "ika:") {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "application/json";
  input.style.display = "none";
  document.body.appendChild(input);
  input.addEventListener(
    "change",
    async () => {
      const file = input.files && input.files[0];
      document.body.removeChild(input);
      if (!file) return;
      try {
        const text = await file.text();
        const bag = JSON.parse(text);
        Object.entries(bag).forEach(([k, v]) => {
          try {
            if (!k.startsWith(prefix)) return;
            localStorage.setItem(k, v ?? "");
          } catch {}
        });
        window.location.reload();
      } catch {
        alert("Invalid or unreadable settings file");
      }
    },
    { once: true }
  );
  input.click();
}

export function resetSettings(prefix = "ika:") {
  if (!confirm("Reset all saved settings?")) return;
  try {
    const keysToRemove = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith(prefix)) keysToRemove.push(key);
    }
    keysToRemove.forEach((k) => {
      try {
        localStorage.removeItem(k);
      } catch {}
    });
  } catch {}
  window.location.reload();
}
