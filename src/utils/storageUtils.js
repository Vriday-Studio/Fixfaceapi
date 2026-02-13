export function getStoredNumber(key) {
  try {
    const v = localStorage.getItem(key);
    if (v == null) return null;
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : null;
  } catch {
    return null;
  }
}

export function encodeDescFloat32ToU8(descF32) {
  try {
    return new Uint8Array(new Float32Array(descF32).buffer);
  } catch {
    return new Uint8Array();
  }
}

export function decodeDescU8ToFloat32(u8) {
  try {
    return new Float32Array(u8.buffer);
  } catch {
    return new Float32Array();
  }
}

export function u8ToB64(u8) {
  try {
    let binary = "";
    const len = u8.byteLength;
    for (let i = 0; i < len; i++) binary += String.fromCharCode(u8[i]);
    return btoa(binary);
  } catch {
    return "";
  }
}

export function b64ToU8(b64) {
  try {
    const binary = atob(b64);
    const len = binary.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
    return bytes;
  } catch {
    return new Uint8Array();
  }
}
