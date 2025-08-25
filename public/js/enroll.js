// ====== CONFIG ======
const MODEL_URL  = "/models";               // must exist on same origin
const LABELS_URL = "/labels/labels.json";   // optional; merged if present

// ====== Helpers (quantize like your app) ======
function encodeDescFloat32ToU8(descF32){
  const out = new Uint8Array(descF32.length);
  for (let i=0;i<descF32.length;i++){
    const clamped = Math.max(-1, Math.min(1, descF32[i]));
    out[i] = Math.round((clamped + 1) * 127.5);
  }
  return out;
}
function u8ToB64(u8){
  let bin=""; const CHUNK=0x8000;
  for(let i=0;i<u8.length;i+=CHUNK){
    bin += String.fromCharCode.apply(null, u8.subarray(i, i+CHUNK));
  }
  return btoa(bin);
}
function log(el, msg){ el.textContent += (el.textContent ? "\n" : "") + msg; }

// ====== State / DOM ======
const picker      = document.getElementById("picker");
const loadBtn     = document.getElementById("loadModels");
const scanBtn     = document.getElementById("scan");
const buildBtn    = document.getElementById("build");
const dlBtn       = document.getElementById("download");
const peopleBox   = document.getElementById("peopleBox");
const progressEl  = document.getElementById("progress");
const summaryEl   = document.getElementById("summary");
const modelStatus = document.getElementById("modelStatus");
const labelsStatus= document.getElementById("labelsStatus");
const perPersonEl = document.getElementById("perPerson");
const dropZone    = document.getElementById("dropZone");

let files = [];
let grouped = new Map(); // name -> File[]
let selectedNames = new Set();
let modelsReady = false;
let existing = []; // existing labels.json (array)
let merged = [];   // final merged array
let builtByName = new Map(); // name -> Float32Array[] (new)

// ====== Load existing labels.json (if any) ======
async function loadExisting(){
  try {
    const res = await fetch(LABELS_URL, { cache:"no-store" });
    if (!res.ok) { labelsStatus.textContent = "none"; return; }
    const data = await res.json();
    existing = Array.isArray(data)
      ? data
      : Object.entries(data).map(([label, descriptors]) => ({ label, descriptors }));
    labelsStatus.innerHTML = `<span class="badge ok">loaded (${existing.length} people)</span>`;
  } catch {
    labelsStatus.textContent = "none";
  }
}
loadExisting();

async function loadFileAsImage(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload  = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = (e) => { URL.revokeObjectURL(url); reject(e); };
    img.src = url;
  });
}

// --- image scaling for robustness ---
function drawScaled(img, maxSide=1024){
  const w = img.naturalWidth || img.width, h = img.naturalHeight || img.height;
  const s = Math.min(1, maxSide / Math.max(w,h));
  if (s === 1) return img;
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(w*s));
  c.height= Math.max(1, Math.round(h*s));
  c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
  return c;
}

function withTimeout(promise, ms=8000){
  let t; const timeout = new Promise((_,rej)=> t=setTimeout(()=>rej(new Error('timeout')), ms));
  return Promise.race([promise, timeout]).finally(()=>clearTimeout(t));
}
async function safeDetectAllFaces(imgOrCanvas, opts){
  return withTimeout(faceapi.detectAllFaces(imgOrCanvas, opts), 8000);
}
async function safeDetectSingle(imgOrCanvas, opts){
  return withTimeout(
    faceapi.detectSingleFace(imgOrCanvas, opts).withFaceLandmarks().withFaceDescriptor(),
    12000
  );
}

// ====== Load models (self-hosted) ======
const tf = faceapi.tf;

loadBtn.addEventListener("click", async () => {
  loadBtn.disabled = true;
  modelStatus.textContent = "loading...";

  try {
    // wasm → cpu (and webgl when available)
    let ok = false;
    try { await tf.setBackend("webgl"); await tf.ready(); ok = tf.getBackend()==="webgl"; } catch {}
    if (!ok) {
      if (tf?.wasm?.setWasmPaths) tf.wasm.setWasmPaths("/tfjs-backend-wasm/");
      try { await tf.setBackend("wasm"); await tf.ready(); ok = tf.getBackend()==="wasm"; } catch {}
    }
    if (!ok) { await tf.setBackend("cpu"); await tf.ready(); }

    // quick probe to surface 404s
    const base = MODEL_URL.replace(/\/$/, "");
    const must = [
      `${base}/tiny_face_detector_model-weights_manifest.json`,
      `${base}/face_landmark_68_model-weights_manifest.json`,
      `${base}/face_recognition_model-weights_manifest.json`,
    ];
    for (const u of must) {
      const r = await fetch(u, { cache: "no-store" });
      if (!r.ok) throw new Error(`Missing: ${u} (HTTP ${r.status})`);
    }

    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
    ]);

    modelsReady = true;
    modelStatus.innerHTML = `<span class="badge ok">ready (${tf.getBackend()})</span>`;
    scanBtn.disabled = files.length === 0;
  } catch (e) {
    console.error(e);
    modelStatus.innerHTML = `<span class="badge err">failed</span>`;
    alert(`Model load failed: ${e.message || e}`);
  } finally {
    loadBtn.disabled = false;
  }
});

// ====== PERSON LABEL DERIVATION ======
// Prefer an explicit hint; else first path segment; else "Name--file.jpg"; else "Unknown"
function derivePersonLabel(file) {
  if (file.personHint && String(file.personHint).trim()) {
    return String(file.personHint).trim();
  }
  const rel = file.webkitRelativePath || "";
  if (rel.includes("/")) return rel.split("/")[0];
  const m = file.name.match(/^(.+?)--/);
  return (m && m[1]) || "Unknown";
}

// Unified grouper used by all ingest paths
function groupFilesFromArray(collected) {
  files = collected;
  grouped = new Map();
  selectedNames = new Set();

  for (const f of files) {
    const person = derivePersonLabel(f);
    if (!grouped.has(person)) grouped.set(person, []);
    grouped.get(person).push(f);
  }

  // UI checklist
  peopleBox.innerHTML = "";
  for (const [name, list] of grouped.entries()) {
    selectedNames.add(name);
    const id = "p_" + Math.random().toString(36).slice(2);
    const row = document.createElement("label");
    row.className = "person";
    row.innerHTML = `
      <input type="checkbox" id="${id}" checked />
      <span><b>${name}</b></span>
      <span class="badge">${list.length} image(s)</span>
    `;
    const cb = row.querySelector("input");
    cb.addEventListener("change", () => {
      if (cb.checked) selectedNames.add(name);
      else selectedNames.delete(name);
    });
    peopleBox.appendChild(row);
  }

  summaryEl.textContent =
    `Selected parent contains ${grouped.size} person folder(s), ${files.length} file(s).`;
  scanBtn.disabled = !(modelsReady && files.length);
}

// ====== FILE INPUT (single folder – Safari/Firefox compatible) ======
picker.addEventListener("change", () => {
  groupFilesFromArray(Array.from(picker.files || []));
});

// ====== DRAG & DROP OF MULTIPLE FOLDERS (Chrome/Edge/Safari) ======
function setDZDragging(on){ dropZone.classList.toggle("drag", !!on); }

dropZone.addEventListener("dragover", (e)=>{ e.preventDefault(); setDZDragging(true); });
dropZone.addEventListener("dragleave", ()=> setDZDragging(false));

dropZone.addEventListener("drop", async (e) => {
  e.preventDefault();
  setDZDragging(false);
  const items = e.dataTransfer?.items ? Array.from(e.dataTransfer.items) : [];
  if (!items.length) return;

  const out = [];

  // Chrome/Edge: DataTransferItem.webkitGetAsEntry
  const roots = items
    .map(it => (typeof it.webkitGetAsEntry === "function" ? it.webkitGetAsEntry() : null))
    .filter(Boolean);

  if (roots.length) {
    // recursive walk using the old webkitEntry API
    async function walkEntry(entry, prefix = "", depth = 0, rootName = entry.name) {
      return new Promise((resolve) => {
        if (entry.isFile) {
          entry.file((file) => {
            file.webkitRelativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
            const firstSeg = prefix.split("/")[0] || "";
            file.personHint = firstSeg || (depth === 0 ? rootName : "");
            out.push(file);
            resolve();
          }, resolve);
        } else if (entry.isDirectory) {
          const reader = entry.createReader();
          const batch = [];
          const readBatch = () => {
            reader.readEntries(async (entries) => {
              if (!entries.length) {
                for (const child of batch) {
                  await walkEntry(
                    child,
                    prefix ? `${prefix}/${entry.name}` : entry.name,
                    depth + 1,
                    rootName
                  );
                }
                resolve();
              } else {
                batch.push(...entries);
                readBatch();
              }
            }, resolve);
          };
          readBatch();
        } else {
          resolve();
        }
      });
    }

    for (const root of roots) {
      await walkEntry(root, "", 0, root.name);
    }
    groupFilesFromArray(out);
    return;
  }

  // Safari fallback: plain files without entries (best effort grouping)
  const fileList = Array.from(e.dataTransfer.files || []);
  groupFilesFromArray(fileList);
});

// ====== Analyze selection ======
scanBtn.addEventListener("click", async () => {
  if (!modelsReady || !grouped.size) return;
  scanBtn.disabled = true;
  buildBtn.disabled = true;
  progressEl.textContent = "Analyzing images (detect 1 face per photo)...";

  let checked = 0, ok = 0, skippedZero = 0, skippedMany = 0, failed = 0;
  const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.3 });

  try {
    for (const [name, list] of grouped.entries()) {
      if (!selectedNames.has(name)) continue;

      for (const f of list) {
        checked++;
        try {
          const img = await loadFileAsImage(f);
          const scaled = drawScaled(img, 1024);
          const dets = await safeDetectAllFaces(scaled, opts);
          if (Array.isArray(dets)) {
            if (dets.length === 1) ok++;
            else if (dets.length === 0) skippedZero++;
            else skippedMany++;
          } else {
            failed++;
          }
        } catch {
          failed++;
        }
        if (checked % 4 === 0) log(progressEl, `...${checked} checked`);
        await new Promise(r => setTimeout(r, 0));
      }
    }

    log(progressEl, `Done. OK: ${ok}, 0-face: ${skippedZero}, >1-face: ${skippedMany}, failed: ${failed}`);
    buildBtn.disabled = false;
  } catch (e) {
    log(progressEl, `\nError while analyzing: ${e.message || e}`);
  } finally {
    scanBtn.disabled = false;
  }
});

// ====== Build descriptors + per-person breakdown ======
buildBtn.addEventListener("click", async () => {
  if (!modelsReady) return;
  buildBtn.disabled = true;
  dlBtn.disabled = true;
  perPersonEl.innerHTML = "";
  progressEl.textContent = "Building descriptors (128-D)…";

  builtByName = new Map();
  const opts = new faceapi.TinyFaceDetectorOptions({ inputSize: 416, scoreThreshold: 0.3 });

  let processed = 0, added = 0, skipped = 0, failed = 0;

  try {
    // 1) Build descriptors from selected images
    for (const [name, list] of grouped.entries()) {
      if (!selectedNames.has(name)) continue;

      for (const f of list) {
        processed++;
        try {
          const img = await loadFileAsImage(f);
          const scaled = drawScaled(img, 1024);
          const det = await safeDetectSingle(scaled, opts);

          if (det?.descriptor && det.descriptor.length === 128) {
            if (!builtByName.has(name)) builtByName.set(name, []);
            builtByName.get(name).push(new Float32Array(det.descriptor));
            added++;
          } else {
            skipped++;
          }
        } catch {
          failed++;
        }

        if (processed % 4 === 0)
          log(progressEl, `processed ${processed}, added ${added}, skipped ${skipped}, failed ${failed}`);
        await new Promise(r => setTimeout(r, 0));
      }
    }

    log(progressEl, `Descriptors built. processed=${processed} added=${added} skipped=${skipped} failed=${failed}`);

    // 2) Existing map -> Set(b64)
    const existingSets = new Map(); // label -> Set(b64)
    (existing || []).forEach(item => {
      const label = (item.label || item.name || "").trim();
      if (!label) return;

      const set = new Set();
      if (Array.isArray(item.descriptors_b64)) {
        for (const s of item.descriptors_b64) if (typeof s === "string" && s.length) set.add(s);
      } else if (Array.isArray(item.descriptors)) {
        for (const arr of item.descriptors) {
          if (!arr) continue;
          const f32 = arr instanceof Float32Array ? arr : new Float32Array(arr);
          set.add(u8ToB64(encodeDescFloat32ToU8(f32)));
        }
      }
      if (set.size) existingSets.set(label, set);
    });

    // 3) Merge & per-person stats
    const mergedMap = new Map(); // label -> { label, descriptors_b64: [] }
    for (const [label, set] of existingSets.entries()) {
      mergedMap.set(label, { label, descriptors_b64: Array.from(set) });
    }

    const resultRows = [];
    let peopleNew = 0, peopleUpdated = 0, peopleUnchanged = 0;

    for (const [name, descsF32] of builtByName.entries()) {
      const label = name.trim();
      if (!label) continue;

      const attemptedNew = descsF32.length;
      const newB64 = descsF32.map(f32 => u8ToB64(encodeDescFloat32ToU8(f32)));

      const beforeSet   = existingSets.get(label) || new Set();
      const beforeCount = beforeSet.size;

      const uniqNewSet = new Set(newB64);
      const dedupedWithinNew = attemptedNew - uniqNewSet.size;

      const reallyNew = [];
      for (const b of uniqNewSet) if (!beforeSet.has(b)) reallyNew.push(b);

      const addedNew       = reallyNew.length;
      const alreadyExisted = uniqNewSet.size - addedNew;
      const dedupedTotal   = dedupedWithinNew + alreadyExisted;

      const base = mergedMap.get(label) || { label, descriptors_b64: [] };
      base.descriptors_b64.push(...reallyNew);
      base.descriptors_b64 = Array.from(new Set(base.descriptors_b64));
      mergedMap.set(label, base);

      const finalCount = beforeCount + addedNew;

      let status, tagClass;
      if (beforeCount === 0 && addedNew > 0) { status = `new person (+${addedNew})`; tagClass = "ok";   peopleNew++; }
      else if (addedNew > 0)                { status = `updated (+${addedNew})`;    tagClass = "ok";   peopleUpdated++; }
      else                                  { status = `no change (all duplicates)`; tagClass = "warn"; peopleUnchanged++; }

      resultRows.push({ label, beforeCount, attemptedNew, addedNew, dedupedTotal, finalCount, status, tagClass });
    }

    for (const [label, set] of existingSets.entries()) {
      if (builtByName.has(label)) continue;
      resultRows.push({
        label, beforeCount:set.size, attemptedNew:0, addedNew:0, dedupedTotal:0, finalCount:set.size,
        status:"kept (no new photos)", tagClass:"small"
      });
    }

    merged = Array.from(mergedMap.values()).sort((a,b)=> a.label.localeCompare(b.label));

    resultRows.sort((a,b)=> a.label.localeCompare(b.label));
    perPersonEl.innerHTML = resultRows.map(r => `
      <div style="display:flex;justify-content:space-between;gap:12px;padding:6px 8px;border-radius:8px">
        <div><b>${r.label}</b> <span class="badge ${r.tagClass || ""}">${r.status}</span></div>
        <div class="small" style="display:flex;gap:10px">
          <span class="badge">before ${r.beforeCount}</span>
          <span class="badge">attempted +${r.attemptedNew}</span>
          <span class="badge">deduped −${r.dedupedTotal}</span>
          <span class="badge">kept +${r.addedNew}</span>
          <span class="badge">final ${r.finalCount}</span>
        </div>
      </div>
    `).join("");

    const totalAdded = resultRows.reduce((s,r)=> s + r.addedNew, 0);
    summaryEl.textContent =
      `Merged labels: ${merged.length} person(s). ` +
      `New descriptors kept: ${totalAdded}. ` +
      `People → new: ${peopleNew}, updated: ${peopleUpdated}, unchanged: ${peopleUnchanged}.`;
    dlBtn.disabled = merged.length === 0 ? true : false;

  } catch (e) {
    log(progressEl, `\nError while building: ${e.message || e}`);
  } finally {
    buildBtn.disabled = false;
  }
});

// ====== Download merged labels.json ======
dlBtn.addEventListener("click", () => {
  if (!merged || !merged.length) return;
  const blob = new Blob([JSON.stringify(merged, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "labels.json";
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});