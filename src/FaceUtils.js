// === CONFIG ===
const BASE = "./"; // adjust if your models/labels folder is in another path
const LABELS_MANIFEST_URL = `${BASE}labels/labels.json`;
const SCORE_TH = 0.5;              // detection threshold
const KNOWN_MATCH_THRESHOLD = 0.5; // lower = stricter
const ENABLE_EPHEMERAL_RECOG = true;
const ENABLE_KNOWN_RECOG = true;

// === LOAD MODELS ===
async function loadModels() {
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(`${BASE}models`),
    faceapi.nets.faceLandmark68Net.loadFromUri(`${BASE}models`),
    faceapi.nets.faceExpressionNet.loadFromUri(`${BASE}models`),
    faceapi.nets.ageGenderNet.loadFromUri(`${BASE}models`),
    (ENABLE_EPHEMERAL_RECOG || ENABLE_KNOWN_RECOG)
      ? faceapi.nets.faceRecognitionNet.loadFromUri(`${BASE}models`)
      : Promise.resolve(),
  ]);

  // Pick backend
  const FORCE_CPU = /forceCpu=1/.test(window.location.search);
  try {
    if (FORCE_CPU) throw new Error("Forced CPU via ?forceCpu=1");
    await faceapi.tf.setBackend("webgl");
    await faceapi.tf.ready();
    console.log("✅ Using WebGL backend");
  } catch (e) {
    console.warn("⚠️ WebGL unavailable, falling back to CPU:", e.message || e);
    await faceapi.tf.setBackend("cpu");
    await faceapi.tf.ready();
    console.log("✅ Using CPU backend");
  }
}

// === LOAD KNOWN FACES ===
async function loadKnownMatcher() {
  try {
    const res = await fetch(LABELS_MANIFEST_URL, { cache: "no-store" });
    if (!res.ok) throw new Error("labels.json not found");
    const data = await res.json();
    const entries = Array.isArray(data.people) ? data.people : [];

    const labeled = [];
    for (const { name, images } of entries) {
      if (!name) continue;
      const count = Math.max(1, Number(images || 1));
      const ds = [];

      for (let i = 1; i <= count; i++) {
        try {
          const url = `${BASE}labels/${name}/${i}.jpg`;
          const img = await faceapi.fetchImage(url);

          const det = await faceapi
            .detectSingleFace(
              img,
              new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: SCORE_TH })
            )
            .withFaceLandmarks()
            .withFaceDescriptor();

          if (det?.descriptor) ds.push(det.descriptor);
          else console.warn(`[labels] No face in ${url} — skipped`);
        } catch (e) {
          console.warn(`[labels] Failed ${name}/${i}.jpg — skipped`, e?.message || e);
        }
      }

      if (ds.length) {
        labeled.push(new faceapi.LabeledFaceDescriptors(name, ds));
      } else {
        console.warn(`[labels] "${name}" had 0 usable images — omitted`);
      }
    }

    if (labeled.length === 0) {
      console.warn("[labels] No valid labeled faces — running without known-face matching.");
      return null;
    }
    return new faceapi.FaceMatcher(labeled, KNOWN_MATCH_THRESHOLD);
  } catch (e) {
    console.warn("[labels] Error loading manifest, skipping known faces:", e.message || e);
    return null;
  }
}

// === USAGE EXAMPLE ===
let faceMatcherRef = { current: null };

async function init() {
  await loadModels();
  faceMatcherRef.current = await loadKnownMatcher();
}

// In your detection loop:
function recognizeFace(d) {
  let name = "unknown";
  if (ENABLE_KNOWN_RECOG && faceMatcherRef.current && d.descriptor) {
    const best = faceMatcherRef.current.findBestMatch(d.descriptor);
    if (best && best.label && best.label !== "unknown" && best.distance <= KNOWN_MATCH_THRESHOLD) {
      name = best.label;
    }
  }
  return name;
}
