import fs from "node:fs/promises";
import path from "node:path";
import glob from "fast-glob";
import * as faceapi from "face-api.js";
import * as tf from "@tensorflow/tfjs-node";
import { Canvas, Image, ImageData } from "canvas";

// --- Patch the env so face-api sees DOM-like canvas in Node
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

/* ====== CONFIG ====== */
const LABELS_DIR  = path.resolve("labels"); // folder with person subfolders
const MODELS_DIR  = path.resolve("models"); // folder with face-api model files
const OUT_JSON    = path.resolve("public/labels/labels-descriptors.json"); // where to write
const MIN_SCORE   = 0.3;   // TinyFaceDetector score threshold
const INPUT_SIZE  = 416;   // 320/416/512 are typical
const MAX_PER_PERSON = 12; // optional cap per label

/* ====== helper: quantize Float32 [-1,1] -> Uint8 (0..255) + base64 ====== */
function encodeDescFloat32ToU8(descF32) {
  const out = new Uint8Array(descF32.length);
  for (let i = 0; i < descF32.length; i++) {
    const clamped = Math.max(-1, Math.min(1, descF32[i]));
    out[i] = Math.round((clamped + 1) * 127.5);
  }
  return out;
}
function u8ToB64(u8) {
  // safe chunked btoa for large arrays
  let bin = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < u8.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, u8.subarray(i, i + CHUNK));
  }
  return Buffer.from(bin, "binary").toString("base64");
}

async function loadModels() {
  await faceapi.nets.tinyFaceDetector.loadFromDisk(MODELS_DIR);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODELS_DIR);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODELS_DIR);
}

async function loadImage(file) {
  const img = new Image();
  img.src = await fs.readFile(file);
  return img;
}

async function descriptorFromImage(img) {
  const det = await faceapi
    .detectSingleFace(
      img,
      new faceapi.TinyFaceDetectorOptions({ inputSize: INPUT_SIZE, scoreThreshold: MIN_SCORE })
    )
    .withFaceLandmarks()
    .withFaceDescriptor();
  return det?.descriptor || null;
}

async function build() {
  console.log("[build] using tf backend:", tf.getBackend()); // should be 'tensorflow' here
  await loadModels();

  // find person folders under labels/
  const peopleDirs = await fs.readdir(LABELS_DIR, { withFileTypes: true });
  const people = peopleDirs.filter(d => d.isDirectory()).map(d => d.name);

  const out = [];

  for (const person of people) {
    const dir = path.join(LABELS_DIR, person);
    const files = await glob(["*.jpg", "*.jpeg", "*.png"], { cwd: dir, absolute: true, caseSensitiveMatch: false });
    if (!files.length) continue;

    const keep = files.slice(0, MAX_PER_PERSON);
    const descs = [];

    for (const file of keep) {
      try {
        const img = await loadImage(file);
        const d = await descriptorFromImage(img);
        if (d && d.length === 128) {
          // quantize for smaller JSON
          const q = encodeDescFloat32ToU8(d);
          descs.push(u8ToB64(q));
          process.stdout.write(".");
        } else {
          process.stdout.write("x");
        }
      } catch (e) {
        process.stdout.write("!");
      }
    }
    process.stdout.write(`  ${person} (${descs.length}/${keep.length})\n`);

    if (descs.length) {
      out.push({ label: person, descriptors_b64: descs });
    }
  }

  // ensure folder exists (e.g. public/labels)
  await fs.mkdir(path.dirname(OUT_JSON), { recursive: true });
  await fs.writeFile(OUT_JSON, JSON.stringify(out, null, 2));
  console.log("\n[build] wrote", OUT_JSON, "labels:", out.length);
}

build().catch(err => {
  console.error(err);
  process.exit(1);
});