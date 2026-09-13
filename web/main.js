import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";

// ---------------------------------------------------------------- DOM refs
const el = (id) => document.getElementById(id);
const startBtn = el("startBtn");
const resetBtn = el("resetBtn");
const statusEl = el("status");
const gameImg = el("gameImg");
const deadFlash = el("deadFlash");
const brainMeta = el("brainMeta");

// ---------------------------------------------------------------- 3D scene
const container = el("brainCanvas");
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(0, 0, 3.1);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
// ACES tone mapping rolls bright values off smoothly instead of clipping to
// a solid white blob where many glowing neurons overlap.
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.9;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;
controls.minDistance = 0.2;   // allow zooming right into individual neurons
controls.maxDistance = 12;
// Stop the auto-spin as soon as the user grabs it, so they can inspect.
controls.addEventListener("start", () => { controls.autoRotate = false; });

let composer, bloom;
function setupComposer(w, h) {
  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  // strength, radius, threshold — only genuinely active neurons bloom now,
  // the base structure stays crisp and readable.
  bloom = new UnrealBloomPass(new THREE.Vector2(w, h), 0.55, 0.4, 0.6);
  composer.addPass(bloom);
}

function resize() {
  const w = container.clientWidth;
  const h = container.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  if (composer) composer.setSize(w, h);
  else setupComposer(w, h);
}
window.addEventListener("resize", resize);

// ---------------------------------------------------------------- shader
// One merged LineSegments. Each vertex carries a base role color and the
// neuron's activation index. A tiny data texture (one texel per spotlight
// neuron) is updated every frame with 0..1 brightness; the vertex shader
// scales the base color by it, and UnrealBloom makes the hot ones glow.
let actTexture = null;
let actArray = null;
let brainMat = null;
let nSpotlight = 0;

function makeMaterial(n) {
  actArray = new Float32Array(n);
  actTexture = new THREE.DataTexture(actArray, n, 1, THREE.RedFormat, THREE.FloatType);
  actTexture.magFilter = THREE.NearestFilter;
  actTexture.minFilter = THREE.NearestFilter;
  actTexture.needsUpdate = true;

  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uAct: { value: actTexture },
      uW: { value: n },
      uGain: { value: 1.6 },
      uBase: { value: 0.05 },
    },
    vertexShader: /* glsl */ `
      attribute vec3 color;
      attribute float aNeuron;
      varying vec3 vCol;
      uniform sampler2D uAct;
      uniform float uW;
      uniform float uGain;
      uniform float uBase;
      void main() {
        float b = texture2D(uAct, vec2((aNeuron + 0.5) / uW, 0.5)).r;
        vCol = color * (uBase + b * uGain);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: /* glsl */ `
      varying vec3 vCol;
      void main() { gl_FragColor = vec4(vCol, 0.9); }
    `,
  });
}

// HSL with a golden-angle hue step so consecutive neuron indices land far
// apart on the color wheel (adjacent trees don't get near-identical hues).
function distinctColor(i) {
  const hue = (i * 137.508) % 360;
  const c = new THREE.Color();
  c.setHSL(hue / 360, 0.7, 0.55);
  return [c.r, c.g, c.b];
}

let roleColorArr = null;
let uniqueColorArr = null;
let lineGeo = null;
let byRole = true;

function buildBrain(data) {
  nSpotlight = data.n_spotlight;
  brainMeta.textContent = `${data.n_rendered} traced neurons`;

  const positions = [];
  const roleColors = [];
  const uniqueColors = [];
  const neuronIdx = [];

  data.neurons.forEach((nrn, ni) => {
    const v = nrn.verts;         // flat [x,y,z, ...]
    const e = nrn.edges;         // flat [i,j, i,j, ...]
    const c = nrn.color;         // [r,g,b] 0..1, by role
    const uc = distinctColor(ni);// [r,g,b] 0..1, unique per neuron
    const ai = nrn.act_index;
    for (let k = 0; k < e.length; k += 2) {
      const a = e[k] * 3;
      const b = e[k + 1] * 3;
      positions.push(v[a], v[a + 1], v[a + 2], v[b], v[b + 1], v[b + 2]);
      roleColors.push(c[0], c[1], c[2], c[0], c[1], c[2]);
      uniqueColors.push(uc[0], uc[1], uc[2], uc[0], uc[1], uc[2]);
      neuronIdx.push(ai, ai);
    }
  });

  roleColorArr = new Float32Array(roleColors);
  uniqueColorArr = new Float32Array(uniqueColors);

  lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  lineGeo.setAttribute("color", new THREE.Float32BufferAttribute(roleColorArr.slice(), 3));
  lineGeo.setAttribute("aNeuron", new THREE.Float32BufferAttribute(neuronIdx, 1));

  // Center + gentle initial orientation so two lobes read clearly.
  lineGeo.computeBoundingSphere();
  brainMat = makeMaterial(nSpotlight);
  const lines = new THREE.LineSegments(lineGeo, brainMat);
  lines.rotation.x = -Math.PI / 2;   // fly CNS: bring the horizontal plane up
  scene.add(lines);

  resize();
}

function setColorMode(useRole) {
  byRole = useRole;
  if (!lineGeo) return;
  lineGeo.attributes.color.array.set(useRole ? roleColorArr : uniqueColorArr);
  lineGeo.attributes.color.needsUpdate = true;
}

// Smoothed mode: rise fast, fall slow — easier to read, but not literal.
// Raw mode: exactly this frame's magnitude, no persistence — flickery but
// 100% true to what the connectome actually computed that instant.
let smoothGlow = true;

function applyActivation(act) {
  if (!actArray) return;
  for (let i = 0; i < actArray.length; i++) {
    const target = i < act.length ? act[i] : 0;
    actArray[i] = smoothGlow
      ? (target > actArray[i] ? target : actArray[i] * 0.82 + target * 0.18)
      : target;
  }
  actTexture.needsUpdate = true;
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if (composer) composer.render();
  else renderer.render(scene, camera);
}
animate();

// ---------------------------------------------------------------- HUD
function setHud(f) {
  el("score").textContent = f.score;
  el("reward").textContent = f.reward.toFixed(2);
  el("epReward").textContent = f.ep_reward.toFixed(2);
  el("action").textContent = f.action === 1 ? "FLAP" : "fall";
  el("step").textContent = f.step;
}

// ---------------------------------------------------------------- WebSocket
let ws = null;
function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => setStatus("connected — press Play", "live");
  ws.onclose = () => setStatus("disconnected — reload to retry", "error");
  ws.onerror = () => setStatus("connection error", "error");
  ws.onmessage = (ev) => {
    const f = JSON.parse(ev.data);
    if (f.type !== "frame") return;
    if (f.img) gameImg.src = f.img;
    applyActivation(f.act);
    setHud(f);
    if (f.done) {
      deadFlash.classList.add("show");
      setTimeout(() => deadFlash.classList.remove("show"), 700);
    }
  };
}

function setStatus(msg, cls) {
  statusEl.textContent = msg;
  statusEl.className = "status" + (cls ? " " + cls : "");
}

// ---------------------------------------------------------------- boot
startBtn.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd: "start" }));
    setStatus("● LIVE — the fly is playing", "live");
    startBtn.textContent = "Playing…";
  }
});
resetBtn.addEventListener("click", () => {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ cmd: "start" }));
  }
});
const smoothBtn = el("smoothBtn");
smoothBtn.addEventListener("click", () => {
  smoothGlow = !smoothGlow;
  smoothBtn.textContent = `Smoothed glow: ${smoothGlow ? "ON" : "OFF"}`;
});
const colorModeBtn = el("colorModeBtn");
colorModeBtn.addEventListener("click", () => {
  setColorMode(!byRole);
  colorModeBtn.textContent = byRole ? "Colors: by role" : "Colors: 400 distinct neurons";
});

(async function init() {
  setStatus("loading brain morphology…");
  try {
    const res = await fetch("/api/brain");
    if (!res.ok) throw new Error(`brain fetch failed (${res.status})`);
    const data = await res.json();
    buildBrain(data);
    connect();
    startBtn.disabled = false;
    startBtn.textContent = "▶ Play";
  } catch (e) {
    setStatus("failed to load brain: " + e.message, "error");
    console.error(e);
  }
})();
