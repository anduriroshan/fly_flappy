import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

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

// ---------------------------------------------------------- fly showcase (3D)
// A dedicated scene: the fly, procedural geometry (no external assets),
// perched over a "screen" that shows the REAL live game frame as a texture
// -- nothing about the actual gameplay is reinterpreted, gameImg already
// holds the true server-streamed frame (see the websocket handler below).
const flyContainer = el("flyShowcase");
const flyScene = new THREE.Scene();
flyScene.background = new THREE.Color(0x05060b);

const flyCamera = new THREE.PerspectiveCamera(42, 1, 0.01, 1000);
flyCamera.position.set(0, 1, 6);

const flyRenderer = new THREE.WebGLRenderer({ antialias: true });
flyRenderer.setPixelRatio(Math.max(1, Math.min(window.devicePixelRatio || 1, 2)));
flyRenderer.setSize(
  flyContainer.clientWidth || 400,
  flyContainer.clientHeight || 400
);   // explicit non-zero initial size — avoids a 0x0 drawing buffer
flyRenderer.outputColorSpace = THREE.SRGBColorSpace;
flyContainer.appendChild(flyRenderer.domElement);

const flyControls = new OrbitControls(flyCamera, flyRenderer.domElement);
flyControls.enableDamping = true;
flyControls.dampingFactor = 0.08;
flyControls.autoRotate = true;
flyControls.autoRotateSpeed = 0.8;
flyControls.addEventListener("start", () => { flyControls.autoRotate = false; });

flyScene.add(new THREE.AmbientLight(0xffffff, 0.9));
const flyKeyLight = new THREE.DirectionalLight(0xfff0e0, 2.0);   // warm key
flyKeyLight.position.set(3, 5, 4);
flyScene.add(flyKeyLight);
const flyFillLight = new THREE.DirectionalLight(0x88aacc, 0.8);  // cool fill
flyFillLight.position.set(-4, 1, -2);
flyScene.add(flyFillLight);

function resizeFly() {
  const w = flyContainer.clientWidth;
  const h = flyContainer.clientHeight;
  if (w === 0 || h === 0) return;
  flyRenderer.setSize(w, h);
  flyCamera.aspect = w / h;
  flyCamera.updateProjectionMatrix();
}
window.addEventListener("resize", resizeFly);
// The panel has no fixed size until layout settles; a one-shot resize on load
// can land while clientHeight is still 0, leaving a 0x0 buffer that never
// recovers. Observe the container so the renderer always tracks its real size.
new ResizeObserver(resizeFly).observe(flyContainer);

// The live game frame is shown as a crisp 2D overlay pinned in this panel's
// corner (see index.html / style.css) rather than as a 3D textured screen —
// keeps it readable and avoids fragile in-scene placement.

// ---- the fly: real NeuroMechFly (flygym) anatomical model, naturalistic ----
// Loaded from a prebuilt GLB (scripts/build_fly_model.py assembles the
// Apache-2.0 flygym STL parts via their rigging.yaml). Materials + animation
// are applied here by node name.
const flyBodyMat = new THREE.MeshStandardMaterial({
  color: 0x6e4a26, roughness: 0.62, metalness: 0.12,   // amber-brown chitin
  side: THREE.DoubleSide,   // STL->GLB winding is unreliable; render both sides
});
const flyEyeMat = new THREE.MeshStandardMaterial({
  color: 0x7a1414, roughness: 0.28, metalness: 0.05,
  emissive: 0x3a0405, emissiveIntensity: 0.4,          // deep red compound eyes
  side: THREE.DoubleSide,
});
const flyWingMat = new THREE.MeshStandardMaterial({
  color: 0xcdb488, transparent: true, opacity: 0.28,   // translucent amber wings
  side: THREE.DoubleSide, roughness: 0.35, depthWrite: false,
});

const fly = new THREE.Group();
flyScene.add(fly);

// The tap button the fly presses to flap — small (bigger than a leg, smaller
// than the fly), sitting on the ground beside its front leg. Positioned after
// the fly loads (see below).
const buttonGroup = new THREE.Group();
flyScene.add(buttonGroup);
const buttonBase = new THREE.Mesh(
  new THREE.CylinderGeometry(0.19, 0.23, 0.06, 36),
  new THREE.MeshStandardMaterial({ color: 0x0c0e16, roughness: 0.5, metalness: 0.35 })
);
buttonGroup.add(buttonBase);
const buttonCapMat = new THREE.MeshStandardMaterial({
  color: 0x00ffc8, emissive: 0x00ffc8, emissiveIntensity: 0.5, roughness: 0.3,
});
const buttonCap = new THREE.Mesh(
  new THREE.CylinderGeometry(0.14, 0.14, 0.07, 36), buttonCapMat
);
const BUTTON_CAP_Y = 0.05;
buttonCap.position.y = BUTTON_CAP_Y;
buttonGroup.add(buttonCap);

let wingPivots = [];     // {pivot, sign} for buzz
let frontLegPivots = []; // {pivot, sign} for tap
let tapLegPivot = null;  // the ONE front-leg pivot resting on the button
let tapAxis = "y";       // which local pivot axis presses the toe downward
let tapSign = 1;         // press direction so that foot swings down onto button
let flyLoaded = false;
const GROUND_Y = -0.55;  // invisible ground plane the fly + button sit on

function materialFor(name) {
  if (name.includes("eye")) return flyEyeMat;
  if (name.includes("wing")) return flyWingMat;
  return flyBodyMat;      // thorax, head, abdomen, legs, antenna, haltere, proboscis
}

// Move a set of named nodes under a new pivot Group at `pivotPos` (model
// coords), preserving world transform, so rotating the pivot articulates them.
function groupUnderPivot(model, pivotPos, nodeNames) {
  const pivot = new THREE.Group();
  pivot.position.fromArray(pivotPos);
  model.add(pivot);
  model.updateWorldMatrix(true, true);
  for (const n of nodeNames) {
    const node = model.getObjectByName(n);
    if (node) pivot.attach(node);
  }
  return pivot;
}

const FRONT_LEG_L = ["lf_coxa", "lf_trochanterfemur", "lf_tibia",
  "lf_tarsus1", "lf_tarsus2", "lf_tarsus3", "lf_tarsus4", "lf_tarsus5"];
const FRONT_LEG_R = FRONT_LEG_L.map((n) => "r" + n.slice(1));

new GLTFLoader().load("/static/assets/fly/fly.glb", async (gltf) => {
  const model = gltf.scene;
  model.traverse((o) => {
    if (o.isMesh) {
      o.geometry.computeVertexNormals();   // GLB from trimesh has no normals
      o.geometry.computeBoundingSphere();
      o.frustumCulled = false;             // don't let a stale bounds cull it
      o.material = materialFor(o.name);
    }
  });

  const meta = await fetch("/static/assets/fly/fly_meta.json").then((r) => r.json());
  const P = meta.pivots;

  // Articulation pivots (built while the model is still at identity, so the
  // model-space pivot coords from meta line up directly).
  fly.add(model);
  wingPivots = [
    { pivot: groupUnderPivot(model, P.l_wing, ["l_wing"]), sign: 1 },
    { pivot: groupUnderPivot(model, P.r_wing, ["r_wing"]), sign: -1 },
  ];
  frontLegPivots = [
    { pivot: groupUnderPivot(model, P.lf_coxa, FRONT_LEG_L), sign: 1 },
    { pivot: groupUnderPivot(model, P.rf_coxa, FRONT_LEG_R), sign: -1 },
  ];

  // Orient (flygym Z is dorso-ventral -> three.js Y-up) and center at origin.
  const c = meta.center;
  model.position.set(-c[0], -c[1], -c[2]);
  fly.rotation.x = -Math.PI / 2;

  // Scale to fit the small inset, then stand the fly on an (invisible) ground
  // plane so it reads like the reference photo: side profile, feet down.
  const v = new THREE.Vector3();
  fly.updateWorldMatrix(true, true);
  let box = new THREE.Box3().setFromObject(fly);
  const maxDim = Math.max(...box.getSize(v).toArray());
  fly.scale.setScalar(1.7 / maxDim);

  fly.updateWorldMatrix(true, true);
  box = new THREE.Box3().setFromObject(fly);
  const ctr = box.getCenter(v.clone());
  fly.position.x = -ctr.x;                 // center horizontally
  fly.position.z = -ctr.z;                 // center in depth
  fly.position.y = GROUND_Y - box.min.y;   // feet on the ground plane

  // ---- place the button under a real FRONT foot and frame it like the ref ----
  fly.updateWorldMatrix(true, true);

  // Lowest world-space vertex of a node's mesh = the actual TOE. The node
  // origin is the joint up the leg, so placing the button there left it
  // floating in the gap between the splayed feet — this uses the real tip.
  const toeOf = (node) => {
    let mesh = null;
    node.traverse((o) => { if (o.isMesh && !mesh) mesh = o; });
    if (!mesh) return null;
    mesh.updateWorldMatrix(true, false);
    const pos = mesh.geometry.attributes.position;
    const v = new THREE.Vector3();
    let best = null;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(mesh.matrixWorld);
      if (!best || v.y < best.y) best = v.clone();
    }
    return best;
  };

  // Forward direction from thorax -> head, using mesh BBOX centers (the node
  // origins sit at the joints and nearly coincide, giving no usable forward).
  // Keeps the camera framing independent of the baked axis convention.
  const centerOf = (name) => {
    const node = fly.getObjectByName(name);
    return node ? new THREE.Box3().setFromObject(node).getCenter(new THREE.Vector3()) : null;
  };
  const headP = centerOf("c_head");
  const thoraxP = centerOf("c_thorax") || new THREE.Vector3();
  const fwd = (headP || new THREE.Vector3(1, 0, 0)).clone().sub(thoraxP); fwd.y = 0;
  if (fwd.lengthSq() < 1e-6) fwd.set(1, 0, 0);
  fwd.normalize();
  const bodyC = new THREE.Vector3(thoraxP.x, 0, thoraxP.z);

  // The two front legs with their real toe positions; pick the one nearest the
  // camera so the button-on-foot reads clearly, and remember it as THE tapper.
  const frontLegs = [
    { pivot: frontLegPivots[0].pivot, node: fly.getObjectByName("lf_tarsus5") },
    { pivot: frontLegPivots[1].pivot, node: fly.getObjectByName("rf_tarsus5") },
  ].map((L) => ({ ...L, toe: L.node ? toeOf(L.node) : null }))
   .filter((L) => L.toe)
   .sort((a, b) => b.toe.z - a.toe.z);
  const chosen = frontLegs[0];

  if (chosen) {
    tapLegPivot = chosen.pivot;
    buttonGroup.position.set(chosen.toe.x, GROUND_Y, chosen.toe.z);

    // Press axis/sign: probe the pivot's local axes, keep whichever moves the
    // toe most DOWNWARD — no reliance on any baked axis convention.
    const baseY = chosen.toe.y;
    let best = { axis: "y", sign: 1, drop: -Infinity };
    for (const axis of ["x", "y", "z"]) {
      for (const sign of [1, -1]) {
        const prev = chosen.pivot.rotation[axis];
        chosen.pivot.rotation[axis] = prev + sign * 0.25;
        fly.updateWorldMatrix(true, true);
        const t = toeOf(chosen.node);
        chosen.pivot.rotation[axis] = prev;
        const drop = baseY - t.y;                 // + = toe dipped down
        if (drop > best.drop) best = { axis, sign, drop };
      }
    }
    fly.updateWorldMatrix(true, true);
    tapAxis = best.axis; tapSign = best.sign;
  }

  // Side-3/4 camera (like the reference photo), built from `fwd` so it frames
  // the fly's profile and the chosen foot on the button regardless of facing.
  flyControls.autoRotate = false;
  const up = new THREE.Vector3(0, 1, 0);
  const side = new THREE.Vector3().crossVectors(fwd, up).normalize();
  const toeOff = chosen
    ? new THREE.Vector3(chosen.toe.x - bodyC.x, 0, chosen.toe.z - bodyC.z)
    : side;
  if (side.dot(toeOff) < 0) side.negate();   // favour the chosen foot's side
  const focus = new THREE.Vector3(
    chosen ? chosen.toe.x * 0.45 : 0, GROUND_Y + 0.4,
    chosen ? chosen.toe.z * 0.45 : 0);
  flyControls.target.copy(focus);
  flyCamera.position.copy(focus)
    .addScaledVector(side, 2.7)
    .addScaledVector(fwd, 1.1)
    .addScaledVector(up, 1.25);
  flyCamera.near = 0.01;
  flyCamera.far = 100;
  flyCamera.updateProjectionMatrix();
  flyLoaded = true;
  resizeFly();
}, undefined, (err) => {
  console.error("fly model load failed", err);
  setStatus("fly model failed to load", "error");
});

let tapPulse = 0;    // set to 1 on each real FLAP action, decays away
let tapAmount = 0;   // eased toward tapPulse -- fast strike down, slow recover
function triggerFlyTap() { tapPulse = 1; }

let flyClock = 0;
function updateFly(dt) {
  if (!flyLoaded) return;
  flyClock += dt;

  const buzz = Math.sin(flyClock * 42) * 0.5;                // fast wing buzz
  for (const w of wingPivots) w.pivot.rotation.x = w.sign * buzz;

  tapAmount += (tapPulse - tapAmount) * (tapPulse > tapAmount ? 0.6 : 0.15);
  tapPulse *= 0.9;
  // Only the ONE front leg resting on the button presses: rotate it at the
  // coxa (about the probed axis) so its toe jabs the button straight DOWN.
  // The body and every other leg stay put.
  if (tapLegPivot) tapLegPivot.rotation[tapAxis] = tapSign * tapAmount * 0.5;

  // The button depresses + flares under that press.
  buttonCap.position.y = BUTTON_CAP_Y - tapAmount * 0.045;
  buttonCapMat.emissiveIntensity = 0.5 + tapAmount * 2.5;

  flyControls.update();
}

resizeFly();

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
let visibleN = 0;        // how many neurons the slider currently reveals
let totalNeurons = 0;    // total neurons in brain.json
// cumulativeVertexCount[k] = total vertex count of first k neurons in the
// (shuffled) render order. setDrawRange(0, cumulativeVertexCount[N])
// draws exactly the first N shuffled neurons — no geometry rebuild,
// no re-upload, one WebGL call per slider tick.
let cumulativeVertexCount = null;

// Deterministic Fisher-Yates shuffle so the same order is picked each
// page-load (avoids the visible cluster jumping around on refresh).
function seededShuffle(arr, seed = 1337) {
  const a = arr.slice();
  let s = seed;
  const rnd = () => {
    // xorshift32 — cheap, deterministic
    s ^= s << 13; s ^= s >>> 17; s ^= s << 5;
    return ((s >>> 0) % 1e9) / 1e9;
  };
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function buildBrain(data) {
  nSpotlight = data.n_spotlight;
  totalNeurons = data.n_rendered;

  // Shuffle the neuron render order so revealing the first N via
  // setDrawRange gives a role-balanced random subset rather than
  // "all motors first, then all sensories" (which is how brain.json
  // is emitted). Each neuron's vertices stay contiguous in the buffer,
  // so setDrawRange stays a simple offset lookup.
  const order = seededShuffle(data.neurons.map((_, i) => i));

  const positions = [];
  const roleColors = [];
  const uniqueColors = [];
  const neuronIdx = [];
  cumulativeVertexCount = new Uint32Array(totalNeurons + 1);

  order.forEach((origIdx, renderIdx) => {
    const nrn = data.neurons[origIdx];
    const v = nrn.verts;         // flat [x,y,z, ...]
    const e = nrn.edges;         // flat [i,j, i,j, ...]
    const c = nrn.color;         // [r,g,b] 0..1, by role
    const uc = distinctColor(origIdx);
    const ai = nrn.act_index;
    for (let k = 0; k < e.length; k += 2) {
      const a = e[k] * 3;
      const b = e[k + 1] * 3;
      positions.push(v[a], v[a + 1], v[a + 2], v[b], v[b + 1], v[b + 2]);
      roleColors.push(c[0], c[1], c[2], c[0], c[1], c[2]);
      uniqueColors.push(uc[0], uc[1], uc[2], uc[0], uc[1], uc[2]);
      neuronIdx.push(ai, ai);
    }
    cumulativeVertexCount[renderIdx + 1] = positions.length / 3;
  });

  roleColorArr = new Float32Array(roleColors);
  uniqueColorArr = new Float32Array(uniqueColors);

  lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  lineGeo.setAttribute("color", new THREE.Float32BufferAttribute(roleColorArr.slice(), 3));
  lineGeo.setAttribute("aNeuron", new THREE.Float32BufferAttribute(neuronIdx, 1));

  lineGeo.computeBoundingSphere();
  brainMat = makeMaterial(nSpotlight);
  const lines = new THREE.LineSegments(lineGeo, brainMat);
  lines.rotation.x = -Math.PI / 2;   // fly CNS: bring the horizontal plane up
  scene.add(lines);

  setVisibleNeurons(Math.min(400, totalNeurons));   // default: 400 (less bright)
  wireSlider();
  resize();
}

function setVisibleNeurons(n) {
  if (!lineGeo || !cumulativeVertexCount) return;
  visibleN = Math.max(1, Math.min(totalNeurons, n));
  lineGeo.setDrawRange(0, cumulativeVertexCount[visibleN]);
  brainMeta.textContent = `${visibleN} of ${totalNeurons} traced neurons`;
  // Re-render the color-mode button label so the count matches too.
  const btn = el("colorModeBtn");
  if (btn && !byRole) {
    btn.textContent = `Colors: ${visibleN} distinct neurons`;
  }
}

function wireSlider() {
  const slider = el("neuronSlider");
  const label = el("neuronSliderValue");
  if (!slider || !label) return;
  slider.max = String(totalNeurons);
  slider.value = String(visibleN);
  label.textContent = String(visibleN);
  slider.addEventListener("input", () => {
    const n = parseInt(slider.value, 10);
    label.textContent = String(n);
    setVisibleNeurons(n);
  });
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

const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  controls.update();
  updateFly(dt);
  if (composer) composer.render();
  else renderer.render(scene, camera);
  // Only render the fly once its canvas has a real size, to avoid
  // GL_INVALID_FRAMEBUFFER_OPERATION (zero-size attachment) spam.
  if (flyContainer.clientWidth > 0 && flyContainer.clientHeight > 0) {
    flyRenderer.render(flyScene, flyCamera);
  }
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
    if (f.action === 1) triggerFlyTap();
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
  colorModeBtn.textContent = byRole
    ? "Colors: by role"
    : `Colors: ${visibleN} distinct neurons`;
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
