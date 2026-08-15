// Aero-Topo Three.js & FastAPI Frontend Engine

let scene, camera, renderer, controls, terrainMesh;
let textureLoader = new THREE.TextureLoader();
let currentFile = null;

// UI Elements
const systemStatusEl = document.getElementById("system-status");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const btnProcess = document.getElementById("btn-process");
const btnSample = document.getElementById("btn-sample");
const loaderOverlay = document.getElementById("loader");

const imgThermal = document.getElementById("img-thermal");
const imgRgb = document.getElementById("img-rgb");
const imgDepth = document.getElementById("img-depth");
const tagStage1 = document.getElementById("tag-stage1");
const tagStage2 = document.getElementById("tag-stage2");

const sliderScale = document.getElementById("displacement-scale");
const scaleValEl = document.getElementById("scale-val");
const sliderSubdiv = document.getElementById("mesh-subdivision");
const subdivValEl = document.getElementById("subdiv-val");
const wireframeToggle = document.getElementById("wireframe-toggle");
const autorotateToggle = document.getElementById("autorotate-toggle");
const failsafeToggle = document.getElementById("failsafe-toggle");

// Initialize Three.js Scene
function initThreeJS() {
  const container = document.getElementById("canvas-container");
  const canvas = document.getElementById("webgl-canvas");

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0d14);

  camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
  camera.position.set(0, -90, 80);

  renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;

  // Orbit Controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.5;
  controls.maxPolarAngle = Math.PI / 2 - 0.05; // Don't go below ground

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(ambientLight);

  const dirLight1 = new THREE.DirectionalLight(0x00f2fe, 0.8);
  dirLight1.position.set(50, 50, 100);
  scene.add(dirLight1);

  const dirLight2 = new THREE.DirectionalLight(0xff0844, 0.4);
  dirLight2.position.set(-50, -50, 50);
  scene.add(dirLight2);

  // Helper Grid
  const gridHelper = new THREE.GridHelper(120, 20, 0x00f2fe, 0x1f293d);
  gridHelper.rotation.x = Math.PI / 2;
  gridHelper.position.z = -1;
  scene.add(gridHelper);

  // Create initial placeholder plane
  createTerrainMesh(
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
  );

  // Window Resize Listener
  window.addEventListener("resize", onWindowResize);

  // Animation Loop
  animate();
}

function createTerrainMesh(rgbDataUrl, depthDataUrl) {
  if (terrainMesh) {
    scene.remove(terrainMesh);
    terrainMesh.geometry.dispose();
    terrainMesh.material.dispose();
  }

  const subdiv = parseInt(sliderSubdiv.value);
  const scale = parseFloat(sliderScale.value);
  const isWireframe = wireframeToggle.checked;

  const geometry = new THREE.PlaneGeometry(100, 100, subdiv, subdiv);

  textureLoader.load(rgbDataUrl, (rgbTex) => {
    textureLoader.load(depthDataUrl, (depthTex) => {
      rgbTex.generateMipmaps = true;
      depthTex.generateMipmaps = true;

      const material = new THREE.MeshStandardMaterial({
        map: rgbTex,
        displacementMap: depthTex,
        displacementScale: scale,
        displacementBias: 0,
        wireframe: isWireframe,
        roughness: 0.4,
        metalness: 0.1,
        side: THREE.DoubleSide
      });

      terrainMesh = new THREE.Mesh(geometry, material);
      scene.add(terrainMesh);
    });
  });
}

function animate() {
  requestAnimationFrame(animate);
  controls.autoRotate = autorotateToggle.checked;
  controls.update();
  renderer.render(scene, camera);
}

function onWindowResize() {
  const container = document.getElementById("canvas-container");
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

// Health Check API
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.status === "online") {
      const modeText = data.pix2pix_weights_loaded ? "GPU (Pre-trained Weights)" : "Adapter Fallback Mode";
      systemStatusEl.innerText = `System Ready [Device: ${data.device} | ${modeText}]`;
    }
  } catch (err) {
    systemStatusEl.innerText = "API Offline / Connecting...";
  }
}

// File Upload Handlers
dropzone.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleFileSelect(e.target.files[0]);
  }
});

dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  if (e.dataTransfer.files.length > 0) {
    handleFileSelect(e.dataTransfer.files[0]);
  }
});

function handleFileSelect(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    imgThermal.src = e.target.result;
    dropzone.querySelector(".dropzone-text").innerText = file.name;
  };
  reader.readAsDataURL(file);
}

// Process Pipeline Handler
btnProcess.addEventListener("click", async () => {
  if (!currentFile) {
    alert("Please upload a thermal image first or click 'Generate Synthetic Sample'.");
    return;
  }

  loaderOverlay.classList.add("active");

  const formData = new FormData();
  formData.append("file", currentFile);
  formData.append("use_failsafe", failsafeToggle.checked);

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Processing failed");
    }

    const data = await response.json();
    
    // Update Previews
    imgThermal.src = data.images.thermal_input;
    imgRgb.src = data.images.rgb_output;
    imgDepth.src = data.images.depth_map;

    tagStage1.innerText = `STAGE 1: ${data.stage1_mode}`;
    tagStage2.innerText = `STAGE 2: ${data.stage2_mode}`;

    // Update Three.js 3D Mesh
    createTerrainMesh(data.images.rgb_output, data.images.depth_map);

  } catch (error) {
    alert(`Error: ${error.message}`);
  } finally {
    loaderOverlay.classList.remove("active");
  }
});

// Synthetic Thermal Image Generator for Instant Testing
btnSample.addEventListener("click", () => {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");

  // Generate synthetic thermal mountain terrain gradient
  const grad = ctx.createRadialGradient(128, 128, 10, 128, 128, 120);
  grad.addColorStop(0, "#ffffff");
  grad.addColorStop(0.4, "#f59e0b");
  grad.addColorStop(0.7, "#ef4444");
  grad.addColorStop(1, "#1e1b4b");

  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 256, 256);

  // Add terrain noise details
  for (let i = 0; i < 400; i++) {
    const x = Math.random() * 256;
    const y = Math.random() * 256;
    const r = Math.random() * 12;
    ctx.fillStyle = `rgba(255, 255, 255, ${Math.random() * 0.3})`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  canvas.toBlob((blob) => {
    const file = new File([blob], "synthetic_thermal_sample.png", { type: "image/png" });
    handleFileSelect(file);
    btnProcess.click();
  }, "image/png");
});

// Interactive UI Sliders & Toggles
sliderScale.addEventListener("input", (e) => {
  scaleValEl.innerText = e.target.value;
  if (terrainMesh && terrainMesh.material) {
    terrainMesh.material.displacementScale = parseFloat(e.target.value);
  }
});

sliderSubdiv.addEventListener("input", (e) => {
  subdivValEl.innerText = e.target.value;
  if (imgRgb.src && imgDepth.src) {
    createTerrainMesh(imgRgb.src, imgDepth.src);
  }
});

wireframeToggle.addEventListener("change", (e) => {
  if (terrainMesh && terrainMesh.material) {
    terrainMesh.material.wireframe = e.target.checked;
  }
});

// Start Three.js on page load
window.addEventListener("DOMContentLoaded", () => {
  initThreeJS();
  checkHealth();
});
