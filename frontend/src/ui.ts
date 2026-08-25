import { fetchSystemStatus, uploadImagePair, runAnalysis, AnalysisResponse } from './api';
import { TerrainViewer3D } from './viewer';

export function setupUI(viewer: TerrainViewer3D) {
  let selectedRgbFile: File | null = null;
  let selectedThermalFile: File | null = null;
  let historyLogs: any[] = [];

  // Tab Switching
  const navItems = document.querySelectorAll('.nav-item');
  const tabs = document.querySelectorAll('.workspace-tab');

  function switchTab(tabName: string) {
    navItems.forEach(item => {
      if (item.getAttribute('data-tab') === tabName) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });

    tabs.forEach(tab => {
      if (tab.id === `tab-${tabName}`) {
        tab.classList.add('active');
      } else {
        tab.classList.remove('active');
      }
    });

    if (tabName === 'scene') {
      setTimeout(() => {
        window.dispatchEvent(new Event('resize'));
      }, 50);
    }
  }

  navItems.forEach(item => {
    item.addEventListener('click', () => {
      const tabName = item.getAttribute('data-tab');
      if (tabName) switchTab(tabName);
    });
  });

  const btnNavSettings = document.getElementById('nav-btn-settings');
  if (btnNavSettings) {
    btnNavSettings.addEventListener('click', () => switchTab('system'));
  }

  // Dropzone Elements (Overview & Inputs)
  const overviewDropRgb = document.getElementById('overview-drop-rgb')!;
  const overviewDropThermal = document.getElementById('overview-drop-thermal')!;
  const overviewFileRgb = document.getElementById('overview-file-rgb') as HTMLInputElement;
  const overviewFileThermal = document.getElementById('overview-file-thermal') as HTMLInputElement;
  const overviewPreviewRgb = document.getElementById('overview-preview-rgb') as HTMLImageElement;
  const overviewPreviewThermal = document.getElementById('overview-preview-thermal') as HTMLImageElement;

  const btnOverviewAnalyze = document.getElementById('btn-overview-analyze') as HTMLButtonElement;
  const chkOverviewGen = document.getElementById('overview-chk-generative') as HTMLInputElement;

  // Workspace Image Elements
  const viewWorkspaceRgb = document.getElementById('view-workspace-rgb') as HTMLImageElement;
  const viewWorkspaceThermal = document.getElementById('view-workspace-thermal') as HTMLImageElement;
  const metaRgbSpecs = document.getElementById('meta-rgb-specs')!;
  const metaThermalSpecs = document.getElementById('meta-thermal-specs')!;
  const placeholderWorkspaceRgb = document.querySelector('#container-workspace-rgb .placeholder-text')!;
  const placeholderWorkspaceThermal = document.querySelector('#container-workspace-thermal .placeholder-text')!;

  // Output Views
  const viewFusionLarge = document.getElementById('view-fusion-large') as HTMLImageElement;
  const fusionPlaceholder = document.getElementById('fusion-placeholder')!;
  const fusionMetaModel = document.getElementById('fusion-meta-model')!;
  const fusionMetaStatus = document.getElementById('fusion-meta-status')!;

  const viewDepthLarge = document.getElementById('view-depth-large') as HTMLImageElement;
  const depthPlaceholder = document.getElementById('depth-placeholder')!;
  const depthMetaStatus = document.getElementById('depth-meta-status')!;
  const depthMetaMin = document.getElementById('depth-meta-min')!;
  const depthMetaMax = document.getElementById('depth-meta-max')!;
  const depthMetaStd = document.getElementById('depth-meta-std')!;

  // 3D Scene Controls
  const sliderExaggeration = document.getElementById('slider-exaggeration') as HTMLInputElement;
  const valExaggeration = document.getElementById('val-exaggeration')!;
  const selectMode = document.getElementById('select-mode') as HTMLSelectElement;
  const selectTexture = document.getElementById('select-texture') as HTMLSelectElement;
  const chkWireframe = document.getElementById('chk-wireframe') as HTMLInputElement;
  const chkAutoRotate = document.getElementById('chk-autorotate') as HTMLInputElement;
  const btnResetCam = document.getElementById('btn-scene-reset-cam') as HTMLButtonElement;

  // 3D Metadata Footer
  const metaVCount = document.getElementById('meta-v-count')!;
  const metaFCount = document.getElementById('meta-f-count')!;
  const metaCamState = document.getElementById('meta-cam-state')!;
  const metaSmokeLevel = document.getElementById('meta-smoke-level')!;

  // Export Buttons
  const dlGlb = document.getElementById('dl-glb') as HTMLAnchorElement;
  const dlObj = document.getElementById('dl-obj') as HTMLAnchorElement;
  const dlPly = document.getElementById('dl-ply') as HTMLAnchorElement;
  const dlNpy = document.getElementById('dl-npy') as HTMLAnchorElement;

  // Compare View
  const compImgRgb = document.getElementById('comp-img-rgb') as HTMLImageElement;
  const compImgFusion = document.getElementById('comp-img-fusion') as HTMLImageElement;
  const compImgGen = document.getElementById('comp-img-gen') as HTMLImageElement;

  // History Table
  const historyTableBody = document.getElementById('history-table-body')!;

  // Intrinsics Controls
  const inputFx = document.getElementById('input-fx') as HTMLInputElement;
  const inputFy = document.getElementById('input-fy') as HTMLInputElement;
  const inputCx = document.getElementById('input-cx') as HTMLInputElement;
  const inputCy = document.getElementById('input-cy') as HTMLInputElement;
  const sysCameraState = document.getElementById('sys-camera-state')!;
  const btnPresetIntrinsics = document.getElementById('btn-preset-intrinsics')!;
  const btnClearIntrinsics = document.getElementById('btn-clear-intrinsics')!;

  const loadingOverlay = document.getElementById('canvas-overlay-loading')!;
  const sidebarStateText = document.getElementById('sidebar-state-text')!;

  // Initial Status Check
  fetchSystemStatus()
    .then(status => {
      sidebarStateText.textContent = "Ready";
    })
    .catch(() => {
      sidebarStateText.textContent = "Offline";
    });

  // Handle RGB Upload
  overviewDropRgb.addEventListener('click', () => overviewFileRgb.click());
  overviewFileRgb.addEventListener('change', (e) => {
    const files = (e.target as HTMLInputElement).files;
    if (files && files.length > 0) {
      selectedRgbFile = files[0];
      const url = URL.createObjectURL(selectedRgbFile);
      overviewPreviewRgb.src = url;
      overviewPreviewRgb.classList.remove('hidden');

      viewWorkspaceRgb.src = url;
      viewWorkspaceRgb.classList.remove('hidden');
      if (placeholderWorkspaceRgb) placeholderWorkspaceRgb.classList.add('hidden');
      metaRgbSpecs.textContent = `${selectedRgbFile.name} (${(selectedRgbFile.size / 1024).toFixed(1)} KB)`;
      checkReadyToAnalyze();
    }
  });

  // Handle Thermal Upload
  overviewDropThermal.addEventListener('click', () => overviewFileThermal.click());
  overviewFileThermal.addEventListener('change', (e) => {
    const files = (e.target as HTMLInputElement).files;
    if (files && files.length > 0) {
      selectedThermalFile = files[0];
      const url = URL.createObjectURL(selectedThermalFile);
      overviewPreviewThermal.src = url;
      overviewPreviewThermal.classList.remove('hidden');

      viewWorkspaceThermal.src = url;
      viewWorkspaceThermal.classList.remove('hidden');
      if (placeholderWorkspaceThermal) placeholderWorkspaceThermal.classList.add('hidden');
      metaThermalSpecs.textContent = `${selectedThermalFile.name} (${(selectedThermalFile.size / 1024).toFixed(1)} KB)`;
      checkReadyToAnalyze();
    }
  });

  function checkReadyToAnalyze() {
    btnOverviewAnalyze.disabled = !(selectedRgbFile && selectedThermalFile);
  }

  // Intrinsics Helpers
  function updateIntrinsicsState() {
    const fx = inputFx.value.trim();
    const fy = inputFy.value.trim();
    if (fx && fy) {
      sysCameraState.textContent = "Calibrated Intrinsics";
      metaCamState.textContent = "Calibrated";
    } else {
      sysCameraState.textContent = "Approximate Intrinsics";
      metaCamState.textContent = "Approximate";
    }
  }

  [inputFx, inputFy, inputCx, inputCy].forEach(input => {
    if (input) input.addEventListener('input', updateIntrinsicsState);
  });

  if (btnPresetIntrinsics) {
    btnPresetIntrinsics.addEventListener('click', () => {
      inputFx.value = "800.0";
      inputFy.value = "800.0";
      inputCx.value = "320.0";
      inputCy.value = "256.0";
      updateIntrinsicsState();
    });
  }

  if (btnClearIntrinsics) {
    btnClearIntrinsics.addEventListener('click', () => {
      inputFx.value = "";
      inputFy.value = "";
      inputCx.value = "";
      inputCy.value = "";
      updateIntrinsicsState();
    });
  }

  // Run Analysis Handler
  btnOverviewAnalyze.addEventListener('click', async () => {
    if (!selectedRgbFile || !selectedThermalFile) return;

    loadingOverlay.classList.remove('hidden');
    btnOverviewAnalyze.disabled = true;
    sidebarStateText.textContent = "Processing";

    const intrinsics = {
      fx: inputFx.value ? parseFloat(inputFx.value) : undefined,
      fy: inputFy.value ? parseFloat(inputFy.value) : undefined,
      cx: inputCx.value ? parseFloat(inputCx.value) : undefined,
      cy: inputCy.value ? parseFloat(inputCy.value) : undefined,
    };

    try {
      const uploadRes = await uploadImagePair(selectedRgbFile, selectedThermalFile);
      const result = await runAnalysis(
        uploadRes.rgb_path,
        uploadRes.thermal_path,
        chkOverviewGen.checked,
        intrinsics
      );

      // Populate Fusion Tab
      viewFusionLarge.src = result.artifacts.fused;
      viewFusionLarge.classList.remove('hidden');
      fusionPlaceholder.classList.add('hidden');
      fusionMetaModel.textContent = result.metadata.fusion_model;
      fusionMetaStatus.textContent = "Complete";

      // Populate Depth Tab
      viewDepthLarge.src = result.artifacts.depth_preview;
      viewDepthLarge.classList.remove('hidden');
      depthPlaceholder.classList.add('hidden');
      depthMetaStatus.textContent = result.metadata.depth_quality.status;
      depthMetaMin.textContent = result.metadata.depth_quality.min_depth.toString();
      depthMetaMax.textContent = result.metadata.depth_quality.max_depth.toString();
      depthMetaStd.textContent = result.metadata.depth_quality.std_depth.toString();

      // Update 3D Metadata Footer
      metaVCount.textContent = result.metadata.mesh.num_vertices.toLocaleString();
      metaFCount.textContent = result.metadata.mesh.num_faces.toLocaleString();
      metaCamState.textContent = result.metadata.camera_intrinsics.calibration_state;
      metaSmokeLevel.textContent = `Heuristic (${result.metadata.smoke_confidence.smoke_level})`;

      // Enable Export Links
      if (result.artifacts.terrain_glb) {
        dlGlb.href = result.artifacts.terrain_glb;
        dlGlb.classList.remove('disabled');
      }
      dlObj.href = result.artifacts.terrain_obj;
      dlObj.classList.remove('disabled');
      dlPly.href = result.artifacts.pointcloud_ply;
      dlPly.classList.remove('disabled');
      dlNpy.href = result.artifacts.depth_npy;
      dlNpy.classList.remove('disabled');

      // Populate Compare Tab
      compImgRgb.src = result.artifacts.input_rgb;
      compImgRgb.classList.remove('hidden');
      compImgFusion.src = result.artifacts.depth_preview;
      compImgFusion.classList.remove('hidden');

      if (result.artifacts.generated_rgb) {
        compImgGen.src = result.artifacts.generated_rgb;
        compImgGen.classList.remove('hidden');
      }

      // Add to History Table Log
      addHistoryRow(result);

      // Fetch Raw Depth NPY for 3D Viewport
      const npyResp = await fetch(result.artifacts.depth_npy);
      const npyBuf = await npyResp.arrayBuffer();
      const depthFloatArray = parseNpyBuffer(npyBuf);

      const fusedImgObj = new Image();
      fusedImgObj.crossOrigin = "Anonymous";
      fusedImgObj.src = result.artifacts.fused;
      fusedImgObj.onload = () => {
        viewer.updateTerrain(depthFloatArray, 640, 512, fusedImgObj);
        loadingOverlay.classList.add('hidden');
        sidebarStateText.textContent = "Complete";
        // Auto navigate to 3D Scene View (The Main Feature)
        switchTab('scene');
      };

    } catch (err: any) {
      alert(`Inference Pipeline Error: ${err.message || err}`);
      loadingOverlay.classList.add('hidden');
      btnOverviewAnalyze.disabled = false;
      sidebarStateText.textContent = "Error";
    }
  });

  function addHistoryRow(res: AnalysisResponse) {
    historyLogs.unshift(res);
    if (historyTableBody) {
      if (historyLogs.length === 1) {
        historyTableBody.innerHTML = '';
      }
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="text-mono">${res.metadata.timestamp.split(' ')[1]}</td>
        <td class="text-mono">${res.run_id}</td>
        <td class="text-mono">${res.metadata.fusion_model}</td>
        <td class="text-mono">Depth Anything V2 Small</td>
        <td class="text-mono">${res.metadata.depth_quality.status}</td>
        <td class="text-mono">${res.processing_time_sec}s</td>
      `;
      historyTableBody.prepend(tr);
    }
  }

  // Viewport Controls
  if (sliderExaggeration) {
    sliderExaggeration.addEventListener('input', (e) => {
      const val = parseFloat((e.target as HTMLInputElement).value);
      if (valExaggeration) valExaggeration.textContent = val.toFixed(1);
      viewer.setExaggeration(val);
    });
  }

  if (selectMode) {
    selectMode.addEventListener('change', (e) => {
      viewer.setDisplayMode((e.target as HTMLSelectElement).value);
    });
  }

  if (chkWireframe) {
    chkWireframe.addEventListener('change', (e) => {
      viewer.setWireframe((e.target as HTMLInputElement).checked);
    });
  }

  if (chkAutoRotate) {
    chkAutoRotate.addEventListener('change', (e) => {
      viewer.setAutoRotate((e.target as HTMLInputElement).checked);
    });
  }

  if (btnResetCam) {
    btnResetCam.addEventListener('click', () => {
      viewer.resetCamera();
    });
  }
}

function parseNpyBuffer(buf: ArrayBuffer): Float32Array {
  const u8 = new Uint8Array(buf);
  let headerLen = u8[8] + (u8[9] << 8);
  let offset = 10 + headerLen;
  return new Float32Array(buf.slice(offset));
}
