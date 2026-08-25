export interface SystemStatus {
  ff_fusion: { ready: boolean; status_message: string };
  depth_anything: { ready: boolean; encoder: string; mode: string; status_message: string };
  robofirefusenet: { enabled: boolean; ready: boolean };
  generative_api: { enabled: boolean; ready: boolean; status_message: string };
}

export interface AnalysisResponse {
  status: string;
  run_id: string;
  processing_time_sec: number;
  artifacts: {
    input_rgb: string;
    input_thermal: string;
    fused: string;
    depth_preview: string;
    depth_npy: string;
    pointcloud_ply: string;
    terrain_obj: string;
    terrain_glb: string | null;
    generated_rgb: string | null;
    metadata_json: string;
  };
  metadata: {
    run_id: string;
    timestamp: string;
    fusion_model: string;
    smoke_confidence: {
      estimate_type: string;
      smoke_level: string;
      visibility_level: string;
      description: string;
    };
    depth_quality: {
      status: string;
      depth_mode: string;
      min_depth: number;
      max_depth: number;
      mean_depth: number;
      std_depth: number;
      p5_depth: number;
      p95_depth: number;
      percentile_range: number;
      pct_nan_inf: number;
      warnings: string[];
    };
    camera_intrinsics: {
      fx: number;
      fy: number;
      cx: number;
      cy: number;
      calibration_state: string;
      description: string;
    };
    point_cloud: {
      total_points: number;
    };
    mesh: {
      num_vertices: number;
      num_faces: number;
    };
  };
}

const API_BASE = "";

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const resp = await fetch(`${API_BASE}/api/status`);
  if (!resp.ok) {
    throw new Error(`Failed to fetch backend status (${resp.status})`);
  }
  return await resp.json();
}

export async function uploadImagePair(rgbFile: File, thermalFile: File): Promise<{ session_id: string; rgb_path: string; thermal_path: string }> {
  const formData = new FormData();
  formData.append("rgb_file", rgbFile);
  formData.append("thermal_file", thermalFile);

  const resp = await fetch(`${API_BASE}/api/uploads`, {
    method: "POST",
    body: formData
  });

  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail || "Failed to upload image pair.");
  }
  return await resp.json();
}

export async function runAnalysis(
  rgbPath: string,
  thermalPath: string,
  enableGenerative: boolean,
  intrinsics?: { fx?: number; fy?: number; cx?: number; cy?: number }
): Promise<AnalysisResponse> {
  const payload: any = {
    rgb_path: rgbPath,
    thermal_path: thermalPath,
    enable_generative: enableGenerative
  };

  if (intrinsics) {
    if (intrinsics.fx !== undefined) payload.fx = intrinsics.fx;
    if (intrinsics.fy !== undefined) payload.fy = intrinsics.fy;
    if (intrinsics.cx !== undefined) payload.cx = intrinsics.cx;
    if (intrinsics.cy !== undefined) payload.cy = intrinsics.cy;
  }

  const resp = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail || "Inference pipeline failed.");
  }
  return await resp.json();
}
