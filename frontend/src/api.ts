export interface SystemStatus {
  cgan: {
    available: boolean;
    checkpoint: string;
    architecture: string;
    input_channels: number;
    output_channels: number;
    image_size: number;
    status_message: string;
  };
  ff_fusion: {
    enabled?: boolean;
    available: boolean;
    mode?: string;
    checkpoint: string;
    status_message: string;
  };
  depth_anything_v2: {
    available: boolean;
    variant: string;
    depth_mode: string;
    checkpoint: string;
    status_message: string;
  };
}

export interface AnalysisResponse {
  status: string;
  run_id: string;
  processing_time_sec: number;
  artifacts: {
    input_thermal: string;
    generated_rgb: string;
    fused: string;
    depth_preview: string;
    depth_npy: string;
    pointcloud_ply: string;
    terrain_obj: string;
    terrain_glb: string | null;
    metadata_json: string;
  };
  metadata: {
    run_id: string;
    timestamp: string;
    cgan_model: string;
    fusion_model: string;
    depth_model: string;
    depth_mode: string;
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

export async function uploadThermalImage(thermalFile: File): Promise<{ session_id: string; thermal_path: string }> {
  const formData = new FormData();
  formData.append("thermal_file", thermalFile);

  const resp = await fetch(`${API_BASE}/api/uploads`, {
    method: "POST",
    body: formData
  });

  if (!resp.ok) {
    const err = await resp.json();
    throw new Error(err.detail || "Failed to upload thermal image.");
  }
  return await resp.json();
}

export async function runAnalysis(
  thermalPath: string,
  intrinsics?: { fx?: number; fy?: number; cx?: number; cy?: number }
): Promise<AnalysisResponse> {
  const payload: any = {
    thermal_path: thermalPath
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
