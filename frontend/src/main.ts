import { TerrainViewer3D } from './viewer';
import { setupUI } from './ui';

document.addEventListener('DOMContentLoaded', () => {
  try {
    const viewer = new TerrainViewer3D('three-container');
    setupUI(viewer);
    console.log("Aero-Topo 3D Viewport initialized successfully.");
  } catch (err) {
    console.error("Failed to initialize Aero-Topo frontend:", err);
  }
});
