import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

export class TerrainViewer3D {
  private container: HTMLElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;

  private meshObject: THREE.Mesh | null = null;
  private pointCloudObject: THREE.Points | null = null;

  private rawDepthData: Float32Array | null = null;
  private depthWidth: number = 0;
  private depthHeight: number = 0;
  private textureImage: HTMLImageElement | null = null;

  private exaggerationScale: number = 2.0;
  private displayMode: string = 'mesh';
  private isWireframe: boolean = false;

  constructor(containerId: string) {
    const el = document.getElementById(containerId);
    if (!el) throw new Error(`Container #${containerId} not found`);
    this.container = el;

    // 1. Scene setup
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0B1017);

    // 2. Camera setup
    const aspect = this.container.clientWidth / (this.container.clientHeight || 400);
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
    this.camera.position.set(0, -140, 120);
    this.camera.up.set(0, 0, 1);

    // 3. Renderer setup
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight || 400);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.container.appendChild(this.renderer.domElement);

    // 4. Orbit Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;

    // 5. Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    this.scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(100, 100, 200);
    this.scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x4C8DFF, 0.3);
    dirLight2.position.set(-100, -100, 50);
    this.scene.add(dirLight2);

    // Grid helper
    const grid = new THREE.GridHelper(200, 20, 0x263241, 0x151D28);
    grid.rotation.x = Math.PI / 2;
    this.scene.add(grid);

    window.addEventListener('resize', () => this.onWindowResize());
    this.animate();
  }

  public updateTerrain(
    depthArray: Float32Array,
    width: number,
    height: number,
    textureImg: HTMLImageElement
  ) {
    this.rawDepthData = depthArray;
    this.depthWidth = width;
    this.depthHeight = height;
    this.textureImage = textureImg;

    this.rebuildGeometry();
    this.resetCamera();
  }

  public setExaggeration(scale: number) {
    this.exaggerationScale = scale;
    if (this.rawDepthData) {
      this.rebuildGeometry();
    }
  }

  public setDisplayMode(mode: string) {
    this.displayMode = mode;
    if (this.meshObject) this.meshObject.visible = (mode === 'mesh' || mode === 'both');
    if (this.pointCloudObject) this.pointCloudObject.visible = (mode === 'pointcloud' || mode === 'both');
  }

  public setWireframe(enabled: boolean) {
    this.isWireframe = enabled;
    if (this.meshObject && this.meshObject.material) {
      (this.meshObject.material as THREE.MeshStandardMaterial).wireframe = enabled;
    }
  }

  public setAutoRotate(enabled: boolean) {
    this.controls.autoRotate = enabled;
    this.controls.autoRotateSpeed = 2.0;
  }

  public resetCamera() {
    this.camera.position.set(0, -140, 120);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  private rebuildGeometry() {
    if (!this.rawDepthData || this.depthWidth === 0 || this.depthHeight === 0) return;

    if (this.meshObject) {
      this.scene.remove(this.meshObject);
      this.meshObject.geometry.dispose();
      this.meshObject = null;
    }
    if (this.pointCloudObject) {
      this.scene.remove(this.pointCloudObject);
      this.pointCloudObject.geometry.dispose();
      this.pointCloudObject = null;
    }

    const w = this.depthWidth;
    const h = this.depthHeight;
    const subsample = 2;
    const sw = Math.floor(w / subsample);
    const sh = Math.floor(h / subsample);

    let dMin = Infinity, dMax = -Infinity;
    for (let i = 0; i < this.rawDepthData.length; i++) {
      const v = this.rawDepthData[i];
      if (isFinite(v)) {
        if (v < dMin) dMin = v;
        if (v > dMax) dMax = v;
      }
    }
    const dRange = (dMax - dMin) > 1e-5 ? (dMax - dMin) : 1.0;

    const numVertices = sw * sh;
    const positions = new Float32Array(numVertices * 3);
    const uvs = new Float32Array(numVertices * 2);
    const colors = new Float32Array(numVertices * 3);

    let canvas: HTMLCanvasElement | null = null;
    let ctx: CanvasRenderingContext2D | null = null;
    if (this.textureImage) {
      canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      ctx = canvas.getContext('2d');
      if (ctx) ctx.drawImage(this.textureImage, 0, 0, w, h);
    }

    const imgData = ctx ? ctx.getImageData(0, 0, w, h).data : null;

    let vIdx = 0;
    let uvIdx = 0;
    for (let r = 0; r < sh; r++) {
      for (let c = 0; c < sw; c++) {
        const origR = r * subsample;
        const origC = c * subsample;
        const depthIdx = origR * w + origC;
        const rawZ = this.rawDepthData[depthIdx];
        const normZ = isFinite(rawZ) ? (rawZ - dMin) / dRange : 0;

        const x = (c - sw / 2) * (100 / sw);
        const y = (sh / 2 - r) * (100 / sh);
        const z = normZ * 20.0 * this.exaggerationScale;

        positions[vIdx * 3] = x;
        positions[vIdx * 3 + 1] = y;
        positions[vIdx * 3 + 2] = z;

        const u = c / (sw - 1);
        const v = 1.0 - (r / (sh - 1));
        uvs[uvIdx * 2] = u;
        uvs[uvIdx * 2 + 1] = v;

        if (imgData) {
          const pxIdx = (origR * w + origC) * 4;
          colors[vIdx * 3] = imgData[pxIdx] / 255.0;
          colors[vIdx * 3 + 1] = imgData[pxIdx + 1] / 255.0;
          colors[vIdx * 3 + 2] = imgData[pxIdx + 2] / 255.0;
        } else {
          colors[vIdx * 3] = 0.3;
          colors[vIdx * 3 + 1] = 0.55;
          colors[vIdx * 3 + 2] = 1.0;
        }

        vIdx++;
        uvIdx++;
      }
    }

    const indices: number[] = [];
    for (let r = 0; r < sh - 1; r++) {
      for (let c = 0; c < sw - 1; c++) {
        const tl = r * sw + c;
        const tr = tl + 1;
        const bl = (r + 1) * sw + c;
        const br = bl + 1;

        indices.push(tl, bl, tr);
        indices.push(tr, bl, br);
      }
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    let textureMap: THREE.Texture | null = null;
    if (this.textureImage) {
      textureMap = new THREE.CanvasTexture(this.textureImage);
    }

    const meshMat = new THREE.MeshStandardMaterial({
      map: textureMap,
      vertexColors: !textureMap,
      wireframe: this.isWireframe,
      roughness: 0.6,
      metalness: 0.1,
      side: THREE.DoubleSide
    });
    this.meshObject = new THREE.Mesh(geometry, meshMat);
    this.meshObject.visible = (this.displayMode === 'mesh' || this.displayMode === 'both');
    this.scene.add(this.meshObject);

    const pointsMat = new THREE.PointsMaterial({
      size: 1.2,
      vertexColors: true
    });
    this.pointCloudObject = new THREE.Points(geometry, pointsMat);
    this.pointCloudObject.visible = (this.displayMode === 'pointcloud' || this.displayMode === 'both');
    this.scene.add(this.pointCloudObject);
  }

  private onWindowResize() {
    if (!this.container) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    if (width === 0 || height === 0) return;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  private animate = () => {
    requestAnimationFrame(this.animate);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };
}
