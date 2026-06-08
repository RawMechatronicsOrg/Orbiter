/**
 * Node-type → three.js object factory registry.
 *
 * This is the heart of the domain-agnostic renderer: it knows how to turn an
 * abstract `SceneNode` into a three.js object, and nothing about Orbiter.
 * New node types are added here only.
 */

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import type { SceneNode } from './protocol';

/* eslint-disable @typescript-eslint/no-explicit-any -- node props are dynamic JSON */
type Props = any;

interface NodeFactory {
  create(props: Props): THREE.Object3D;
}

// ── shared builders ─────────────────────────────────────────────────────────

function makeMaterial(spec: Props): THREE.Material {
  const color = new THREE.Color((spec?.color as string) ?? '#cccccc');
  const side =
    spec?.side === 'double'
      ? THREE.DoubleSide
      : spec?.side === 'back'
        ? THREE.BackSide
        : THREE.FrontSide;
  const common = {
    color,
    side,
    transparent: !!spec?.transparent,
    opacity: spec?.opacity ?? 1,
    // Optional `depthWrite: false` — required on translucent overlays
    // that should not occlude opaque-from-the-other-side geometry.
    depthWrite: spec?.depthWrite !== false,
  };
  if (spec?.type === 'basic') return new THREE.MeshBasicMaterial(common);
  return new THREE.MeshStandardMaterial({ ...common, metalness: 0.1, roughness: 0.7 });
}

function makeGeometry(prim: Props): THREE.BufferGeometry {
  switch (prim?.kind) {
    case 'box':
      return new THREE.BoxGeometry(prim.width ?? 1, prim.height ?? 1, prim.depth ?? 1);
    case 'sphere':
      return new THREE.SphereGeometry(prim.radius ?? 1, prim.segments ?? 24, prim.segments ?? 16);
    case 'cylinder':
      return new THREE.CylinderGeometry(
        prim.radiusTop ?? prim.radius ?? 1,
        prim.radiusBottom ?? prim.radius ?? 1,
        prim.height ?? 1,
        prim.segments ?? 24,
      );
    case 'circle':
      return new THREE.CircleGeometry(prim.radius ?? 1, prim.segments ?? 48);
    case 'plane':
      return new THREE.PlaneGeometry(prim.width ?? 1, prim.height ?? 1);
    case 'torus':
      return new THREE.TorusGeometry(
        prim.radius ?? 1, prim.tube ?? 0.3, prim.radialSegments ?? 8, prim.segments ?? 24,
      );
    default:
      return new THREE.BoxGeometry(1, 1, 1);
  }
}

function geometryFromBuffers(props: Props): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(props.vertices as number[], 3));
  if (props.indices) geo.setIndex(props.indices as number[]);
  if (props.normals) {
    geo.setAttribute('normal', new THREE.Float32BufferAttribute(props.normals as number[], 3));
  } else {
    geo.computeVertexNormals();
  }
  return geo;
}

/** Sprite with rasterised text — the `label` node renderer. */
function makeTextSprite(text: string, fontSize: number, color: string): THREE.Sprite {
  const FONT_PX = 64;
  const font = `${FONT_PX}px ui-monospace, monospace`;
  const measure = document.createElement('canvas').getContext('2d')!;
  measure.font = font;
  const textW = Math.max(1, Math.ceil(measure.measureText(text).width));
  const canvas = document.createElement('canvas');
  canvas.width = textW + 16;
  canvas.height = FONT_PX + 16;
  const ctx = canvas.getContext('2d')!;
  ctx.font = font;
  ctx.fillStyle = color;
  ctx.textBaseline = 'middle';
  ctx.fillText(text, 8, canvas.height / 2);
  const tex = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }),
  );
  sprite.scale.set((canvas.width / canvas.height) * fontSize, fontSize, 1);
  return sprite;
}

// ── factories ───────────────────────────────────────────────────────────────

const frameFactory: NodeFactory = {
  create(props) {
    const group = new THREE.Group();
    if (props?.axesLength) group.add(new THREE.AxesHelper(props.axesLength));
    return group;
  },
};

const gridFactory: NodeFactory = {
  create(props) {
    // GridHelper lies in the XZ plane; the world is Z-up, so wrap it in a
    // group rotated into XY — the node transform then applies cleanly.
    const group = new THREE.Group();
    const helper = new THREE.GridHelper(
      props?.size ?? 100,
      props?.divisions ?? 10,
      props?.color ?? '#444444',
      props?.color ?? '#444444',
    );
    helper.rotation.x = Math.PI / 2;
    group.add(helper);
    return group;
  },
};

const meshFactory: NodeFactory = {
  create(props) {
    const geometry =
      props?.vertices !== undefined
        ? geometryFromBuffers(props)
        : makeGeometry(props?.primitive);
    return new THREE.Mesh(geometry, makeMaterial(props?.material));
  },
};

const lineSegmentsFactory: NodeFactory = {
  create(props) {
    const flat: number[] = [];
    for (const p of (props?.points as number[][]) ?? []) {
      flat.push(p[0], p[1], p[2]);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));
    return new THREE.LineSegments(
      geo,
      new THREE.LineBasicMaterial({
        color: new THREE.Color((props?.color as string) ?? '#ffffff'),
        transparent: !!props?.transparent,
        opacity: props?.opacity ?? 1,
      }),
    );
  },
};

const pointCloudFactory: NodeFactory = {
  create(props) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute(
      'position',
      new THREE.Float32BufferAttribute((props?.positions as number[]) ?? [], 3),
    );
    const hasColors = Array.isArray(props?.colors);
    if (hasColors) {
      geo.setAttribute('color', new THREE.Float32BufferAttribute(props.colors as number[], 3));
    }
    return new THREE.Points(
      geo,
      new THREE.PointsMaterial({
        size: props?.pointSize ?? 3,
        color: new THREE.Color((props?.color as string) ?? '#ffffff'),
        vertexColors: hasColors,
        sizeAttenuation: true,
      }),
    );
  },
};

/** Wireframe camera frustum: apex at the camera, base forward along -Z. */
const cameraFrustumFactory: NodeFactory = {
  create(props) {
    const group = new THREE.Group();
    const d = props?.scale ?? 18;
    // Base sized to the camera frame aspect (width / height): >1 landscape,
    // <1 portrait. The larger half-extent is pinned to 0.6·d so the frustum
    // keeps a consistent size regardless of orientation.
    const aspect =
      typeof props?.aspect === 'number' && props.aspect > 0 ? props.aspect : 1.4;
    const w = aspect >= 1 ? 0.6 * d : 0.6 * d * aspect;
    const h = aspect >= 1 ? (0.6 * d) / aspect : 0.6 * d;
    const apex: [number, number, number] = [0, 0, 0];
    const tl: [number, number, number] = [-w, h, -d];
    const tr: [number, number, number] = [w, h, -d];
    const br: [number, number, number] = [w, -h, -d];
    const bl: [number, number, number] = [-w, -h, -d];
    const edges: [number, number, number][][] = [
      [apex, tl], [apex, tr], [apex, br], [apex, bl],
      [tl, tr], [tr, br], [br, bl], [bl, tl],
    ];
    const flat: number[] = [];
    for (const [a, b] of edges) flat.push(...a, ...b);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));
    const color = new THREE.Color((props?.color as string) ?? '#f97316');
    group.add(
      new THREE.LineSegments(
        geo,
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }),
      ),
    );
    group.add(
      new THREE.Mesh(
        new THREE.SphereGeometry(1.2, 8, 8),
        new THREE.MeshBasicMaterial({ color }),
      ),
    );
    return group;
  },
};

/** Textured quad — a captured photo at its camera plane. */
const imagePlaneFactory: NodeFactory = {
  create(props) {
    const mesh = new THREE.Mesh(
      new THREE.PlaneGeometry(props?.width ?? 1, props?.height ?? 1),
      new THREE.MeshBasicMaterial({
        color: 0xffffff,
        side: THREE.DoubleSide,
        transparent: true,
      }),
    );
    if (props?.url) {
      new THREE.TextureLoader().load(props.url, (tex) => {
        const mat = mesh.material as THREE.MeshBasicMaterial;
        mat.map = tex;
        mat.needsUpdate = true;
      });
    }
    return mesh;
  },
};

const labelFactory: NodeFactory = {
  create(props) {
    return makeTextSprite(
      String(props?.text ?? ''),
      props?.fontSize ?? 8,
      (props?.color as string) ?? '#e2e8f0',
    );
  },
};

/** Cached glTF/GLB loader. Front-end loads the same `Orbiter.glb` the back-end
 *  uses for geometry queries, so part names line up 1:1 with `rig_model.py`.
 *  Every node sharing a URL waits on one Promise → `cad_part` re-creates are
 *  cheap. */
const _cadCache: Map<string, Promise<THREE.Group>> = new Map();
function loadCadCached(url: string): Promise<THREE.Group> {
  let p = _cadCache.get(url);
  if (!p) {
    p = new Promise<THREE.Group>((resolve, reject) => {
      new GLTFLoader().load(
        url,
        (gltf) => resolve(gltf.scene),
        undefined,
        reject,
      );
    });
    _cadCache.set(url, p);
  }
  return p;
}

/** Deep clone an Object3D subtree including geometries and materials.
 *  Plain `.clone(true)` shares geometry/material refs with the source;
 *  if a wrapper made from such a shallow clone is later passed through
 *  `disposeObject`, the shared resources die and every other wrapper that
 *  references the cached model renders blank. We pay the duplication cost
 *  here so the scene reconciler can dispose freely. */
function deepCloneObject<T extends THREE.Object3D>(src: T): T {
  const dst = src.clone(true) as T;
  dst.traverse((node) => {
    const mesh = node as THREE.Mesh;
    if (!(mesh as unknown as { isMesh?: boolean }).isMesh) return;
    if (mesh.geometry) mesh.geometry = mesh.geometry.clone();
    const m = mesh.material;
    if (Array.isArray(m)) {
      mesh.material = m.map((mm) => mm.clone());
    } else if (m) {
      mesh.material = (m as THREE.Material).clone();
    }
  });
  return dst;
}

/** Async-loaded CAD assembly — same `Orbiter.glb` the back-end queries via
 *  `geom/rig_model.py`. Part names line up 1:1 with `trimesh.scene.graph`.
 *  `convert_y_up` rotates Y-up glTF → our Z-up scene (default true).
 *  `scale` is uniform; for our mm-based scene with a glTF stored in metres,
 *  pass 1000. `hide_parts` masks subtrees we render separately under
 *  rotation frames via `cad_part`. */
const cadFactory: NodeFactory = {
  create(props) {
    const group = new THREE.Group();
    const url = props?.url;
    if (typeof url !== 'string') return group;
    const scale = typeof props?.scale === 'number' ? props.scale : 1;
    const convertYUp = props?.convert_y_up !== false;
    const hideParts: string[] = Array.isArray(props?.hide_parts)
      ? (props.hide_parts as string[])
      : [];
    loadCadCached(url)
      .then((root) => {
        // Deep-clone so reconciler dispose can't poison the shared cache.
        const clone = deepCloneObject(root);
        if (convertYUp) clone.rotation.x = Math.PI / 2;
        clone.scale.setScalar(scale);
        for (const name of hideParts) {
          const obj = clone.getObjectByName(name);
          if (obj) {
            obj.visible = false;
          } else {
            console.warn('cad_model: hide_parts name not found:', name);
          }
        }
        group.add(clone);
      })
      .catch((err) => console.error('GLTFLoader failed:', url, err));
    return group;
  },
};

/** A single named subtree of the cached CAD model, re-anchored under an
 *  arbitrary parent frame. Used to drive part rotation from the server
 *  scene-graph (`platform_spin`, `orbit_spin`) without reloading the model
 *  on every AZ/EL update — only the parent frame's transform changes.
 *
 *  Pivot is the parent frame's world position (mm in our Z-up world),
 *  computed by the server from `geom/rig_model.py` and passed in as
 *  `pivot_offset`. The factory subtracts it from the part's rig-world
 *  transform so the clone lands at the right place when the parent is at
 *  the pivot and unrotated.
 *
 *  NOTE: per project rule (no geometry math on the front-end), this factory
 *  performs ONE local computation — reading the part's `matrixWorld` from
 *  the CAD it just rendered. That's a presentation concern (placing the
 *  rendered subtree); the *pivot point itself* is server-authoritative. */
const cadPartFactory: NodeFactory = {
  create(props) {
    const root = new THREE.Group();
    const url = props?.url;
    const partName = props?.part_name;
    if (typeof url !== 'string' || typeof partName !== 'string') return root;
    const scale = typeof props?.scale === 'number' ? props.scale : 1;
    const convertYUp = props?.convert_y_up !== false;
    const pivotOff = (props?.pivot_offset as number[] | undefined) ?? [0, 0, 0];

    loadCadCached(url)
      .then((cad) => {
        const part = cad.getObjectByName(partName);
        if (!part) {
          console.warn('cad_part: child not found', partName, 'in', url);
          return;
        }
        // Compute part's world matrix as it would sit inside the static rig:
        //   rigOuter = rotX(π/2) * scale(s)
        //   effective = rigOuter * part.matrixWorld(within CAD)
        cad.updateMatrixWorld(true);
        const rigRot = convertYUp
          ? new THREE.Matrix4().makeRotationX(Math.PI / 2)
          : new THREE.Matrix4();
        const rigScale = new THREE.Matrix4().makeScale(scale, scale, scale);
        const rigOuter = new THREE.Matrix4().multiplyMatrices(rigRot, rigScale);
        const effective = new THREE.Matrix4().multiplyMatrices(rigOuter, part.matrixWorld);
        // Subtract pivot so the parent frame sits at the part's rotation axis.
        const offsetMat = new THREE.Matrix4().makeTranslation(
          -pivotOff[0], -pivotOff[1], -pivotOff[2],
        );
        const local = new THREE.Matrix4().multiplyMatrices(offsetMat, effective);
        const pos = new THREE.Vector3();
        const quat = new THREE.Quaternion();
        const scl = new THREE.Vector3();
        local.decompose(pos, quat, scl);

        // Deep-clone the subtree so the reconciler can dispose it without
        // wrecking the cached FBX's shared geometries/materials.
        const clone = deepCloneObject(part);
        clone.position.copy(pos);
        clone.quaternion.copy(quat);
        clone.scale.copy(scl);
        root.add(clone);
      })
      .catch((err) => console.error('cad_part load failed:', url, partName, err));
    return root;
  },
};

/** Subject-disk geometry with a procedurally-rendered dial texture on top.
 *  High-contrast radial lines through the entire grid plus dense rim labels.
 *  Renders the side + bottom as a plain coloured cylinder, the top cap as
 *  a CircleGeometry textured with a CanvasTexture. */
const discDialFactory: NodeFactory = {
  create(props) {
    const radius = (props?.radius as number | undefined) ?? 140;
    const height = (props?.height as number | undefined) ?? 15;
    const ground = (props?.ground_color as string | undefined) ?? '#0e0e12';
    const marks = (props?.marks_color as string | undefined) ?? '#f5f5f7';
    const labelClr = (props?.label_color as string | undefined) ?? '#ffffff';
    const tickStep = (props?.tick_step_deg as number | undefined) ?? 5;
    const majorStep = (props?.major_step_deg as number | undefined) ?? 30;
    const labelStep = (props?.label_step_deg as number | undefined) ?? 10;
    const labelMajorPx = (props?.label_major_px as number | undefined) ?? 44;
    const labelMinorPx = (props?.label_minor_px as number | undefined) ?? 22;

    const group = new THREE.Group();

    // Side + bottom — uniform colour, shared material.
    const baseMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(ground),
      metalness: 0.15,
      roughness: 0.7,
    });
    const side = new THREE.Mesh(
      new THREE.CylinderGeometry(radius, radius, height, 96, 1, true),
      baseMat,
    );
    side.rotation.x = Math.PI / 2; // stand cylinder along world +Z
    group.add(side);
    const bot = new THREE.Mesh(new THREE.CircleGeometry(radius, 96), baseMat);
    bot.position.z = -height / 2;
    bot.rotation.x = Math.PI; // face -Z
    group.add(bot);

    // Top — CanvasTexture mapped on a circle.
    const SIZE = 1024;
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = SIZE;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      const cx = SIZE / 2;
      const cy = SIZE / 2;
      const R = SIZE / 2 - 4;
      ctx.fillStyle = ground;
      ctx.fillRect(0, 0, SIZE, SIZE);
      // Concentric rings — subtle radial grid.
      ctx.strokeStyle = marks;
      ctx.globalAlpha = 0.45;
      for (const t of [0.2, 0.4, 0.6, 0.8]) {
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(cx, cy, R * t, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(cx, cy, R - 1, 0, Math.PI * 2);
      ctx.stroke();

      // Radial ticks — major lines run from centre to rim, mid lines from
      // mid-radius, minor lines just on the outer band.
      for (let deg = 0; deg < 360; deg += tickStep) {
        const isMajor = deg % majorStep === 0;
        const isMid = !isMajor && deg % 10 === 0;
        const r0 = isMajor ? 0 : isMid ? R * 0.45 : R * 0.84;
        const r1 = R;
        ctx.lineWidth = isMajor ? 3.5 : isMid ? 1.8 : 0.9;
        ctx.globalAlpha = isMajor ? 1 : isMid ? 0.9 : 0.65;
        // canvas y is downward — flip so 0° is at top of canvas.
        const a = ((deg - 90) * Math.PI) / 180;
        ctx.beginPath();
        ctx.moveTo(cx + r0 * Math.cos(a), cy + r0 * Math.sin(a));
        ctx.lineTo(cx + r1 * Math.cos(a), cy + r1 * Math.sin(a));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      // Rim labels — placed ALONG the radial tick lines they label (not
      // tangent to the rim). On the left half (cos a < 0) we rotate by a+π
      // so labels read outward-then-inward but stay right-side-up to the
      // viewer. Font is a tighter, less rounded sans for an engineering
      // dial vibe.
      ctx.fillStyle = labelClr;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const labelFamily =
        '"Tahoma", "Arial Narrow", "DejaVu Sans Condensed", "Helvetica Neue Condensed", sans-serif';
      for (let deg = 0; deg < 360; deg += labelStep) {
        const isMajor = deg % majorStep === 0;
        const px = isMajor ? labelMajorPx : labelMinorPx;
        ctx.font = `${isMajor ? 'bold ' : ''}${px}px ${labelFamily}`;
        const a = ((deg - 90) * Math.PI) / 180;
        // Place text centre between the outer tick start and the rim so it
        // sits along the line without overlapping its end.
        const r = R * 0.78;
        const x = cx + r * Math.cos(a);
        const y = cy + r * Math.sin(a);
        ctx.save();
        ctx.translate(x, y);
        // Flip on the bottom-left half so the digits don't read upside-down.
        const flip = Math.cos(a) < -1e-6;
        ctx.rotate(flip ? a + Math.PI : a);
        ctx.fillText(`${deg}°`, 0, 0);
        ctx.restore();
      }
    }
    const tex = new THREE.CanvasTexture(canvas);
    tex.anisotropy = 8;
    tex.needsUpdate = true;
    const topMat = new THREE.MeshStandardMaterial({
      map: tex,
      color: 0xffffff,
      metalness: 0.05,
      roughness: 0.6,
    });
    const top = new THREE.Mesh(new THREE.CircleGeometry(radius, 96), topMat);
    top.position.z = height / 2;
    group.add(top);

    return group;
  },
};

/** Fallback for unimplemented node types. */
const unknownFactory: NodeFactory = {
  create() {
    return new THREE.Group();
  },
};

const REGISTRY: Record<string, NodeFactory> = {
  frame: frameFactory,
  grid: gridFactory,
  mesh: meshFactory,
  line_segments: lineSegmentsFactory,
  point_cloud: pointCloudFactory,
  camera_frustum: cameraFrustumFactory,
  image_plane: imagePlaneFactory,
  label: labelFactory,
  cad_model: cadFactory,
  cad_part: cadPartFactory,
  disc_dial: discDialFactory,
};

// ── public API ──────────────────────────────────────────────────────────────

/** Build a three.js object for a node (transform/visibility applied by the
 *  reconciler, not here). */
export function createObject(node: SceneNode): THREE.Object3D {
  const factory = REGISTRY[node.type] ?? unknownFactory;
  return factory.create(node.props);
}

/** Recursively release GPU resources held by an object subtree. */
export function disposeObject(obj: THREE.Object3D): void {
  obj.traverse((child) => {
    const withGeo = child as THREE.Mesh;
    if (withGeo.geometry) withGeo.geometry.dispose();
    const mat = (child as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
    for (const m of Array.isArray(mat) ? mat : mat ? [mat] : []) {
      const tex = (m as THREE.MeshBasicMaterial).map;
      if (tex) tex.dispose();
      m.dispose();
    }
  });
}
