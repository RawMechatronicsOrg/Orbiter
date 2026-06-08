/**
 * Raycast picking for the generic viewer. Walks up from the hit object to the
 * nearest node marked `pickable`, so clicking any sub-mesh of a node reports
 * that node's id back to the server.
 */

import * as THREE from 'three';

export interface PickHit {
  nodeId: string;
  point: [number, number, number];
}

export function raycastPick(
  raycaster: THREE.Raycaster,
  camera: THREE.Camera,
  ndc: THREE.Vector2,
  root: THREE.Object3D,
): PickHit | null {
  raycaster.setFromCamera(ndc, camera);
  for (const hit of raycaster.intersectObject(root, true)) {
    let obj: THREE.Object3D | null = hit.object;
    while (obj) {
      if (obj.userData?.pickable && obj.userData?.nodeId) {
        return {
          nodeId: obj.userData.nodeId as string,
          point: [hit.point.x, hit.point.y, hit.point.z],
        };
      }
      obj = obj.parent;
    }
  }
  return null;
}
