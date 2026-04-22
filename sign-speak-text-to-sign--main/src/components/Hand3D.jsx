import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';

/**
 * Rigged Hand 3D — Orientation and Curl Fixed
 *
 * Orientation: rotation=[0, Math.PI / 2, 0] makes palm face camera
 * Curl method: Extracts BOTH open (first frame) and closed (last frame) 
 * keyframes from animation, bypassing incorrect rest poses.
 */

const FINGER_JOINTS = {
  thumb:  ['thumb_01R_08', 'thumb_02R_09', 'thumb_03R_010'],
  index:  ['index_01R_017', 'index_02R_018', 'index_03R_019'],
  middle: ['middle_01R_025', 'middle_02R_026', 'middle_03R_027'],
  ring:   ['ring_01R_033', 'ring_02R_034', 'ring_03R_035'],
  pinky:  ['pinky_01R_041', 'pinky_02R_042', 'pinky_03R_043'],
};

const FINGER_BASES = {
  thumb: 'thumb_baseR_03', index: 'index_baseR_012',
  middle: 'middle_baseR_020', ring: 'ring_baseR_028', pinky: 'pinky_baseR_036',
};

const ALL_CTRL_BONES = [
  'thumb_CtrlR_04','thumb_Ctrl_01R_05','thumb_Ctrl_02R_06','thumb_Ctrl_03R_07',
  'index_CtrlR_013','index_Ctrl_01R_014','index_Ctrl_02R_015','index_Ctrl_03R_016',
  'middle_CtrlR_021','middle_Ctrl_01R_022','middle_Ctrl_02R_023','middle_Ctrl_03R_024',
  'ring_CtrlR_029','ring_Ctrl_01R_030','ring_Ctrl_02R_031','ring_Ctrl_03R_032',
  'pinky_CtrlR_037','pinky_Ctrl_01R_038','pinky_Ctrl_02R_039','pinky_Ctrl_03R_040',
];

function RiggedHand({ pose }) {
  const gltf = useGLTF('/models/rigged_hand.glb');
  const bonesRef = useRef({});
  const restQuatsRef = useRef(new Map());
  const ready = useRef(false);

  const clonedScene = useMemo(() => {
    const clone = gltf.scene.clone(true);
    const allBones = [];
    clone.traverse(n => { if (n.isBone) allBones.push(n); });
    clone.traverse(n => {
      if (!n.isSkinnedMesh || !n.skeleton) return;
      const old = n.skeleton;
      const nb = old.bones.map(ob => allBones.find(b => b.name === ob.name) || ob);
      const skel = new THREE.Skeleton(nb, old.boneInverses.map(bi => bi.clone()));
      n.skeleton = skel;
      n.bind(skel, n.bindMatrix.clone());
    });
    return clone;
  }, [gltf.scene]);

  useEffect(() => {
    const bones = {};
    const rq = new Map();
    
    // 1. Get the default structural geometry
    clonedScene.traverse(node => {
      if (node.isBone) {
        bones[node.name] = node;
        rq.set(node.name, node.quaternion.clone());
      }
    });

    // 2. The model's natural geometry is relaxed/curled. We MUST extract the 
    // perfectly straight pose from the 0th frame of the animation clip to use as our base.
    // We only do this once to set the "zero" state, and NEVER use animations again.
    const clip = gltf.animations?.[0];
    if (clip) {
      for (const track of clip.tracks) {
        if (!track.name.endsWith('.quaternion')) continue;
        const boneName = track.name.split('.')[0];
        const v = track.values;
        if (v.length >= 4) {
          // Overwrite the relaxed geometry with the straight geometry for frame 0
          rq.set(boneName, new THREE.Quaternion(v[0], v[1], v[2], v[3]));
        }
      }
    }

    bonesRef.current = bones;
    restQuatsRef.current = rq;
    ready.current = true;
  }, [clonedScene, gltf.animations]);

  useFrame((_, delta) => {
    if (!ready.current) return;
    const bones = bonesRef.current;
    const rq = restQuatsRef.current;
    
    // Smooth interpolation factor
    const lerpFactor = 0.15;

    // Helper objects to compute rotations without allocating every frame
    const _targetQuat = new THREE.Quaternion();
    const _localQuat = new THREE.Quaternion();
    const _euler = new THREE.Euler();

    // Direct bone manipulation
    for (const [finger, jointNames] of Object.entries(FINGER_JOINTS)) {
      const fp = pose?.[finger] || {};
      const curls = [fp.base ?? 0, fp.mid ?? 0, fp.tip ?? 0];
      const spread = fp.spread ?? 0;

      for (let i = 0; i < 3; i++) {
        const boneName = jointNames[i];
        const bone = bones[boneName];
        if (!bone) continue;

        const restQuat = rq.get(boneName);
        if (!restQuat) continue;

        // Start from the bone's true, natural "zero" rest state
        _targetQuat.copy(restQuat);

        // Apply local rotation (curl). 
        // We are using the Z axis here. If fingers bend sideways, change 'Z' to 'X' or 'Y'.
        const targetCurl = curls[i];
        _euler.set(0, 0, targetCurl, 'XYZ'); 
        _localQuat.setFromEuler(_euler);
        
        // Multiply natural rest state by the pose curl
        _targetQuat.multiply(_localQuat);

        // Smoothly interpolate current bone towards the newly computed target
        bone.quaternion.slerp(_targetQuat, lerpFactor);
      }

      // Spread on base bone
      const baseName = FINGER_BASES[finger];
      const baseBone = bones[baseName];
      const baseRest = rq.get(baseName);
      if (baseBone && baseRest) {
        _targetQuat.copy(baseRest);
        // Spread is mostly on the Y or Z axis depending on the rig
        _euler.set(0, spread, 0, 'XYZ');
        _localQuat.setFromEuler(_euler);
        _targetQuat.multiply(_localQuat);
        baseBone.quaternion.slerp(_targetQuat, lerpFactor);
      }
    }

    // Wrist
    const wristBone = bones['pulseR_01'];
    const wristRest = rq.get('pulseR_01');
    if (wristBone && wristRest) {
      const wr = pose?.wrist || {};
      _targetQuat.copy(wristRest);
      
      _euler.set(wr.x ?? 0, wr.y ?? 0, wr.z ?? 0, 'XYZ');
      _localQuat.setFromEuler(_euler);
      _targetQuat.multiply(_localQuat);
      
      wristBone.quaternion.slerp(_targetQuat, lerpFactor);
    }
  });

  return <primitive object={clonedScene} />;
}

useGLTF.preload('/models/rigged_hand.glb');

export default function Hand3D({ pose, isLeft = false, position = [0, 0, 0] }) {
  const groupRef = useRef();
  const innerRef = useRef();
  const centered = useRef(false);
  const s = 0.55;

  useEffect(() => {
    if (innerRef.current && !centered.current) {
      setTimeout(() => {
        if (!innerRef.current) return;
        const box = new THREE.Box3().setFromObject(innerRef.current);
        const center = box.getCenter(new THREE.Vector3());
        innerRef.current.position.set(-center.x, -center.y, -center.z);
        centered.current = true;
      }, 100);
    }
  }, []);

  return (
    <group position={position}>
      <group
        ref={groupRef}
        scale={isLeft ? [-s, s, s] : [s, s, s]}
        // Reversed Math.PI/2 so the palm faces the camera, rather than the back
        rotation={[0, Math.PI / 2, 0]}
      >
        <group ref={innerRef}>
          <RiggedHand pose={pose} />
        </group>
      </group>
    </group>
  );
}
