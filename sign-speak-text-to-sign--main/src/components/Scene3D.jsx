import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Environment, ContactShadows, PerspectiveCamera } from '@react-three/drei';
import Hand3D from './Hand3D';

function SceneContent({ pose }) {
  return (
    <>
      {/* Camera positioned to frame the hand nicely */}
      <PerspectiveCamera makeDefault position={[0, 0.5, 3.5]} fov={38} />
      <OrbitControls
        enablePan={false}
        enableZoom={true}
        minDistance={2}
        maxDistance={6}
        minPolarAngle={Math.PI / 6}
        maxPolarAngle={Math.PI * 0.75}
        autoRotate={false}
        target={[0, 0.5, 0]}
      />
      
      {/* Lighting optimized for skin rendering */}
      <ambientLight intensity={0.6} color="#e8ddd0" />
      <directionalLight position={[3, 5, 5]} intensity={1.6} color="#fff5e6" castShadow
        shadow-mapSize={[2048, 2048]} shadow-bias={-0.0001} />
      <directionalLight position={[-3, 3, -2]} intensity={0.5} color="#c4d4ff" />
      {/* Rim light for depth */}
      <pointLight position={[0, 4, -3]} intensity={0.6} color="#818cf8" />
      {/* Fill light from below */}
      <pointLight position={[0, -2, 3]} intensity={0.3} color="#06b6d4" />
      {/* Warm key light on the palm side */}
      <pointLight position={[2, 1, 4]} intensity={0.4} color="#ffd4a8" />
      
      {/* Hand placement - centered in viewport */}
      {pose?.left && <Hand3D pose={pose.left} isLeft={true} position={[-0.8, 0.5, 0]} />}
      {pose?.right ? (
        <Hand3D pose={pose.right} isLeft={false} position={[0.8, 0.5, 0]} />
      ) : (pose && !pose.left && !pose.right) ? (
        <Hand3D pose={pose} isLeft={false} position={[0, 0.5, 0]} />
      ) : null}
      
      <ContactShadows position={[0, -0.8, 0]} opacity={0.4} scale={5} blur={2.5} far={3} color="#1a1a3f" />
      <Environment preset="city" environmentIntensity={0.4} />
    </>
  );
}

export default function Scene3D({ pose }) {
  return (
    <div className="w-full h-full" id="scene-3d-container">
      <Canvas shadows dpr={[1, 2]}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance', stencil: false }}
        style={{ background: 'transparent' }}>
        <Suspense fallback={
          <mesh><boxGeometry args={[0.4,0.4,0.4]} /><meshStandardMaterial color="#6366f1" wireframe /></mesh>
        }>
          <SceneContent pose={pose} />
        </Suspense>
      </Canvas>
    </div>
  );
}
