import React, { useRef, useEffect, useState } from 'react';
import { FilesetResolver, HandLandmarker } from '@mediapipe/tasks-vision';

/**
 * Calculates the curl angle between three points in 3D space: A -> B -> C
 * B is the joint.
 * Returns the angle in radians representing how much the joint is bent.
 * 0 means completely straight.
 */
function calculateCurlAngle(p1, p2, p3) {
  // Vector p1 -> p2
  const v1 = { x: p2.x - p1.x, y: p2.y - p1.y, z: p2.z - p1.z };
  // Vector p2 -> p3
  const v2 = { x: p3.x - p2.x, y: p3.y - p2.y, z: p3.z - p2.z };

  const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
  const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);

  if (mag1 === 0 || mag2 === 0) return 0;

  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  let cosTheta = dot / (mag1 * mag2);
  cosTheta = Math.max(-1.0, Math.min(1.0, cosTheta)); // clamp
  
  // Angle between the direction vectors
  let angle = Math.acos(cosTheta);
  
  // Hand3D uses mostly negative Z for curl in its coordinate space
  return -angle; 
}

/**
 * Converts MediaPipe 21 landmarks into our Hand3D ASL Pose format
 */
function mapLandmarksToPose(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;

  // Joint Indices based on MediaPipe Hand tracking:
  // Thumb: 1, 2, 3, 4 (Wrist to Tip)
  // Index: 5, 6, 7, 8
  // Middle: 9, 10, 11, 12
  // Ring: 13, 14, 15, 16
  // Pinky: 17, 18, 19, 20
  const lm = landmarks;
  
  return {
    thumb: {
      base: calculateCurlAngle(lm[0], lm[1], lm[2]),
      mid: calculateCurlAngle(lm[1], lm[2], lm[3]),
      tip: calculateCurlAngle(lm[2], lm[3], lm[4]),
    },
    index: {
      base: calculateCurlAngle(lm[0], lm[5], lm[6]),
      mid: calculateCurlAngle(lm[5], lm[6], lm[7]),
      tip: calculateCurlAngle(lm[6], lm[7], lm[8]),
    },
    middle: {
      base: calculateCurlAngle(lm[0], lm[9], lm[10]),
      mid: calculateCurlAngle(lm[9], lm[10], lm[11]),
      tip: calculateCurlAngle(lm[10], lm[11], lm[12]),
    },
    ring: {
      base: calculateCurlAngle(lm[0], lm[13], lm[14]),
      mid: calculateCurlAngle(lm[13], lm[14], lm[15]),
      tip: calculateCurlAngle(lm[14], lm[15], lm[16]),
    },
    pinky: {
      base: calculateCurlAngle(lm[0], lm[17], lm[18]),
      mid: calculateCurlAngle(lm[17], lm[18], lm[19]),
      tip: calculateCurlAngle(lm[18], lm[19], lm[20]),
    },
    wrist: { x: 0, y: 0, z: 0 } // Wrist can be expanded later if need be
  };
}

export default function MocapStudio({ onPoseDetected }) {
  const videoRef = useRef(null);
  const [isReady, setIsReady] = useState(false);
  const [isTracking, setIsTracking] = useState(false);
  const landmarkerRef = useRef(null);
  const rafId = useRef(null);

  useEffect(() => {
    async function initMediaPipe() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/models/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        landmarkerRef.current = handLandmarker;
        setIsReady(true);
      } catch (err) {
        console.error("Error initializing MediaPipe:", err);
      }
    }
    initMediaPipe();
    
    return () => {
      if (landmarkerRef.current) landmarkerRef.current.close();
      if (rafId.current) cancelAnimationFrame(rafId.current);
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
        setIsTracking(true);
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
    }
    setIsTracking(false);
    if (rafId.current) cancelAnimationFrame(rafId.current);
    if (onPoseDetected) onPoseDetected(null);
  };

  const processVideoFrame = () => {
    if (!videoRef.current || !landmarkerRef.current || !isTracking) return;
    
    // We only process if the video has started playing
    if (videoRef.current.currentTime > 0) {
      const results = landmarkerRef.current.detectForVideo(videoRef.current, performance.now());
      
      if (results.landmarks && results.landmarks.length > 0) {
        // Grab the first hand detected
        const rawLandmarks = results.landmarks[0];
        const mappedPose = mapLandmarksToPose(rawLandmarks);
        if (mappedPose && onPoseDetected) {
          onPoseDetected(mappedPose);
        }
      }
    }
    rafId.current = requestAnimationFrame(processVideoFrame);
  };

  useEffect(() => {
    if (isTracking) {
      rafId.current = requestAnimationFrame(processVideoFrame);
    } else {
      if (rafId.current) cancelAnimationFrame(rafId.current);
    }
  }, [isTracking]);

  return (
    <div className="glass p-4 rounded-xl flex flex-col items-center justify-center gap-4">
      <div className="flex items-center justify-between w-full h-8 px-2">
        <h3 className="font-bold text-text-primary text-sm flex items-center gap-2">
          <span className="text-secondary tracking-widest uppercase text-xs">Live</span>
          Mocap Studio
        </h3>
        <span className={`text-[10px] font-mono px-2 py-1 rounded-full ${isReady ? 'bg-success/20 text-success' : 'bg-warning/20 text-warning'}`}>
          {isReady ? "AI Ready" : "Loading Model..."}
        </span>
      </div>
      
      <div className="w-full h-48 bg-surface-dark rounded-lg overflow-hidden relative border border-surface-light border-dashed flex items-center justify-center">
        <video 
          ref={videoRef} 
          className="absolute inset-0 w-full h-full object-cover transform -scale-x-100" 
          playsInline
          muted
          autoPlay
        />
        {!isTracking && (
          <p className="text-xs text-text-muted absolute z-10 text-center px-4">
            Camera is off. Start tracking to map your hand to the 3D model.
          </p>
        )}
      </div>

      <button
        disabled={!isReady}
        onClick={isTracking ? stopCamera : startCamera}
        className={`w-full py-2 rounded-lg font-bold text-sm transition-all duration-300 ${
          isTracking 
          ? 'bg-error/20 text-error hover:bg-error/30' 
          : 'bg-primary text-surface hover:bg-primary-light disabled:opacity-50 disabled:cursor-not-allowed'
        }`}
      >
        {isTracking ? "Stop Tracking" : "Start Live Mocap"}
      </button>
    </div>
  );
}
