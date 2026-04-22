/**
 * Animation Controller Hook
 * 
 * Manages the playback of ASL animation sequences.
 * Handles play/pause, speed control, and smooth transitions.
 */
import { useState, useCallback, useRef, useEffect } from 'react';
import { translateToASL, interpolatePose, getTransitionDuration } from './translator';
import { REST_POSE } from '../data/aslPoses';

export function useAnimationController() {
  const [sequence, setSequence] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [currentPose, setCurrentPose] = useState(REST_POSE);
  const [currentLabel, setCurrentLabel] = useState('');
  const [progress, setProgress] = useState(0);
  const [inputText, setInputText] = useState('');

  const animFrameRef = useRef(null);
  const startTimeRef = useRef(0);
  const currentIndexRef = useRef(0);
  const sequenceRef = useRef([]);
  const isPlayingRef = useRef(false);
  const speedRef = useRef(1);
  const prevPoseRef = useRef(REST_POSE);

  // Keep refs in sync
  useEffect(() => { speedRef.current = speed; }, [speed]);
  useEffect(() => { isPlayingRef.current = isPlaying; }, [isPlaying]);
  useEffect(() => { sequenceRef.current = sequence; }, [sequence]);

  const animate = useCallback((timestamp) => {
    if (!isPlayingRef.current) return;

    const seq = sequenceRef.current;
    const idx = currentIndexRef.current;

    if (idx >= seq.length) {
      setIsPlaying(false);
      setCurrentPose(REST_POSE);
      setCurrentLabel('Complete');
      setProgress(1);
      return;
    }

    const step = seq[idx];
    const elapsed = (timestamp - startTimeRef.current) * speedRef.current;
    const transitionTime = getTransitionDuration(prevPoseRef.current, step.pose);
    const totalDuration = step.duration + transitionTime;
    const t = Math.min(elapsed / totalDuration, 1);

    // Calculate interpolated pose
    if (elapsed < transitionTime) {
      // Transition phase
      const transT = elapsed / transitionTime;
      const interpolated = interpolatePose(prevPoseRef.current, step.pose, transT);
      setCurrentPose(interpolated);
    } else {
      // Hold phase - stay at target pose
      setCurrentPose(step.pose);
    }

    setCurrentLabel(step.label || '');
    setProgress((idx + t) / seq.length);

    if (t >= 1) {
      // Move to next step
      prevPoseRef.current = step.pose;
      currentIndexRef.current = idx + 1;
      setCurrentIndex(idx + 1);
      startTimeRef.current = timestamp;
    }

    animFrameRef.current = requestAnimationFrame(animate);
  }, []);

  const play = useCallback(() => {
    if (sequence.length === 0) return;
    
    setIsPlaying(true);
    startTimeRef.current = performance.now();
    
    if (currentIndexRef.current >= sequence.length) {
      currentIndexRef.current = 0;
      setCurrentIndex(0);
      prevPoseRef.current = REST_POSE;
    }

    animFrameRef.current = requestAnimationFrame(animate);
  }, [sequence, animate]);

  const pause = useCallback(() => {
    setIsPlaying(false);
    if (animFrameRef.current) {
      cancelAnimationFrame(animFrameRef.current);
    }
  }, []);

  const reset = useCallback(() => {
    pause();
    currentIndexRef.current = 0;
    setCurrentIndex(0);
    setProgress(0);
    setCurrentPose(REST_POSE);
    setCurrentLabel('');
    prevPoseRef.current = REST_POSE;
  }, [pause]);

  const translateText = useCallback((text) => {
    setInputText(text);
    const newSequence = translateToASL(text);
    setSequence(newSequence);
    reset();
    
    if (newSequence.length > 0) {
      // Auto-play
      setTimeout(() => {
        currentIndexRef.current = 0;
        sequenceRef.current = newSequence;
        prevPoseRef.current = REST_POSE;
        setIsPlaying(true);
        isPlayingRef.current = true;
        startTimeRef.current = performance.now();
        animFrameRef.current = requestAnimationFrame(animate);
      }, 100);
    }
  }, [reset, animate]);

  const adjustSpeed = useCallback((newSpeed) => {
    setSpeed(Math.max(0.25, Math.min(3, newSpeed)));
  }, []);

  // Cleanup
  useEffect(() => {
    return () => {
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, []);

  return {
    currentPose,
    currentLabel,
    currentIndex,
    sequence,
    isPlaying,
    speed,
    progress,
    inputText,
    play,
    pause,
    reset,
    translateText,
    adjustSpeed,
  };
}
