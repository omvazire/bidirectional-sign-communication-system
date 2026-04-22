/**
 * Sign Language Translation Engine
 * 
 * Converts English text into a sequence of ASL poses.
 * Uses word-level signs when available, falls back to fingerspelling.
 */
import { ASL_POSES, WORD_SIGNS, REST_POSE } from '../data/aslPoses';

const LETTER_DURATION = 700; // ms per letter
const SPACE_DURATION = 500; // ms for spaces between words
const TRANSITION_DURATION = 250; // ms for smooth transitions

/**
 * Parse input text into a sequence of animation steps
 */
export function translateToASL(text) {
  if (!text || !text.trim()) return [];

  const words = text.trim().toLowerCase().split(/\s+/);
  const sequence = [];

  words.forEach((word, wordIndex) => {
    // Check if word has a dedicated sign
    const cleanWord = word.replace(/[^a-z0-9]/g, '');
    
    if (WORD_SIGNS[cleanWord]) {
      const sign = WORD_SIGNS[cleanWord];
      sign.poses.forEach((pose, i) => {
        sequence.push({
          type: 'word_sign',
          word: cleanWord,
          pose,
          duration: pose.duration || 600,
          label: `"${cleanWord}" (${sign.description})`,
          isFirst: i === 0,
          isLast: i === sign.poses.length - 1,
        });
      });
    } else {
      // Fingerspell the word
      const letters = cleanWord.split('');
      letters.forEach((letter, letterIndex) => {
        const upperLetter = letter.toUpperCase();
        const pose = ASL_POSES[upperLetter] || ASL_POSES[letter];
        
        if (pose) {
          sequence.push({
            type: 'fingerspell',
            letter: upperLetter,
            pose,
            duration: LETTER_DURATION,
            label: `Letter "${upperLetter}"`,
            isFirst: letterIndex === 0,
            isLast: letterIndex === letters.length - 1,
            word: cleanWord,
          });
        }
      });
    }

    // Add space between words
    if (wordIndex < words.length - 1) {
      sequence.push({
        type: 'space',
        pose: REST_POSE,
        duration: SPACE_DURATION,
        label: '(pause)',
      });
    }
  });

  return sequence;
}

/**
 * Interpolate between two poses for smooth animation
 */
export function interpolatePose(poseA, poseB, t) {
  if (!poseA || !poseB) return poseA || poseB || REST_POSE;

  const eased = easeInOutCubic(t);
  const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
  const result = { wrist: {} };

  fingers.forEach(finger => {
    const a = poseA[finger] || { base: 0, mid: 0, tip: 0, spread: 0 };
    const b = poseB[finger] || { base: 0, mid: 0, tip: 0, spread: 0 };

    result[finger] = {
      base: lerp(a.base, b.base, eased),
      mid: lerp(a.mid, b.mid, eased),
      tip: lerp(a.tip, b.tip, eased),
      spread: lerp(a.spread, b.spread, eased),
    };
  });

  // Interpolate wrist
  const wA = poseA.wrist || { x: 0, y: 0, z: 0 };
  const wB = poseB.wrist || { x: 0, y: 0, z: 0 };
  result.wrist = {
    x: lerp(wA.x, wB.x, eased),
    y: lerp(wA.y, wB.y, eased),
    z: lerp(wA.z, wB.z, eased),
  };

  return result;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function easeInOutCubic(t) {
  return t < 0.5
    ? 4 * t * t * t
    : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

/**
 * Get transition duration based on pose difference
 */
export function getTransitionDuration(poseA, poseB) {
  if (!poseA || !poseB) return TRANSITION_DURATION;
  
  // Calculate the "distance" between poses to adjust transition time
  const fingers = ['thumb', 'index', 'middle', 'ring', 'pinky'];
  let totalDiff = 0;
  
  fingers.forEach(finger => {
    const a = poseA[finger] || { base: 0, mid: 0, tip: 0, spread: 0 };
    const b = poseB[finger] || { base: 0, mid: 0, tip: 0, spread: 0 };
    totalDiff += Math.abs(a.base - b.base);
    totalDiff += Math.abs(a.mid - b.mid);
    totalDiff += Math.abs(a.tip - b.tip);
    totalDiff += Math.abs(a.spread - b.spread);
  });

  // Scale transition time: bigger changes need more time
  return Math.min(TRANSITION_DURATION + totalDiff * 30, 500);
}
