import os
import cv2
import json
import numpy as np
import mediapipe as mp
import math

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

dataset_dir = 'src/data/DATASET'
output_file = 'src/data/actualPoses.js'

poses = {}

def get_angle(v1, v2):
    # Angle between two 3D vectors
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0: return 0
    # Add a small epsilon to prevent Floating-point errors
    cos_theta = max(min(dot / norm, 1.0), -1.0)
    return math.acos(cos_theta)

def calculate_angles_from_landmarks(landmarks):
    # Calculate angles for base, mid, tip, spread
    # landmarks: list of 21 (x, y, z)
    def normalize(v):
        n = np.linalg.norm(v)
        return v / n if n != 0 else v

    angles = {}
    finger_indices = {
        'thumb': [1, 2, 3, 4],
        'index': [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring': [13, 14, 15, 16],
        'pinky': [17, 18, 19, 20]
    }
    
    palm_base = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    
    for finger, idx in finger_indices.items():
        # Get coordinates for joints
        j0 = palm_base if finger != 'thumb' else np.array([landmarks[1].x, landmarks[1].y, landmarks[1].z])
        j1 = np.array([landmarks[idx[0]].x, landmarks[idx[0]].y, landmarks[idx[0]].z])
        j2 = np.array([landmarks[idx[1]].x, landmarks[idx[1]].y, landmarks[idx[1]].z])
        j3 = np.array([landmarks[idx[2]].x, landmarks[idx[2]].y, landmarks[idx[2]].z])
        j4 = np.array([landmarks[idx[3]].x, landmarks[idx[3]].y, landmarks[idx[3]].z])

        # Vectors
        v_base = j1 - j0
        v_mid = j2 - j1
        v_tip = j3 - j2
        v_end = j4 - j3
        
        # Flexion angles
        a_base = get_angle(v_base, v_mid)
        a_mid = get_angle(v_mid, v_tip)
        a_tip = get_angle(v_tip, v_end)
        
        # Spread angle (relative to middle finger base if not thumb)
        a_spread = 0.0
        if finger != 'middle' and finger != 'thumb':
            mid_base = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
            v_ref = mid_base - palm_base
            v_finger = j1 - palm_base
            spread_angle = get_angle(v_ref, v_finger)
            if finger in ['index']:
                a_spread = spread_angle
            elif finger in ['ring', 'pinky']:
                a_spread = -spread_angle
                
        if finger == 'thumb':
            # Thumb is complex, simplified spread
            index_base = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])
            a_spread = get_angle(j1 - palm_base, index_base - palm_base)
            
        angles[finger] = {
            'base': a_base,
            'mid': a_mid,
            'tip': a_tip,
            'spread': a_spread
        }
    return angles

for root, _, files in os.walk(dataset_dir):
    sign_name = os.path.basename(root)
    if not sign_name or sign_name == 'DATASET':
        continue
        
    print(f"Processing {sign_name}...")
    sign_anglesList = []
    
    for file in files:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(root, file)
            image = cv2.flip(cv2.imread(image_path), 1)
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    angles = calculate_angles_from_landmarks(hand_landmarks.landmark)
                    sign_anglesList.append(angles)
                    break # Take first hand only
                    
    # Average the angles for the sign
    if sign_anglesList:
        avg_angles = {}
        fingers = sign_anglesList[0].keys()
        for finger in fingers:
            avg_angles[finger] = {
                'base': np.mean([entry[finger]['base'] for entry in sign_anglesList]),
                'mid': np.mean([entry[finger]['mid'] for entry in sign_anglesList]),
                'tip': np.mean([entry[finger]['tip'] for entry in sign_anglesList]),
                'spread': np.mean([entry[finger]['spread'] for entry in sign_anglesList])
            }
        poses[sign_name] = avg_angles

js_content = "export const ACTUAL_POSES = {\n"
for sign, angles in poses.items():
    js_content += f"  '{sign}': {{\n"
    for finger, data in angles.items():
        js_content += f"    {finger}: {{ base: {data['base']:.4f}, mid: {data['mid']:.4f}, tip: {data['tip']:.4f}, spread: {data['spread']:.4f} }},\n"
    js_content += f"    wrist: {{ x: 0, y: 0, z: 0 }}\n"
    js_content += "  },\n"
js_content += "};\n"

with open(output_file, 'w') as f:
    f.write(js_content)
    
print("Done extracting actual poses to actualPoses.js")
