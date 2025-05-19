import cv2
import numpy as np
import random
import os
import mediapipe as mp
from collections import deque

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Constants
FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_SIZE = 0.1  # Minimum face size relative to frame size
MAX_FACE_SIZE = 0.8  # Maximum face size relative to frame size

# Input video path
input_video = "DecOut.mp4"  # Using the existing video file
output_dir = "face_detection_samples"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def detect_faces(image):
    """Detect faces in an image using MediaPipe"""
    with mp_face_detection.FaceDetection(
        model_selection=1,  # 1 for short-range, 0 for long-range
        min_detection_confidence=FACE_DETECTION_CONFIDENCE
    ) as face_detection:
        # Convert the BGR image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect faces
        results = face_detection.process(rgb_image)
        
        # Draw face detections
        if results.detections:
            for detection in results.detections:
                # Get face bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Draw rectangle around the face
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw face landmarks
                mp_drawing.draw_detection(image, detection)
                
                # Add confidence score
                cv2.putText(image, 
                           f'{int(detection.score[0] * 100)}%', 
                           (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
        
        return image, results.detections

def extract_random_frames(video_path, num_frames=5):
    """Extract random frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
    
    # Select random frame indices
    if total_frames < num_frames:
        frame_indices = range(total_frames)
    else:
        frame_indices = sorted(random.sample(range(total_frames), num_frames))
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            timestamp = idx / fps
            frames.append((idx, timestamp, frame))
    
    cap.release()
    return frames

def main():
    # Extract random frames
    print(f"Extracting frames from {input_video}...")
    frames = extract_random_frames(input_video, num_frames=5)
    
    if not frames:
        print("No frames could be extracted from the video.")
        return
    
    print(f"Processing {len(frames)} frames...")
    
    # Process each frame
    for i, (frame_idx, timestamp, frame) in enumerate(frames):
        print(f"Processing frame {frame_idx} at {timestamp:.2f}s")
        
        # Detect faces in the frame
        processed_frame, detections = detect_faces(frame.copy())
        
        # Save the frame with detections
        output_path = os.path.join(output_dir, f"frame_{i:02d}_t{timestamp:.1f}s.jpg")
        cv2.imwrite(output_path, processed_frame)
        print(f"  - Saved {output_path} with {len(detections) if detections else 'no'} faces")
    
    print(f"\nFace detection complete! Check the '{output_dir}' directory for results.")
    print(f"Total frames processed: {len(frames)}")

if __name__ == "__main__":
    main()
