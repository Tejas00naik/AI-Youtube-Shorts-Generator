import cv2
import numpy as np
import webrtcvad
import wave
import contextlib
from pydub import AudioSegment
import os
import mediapipe as mp
from collections import deque

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Constants
temp_audio_path = "temp_audio.wav"
FACE_DETECTION_CONFIDENCE = 0.5
MIN_FACE_SIZE = 0.1  # Minimum face size relative to frame size
MAX_FACE_SIZE = 0.8  # Maximum face size relative to frame size
FACE_TRACKING_HISTORY = 10  # Number of frames to keep in tracking history

# Initialize VAD
vad = webrtcvad.Vad(2)  # Aggressiveness mode from 0 to 3

# Face tracking state
tracked_face = None
face_history = deque(maxlen=FACE_TRACKING_HISTORY)

def voice_activity_detection(audio_frame, sample_rate=16000):
    return vad.is_speech(audio_frame, sample_rate)

def extract_audio_from_video(video_path, audio_path):
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(audio_path, format="wav")

def process_audio_frame(audio_data, sample_rate=16000, frame_duration_ms=30):
    n = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
    offset = 0
    while offset + n <= len(audio_data):
        frame = audio_data[offset:offset + n]
        offset += n
        yield frame

global Frames
Frames = [] # [x,y,w,h]

def detect_faces_and_speakers(input_video_path, output_video_path):
    global Frames, tracked_face, face_history
    
    # Extract audio from the video
    extract_audio_from_video(input_video_path, temp_audio_path)

    # Read the extracted audio
    with contextlib.closing(wave.open(temp_audio_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())

    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_duration_ms = 30  # 30ms frames
    audio_generator = process_audio_frame(audio_data, sample_rate, frame_duration_ms)
    
    # Initialize face detection
    with mp_face_detection.FaceDetection(
        model_selection=1,  # 1 for short-range, 0 for long-range
        min_detection_confidence=FACE_DETECTION_CONFIDENCE
    ) as face_detection:
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            h, w = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get audio data for this frame
            audio_frame = next(audio_generator, None) if frame_count % (fps * frame_duration_ms / 1000) < 1 else None
            is_speaking_audio = voice_activity_detection(audio_frame, sample_rate) if audio_frame is not None else False
            
            # Detect faces using MediaPipe
            results = face_detection.process(rgb_frame)
            
            # Reset face tracking if no faces detected
            if not results.detections:
                tracked_face = None
                face_history.clear()
                
            # Process detected faces
            faces = []
            if results.detections:
                for detection in results.detections:
                    # Get face bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w_face, h_face = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                                         int(bboxC.width * iw), int(bboxC.height * ih)
                    
                    # Skip faces that are too small or too large
                    face_size = (w_face * h_face) / (frame_width * frame_height)
                    if face_size < MIN_FACE_SIZE or face_size > MAX_FACE_SIZE:
                        continue
                        
                    # Get face landmarks for mouth detection
                    keypoints = detection.location_data.relative_keypoints
                    if len(keypoints) > 3:  # Ensure we have mouth landmarks
                        mouth_top = keypoints[3].y * ih
                        mouth_bottom = keypoints[4].y * ih
                        mouth_openness = abs(mouth_bottom - mouth_top)
                    else:
                        mouth_openness = 0
                        
                    # Calculate face center and size
                    face_center = (x + w_face // 2, y + h_face // 2)
                    face_size = (w_face + h_face) / 2
                    
                    faces.append({
                        'bbox': (x, y, x + w_face, y + h_face),
                        'center': face_center,
                        'size': face_size,
                        'mouth_openness': mouth_openness,
                        'confidence': detection.score[0]
                    })
            
            # Track the most likely speaker
            active_face = None
            
            if faces:
                # If we have a tracked face, try to match it
                if tracked_face is not None:
                    # Find the face closest to the tracked face
                    min_dist = float('inf')
                    for face in faces:
                        dist = np.linalg.norm(np.array(face['center']) - np.array(tracked_face['center']))
                        if dist < min_dist:
                            min_dist = dist
                            active_face = face
                    
                    # If no good match, use the face with most movement (speaking)
                    if min_dist > tracked_face['size'] * 0.5:  # Threshold for face tracking
                        active_face = max(faces, key=lambda f: f['mouth_openness'])
                else:
                    # No tracked face, use the most prominent face
                    active_face = max(faces, key=lambda f: f['confidence'])
                
                # Update tracked face
                tracked_face = active_face.copy()
                face_history.append(active_face)
                
                # If we have audio, use it to confirm the speaker
                if is_speaking_audio and len(face_history) > 1:
                    # Find the face with most mouth movement
                    speaking_face = max(faces, key=lambda f: f['mouth_openness'])
                    if speaking_face['mouth_openness'] > 5:  # Threshold for mouth movement
                        active_face = speaking_face
            
            # If no active face, use the last known position or center of frame
            if active_face is None:
                if face_history:
                    # Use the last known position
                    active_face = face_history[-1]
                else:
                    # Default to center of frame
                    center_x, center_y = frame_width // 2, frame_height // 2
                    box_size = min(frame_width, frame_height) // 3
                    active_face = {
                        'bbox': (center_x - box_size//2, center_y - box_size//2,
                               center_x + box_size//2, center_y + box_size//2),
                        'center': (center_x, center_y),
                        'size': box_size
                    }
            
            # Draw debug information
            x1, y1, x2, y2 = active_face['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if is_speaking_audio:
                cv2.putText(frame, "SPEAKING", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Store the face coordinates
            Frames.append([x1, y1, x2, y2])
            
            # Write the frame
            out.write(frame)
            
            # Print progress
            if frame_count % 30 == 0:  # Every second at 30fps
                print(f"Processed {frame_count} frames")
        
        cap.release()
        out.release()
        
        # Clean up
        try:
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary file: {e}")

    # All cleanup already handled above
    # No need to release cap and out again or remove temp_audio_path again



if __name__ == "__main__":
    detect_faces_and_speakers()
    print(Frames)
    print(len(Frames))
    print(Frames[1:5])
