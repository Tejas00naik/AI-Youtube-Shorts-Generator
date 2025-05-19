#!/usr/bin/env python
import cv2
import numpy as np
import os
import webrtcvad
import wave
import contextlib
from pydub import AudioSegment
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Paths to the model files
prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
temp_audio_path = "temp_audio.wav"
output_dir = "debug_crops"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load DNN model
try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    print(f"✅ Successfully loaded face detection model")
except Exception as e:
    print(f"❌ Error loading face detection model: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Make sure the model files are in the correct location")
    exit(1)

# Initialize VAD
vad = webrtcvad.Vad(2)  # Aggressiveness mode from 0 to 3

def voice_activity_detection(audio_frame, sample_rate=16000):
    """Detect if there is voice activity in the audio frame"""
    try:
        return vad.is_speech(audio_frame, sample_rate)
    except Exception as e:
        print(f"❌ Error in voice activity detection: {e}")
        return False

def extract_audio_from_video(video_path, audio_path):
    """Extract audio from video file and save it as WAV"""
    print(f"🔊 Extracting audio from {video_path}...")
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(audio_path, format="wav")
    print(f"✅ Audio extracted to {audio_path}")

def process_audio_frame(audio_data, sample_rate=16000, frame_duration_ms=30):
    """Process audio data into frames for voice activity detection"""
    n = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
    offset = 0
    while offset + n <= len(audio_data):
        frame = audio_data[offset:offset + n]
        offset += n
        yield frame

def debug_speaker_and_crop(video_path, num_snapshots=10):
    """
    Debug face detection, speaker detection, and video cropping
    
    Args:
        video_path: Path to the video file
        num_snapshots: Number of snapshots to save
    """
    print(f"🎬 Starting speaker and crop detection debug for {video_path}")
    start_time = time.time()
    
    # Extract audio from the video
    extract_audio_from_video(video_path, temp_audio_path)
    
    # Read the extracted audio
    with contextlib.closing(wave.open(temp_audio_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        audio_data = wf.readframes(wf.getnframes())
        print(f"📊 Audio: {wf.getnchannels()} channels, {sample_rate} Hz, {wf.getnframes()} frames")
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video: {original_width}x{original_height}, {fps} FPS, {total_frames} frames")
    
    # Calculate crop dimensions for 9:16 aspect ratio
    vertical_height = original_height
    vertical_width = int(vertical_height * 9 / 16)
    print(f"📐 Vertical crop dimensions: {vertical_width}x{vertical_height} (9:16 ratio)")
    
    if original_width < vertical_width:
        print(f"⚠️ Warning: Original video width ({original_width}) is less than the desired vertical width ({vertical_width})")
    
    # Default crop centered in the frame
    x_start = (original_width - vertical_width) // 2
    x_end = x_start + vertical_width
    half_width = vertical_width // 2
    
    # Initialize tracking variables
    frame_info = []
    active_speaker_faces = []
    crop_windows = []
    
    # Determine frames to capture as snapshots
    snapshot_frames = [int(i * total_frames / num_snapshots) for i in range(num_snapshots)]
    
    # Initialize audio frame generator
    audio_generator = process_audio_frame(audio_data, sample_rate)
    
    # Process each frame
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Get next audio frame
        audio_frame = next(audio_generator, None)
        if audio_frame is None:
            break
            
        # Detect voice activity in audio
        is_speaking_audio = False
        if audio_frame:
            is_speaking_audio = voice_activity_detection(audio_frame, sample_rate)
        
        # Prepare for face detection
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        
        # Process face detections
        faces = []
        max_lip_distance = 0
        
        # First pass: find max lip distance and collect faces
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                face_width = x1 - x
                face_height = y1 - y
                
                # Assuming lips are approximately at the bottom third of the face
                lip_distance = abs((y + 2 * face_height // 3) - (y1))
                faces.append({
                    "box": [x, y, x1, y1],
                    "confidence": confidence,
                    "lip_distance": lip_distance
                })
                max_lip_distance = max(max_lip_distance, lip_distance)
        
        # Determine active speaker
        active_speaker_face = None
        
        # Second pass: find active speaker
        for face in faces:
            box = face["box"]
            if face["lip_distance"] >= max_lip_distance * 0.9 and is_speaking_audio:
                active_speaker_face = box
                break
        
        # If no active speaker found but faces exist, use the one with max lip movement
        if not active_speaker_face and faces:
            active_speaker_face = max(faces, key=lambda x: x["lip_distance"])["box"]
        
        # If no faces detected, use default box in center
        if not active_speaker_face:
            center_x = w // 2
            center_y = h // 2
            box_width = w // 3
            box_height = h // 3
            active_speaker_face = [center_x - box_width//2, center_y - box_height//2, 
                                  center_x + box_width//2, center_y + box_height//2]
        
        # Store active speaker face
        active_speaker_faces.append(active_speaker_face)
        
        # ===== CROPPING LOGIC FROM FACECROP.PY =====
        # Calculate the center of the active speaker face
        centerX = active_speaker_face[0] + (active_speaker_face[2] - active_speaker_face[0]) // 2
        
        # Only adjust crop window if this isn't the first frame and the difference is significant
        if frame_count > 0 and abs(x_start - (centerX - half_width)) > 1:
            # Update crop window to center on the active speaker
            x_start = centerX - half_width
            x_end = centerX + half_width
            
            # Handle edge cases where crop window goes outside video boundaries
            if x_end > w:
                x_start -= (x_end - w)
                x_end = w
            if x_start < 0:
                x_end += abs(x_start)
                x_start = 0
                
            # Ensure crop width is correct
            if x_end - x_start != vertical_width:
                x_end = x_start + vertical_width
                # Handle edge cases again
                if x_end > w:
                    x_start = w - vertical_width
                    x_end = w
                if x_start < 0:
                    x_start = 0
                    x_end = vertical_width
        
        # Store the crop window for this frame
        crop_windows.append([x_start, 0, x_end, h])
        
        # Calculate cropped frame
        cropped_frame = frame[:, x_start:x_end]
        
        # If crop failed, use default center crop
        if cropped_frame.shape[1] == 0:
            x_start = (w - vertical_width) // 2
            x_end = x_start + vertical_width
            cropped_frame = frame[:, x_start:x_end]
        
        # Create a debug visualization if this is a snapshot frame
        if frame_count in snapshot_frames:
            # Make a copy of the frame for visualization
            debug_frame = frame.copy()
            
            # Draw all detected faces in green
            for face in faces:
                box = face["box"]
                cv2.rectangle(debug_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
            # Draw active speaker face in red
            cv2.rectangle(debug_frame, 
                        (active_speaker_face[0], active_speaker_face[1]), 
                        (active_speaker_face[2], active_speaker_face[3]), 
                        (0, 0, 255), 3)
            cv2.putText(debug_frame, "Active Speaker", 
                      (active_speaker_face[0], active_speaker_face[1] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw crop window in blue
            cv2.rectangle(debug_frame, 
                        (x_start, 0), 
                        (x_end, h), 
                        (255, 0, 0), 3)
            cv2.putText(debug_frame, "9:16 Crop Window", 
                      (x_start, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Add audio info
            audio_status = "🎤 Speaking" if is_speaking_audio else "🔇 Silent"
            cv2.putText(debug_frame, audio_status, (20, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Also create a separate image showing just the cropped result
            result_img = np.hstack([
                cv2.resize(debug_frame, (round(debug_frame.shape[1] * 0.5), round(debug_frame.shape[0] * 0.5))),
                cv2.resize(cropped_frame, (round(vertical_width * 0.5), round(vertical_height * 0.5)))
            ])
            
            # Add labels
            cv2.putText(result_img, "Original with Detections", 
                       (20, result_img.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(result_img, "Cropped Result (9:16)", 
                       (debug_frame.shape[1]//2 + 20, result_img.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save the combined visualization
            snapshot_index = snapshot_frames.index(frame_count)
            snapshot_path = os.path.join(output_dir, f"crop_{snapshot_index+1}_frame_{frame_count}.jpg")
            cv2.imwrite(snapshot_path, result_img)
            print(f"📸 Saved crop snapshot {snapshot_index+1}/{num_snapshots} (frame {frame_count})")
            
            # Record frame info for diagnostics
            frame_info.append({
                "frame": frame_count,
                "active_speaker": active_speaker_face,
                "crop_window": [x_start, 0, x_end, h],
                "is_speaking": is_speaking_audio,
                "faces_detected": len(faces)
            })
        
        frame_count += 1
        
        # Show progress
        if frame_count % 100 == 0:
            print(f"⏳ Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
    
    # Clean up
    cap.release()
    os.remove(temp_audio_path)
    
    # Calculate statistics
    processing_time = time.time() - start_time
    
    # Generate crop tracking visualization
    generate_crop_tracking_visualization(frame_info, original_width, original_height, output_dir)
    
    # Print summary
    print(f"\n📊 Cropping Analysis Summary:")
    print(f"   - Total frames processed: {frame_count}")
    print(f"   - Original dimensions: {original_width}x{original_height}")
    print(f"   - Crop dimensions (9:16): {vertical_width}x{vertical_height}")
    print(f"   - Processing time: {processing_time:.2f} seconds ({frame_count/processing_time:.2f} FPS)")
    print(f"\n✅ Debug snapshots saved to {output_dir}")
    
    return frame_info

def generate_crop_tracking_visualization(frame_info, video_width, video_height, output_dir):
    """Generate visualization showing how the crop window moves throughout the video"""
    if not frame_info:
        return
    
    plt.figure(figsize=(12, 8))
    
    # Plot frame number vs. crop window center position
    frames = [info['frame'] for info in frame_info]
    
    # Extract crop window centers
    crop_centers = [(info['crop_window'][0] + info['crop_window'][2]) // 2 for info in frame_info]
    
    # Extract face centers
    face_centers = [(info['active_speaker'][0] + info['active_speaker'][2]) // 2 for info in frame_info]
    
    # Plot
    plt.subplot(2, 1, 1)
    plt.plot(frames, crop_centers, 'b-', label='Crop Window Center')
    plt.plot(frames, face_centers, 'r--', label='Active Speaker Center')
    plt.title('Horizontal Position Tracking')
    plt.xlabel('Frame Number')
    plt.ylabel('X Position (pixels)')
    plt.legend()
    plt.grid(True)
    
    # Add reference line for center of video
    plt.axhline(y=video_width//2, color='g', linestyle='-', alpha=0.3, label='Video Center')
    
    # Calculate crop window movement stats
    crop_movements = []
    for i in range(1, len(crop_centers)):
        crop_movements.append(abs(crop_centers[i] - crop_centers[i-1]))
    
    # Plot histogram of crop window movements
    plt.subplot(2, 1, 2)
    plt.hist(crop_movements, bins=20, alpha=0.7, color='blue')
    plt.title('Histogram of Crop Window Movements')
    plt.xlabel('Movement Size (pixels)')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    # Add stats as text
    if crop_movements:
        avg_movement = sum(crop_movements) / len(crop_movements)
        max_movement = max(crop_movements)
        plt.figtext(0.02, 0.02, 
                   f'Average movement: {avg_movement:.1f} px\nMaximum movement: {max_movement:.1f} px',
                   fontsize=10)
    
    plt.tight_layout()
    
    # Save the visualization
    tracking_viz_path = os.path.join(output_dir, 'crop_tracking_visualization.png')
    plt.savefig(tracking_viz_path)
    print(f"📊 Crop tracking visualization saved to {tracking_viz_path}")
    
    return tracking_viz_path

if __name__ == "__main__":
    video_path = "videos/Jordan Peterson on Andrew Tate | Lex Fridman Podcast Clips.mp4"
    
    # Check if the video exists
    if not os.path.exists(video_path):
        print(f"⬇️ Video file not found at {video_path}. Please download it first using the debug_speaker_detection.py script.")
        exit(1)
    
    # Run speaker and crop detection debug
    print(f"🔍 Running speaker and crop detection debug on {video_path}")
    frame_info = debug_speaker_and_crop(video_path)
    
    print(f"\n✅ Debug complete! 10 snapshots with crop visualization have been saved to the '{output_dir}' folder.")
    print(f"🔍 Examine these snapshots to understand how the cropping algorithm follows the active speaker.")
