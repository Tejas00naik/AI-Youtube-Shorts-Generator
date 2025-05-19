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
output_dir = "debug_snapshots"

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

def debug_speaker_detection(video_path, num_snapshots=10):
    """
    Debug face and speaker detection for a video file
    
    Args:
        video_path: Path to the video file
        num_snapshots: Number of snapshots to save
    """
    print(f"🎬 Starting speaker detection debug for {video_path}")
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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📊 Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Determine frames to capture as snapshots
    snapshot_frames = [int(i * total_frames / num_snapshots) for i in range(num_snapshots)]
    
    # Initialize audio frame generator
    audio_generator = process_audio_frame(audio_data, sample_rate)
    
    # Variables for tracking active speakers
    frames_info = []  # Store detection data for each frame
    frames_with_faces = 0
    frames_with_active_speaker = 0
    total_faces_detected = 0
    
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
        active_speaker_face = None
        
        # First pass: find max lip distance
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
        if not faces:
            center_x = w // 2
            center_y = h // 2
            box_width = w // 3
            box_height = h // 3
            active_speaker_face = [center_x - box_width//2, center_y - box_height//2, 
                                  center_x + box_width//2, center_y + box_height//2]
        
        # Save this frame's information
        frame_info = {
            "frame_number": frame_count,
            "faces_detected": len(faces),
            "is_speaking_audio": is_speaking_audio,
            "max_lip_distance": max_lip_distance,
            "active_speaker": active_speaker_face
        }
        frames_info.append(frame_info)
        
        # Update statistics
        if len(faces) > 0:
            frames_with_faces += 1
        if active_speaker_face:
            frames_with_active_speaker += 1
        total_faces_detected += len(faces)
        
        # Create debug visualization
        debug_frame = frame.copy()
        
        # Draw all detected faces
        for face in faces:
            box = face["box"]
            # Green for regular faces
            cv2.rectangle(debug_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            
        # Draw active speaker face in red
        if active_speaker_face:
            cv2.rectangle(debug_frame, 
                         (active_speaker_face[0], active_speaker_face[1]), 
                         (active_speaker_face[2], active_speaker_face[3]), 
                         (0, 0, 255), 3)
            cv2.putText(debug_frame, "Active Speaker", 
                       (active_speaker_face[0], active_speaker_face[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add audio info overlay
        audio_status = "🎤 Speaking" if is_speaking_audio else "🔇 Silent"
        cv2.putText(debug_frame, audio_status, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save snapshot if this is one of our target frames
        if frame_count in snapshot_frames:
            snapshot_index = snapshot_frames.index(frame_count)
            snapshot_path = os.path.join(output_dir, f"snapshot_{snapshot_index+1}_frame_{frame_count}.jpg")
            cv2.imwrite(snapshot_path, debug_frame)
            print(f"📸 Saved snapshot {snapshot_index+1}/{num_snapshots} (frame {frame_count})")
        
        frame_count += 1
        
        # Show progress
        if frame_count % 100 == 0:
            print(f"⏳ Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
    
    # Clean up
    cap.release()
    os.remove(temp_audio_path)
    
    # Calculate statistics
    processing_time = time.time() - start_time
    
    # Print summary
    print(f"\n📊 Speaker Detection Summary:")
    print(f"   - Total frames processed: {frame_count}")
    print(f"   - Frames with faces: {frames_with_faces} ({frames_with_faces/frame_count*100:.1f}%)")
    print(f"   - Frames with active speaker: {frames_with_active_speaker} ({frames_with_active_speaker/frame_count*100:.1f}%)")
    print(f"   - Total faces detected: {total_faces_detected}")
    print(f"   - Average faces per frame: {total_faces_detected/frame_count:.2f}")
    print(f"   - Processing time: {processing_time:.2f} seconds ({frame_count/processing_time:.2f} FPS)")
    print(f"\n✅ Debug snapshots saved to {output_dir}")

    return frames_info

def generate_diagnostic_plots(frames_info, output_dir):
    """Generate diagnostic plots from the speaker detection data"""
    if not frames_info:
        print("❌ No frame data to generate diagnostics")
        return
        
    # Prepare data for plotting
    frame_numbers = [f["frame_number"] for f in frames_info]
    faces_detected = [f["faces_detected"] for f in frames_info]
    voice_activity = [1 if f["is_speaking_audio"] else 0 for f in frames_info]
    lip_distances = [f["max_lip_distance"] if f["max_lip_distance"] else 0 for f in frames_info]
    
    # Create figure with multiple subplots
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Faces detected per frame
    plt.subplot(3, 1, 1)
    plt.plot(frame_numbers, faces_detected, 'g-')
    plt.title('Faces Detected per Frame')
    plt.ylabel('Number of Faces')
    plt.grid(True)
    
    # Plot 2: Voice activity
    plt.subplot(3, 1, 2)
    plt.plot(frame_numbers, voice_activity, 'r-')
    plt.title('Voice Activity Detection')
    plt.ylabel('Speaking (1) / Silent (0)')
    plt.grid(True)
    
    # Plot 3: Lip movement (max distance)
    plt.subplot(3, 1, 3)
    plt.plot(frame_numbers, lip_distances, 'b-')
    plt.title('Maximum Lip Movement Distance')
    plt.xlabel('Frame Number')
    plt.ylabel('Lip Distance (pixels)')
    plt.grid(True)
    
    # Save the figure
    diagnostic_plot_path = os.path.join(output_dir, 'speaker_detection_diagnostics.png')
    plt.tight_layout()
    plt.savefig(diagnostic_plot_path)
    print(f"📊 Diagnostic plots saved to {diagnostic_plot_path}")
    
    return diagnostic_plot_path

if __name__ == "__main__":
    video_url = "https://youtu.be/Q0LqcoeWBBk?si=mKZDZq9pmUd8vjRC"
    
    # Check if the video has already been downloaded
    video_path = "youtube_video.mp4"
    if not os.path.exists(video_path):
        print(f"⬇️ Downloading video from {video_url}...")
        
        # Import the YoutubeDownloader module
        from Components.YoutubeDownloader import download_youtube_video
        video_path = download_youtube_video(video_url)
        
        if not os.path.exists(video_path):
            print(f"❌ Failed to download video. Please check the URL or download it manually.")
            exit(1)
    
    # Run speaker detection debug
    print(f"🔍 Running speaker detection debug on {video_path}")
    frames_info = debug_speaker_detection(video_path)
    
    # Generate diagnostic plots
    generate_diagnostic_plots(frames_info, output_dir)
    
    print(f"\n✅ Debug complete! 10 snapshots have been saved to the '{output_dir}' folder.")
    print(f"🔍 Next steps:")
    print(f"   1. Review the snapshots and diagnostic plots")
    print(f"   2. Check for issues with face detection and active speaker tracking")
    print(f"   3. Adjust thresholds in the Speaker.py module if needed")
