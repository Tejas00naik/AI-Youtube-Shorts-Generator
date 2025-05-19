import cv2
import numpy as np
from moviepy.editor import *
from Components.Speaker import detect_faces_and_speakers, Frames
global Fps

def crop_to_vertical(input_video_path, output_video_path):
    # Run face and speaker detection
    detect_faces_and_speakers(input_video_path, "DecOut.mp4")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Open the video file
    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate vertical dimensions (9:16 aspect ratio)
    vertical_height = int(original_height)
    vertical_width = int(vertical_height * 9 / 16)
    print(f"Crop dimensions: {vertical_width}x{vertical_height} (9:16 ratio)")

    if original_width < vertical_width:
        print("Error: Original video width is less than the desired vertical width.")
        return

    # Initialize crop window at the center of the frame
    x_start = (original_width - vertical_width) // 2
    x_end = x_start + vertical_width
    half_width = vertical_width // 2

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (vertical_width, vertical_height))
    global Fps
    Fps = fps
    
    # Smooth tracking parameters
    target_positions = []  # History of target positions
    smooth_window_size = int(fps / 2)  # Half-second window for smoothing
    position_threshold = vertical_width * 0.05  # 5% of width movement threshold
    
    # Process each frame
    count = 0
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
            
        # Get the active speaker's face for this frame
        try:
            # Check if active speaker face coordinates are available
            if count < len(Frames):
                # Get face coordinates from Speaker.py results
                try:
                    (X, Y, W, H) = Frames[count]
                except Exception as e:
                    try:
                        (X, Y, W, H) = Frames[count][0]
                    except:
                        # Default to center if no face detected
                        center_x = original_width // 2
                        X, Y = center_x - 50, original_height // 3
                        W, H = 100, 100
                
                # Calculate center of the face
                centerX = X + (W - X) // 2
                
                # Add to target position history
                target_positions.append(centerX)
                
                # Keep history limited to smooth window size
                if len(target_positions) > smooth_window_size:
                    target_positions.pop(0)
                
                # Calculate smoothed position (weighted average with recent positions having more weight)
                if len(target_positions) > 1:
                    # Calculate weights: more recent positions have higher weights
                    weights = [i/sum(range(1, len(target_positions)+1)) for i in range(1, len(target_positions)+1)]
                    smoothed_center = int(sum(p*w for p, w in zip(target_positions, weights)))
                else:
                    smoothed_center = centerX
                
                # Determine if we should move the crop window
                current_center = (x_start + x_end) // 2
                position_diff = abs(current_center - smoothed_center)
                
                # Only move the crop window if the face has moved significantly
                if count == 0 or position_diff > position_threshold:
                    # Calculate new crop window position
                    target_x_start = smoothed_center - half_width
                    target_x_end = smoothed_center + half_width
                    
                    # Ensure crop window stays within video boundaries
                    if target_x_end > original_width:
                        target_x_start = original_width - vertical_width
                        target_x_end = original_width
                    if target_x_start < 0:
                        target_x_start = 0
                        target_x_end = vertical_width
                    
                    # Apply gradual movement (interpolate between current and target)
                    # More aggressive movement when large changes are needed
                    lerp_factor = min(0.2, position_diff / (original_width/2))
                    x_start = int((1-lerp_factor) * x_start + lerp_factor * target_x_start)
                    x_end = int((1-lerp_factor) * x_end + lerp_factor * target_x_end)
                    
                    # Ensure crop width remains correct
                    if x_end - x_start != vertical_width:
                        x_end = x_start + vertical_width
                        
                        # Final boundary check
                        if x_end > original_width:
                            x_start = original_width - vertical_width
                            x_end = original_width
                        if x_start < 0:
                            x_start = 0
                            x_end = vertical_width
            else:
                # If we've run out of face data, use the last crop position
                pass
                
        except Exception as e:
            print(f"Error tracking face: {e}")
        
        # Crop the frame
        cropped_frame = frame[:, x_start:x_end]
        
        # Fallback if cropping failed
        if cropped_frame.shape[1] == 0:
            print("Crop failed, using default center crop")
            x_start = (original_width - vertical_width) // 2
            x_end = x_start + vertical_width
            cropped_frame = frame[:, x_start:x_end]
        
        # Avoid excessive debug output
        if count % 20 == 0:
            print(f"Processing frame {count}/{total_frames} - Crop position: {x_start}:{x_end}")
            
        # Write the cropped frame to output
        out.write(cropped_frame)
        count += 1

    # Clean up
    cap.release()
    out.release()
    print("Cropping complete. The video has been saved to", output_video_path, count)



def combine_videos(video_with_audio, video_without_audio, output_filename):
    try:
        # Load video clips
        clip_with_audio = VideoFileClip(video_with_audio)
        clip_without_audio = VideoFileClip(video_without_audio)

        audio = clip_with_audio.audio

        combined_clip = clip_without_audio.set_audio(audio)

        global Fps
        combined_clip.write_videofile(output_filename, codec='libx264', audio_codec='aac', fps=Fps, preset='medium', bitrate='3000k')
        print(f"Combined video saved successfully as {output_filename}")
    
    except Exception as e:
        print(f"Error combining video and audio: {str(e)}")



if __name__ == "__main__":
    input_video_path = r'Out.mp4'
    output_video_path = 'Croped_output_video.mp4'
    final_video_path = 'final_video_with_audio.mp4'
    detect_faces_and_speakers(input_video_path, "DecOut.mp4")
    crop_to_vertical(input_video_path, output_video_path)
    combine_videos(input_video_path, output_video_path, final_video_path)



