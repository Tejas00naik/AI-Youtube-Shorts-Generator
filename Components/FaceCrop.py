import cv2
import numpy as np
from moviepy.editor import *
from Components.Speaker import detect_faces_and_speakers, Frames
global Fps

def crop_to_vertical(input_video_path, output_video_path, target_aspect_ratio=9/16):
    """
    Crop video to vertical format (9:16) while keeping the active speaker in frame.
    No padding is added - only horizontal movement of the crop window.
    
    Args:
        input_video_path (str): Path to input video
        output_video_path (str): Path to save output video
        target_aspect_ratio (float): Target aspect ratio (width/height)
    """
    # First detect all faces and speakers in the video
    print("Detecting faces and speakers...")
    detect_faces_and_speakers(input_video_path, "DecOut.mp4")
    
    # Open the video file
    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return False
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate target dimensions (maintain original height, adjust width for 9:16)
    target_height = original_height
    target_width = int(target_height * target_aspect_ratio)
    
    # Ensure target width doesn't exceed original width
    if target_width > original_width:
        target_width = original_width
        print(f"⚠️ Warning: Original width ({original_width}px) is less than target width ({target_width}px)")
        print(f"Using maximum possible width: {target_width}px")
    
    print(f"Original resolution: {original_width}x{original_height}")
    print(f"Cropping to: {target_width}x{target_height} (Aspect: ~{target_aspect_ratio:.2f})")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (target_width, target_height))
    
    # Store FPS as global for later use in combine_videos
    global Fps
    Fps = fps
    
    # Track previous crop position for smooth transitions
    prev_crop_x = (original_width - target_width) // 2
    
    # Process each frame
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Warning: Could not read frame {frame_idx}")
            break
        
        # Default to center crop if no face data
        crop_x = (original_width - target_width) // 2
        
        # Get the face coordinates for this frame if available
        if frame_idx < len(Frames):
            try:
                x1, y1, x2, y2 = Frames[frame_idx]
                face_center_x = (x1 + x2) // 2
                
                # Calculate ideal crop position to center the face horizontally
                ideal_crop_x = face_center_x - (target_width // 2)
                
                # Check if we can center the face without exceeding video bounds
                if ideal_crop_x < 0:
                    # Face is too far left, align crop with left edge
                    crop_x = 0
                elif ideal_crop_x + target_width > original_width:
                    # Face is too far right, align crop with right edge
                    crop_x = original_width - target_width
                else:
                    # Face can be perfectly centered
                    crop_x = ideal_crop_x
                
                # Smooth transition (reduces rapid jumps)
                if frame_idx > 0:
                    max_movement = target_width // 10  # Max 1/10th of frame width per frame for smoother motion
                    crop_x = max(prev_crop_x - max_movement, 
                               min(prev_crop_x + max_movement, crop_x))
                
                prev_crop_x = crop_x
                
            except Exception as e:
                print(f"⚠️ Warning in frame {frame_idx}: {str(e)}")
                crop_x = prev_crop_x  # Use previous position if error occurs
        
        # Ensure crop_x is within bounds (sanity check)
        crop_x = max(0, min(crop_x, original_width - target_width))
        
        # Crop the frame
        cropped_frame = frame[0:target_height, crop_x:crop_x + target_width]
        
        # Ensure output dimensions match exactly (should always be true, but just in case)
        if cropped_frame.shape[1] != target_width or cropped_frame.shape[0] != target_height:
            cropped_frame = cv2.resize(cropped_frame, (target_width, target_height))
        
        # Write the frame
        out.write(cropped_frame)
        
        # Print progress
        if (frame_idx + 1) % 30 == 0:  # Every second at 30fps
            print(f"Processed {frame_idx + 1}/{total_frames} frames (x={crop_x})")
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"\n✅ Cropping complete. Video saved to {output_video_path}")
    print(f"Processed {frame_idx + 1} frames in total")
    
    return True



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



