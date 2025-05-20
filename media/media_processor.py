"""
Media processor module for AI YouTube Shorts Generator.

This module handles video processing, including clip extraction, pausing,
text overlays, and concatenation for the final output video.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from moviepy.editor import VideoFileClip, concatenate_videoclips, CompositeVideoClip, TextClip

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_pause_narration(video_path: str, segments: List[Dict[str, Any]], 
                           output_path: Optional[str] = None) -> str:
    """
    Process a video with pause-and-narration format based on script segments.
    
    Args:
        video_path: Path to the input video file
        segments: List of segment dictionaries with clip and narration information
        output_path: Path to save the output video (default: auto-generate)
    
    Returns:
        Path to the output video file
    
    Each segment should have one of these formats:
    - Clip: {"type": "clip", "start": 10, "end": 20}
    - Narration: {"type": "narration", "text": "Your text here", "duration": 2.5}
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")
    
    # Create output directory if needed
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output")
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}_processed.mp4")
    
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Output will be saved to: {output_path}")
    
    # Load the input video
    source_video = VideoFileClip(video_path)
    processed_clips = []
    
    for segment in segments:
        if segment["type"] == "clip":
            # Extract clip from video
            start_time = segment["start"]
            end_time = segment["end"]
            
            if end_time > source_video.duration:
                logger.warning(f"Clip end time {end_time} exceeds video duration {source_video.duration}")
                end_time = source_video.duration
                
            clip = source_video.subclip(start_time, end_time)
            processed_clips.append(clip)
            
        elif segment["type"] == "narration":
            # Create a narration pause with text overlay
            
            # 1. If we have previous clips, freeze the last frame
            if processed_clips:
                last_clip = processed_clips[-1]
                frozen_frame = last_clip.to_ImageClip(last_clip.duration).set_duration(segment["duration"])
                
                # 2. Create text overlay
                text = segment["text"]
                txt_clip = TextClip(
                    text, 
                    fontsize=36, 
                    color='white', 
                    bg_color='rgba(0,0,0,0.5)',
                    method='caption',
                    size=(source_video.w * 0.9, None),
                    align='center'
                )
                txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(segment["duration"])
                
                # 3. Combine frozen frame and text
                narration_clip = CompositeVideoClip([frozen_frame, txt_clip])
                processed_clips.append(narration_clip)
            else:
                logger.warning("Narration segment with no preceding video clip, skipping")
    
    # Concatenate all processed clips
    if processed_clips:
        final_clip = concatenate_videoclips(processed_clips)
        
        # Write the final video
        final_clip.write_videofile(output_path)
        logger.info(f"Successfully processed video and saved to {output_path}")
        
        # Clean up
        final_clip.close()
        source_video.close()
        
        return output_path
    else:
        logger.error("No valid clips found to process")
        return ""

def add_captions(video_path: str, captions_data: List[Dict[str, Any]], 
                output_path: Optional[str] = None) -> str:
    """
    Add captions to a video based on provided captions data.
    
    Args:
        video_path: Path to the input video file
        captions_data: List of caption dictionaries
        output_path: Path to save the output video (default: auto-generate)
    
    Returns:
        Path to the output video file with captions
    
    Each caption should have this format:
    {"text": "Caption text", "start": 0.0, "end": 3.5, "position": "bottom"}
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")
    
    # Create output directory if needed
    if output_path is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output")
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{video_name}_captioned.mp4")
    
    logger.info(f"Adding captions to video: {video_path}")
    
    # Load the input video
    video = VideoFileClip(video_path)
    caption_clips = []
    
    # Create each caption as a TextClip
    for caption in captions_data:
        text = caption["text"]
        start = caption["start"]
        end = caption["end"]
        position = caption.get("position", "bottom")
        
        # Create the text clip
        txt_clip = TextClip(
            text, 
            fontsize=24, 
            color='white',
            bg_color='rgba(0,0,0,0.5)',
            method='caption',
            size=(video.w * 0.9, None),
            align='center'
        )
        
        # Set position and timing
        pos_mapping = {
            "top": ('center', 'top'),
            "bottom": ('center', 'bottom'),
            "center": 'center'
        }
        txt_clip = txt_clip.set_position(pos_mapping.get(position, 'bottom'))
        txt_clip = txt_clip.set_start(start).set_end(end)
        
        caption_clips.append(txt_clip)
    
    # Composite the video and captions
    final_clip = CompositeVideoClip([video] + caption_clips)
    
    # Write the final video
    final_clip.write_videofile(output_path)
    logger.info(f"Successfully added captions and saved to {output_path}")
    
    # Clean up
    final_clip.close()
    video.close()
    
    return output_path

def process_shorts(narrative_plan: Dict[str, Any], video_path: str, output_path: Optional[str] = None) -> str:
    """
    Process a full shorts video based on the narrative plan.
    
    Args:
        narrative_plan: The narrative plan dict from the narrative planner
        video_path: Path to the input video file
        output_path: Path to save the output video (default: auto-generate)
        
    Returns:
        Path to the processed shorts video
    """
    # Convert narrative plan to segments for processing
    segments = []
    
    # Extract segments from the narrative plan
    narrative_segments = narrative_plan.get("segments", [])
    
    for i, segment in enumerate(narrative_segments):
        # Add clip segment
        segments.append({
            "type": "clip",
            "start": segment["start_time"],
            "end": segment["end_time"]
        })
        
        # Add narration after each clip except the last one
        if i < len(narrative_segments) - 1:
            segments.append({
                "type": "narration",
                "text": segment.get("description", ""),
                "duration": 2.0  # Default 2 second pause for narration
            })
    
    # Process the video
    return process_pause_narration(video_path, segments, output_path)
