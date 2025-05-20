"""
Director Agent for AI YouTube Shorts Generator.

This module assembles the final video from action clips and pause texts,
applying appropriate transitions, effects, and ensuring technical quality.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from moviepy.editor import (
    VideoFileClip, concatenate_videoclips, CompositeVideoClip, 
    TextClip, ImageClip, ColorClip, vfx
)

from core.error_handler import Result
from core.config import get_config
from llm.llm_client import get_llm_client

# Configure logging
logger = logging.getLogger(__name__)

class DirectorAgent:
    """
    Assembles the final video from action clips and pause texts,
    applying appropriate transitions and effects.
    """
    
    def __init__(self, llm_client=None):
        """Initialize the director agent."""
        self.llm_client = llm_client
        self.config = get_config()
    
    def assemble_video(self, video_path: str, 
                      clips_data: Dict[str, Any],
                      texts_data: Dict[str, Any],
                      output_path: Optional[str] = None) -> Result:
        """
        Assemble the final video from clips and pause texts.
        
        Args:
            video_path: Path to the source video
            clips_data: Data about selected clips
            texts_data: Data about pause texts
            output_path: Optional path for output video
            
        Returns:
            Result object with output video path or error
        """
        # Validate inputs
        if not os.path.exists(video_path):
            return Result.failure(f"Video file not found: {video_path}")
            
        if not isinstance(clips_data, dict) or "clips" not in clips_data:
            return Result.failure("Invalid clips data format")
            
        if not isinstance(texts_data, dict) or "texts" not in texts_data:
            return Result.failure("Invalid texts data format")
        
        clips = clips_data["clips"]
        texts = texts_data["texts"]
        
        if len(texts) != len(clips) - 1:
            logger.warning(
                f"Mismatch between number of clips ({len(clips)}) and " +
                f"texts ({len(texts)}). Should have exactly one less text than clips."
            )
        
        # Create output path if not provided
        if output_path is None:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            
            video_name = Path(video_path).stem
            output_path = str(output_dir / f"{video_name}_shorts.mp4")
        
        # Try to use LLM for direction if available
        if self.llm_client and self.llm_client.is_available():
            try:
                # Get editing instructions from LLM
                timeline_result = self._get_editing_timeline(clips, texts)
                if timeline_result.is_success:
                    return self._assemble_with_timeline(video_path, timeline_result.value, output_path)
            except Exception as e:
                logger.error(f"LLM direction failed: {str(e)}")
                logger.warning("Falling back to default direction")
        
        # Fallback to basic assembly
        return self._assemble_basic_video(video_path, clips, texts, output_path)
    
    def _get_editing_timeline(self, clips: List[Dict[str, Any]], 
                            texts: List[Dict[str, Any]]) -> Result:
        """Get detailed editing timeline from LLM."""
        tech_prompt = f'''
        Assemble video from:
        - Action clips: {json.dumps(clips, indent=2)}
        - Pause texts: {json.dumps(texts, indent=2)}
        
        Technical rules:
        1. Crossfade last 0.5s of each action clip
        2. Text appears instantly, fades out 0.3s
        3. No audio overlap between clips
        
        Return editing timeline JSON:
        {{
          "timeline": [
            {{"type": "clip", "file": "clip_1", "start": 0.0, "end": 7.2, "effects": {{"fade_in": 0.0, "fade_out": 0.5}}}},
            {{"type": "text", "content": "...", "start": 7.2, "end": 9.7, "effects": {{"fade_in": 0.0, "fade_out": 0.3}}}},
            ...
          ]
        }}
        '''
        
        try:
            # Make API call using generic LLM client
            response = self.llm_client.chat_completion(
                system_prompt=tech_prompt,
                user_messages="Generate an editing timeline for the video",
                temperature=0.3,
                json_response=True
            )
            
            # Check for errors
            if "error" in response:
                return Result.failure(f"LLM API error: {response['error']}")
                
            # Parse response
            response_text = response["content"].strip()
            timeline_data = json.loads(response_text)
            
            # Validate the timeline
            validation_result = self._validate_timeline_contract(timeline_data)
            if not validation_result.is_success:
                logger.error(f"LLM-generated timeline failed validation: {validation_result.error.message}")
                return validation_result
            
            return Result.success(timeline_data)
            
        except Exception as e:
            logger.error(f"Error in LLM timeline generation: {str(e)}")
            return Result.failure(f"LLM timeline generation error: {str(e)}")
    
    def _validate_timeline_contract(self, timeline_data: Dict[str, Any]) -> Result:
        """Validate that the timeline follows the contract."""
        # Check basic structure
        if not isinstance(timeline_data, dict):
            return Result.failure("Timeline data must be a dictionary")
        
        if "timeline" not in timeline_data:
            return Result.failure("Timeline data must contain 'timeline' key")
        
        timeline = timeline_data.get("timeline", [])
        if not timeline or not isinstance(timeline, list):
            return Result.failure("Timeline must be a non-empty list")
        
        # Validate each timeline entry
        current_time = 0.0
        for i, entry in enumerate(timeline):
            if not isinstance(entry, dict):
                return Result.failure(f"Timeline entry {i} must be a dictionary")
            
            # Check required fields
            if "type" not in entry:
                return Result.failure(f"Timeline entry {i} missing 'type'")
                
            if "start" not in entry or "end" not in entry:
                return Result.failure(f"Timeline entry {i} missing start/end times")
                
            entry_type = entry["type"]
            start = entry["start"]
            end = entry["end"]
            
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                return Result.failure(f"Timeline entry {i} start/end must be numbers")
                
            if start >= end:
                return Result.failure(f"Timeline entry {i} start must be less than end")
                
            # Check sequential timing
            if i > 0 and abs(start - current_time) > 0.001:  # Small tolerance for floating-point errors
                return Result.failure(
                    f"Timeline entry {i} start time ({start}s) doesn't match " +
                    f"previous entry end time ({current_time}s)"
                )
            
            current_time = end
                
            # Type-specific validation
            if entry_type == "clip":
                if "file" not in entry:
                    return Result.failure(f"Clip entry {i} missing 'file'")
            
            elif entry_type == "text":
                if "content" not in entry:
                    return Result.failure(f"Text entry {i} missing 'content'")
            
            else:
                return Result.failure(f"Timeline entry {i} has invalid type: {entry_type}")
            
            # Check effects
            if "effects" in entry:
                effects = entry["effects"]
                if not isinstance(effects, dict):
                    return Result.failure(f"Timeline entry {i} effects must be a dictionary")
                
                # Validate fade effects
                if "fade_in" in effects and not isinstance(effects["fade_in"], (int, float)):
                    return Result.failure(f"Timeline entry {i} fade_in must be a number")
                    
                if "fade_out" in effects and not isinstance(effects["fade_out"], (int, float)):
                    return Result.failure(f"Timeline entry {i} fade_out must be a number")
            else:
                # Add default effects
                if entry_type == "clip":
                    entry["effects"] = {"fade_in": 0.0, "fade_out": 0.5}
                else:  # text
                    entry["effects"] = {"fade_in": 0.0, "fade_out": 0.3}
        
        return Result.success(timeline_data)
    
    def _assemble_with_timeline(self, video_path: str, timeline_data: Dict[str, Any], 
                              output_path: str) -> Result:
        """Assemble the video based on a detailed timeline."""
        try:
            # Load the source video
            source_video = VideoFileClip(video_path)
            
            # Process the timeline
            timeline = timeline_data["timeline"]
            processed_clips = []
            
            for entry in timeline:
                entry_type = entry["type"]
                start = entry["start"]
                end = entry["end"]
                effects = entry.get("effects", {})
                
                if entry_type == "clip":
                    # For "clip" entries, extract the clip from source video
                    # The "file" field is actually a reference to clip index
                    # (e.g., "clip_1" refers to clips[0])
                    file_ref = entry["file"]
                    try:
                        clip_index = int(file_ref.replace("clip_", "")) - 1
                    except:
                        clip_index = 0  # Default to first clip if parse fails
                    
                    # Find the corresponding clip data
                    from_clips_data = timeline_data.get("_from_clips_data", [])
                    if clip_index < len(from_clips_data):
                        clip_data = from_clips_data[clip_index]
                        clip_start = clip_data["start"]
                        clip_end = clip_data["end"]
                    else:
                        # Fallback: Use the entry start/end as clip timecodes
                        logger.warning(f"Clip reference {file_ref} not found in data, using timeline times")
                        clip_start = start
                        clip_end = end
                    
                    # Extract the clip
                    clip = source_video.subclip(clip_start, clip_end)
                    duration = end - start
                    
                    # Apply effects
                    fade_in = effects.get("fade_in", 0.0)
                    fade_out = effects.get("fade_out", 0.5)
                    
                    if fade_in > 0:
                        clip = clip.fadein(fade_in)
                    if fade_out > 0:
                        clip = clip.fadeout(fade_out)
                    
                    processed_clips.append(clip)
                    
                elif entry_type == "text":
                    # For "text" entries, create a text overlay on frozen frame
                    content = entry["content"]
                    duration = end - start
                    
                    # Get the last frame of the previous clip
                    if processed_clips:
                        last_clip = processed_clips[-1]
                        frozen_frame = last_clip.to_ImageClip(last_clip.duration).set_duration(duration)
                        
                        # Create text overlay
                        txt_clip = TextClip(
                            content, 
                            fontsize=36, 
                            color='white', 
                            bg_color='rgba(0,0,0,0.5)',
                            method='caption',
                            size=(source_video.w * 0.9, None),
                            align='center'
                        )
                        txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(duration)
                        
                        # Apply effects
                        fade_out = effects.get("fade_out", 0.3)
                        if fade_out > 0:
                            txt_clip = txt_clip.fadeout(fade_out)
                        
                        # Combine frozen frame and text
                        text_composite = CompositeVideoClip([frozen_frame, txt_clip])
                        processed_clips.append(text_composite)
                    else:
                        logger.warning("Text entry with no preceding clip, skipping")
            
            # Concatenate all clips
            if processed_clips:
                final_clip = concatenate_videoclips(processed_clips)
                
                # Write the output file
                final_clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    preset="medium",
                    fps=30
                )
                
                # Clean up
                final_clip.close()
                source_video.close()
                
                logger.info(f"Successfully assembled video at {output_path}")
                return Result.success(output_path)
            else:
                return Result.failure("No clips were generated")
                
        except Exception as e:
            logger.error(f"Error assembling video with timeline: {str(e)}")
            return Result.failure(f"Video assembly error: {str(e)}")
    
    def _assemble_basic_video(self, video_path: str, 
                            clips: List[Dict[str, Any]],
                            texts: List[Dict[str, Any]], 
                            output_path: str) -> Result:
        """
        Assemble a basic video with alternating clips and text overlays.
        This is a fallback when LLM direction is not available.
        """
        try:
            # Load the source video
            source_video = VideoFileClip(video_path)
            processed_clips = []
            
            # Process clips and texts, alternating between them
            for i, clip_data in enumerate(clips):
                # Extract action clip
                clip_start = clip_data["start"]
                clip_end = clip_data["end"]
                
                # Validate time boundaries
                if clip_end > source_video.duration:
                    logger.warning(f"Clip end time ({clip_end}) exceeds video duration ({source_video.duration})")
                    clip_end = min(clip_end, source_video.duration)
                
                # Extract the clip
                try:
                    clip = source_video.subclip(clip_start, clip_end)
                    
                    # Apply subtle fadeout to the end of each clip
                    fadeout_duration = min(0.5, clip.duration / 4)  # Max 0.5s or 1/4 of clip duration
                    clip = clip.fadeout(fadeout_duration)
                    
                    processed_clips.append(clip)
                except Exception as e:
                    logger.error(f"Error extracting clip {i}: {str(e)}")
                    continue
                
                # Add text pause after each clip except the last one
                if i < len(clips) - 1 and i < len(texts):
                    text_data = texts[i]
                    text = text_data["text"]
                    duration = text_data.get("duration", 2.5)
                    position = text_data.get("position", "bottom_center")
                    
                    # Create frozen frame with text overlay
                    try:
                        # Create frozen frame from last frame of clip
                        frozen_frame = clip.to_ImageClip(clip.duration).set_duration(duration)
                        
                        # Create text overlay
                        txt_clip = TextClip(
                            text, 
                            fontsize=36, 
                            color='white', 
                            bg_color='rgba(0,0,0,0.5)',
                            method='caption',
                            size=(source_video.w * 0.9, None),
                            align='center'
                        )
                        
                        # Set position based on position parameter
                        if position == "top_center":
                            txt_pos = ('center', 'top')
                        elif position == "center":
                            txt_pos = 'center'
                        else:  # bottom_center
                            txt_pos = ('center', 'bottom')
                            
                        txt_clip = txt_clip.set_position(txt_pos).set_duration(duration)
                        
                        # Apply fadeout to text
                        fadeout_duration = min(0.3, duration / 3)  # Max 0.3s or 1/3 of pause duration
                        txt_clip = txt_clip.fadeout(fadeout_duration)
                        
                        # Combine frozen frame and text
                        text_composite = CompositeVideoClip([frozen_frame, txt_clip])
                        processed_clips.append(text_composite)
                    except Exception as e:
                        logger.error(f"Error creating text overlay {i}: {str(e)}")
                        continue
            
            # Concatenate all clips
            if processed_clips:
                final_clip = concatenate_videoclips(processed_clips)
                
                # Write the output file
                final_clip.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    preset="medium",
                    fps=30
                )
                
                # Clean up
                final_clip.close()
                source_video.close()
                
                logger.info(f"Successfully assembled video at {output_path}")
                return Result.success(output_path)
            else:
                return Result.failure("No clips were generated")
                
        except Exception as e:
            logger.error(f"Error in basic video assembly: {str(e)}")
            return Result.failure(f"Video assembly error: {str(e)}")


# Singleton instance
_director_agent = None

def get_director_agent(llm_client=None) -> DirectorAgent:
    """
    Get the singleton director agent instance.
    
    Args:
        llm_client: Optional LLMClient instance
        
    Returns:
        DirectorAgent instance
    """
    global _director_agent
    if _director_agent is None:
        _director_agent = DirectorAgent(llm_client)
    return _director_agent

def assemble_video(video_path: str, 
                  clips_data: Dict[str, Any],
                  texts_data: Dict[str, Any],
                  output_path: Optional[str] = None,
                  llm_provider: str = None,
                  api_key: str = None,
                  model: str = None) -> Result:
    """
    Convenience function to assemble a video from clips and texts.
    
    Args:
        video_path: Path to the source video
        clips_data: Data about selected clips
        texts_data: Data about pause texts
        output_path: Optional path for output video
        llm_provider: Optional LLM provider name (openai, deepseek, anthropic, local)
        api_key: Optional API key for the LLM provider
        model: Optional model name to use
        
    Returns:
        Result object with output video path or error
    """
    # Get LLM client with specified provider, if any
    llm_client = get_llm_client(
        provider=llm_provider,
        api_key=api_key,
        model=model
    )
    
    director = get_director_agent(llm_client)
    return director.assemble_video(video_path, clips_data, texts_data, output_path)
