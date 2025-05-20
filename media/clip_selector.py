"""
Clip Selector for AI YouTube Shorts Generator.

This module selects the best clips from a video based on the narrative plan,
focusing on visual quality, speaker visibility, and audio clarity.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
from moviepy.editor import VideoFileClip

from media.clip_validator import ClipValidator
from core.error_handler import Result

# Configure logging
logger = logging.getLogger(__name__)

class ClipSelector:
    """
    Selects optimal video clips based on quality metrics and the narrative plan.
    """
    
    def __init__(self, openai_client=None):
        """Initialize the clip selector."""
        self.openai_client = openai_client
        
    def select_clips(self, video_path: str, narrative_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select clips from the video based on the narrative plan.
        First validates the plan against video constraints to avoid
        overlapping clips and ensure all clips fit within video duration.
        
        Args:
            video_path: Path to the source video file
            narrative_plan: The narrative plan with segments
            
        Returns:
            Dict with selected clips information
        """
        # Validate and adjust the narrative plan against video constraints
        validation_result = ClipValidator.validate_narrative_plan(narrative_plan, video_path)
        
        if not validation_result.is_success:
            logger.error(f"Clip validation failed: {validation_result.error}")
            raise ValueError(f"Clip validation error: {validation_result.error}")
            
        # Use the validated and adjusted narrative plan
        validated_plan = validation_result.value
        
        # Get video properties (already done by validator, but needed here too)
        video = VideoFileClip(video_path)
        fps = video.fps if video.fps else 30  # Default to 30 if not available
        
        # Extract clips based on validated narrative plan
        clips = []
        for segment in validated_plan.get('segments', []):
            if segment['type'] == 'action':
                start_time = segment['start_time']
                end_time = segment['end_time']
                
                # Add a small buffer at the end to ensure complete speech
                speech_completion_buffer = 0.5  # 0.5 seconds
                if end_time + speech_completion_buffer <= video.duration:
                    end_time += speech_completion_buffer
                
                clip_data = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'content': segment.get('content', '')
                }
                clips.append(clip_data)
        
        video.close()
        
        result = {
            'clips': clips,
            'video_path': video_path,
            'fps': fps
        }
        
        # Log the selected clips for debugging
        for i, clip in enumerate(clips):
            logger.info(f"Clip {i+1}: {clip['start_time']:.1f}s to {clip['end_time']:.1f}s ({clip['duration']:.1f}s duration)")
        
        return result

    def _select_with_llm(self, video_path: str, narrative_plan: Dict[str, Any], 
                       action_segments: List[Dict[str, Any]]) -> Result:
        """Select clips using LLM assistance."""
        # This would typically involve:
        # 1. Extracting frames/features from video
        # 2. Sending to LLM with appropriate prompt
        # 3. Processing the response
        
        # For now, we'll use a simplified prompt
        prompt = f'''
        For action segments in this plan:
        {json.dumps(narrative_plan, indent=2)}
        
        Select BEST clips matching:
        - Clear speaker face visible
        - High audio clarity
        - No rapid scene changes
        
        Return JSON array with exact frame times:
        {{
          "clips": [
            {{"start": 12.3, "end": 19.8, "focus_point": "speaker_face"}},
            ...
          ]
        }}
        '''
        
        try:
            # Make API call
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={ "type": "json_object" }
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            clips_data = json.loads(response_text)
            
            # Validate the clips
            validation_result = self._validate_clips_contract(clips_data, video_path)
            if not validation_result.is_success:
                logger.error(f"LLM-selected clips failed validation: {validation_result.error.message}")
                return validation_result
            
            return Result.success(clips_data)
            
        except Exception as e:
            logger.error(f"Error in LLM clip selection: {str(e)}")
            return Result.failure(f"LLM clip selection error: {str(e)}")
    
    def _select_clips_with_analysis(self, video_path: str, 
                                  action_segments: List[Dict[str, Any]]) -> Result:
        """
        Select best clips using computer vision analysis.
        This is a fallback when LLM selection is not available.
        """
        try:
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return Result.failure(f"Could not open video file: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            logger.info(f"Video properties: {fps} fps, {frame_count} frames, {duration:.2f}s duration")
            
            # Prepare result
            clips = []
            
            # Process each action segment
            for i, segment in enumerate(action_segments):
                # Get segment times
                start_time = segment["start"]
                end_time = segment["end"]
                
                # Validate times against video duration
                if start_time >= duration:
                    logger.warning(f"Segment {i} start time ({start_time}s) exceeds video duration ({duration:.2f}s)")
                    start_time = max(0, duration - 10)  # Default to last 10 seconds
                
                if end_time > duration:
                    logger.warning(f"Segment {i} end time ({end_time}s) exceeds video duration ({duration:.2f}s)")
                    end_time = duration
                
                # For simplicity in fallback mode, we'll just use the segment times as-is
                # In a full implementation, we'd analyze video quality here
                clip_info = {
                    "start": start_time,
                    "end": end_time,
                    "duration": end_time - start_time,
                    "focus_point": "speaker_face"  # Default assumption
                }
                
                clips.append(clip_info)
            
            # Clean up
            cap.release()
            
            result = {"clips": clips}
            return Result.success(result)
            
        except Exception as e:
            logger.error(f"Error in clip analysis: {str(e)}")
            return Result.failure(f"Clip analysis error: {str(e)}")
    
    def _validate_clips_contract(self, clips_data: Dict[str, Any], video_path: str) -> Result:
        """Validate that the clips data follows the contract."""
        # Open video to get duration
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return Result.failure(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frame_count / fps
        
        cap.release()
        
        # Check basic structure
        if not isinstance(clips_data, dict):
            return Result.failure("Clips data must be a dictionary")
        
        if "clips" not in clips_data:
            return Result.failure("Clips data must contain 'clips' key")
        
        clips = clips_data.get("clips", [])
        if not clips or not isinstance(clips, list):
            return Result.failure("Clips must be a non-empty list")
        
        # Validate each clip
        for i, clip in enumerate(clips):
            if not isinstance(clip, dict):
                return Result.failure(f"Clip {i} must be a dictionary")
            
            # Check required fields
            if "start" not in clip:
                return Result.failure(f"Clip {i} missing 'start' time")
                
            if "end" not in clip:
                return Result.failure(f"Clip {i} missing 'end' time")
                
            start = clip["start"]
            end = clip["end"]
            
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                return Result.failure(f"Clip {i} start/end must be numbers")
            
            if start < 0:
                return Result.failure(f"Clip {i} start time ({start}s) must be non-negative")
            
            if end > video_duration:
                logger.warning(f"Clip {i} end time ({end}s) exceeds video duration ({video_duration:.2f}s), will truncate")
                clip["end"] = video_duration
            
            if start >= end:
                return Result.failure(f"Clip {i} start time ({start}s) must be less than end time ({end}s)")
            
            duration = end - start
            clip["duration"] = duration
            
            if duration < 5.0 or duration > 8.0:
                logger.warning(f"Clip {i} duration ({duration:.2f}s) outside recommended range (5-8s)")
            
            # Check focus point
            if "focus_point" not in clip:
                clip["focus_point"] = "speaker_face"  # Default
            else:
                focus_point = clip["focus_point"]
                if focus_point not in ["speaker_face", "slide", "demo_screen"]:
                    logger.warning(f"Clip {i} has unknown focus point: {focus_point}, defaulting to 'speaker_face'")
                    clip["focus_point"] = "speaker_face"
        
        return Result.success(clips_data)


# Singleton instance
_clip_selector = None

def get_clip_selector(openai_client=None) -> ClipSelector:
    """Get the singleton clip selector instance."""
    global _clip_selector
    if _clip_selector is None:
        _clip_selector = ClipSelector(openai_client)
    return _clip_selector

def select_best_clips(video_path: str, narrative_plan: Dict[str, Any], 
                     openai_client=None) -> Result:
    """
    Convenience function to select the best clips from a video.
    
    Args:
        video_path: Path to the video file
        narrative_plan: The narrative plan with action segments
        openai_client: Optional OpenAI client instance
        
    Returns:
        Result object with optimized clips or error
    """
    selector = get_clip_selector(openai_client)
    # Pass parameters in the right order (video_path, narrative_plan) as expected by select_clips
    return selector.select_clips(video_path, narrative_plan)
