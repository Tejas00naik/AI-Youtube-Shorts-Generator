"""
Clip Validator for AI YouTube Shorts Generator.

This module validates clip selections against video constraints
to prevent overlaps and ensure clips fit within the video duration.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import numpy as np
from moviepy.editor import VideoFileClip

from core.error_handler import Result

# Configure logging
logger = logging.getLogger(__name__)

class ClipValidator:
    """
    Validates clip selections to ensure they meet video constraints.
    This class helps prevent common issues like overlapping timestamps
    and clips exceeding video duration.
    """
    
    @staticmethod
    def validate_narrative_plan(narrative_plan: Dict[str, Any], video_path: str) -> Result:
        """
        Validate that a narrative plan's clips fit within the video bounds
        and don't contain overlapping timestamps.
        
        Args:
            narrative_plan: The narrative plan to validate
            video_path: Path to the source video file
            
        Returns:
            Result object with the validated plan or error.
            The validated plan is accessible via result.value
        """
        try:
            # Check that the video exists
            if not os.path.exists(video_path):
                return Result.failure(f"Video file not found: {video_path}")

            # Get video properties
            try:
                video = VideoFileClip(video_path)
                video_duration = video.duration
                fps = video.fps if video.fps else 30  # Default to 30 if not available
                logger.info(f"Video properties: {fps} fps, {int(video_duration * fps)} frames, {video_duration:.2f}s duration")
            except Exception as e:
                return Result.failure(f"Error loading video: {str(e)}")
            
            # Check that all segments fit within video duration
            adjusted_plan = ClipValidator._adjust_segments_to_video(narrative_plan, video_duration)
            
            # Check for and fix overlapping segments
            final_plan = ClipValidator._fix_overlapping_segments(adjusted_plan)
            
            # Close the video file
            video.close()
            
            return Result.success(final_plan)
            
        except Exception as e:
            logger.error(f"Error validating narrative plan: {str(e)}")
            return Result.failure(f"Clip validation error: {str(e)}")
    
    @staticmethod
    def _adjust_segments_to_video(narrative_plan: Dict[str, Any], video_duration: float) -> Dict[str, Any]:
        """
        Adjust segments to fit within the video duration.
        
        Args:
            narrative_plan: The narrative plan with segments to adjust
            video_duration: Total duration of the source video
            
        Returns:
            Updated narrative plan with adjusted segments
        """
        adjusted_plan = narrative_plan.copy()
        adjusted_segments = []
        
        for segment in narrative_plan.get('segments', []):
            if segment['type'] == 'action':
                # Handle clips that exceed video duration
                if segment['start_time'] >= video_duration:
                    logger.warning(f"Segment start time ({segment['start_time']}s) exceeds video duration ({video_duration}s)")
                    # Skip this segment as it's completely outside the video
                    continue
                    
                if segment['end_time'] > video_duration:
                    logger.warning(f"Segment end time ({segment['end_time']}s) exceeds video duration ({video_duration}s)")
                    # Adjust end time to fit within video, maintaining at least 1s duration
                    new_end_time = video_duration
                    new_start_time = min(segment['start_time'], video_duration - 1.0)
                    segment = segment.copy()
                    segment['start_time'] = new_start_time
                    segment['end_time'] = new_end_time
            
            # Keep non-action segments as they are
            adjusted_segments.append(segment)
        
        adjusted_plan['segments'] = adjusted_segments
        
        # Recalculate total duration
        total_duration = ClipValidator._calculate_total_duration(adjusted_segments)
        adjusted_plan['total_duration'] = total_duration
        
        return adjusted_plan
    
    @staticmethod
    def _fix_overlapping_segments(narrative_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fix overlapping segments by adjusting their start and end times.
        
        Args:
            narrative_plan: The narrative plan with segments to fix
            
        Returns:
            Updated narrative plan with non-overlapping segments
        """
        fixed_plan = narrative_plan.copy()
        segments = narrative_plan.get('segments', [])
        fixed_segments = []
        
        # Track last end time for action segments
        last_end_time = 0
        
        for i, segment in enumerate(segments):
            if segment['type'] == 'action':
                # If this segment starts before the last one ended, adjust it
                if segment['start_time'] < last_end_time:
                    logger.warning(f"Overlapping segments detected: segment {i} starts at {segment['start_time']}s but previous segment ended at {last_end_time}s")
                    
                    # Make a copy to avoid modifying the original
                    segment = segment.copy()
                    
                    # Add a small buffer between segments
                    buffer = 0.2  # 200ms buffer
                    segment['start_time'] = last_end_time + buffer
                    
                    # Ensure the clip is at least 1 second long
                    min_duration = 1.0
                    if segment['end_time'] - segment['start_time'] < min_duration:
                        segment['end_time'] = segment['start_time'] + min_duration
                
                # Update the last end time
                last_end_time = segment['end_time']
            
            # Add the segment (original or adjusted) to the fixed list
            fixed_segments.append(segment)
        
        fixed_plan['segments'] = fixed_segments
        
        # Recalculate total duration
        total_duration = ClipValidator._calculate_total_duration(fixed_segments)
        fixed_plan['total_duration'] = total_duration
        
        return fixed_plan
    
    @staticmethod
    def _calculate_total_duration(segments: List[Dict[str, Any]]) -> float:
        """
        Calculate the total duration of all segments.
        
        Args:
            segments: List of segments to calculate duration for
            
        Returns:
            Total duration in seconds
        """
        # For action segments, calculate from start_time to end_time
        # For pause segments, use the duration field
        
        total_duration = 0.0
        action_segments = [s for s in segments if s['type'] == 'action']
        
        if action_segments:
            # Find the earliest start time and latest end time
            earliest_start = min(s['start_time'] for s in action_segments)
            latest_end = max(s['end_time'] for s in action_segments)
            
            # Base calculation on action segments
            action_duration = latest_end - earliest_start
            
            # Add durations of pause segments
            pause_duration = sum(s.get('duration', 0) for s in segments if s['type'] != 'action')
            
            total_duration = action_duration + pause_duration
        else:
            # If no action segments, just sum the durations of pause segments
            total_duration = sum(s.get('duration', 0) for s in segments if s['type'] != 'action')
        
        return total_duration
