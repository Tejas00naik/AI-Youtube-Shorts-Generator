"""
Script validator for AI YouTube Shorts Generator.

This module validates script structure and ensures timing 
constraints are met for YouTube Shorts format.
"""

import logging
from typing import Dict, Any, List, Tuple

from core.error_handler import Result

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_script(script: List[Dict[str, Any]]) -> Result:
    """
    Validate a script for YouTube Shorts.
    
    Args:
        script: List of segment dictionaries (clip and narration segments)
        
    Returns:
        Result object indicating if validation passed or failed
    """
    if not script or not isinstance(script, list):
        return Result.failure("Invalid script format - must be a non-empty list")
    
    # Calculate total duration
    total_duration = 0
    for item in script:
        if not isinstance(item, dict):
            return Result.failure(f"Invalid segment format: {item}")
            
        if "type" not in item:
            return Result.failure(f"Missing 'type' in segment: {item}")
            
        if item["type"] == "clip":
            if "start" not in item or "end" not in item:
                return Result.failure(f"Clip segment missing start/end time: {item}")
                
            if not isinstance(item["start"], (int, float)) or not isinstance(item["end"], (int, float)):
                return Result.failure(f"Clip start/end must be numbers: {item}")
                
            if item["start"] >= item["end"]:
                return Result.failure(f"Clip start must be before end: {item}")
                
            clip_duration = item["end"] - item["start"]
            total_duration += clip_duration
            
            # Check for reasonable clip length
            if clip_duration < 2:
                logger.warning(f"Very short clip detected ({clip_duration}s): {item}")
            elif clip_duration > 30:
                logger.warning(f"Very long clip detected ({clip_duration}s): {item}")
                
        elif item["type"] == "narration":
            if "text" not in item or "duration" not in item:
                return Result.failure(f"Narration segment missing text or duration: {item}")
                
            if not isinstance(item["duration"], (int, float)):
                return Result.failure(f"Narration duration must be a number: {item}")
                
            if item["duration"] <= 0:
                return Result.failure(f"Narration duration must be positive: {item}")
                
            total_duration += item["duration"]
            
            # Check narration text length
            text = item["text"]
            word_count = len(text.split())
            if word_count > 15:
                logger.warning(f"Narration exceeds recommended 15 words ({word_count}): {text}")
                
            # Check for reasonable narration duration
            if item["duration"] < 1:
                logger.warning(f"Very short narration detected ({item['duration']}s): {item}")
            elif item["duration"] > 5:
                logger.warning(f"Very long narration detected ({item['duration']}s): {item}")
                
        else:
            return Result.failure(f"Unknown segment type '{item['type']}': {item}")
    
    # Check total duration
    if total_duration > 60:
        return Result.failure(f"Script duration ({total_duration:.2f}s) exceeds 60s limit for Shorts")
        
    logger.info(f"Script validation passed - total duration: {total_duration:.2f}s")
    return Result.success({"total_duration": total_duration})

def check_script_continuity(script: List[Dict[str, Any]]) -> Result:
    """
    Check for continuity issues in the script.
    
    Args:
        script: List of segment dictionaries
        
    Returns:
        Result object with validation results and any warnings
    """
    if not script:
        return Result.failure("Empty script")
        
    warnings = []
    clip_segments = [s for s in script if s["type"] == "clip"]
    
    # Check for clip continuity
    for i in range(len(clip_segments) - 1):
        current = clip_segments[i]
        next_clip = clip_segments[i + 1]
        
        # Check for gaps
        if current["end"] != next_clip["start"]:
            warnings.append(
                f"Gap or overlap between clips at {current['end']}s - " +
                f"Clip {i} ends at {current['end']}s but clip {i+1} starts at {next_clip['start']}s"
            )
            
    if warnings:
        for warning in warnings:
            logger.warning(warning)
        return Result.success({"warnings": warnings})
        
    return Result.success({"warnings": []})

def get_total_duration(script: List[Dict[str, Any]]) -> float:
    """Calculate total duration of a script."""
    total = 0
    for item in script:
        if item["type"] == "clip":
            total += item["end"] - item["start"]
        else:  # narration
            total += item["duration"]
    return total
