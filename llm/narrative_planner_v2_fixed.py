"""
Narrative Planner v2 for AI YouTube Shorts Generator.

This module generates an alternating sequence of action clips and narration pauses
using a more structured contract-based approach.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

from core.error_handler import Result
from core.config import get_config

# Configure logging
logger = logging.getLogger(__name__)

class NarrativePlannerV2:
    """
    Narrative Planner that creates an alternating sequence of 
    action clips and narration pauses for a YouTube Short.
    """
    
    def __init__(self, openai_client=None):
        """Initialize the narrative planner."""
        self.openai_client = openai_client
        self.config = get_config()
    
    def generate_plan(self, transcript: str, user_directions: str, 
                     clip_count: Optional[int] = None, 
                     tone: Optional[str] = "professional",
                     interruption_style: str = "pause",
                     interruption_frequency: Optional[int] = None,
                     max_duration: Optional[float] = 60.0) -> Result:
        """
        Generate a narrative plan with dynamically determined structure based on user prompt.
        
        Args:
            transcript: The transcript of the video
            user_directions: User's directions for the narrative
            clip_count: Optional number of action clips to include (default: auto-determined)
            tone: Tone of the narrative (default: professional)
            interruption_style: Style of interruptions ('pause' or 'continuous')
            interruption_frequency: Number of interruptions to include
            max_duration: Maximum duration of the video in seconds
            
        Returns:
            Result object with the narrative plan or error
        """
        # Auto-determine clip count if not specified (based on transcript length)
        if clip_count is None:
            # Simple heuristic: ~5 words per second, aiming for ~7 sec clips
            word_count = len(transcript.split())
            estimated_duration = word_count / 5  # in seconds
            clip_count = min(max(3, int(estimated_duration / 15)), 6)  # Between 3-6 clips
            logger.info(f"Auto-determined clip count: {clip_count}")
            
        # Auto-determine interruption frequency if not specified
        if interruption_frequency is None:
            if interruption_style == "pause":
                # Default: one less than clip count (pauses between clips)
                interruption_frequency = clip_count - 1
            else:
                # Continuous style has no interruptions
                interruption_frequency = 0
                
        # Validate max duration
        if max_duration is None or max_duration > 60.0:
            max_duration = 60.0  # Cap at 60 seconds
            
        logger.info(f"Using clip_count={clip_count}, style={interruption_style}, interruptions={interruption_frequency}, max_duration={max_duration}")
            
        # Build the system prompt based on interruption style
        if interruption_style == "continuous":
            # Continuous style: no text interruptions
            system_prompt = f'''
            You're a YouTube Shorts director analyzing this podcast/video:
            {transcript}
            
            Create a continuous {clip_count}-clip narrative without interruptions.
            - Action Clips: {clip_count} clips, each 5-10s
            - No text interruptions between clips
            
            User directions: {user_directions}
            Tone: {tone}
            
            Rules:
            1. Total duration: {max_duration-5}-{max_duration}s
            2. Select the most engaging moments
            3. Ensure narrative flow between clips
            
            Return JSON:
            {{
              "segments": [
                {{"type": "action", "start": float, "end": float}},
                {{"type": "action", "start": float, "end": float}},
                ...
              ]
            }}
            '''
        else:
            # Pause style: text interruptions between clips
            system_prompt = f'''
            You're a YouTube Shorts director analyzing this podcast/video:
            {transcript}
            
            Create a narrative with {clip_count} action clips and {interruption_frequency} text interruptions:
            - Action Clips: speaker footage, 5-8s each
            - Text Pauses: overlay text, 2-3s each
            
            User directions: {user_directions}
            Tone: {tone}
            
            Rules:
            1. Total duration: {max_duration-5}-{max_duration}s
            2. {'Start and end with action clips' if interruption_frequency < clip_count else 'Alternate between action and pause'}
            3. Pauses must explain next clip's value
            4. Text under 15 words, no markdown
            
            Return JSON:
            {{
              "segments": [
                {{"type": "action", "start": float, "end": float}},
                {{"type": "pause", "text": "Hook: Why startups fail", "duration": 2.5}},
                ...
              ]
            }}
            '''
            
        # Try to use OpenAI API if available
        if self.openai_client:
            try:
                return self._generate_with_llm(system_prompt)
            except Exception as e:
                logger.error(f"LLM generation failed: {str(e)}")
                logger.warning("Falling back to default plan generation")
        
        # Fallback to simple plan generation
        return self._generate_fallback_plan(transcript, clip_count, interruption_style, interruption_frequency)
    
    def _generate_with_llm(self, system_prompt: str) -> Result:
        """Generate narrative plan using LLM."""
        try:
            from llm.llm_client import LLMClient
            
            # Create LLM client if needed
            llm_client = LLMClient()
            
            # Make API call
            response = llm_client.chat_completion(
                system_prompt=system_prompt,
                user_messages="Generate a narrative plan as specified",
                temperature=0.7,
                json_response=True
            )
            
            # Check for errors
            if "error" in response:
                return Result.failure(f"LLM API error: {response['error']}")
            
            # Parse response
            content = response["content"].strip()
            plan = json.loads(content)
            
            # Validate the plan
            validation_result = self._validate_plan_contract(plan)
            if not validation_result.is_success:
                logger.error(f"LLM-generated plan failed validation: {str(validation_result.error)}")
                return validation_result
            
            return Result.success(plan)
            
        except Exception as e:
            logger.error(f"Error in LLM generation: {str(e)}")
            return Result.failure(f"LLM generation error: {str(e)}")
    
    def _generate_fallback_plan(self, transcript: str, clip_count: int, 
                              interruption_style: str = "pause", 
                              interruption_frequency: Optional[int] = None) -> Result:
        """Generate a simple fallback plan when LLM generation fails."""
        try:
            # Simple approach: divide transcript into equal segments
            words = transcript.split()
            total_words = len(words)
            
            # Determine interruption frequency if not specified
            if interruption_frequency is None:
                if interruption_style == "pause":
                    interruption_frequency = clip_count - 1
                else:
                    interruption_frequency = 0
            
            # Aim for ~30-35 seconds total video
            total_duration = min(35.0, max(25.0, total_words / 12))
            
            # Each action clip should be ~7 seconds
            clip_duration = 7.0
            
            # Each pause should be ~2.5 seconds
            pause_duration = 2.5
            
            # Calculate total segments (action + pause)
            segments = []
            current_time = 0.0
            
            if interruption_style == "continuous" or interruption_frequency == 0:
                # Continuous style: only action clips, no pauses
                for i in range(clip_count):
                    end_time = current_time + clip_duration
                    segments.append({
                        "type": "action",
                        "start": current_time,
                        "end": end_time
                    })
                    current_time = end_time
            else:
                # Pause style: alternating action and pause segments
                pause_count = min(interruption_frequency, clip_count - 1)
                
                # Add first action segment
                segments.append({
                    "type": "action",
                    "start": 0.0,
                    "end": clip_duration
                })
                current_time = clip_duration
                
                # Add remaining segments with pauses
                for i in range(1, clip_count):
                    # Add a pause before this action clip if we haven't reached max pauses
                    if i <= pause_count:
                        # Extract some text from the transcript for this pause
                        start_idx = int(i * total_words / clip_count) - 15
                        end_idx = start_idx + 30
                        start_idx = max(0, start_idx)
                        end_idx = min(total_words, end_idx)
                        
                        pause_text = " ".join(words[start_idx:end_idx])
                        
                        segments.append({
                            "type": "pause",
                            "text": pause_text,
                            "duration": pause_duration
                        })
                        current_time += pause_duration
                    
                    # Add the action clip
                    end_time = current_time + clip_duration
                    segments.append({
                        "type": "action",
                        "start": current_time,
                        "end": end_time
                    })
                    current_time = end_time
            
            # Create the narrative plan
            narrative_plan = {
                "segments": segments,
                "total_duration": current_time,
                "is_fallback": True,
                "interruption_style": interruption_style,
                "interruption_frequency": interruption_frequency if interruption_style == "pause" else 0
            }
            
            return Result.success(narrative_plan)
            
        except Exception as e:
            logger.error(f"Error in fallback plan generation: {str(e)}")
            return Result.failure(f"Fallback plan generation failed: {str(e)}")
    
    def _validate_plan_contract(self, plan: Dict[str, Any]) -> Result:
        """Validate that the generated plan follows the contract."""
        if not isinstance(plan, dict):
            return Result.failure("Plan must be a dictionary")
        
        if "segments" not in plan:
            return Result.failure("Plan must contain 'segments' key")
        
        segments = plan["segments"]
        if not isinstance(segments, list):
            return Result.failure("Segments must be a list")
        
        if len(segments) < 3:
            return Result.failure("Plan must have at least 3 segments")
        
        # Track total duration
        total_duration = 0.0
        
        # Validate each segment
        for i, segment in enumerate(segments):
            if not isinstance(segment, dict):
                return Result.failure(f"Segment {i} must be a dictionary")
            
            if "type" not in segment:
                return Result.failure(f"Segment {i} missing type")
            
            segment_type = segment["type"]
            
            if segment_type == "action":
                if "start" not in segment or "end" not in segment:
                    return Result.failure(f"Action segment {i} missing start/end")
                
                start = segment["start"]
                end = segment["end"]
                
                if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                    return Result.failure(f"Segment {i} start/end must be numbers")
                
                if start >= end:
                    return Result.failure(f"Segment {i} start must be less than end")
                
                duration = end - start
                if duration < 5.0 or duration > 8.0:
                    logger.warning(f"Segment {i} duration ({duration}s) outside recommended range (5-8s)")
                
                total_duration += duration
                
            elif segment_type == "pause":
                if "text" not in segment or "duration" not in segment:
                    return Result.failure(f"Pause segment {i} missing text or duration")
                
                text = segment["text"]
                duration = segment["duration"]
                
                if not isinstance(text, str):
                    return Result.failure(f"Segment {i} text must be a string")
                
                if not isinstance(duration, (int, float)):
                    return Result.failure(f"Segment {i} duration must be a number")
                
                if duration < 2.0 or duration > 3.0:
                    logger.warning(f"Segment {i} duration ({duration}s) outside recommended range (2-3s)")
                
                word_count = len(text.split())
                if word_count > 15:
                    logger.warning(f"Segment {i} text exceeds 15 words ({word_count} words)")
                
                total_duration += duration
                
            else:
                return Result.failure(f"Segment {i} has invalid type: {segment_type}")
        
        # Check first and last segment types
        if segments[0]["type"] != "action":
            return Result.failure("First segment must be an action clip")
            
        if segments[-1]["type"] != "action":
            return Result.failure("Last segment must be an action clip")
        
        # Check total duration
        if total_duration < 55.0 or total_duration > 59.5:
            logger.warning(f"Total duration ({total_duration:.1f}s) outside recommended range (55-59.5s)")
        
        # Add total_duration to plan if not already there
        if "total_duration" not in plan:
            plan["total_duration"] = total_duration
        
        return Result.success(plan)


# Singleton instance
_narrative_planner = None

def get_narrative_planner(openai_client=None) -> NarrativePlannerV2:
    """Get the singleton narrative planner instance."""
    global _narrative_planner
    if _narrative_planner is None:
        _narrative_planner = NarrativePlannerV2(openai_client)
    return _narrative_planner

def generate_narrative_plan(transcript: str, user_directions: str, 
                           clip_count: Optional[int] = None,
                           tone: Optional[str] = "professional",
                           interruption_style: str = "pause",
                           interruption_frequency: Optional[int] = None,
                           max_duration: Optional[float] = 60.0,
                           openai_client=None) -> Result:
    """
    Convenience function to generate a narrative plan.
    
    Args:
        transcript: The transcript of the video
        user_directions: User's directions for the narrative
        clip_count: Optional number of action clips to include
        tone: Tone of the narrative
        interruption_style: Style of interruptions ('pause' or 'continuous')
        interruption_frequency: Number of interruptions to include
        max_duration: Maximum duration of the video in seconds
        openai_client: Optional OpenAI client instance
        
    Returns:
        Result object with the narrative plan or error
    """
    planner = get_narrative_planner(openai_client)
    return planner.generate_plan(
        transcript, 
        user_directions, 
        clip_count, 
        tone,
        interruption_style,
        interruption_frequency,
        max_duration
    )
