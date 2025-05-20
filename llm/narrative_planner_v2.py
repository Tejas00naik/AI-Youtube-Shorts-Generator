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
                     tone: Optional[str] = "professional") -> Result:
        """
        Generate a narrative plan with alternating action clips and narration pauses.
        
        Args:
            transcript: The transcript of the video
            user_directions: User's directions for the narrative
            clip_count: Optional number of action clips to include (default: auto-determined)
            tone: Tone of the narrative (default: professional)
            
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
        
        # Build the system prompt
        system_prompt = f'''
        You're a YouTube Shorts director analyzing this podcast/video:
        {transcript}
        
        Create {clip_count} video segments alternating between:
        - Action Clips (speaker footage, 5-8s)
        - Narration Pauses (text overlay, 2-3s)
        
        User directions: {user_directions}
        Tone: {tone}
        
        Rules:
        1. Total duration: 55-59.5s
        2. Start/end with action clips
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
        return self._generate_fallback_plan(transcript, clip_count)
    
    def _generate_with_llm(self, system_prompt: str) -> Result:
        """Generate narrative plan using LLM."""
        try:
            # Make API call
            response = self.openai_client.chat.completions.create(
                model=self.config.get('OPENAI_MODEL', 'gpt-4'),
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                response_format={ "type": "json_object" }
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            plan = json.loads(response_text)
            
            # Validate the plan
            validation_result = self._validate_plan_contract(plan)
            if not validation_result.is_success:
                logger.error(f"LLM-generated plan failed validation: {validation_result.error.message}")
                return validation_result
            
            return Result.success(plan)
            
        except Exception as e:
            logger.error(f"Error in LLM generation: {str(e)}")
            return Result.failure(f"LLM generation error: {str(e)}")
    
    def _generate_fallback_plan(self, transcript: str, clip_count: int) -> Result:
        """Generate a simple fallback narrative plan."""
        # Create a plan with evenly distributed clips
        segments = []
        total_words = len(transcript.split())
        words_per_second = 5  # Assumption
        estimated_duration = min(total_words / words_per_second, 120)  # Cap at 120s
        
        clip_duration = 7.0  # target seconds per action clip
        pause_duration = 2.5  # seconds per pause
        
        current_time = 0.0
        words = transcript.split()
        words_per_clip = len(words) // clip_count
        
        for i in range(clip_count):
            # Add action clip
            clip_end = min(current_time + clip_duration, estimated_duration)
            segments.append({
                "type": "action",
                "start": current_time,
                "end": clip_end
            })
            current_time = clip_end
            
            # Add pause (except after last clip)
            if i < clip_count - 1:
                # Get text from transcript for this section
                start_word = i * words_per_clip
                end_word = min(start_word + 15, len(words))
                text = " ".join(words[start_word:end_word])
                if len(text) > 100:  # Truncate if too long
                    text = text[:100] + "..."
                
                segments.append({
                    "type": "pause",
                    "text": text,
                    "duration": pause_duration
                })
                current_time += pause_duration
        
        # Calculate total duration
        total_duration = sum(
            seg["end"] - seg["start"] if seg["type"] == "action" else seg["duration"] 
            for seg in segments
        )
        
        plan = {
            "segments": segments,
            "total_duration": total_duration,
            "is_fallback": True
        }
        
        return Result.success(plan)
    
    def _validate_plan_contract(self, plan: Dict[str, Any]) -> Result:
        """Validate that the generated plan follows the contract."""
        # Check basic structure
        if not isinstance(plan, dict):
            return Result.failure("Plan must be a dictionary")
        
        if "segments" not in plan:
            return Result.failure("Plan must contain 'segments' key")
        
        segments = plan.get("segments", [])
        if not segments or not isinstance(segments, list):
            return Result.failure("Segments must be a non-empty list")
        
        # Validate segments
        total_duration = 0
        prev_type = None
        
        for i, segment in enumerate(segments):
            if not isinstance(segment, dict):
                return Result.failure(f"Segment {i} must be a dictionary")
            
            # Check required fields
            if "type" not in segment:
                return Result.failure(f"Segment {i} missing 'type'")
            
            segment_type = segment["type"]
            
            # Alternating pattern check
            if prev_type == segment_type and i > 0:
                return Result.failure(f"Segment {i} has same type as previous segment, must alternate")
            
            prev_type = segment_type
            
            # Type-specific validation
            if segment_type == "action":
                if "start" not in segment or "end" not in segment:
                    return Result.failure(f"Action segment {i} missing start/end times")
                
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
                           openai_client=None) -> Result:
    """
    Convenience function to generate a narrative plan.
    
    Args:
        transcript: The transcript of the video
        user_directions: User's directions for the narrative
        clip_count: Optional number of action clips to include
        tone: Tone of the narrative
        openai_client: Optional OpenAI client instance
        
    Returns:
        Result object with the narrative plan or error
    """
    planner = get_narrative_planner(openai_client)
    return planner.generate_plan(transcript, user_directions, clip_count, tone)
