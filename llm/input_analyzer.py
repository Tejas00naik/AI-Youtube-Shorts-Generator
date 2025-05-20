#!/usr/bin/env python3
"""
Input analyzer for the AI YouTube Shorts Generator.
Analyzes user prompts to dynamically determine optimal parameters for video generation.
"""

import os
import re
import json
import logging
from typing import Dict, Any, Tuple, Optional

from llm.prompt_templates import INPUT_ANALYZER_PROMPT

from core.error_handler import Result
from llm.llm_client import LLMClient

# Set up logging
logger = logging.getLogger(__name__)

class InputAnalyzer:
    """
    Analyzes user input to determine optimal parameters for video generation.
    Uses LLM to extract key parameters from natural language input.
    """
    
    def __init__(self, llm_client=None):
        """Initialize the input analyzer with an optional LLM client."""
        self.llm_client = llm_client or LLMClient()
    
    def analyze_prompt(self, prompt: str, transcript_length: int) -> Result:
        """
        Analyze the user prompt to determine optimal video parameters.
        
        Args:
            prompt: The user's prompt text
            transcript_length: Length of the source video transcript
            
        Returns:
            Result object with video parameters or error
        """
        try:
            # Direct pattern matching for explicit parameters
            parameters = self._extract_direct_parameters(prompt)
            
            if parameters.get('clip_count') is None or parameters.get('interruption_style') is None:
                # Use LLM for more sophisticated analysis
                llm_parameters = self._analyze_with_llm(prompt, transcript_length)
                
                # Merge direct and LLM-derived parameters, prioritizing direct ones
                for key, value in llm_parameters.items():
                    if key not in parameters or parameters[key] is None:
                        parameters[key] = value
            
            # Apply validations and constraints
            self._validate_parameters(parameters, transcript_length)
            
            logger.info(f"Analyzed prompt, determined parameters: {parameters}")
            return Result.success(parameters)
            
        except Exception as e:
            logger.error(f"Error analyzing prompt: {str(e)}")
            return Result.failure(f"Prompt analysis error: {str(e)}")
    
    def _extract_direct_parameters(self, prompt: str) -> Dict[str, Any]:
        """Extract explicit parameters from the prompt using regex patterns."""
        parameters = {
            'clip_count': None,
            'interruption_style': None,
            'interruption_frequency': None,
            'max_duration': None
        }
        
        # Extract clip count
        clip_count_match = re.search(r'(\d+)\s*clips?', prompt, re.IGNORECASE)
        if clip_count_match:
            parameters['clip_count'] = int(clip_count_match.group(1))
        
        # Extract interruption style preference
        if re.search(r'continuous|no\s*interruptions?|no\s*pauses?|without\s*stopping', prompt, re.IGNORECASE):
            parameters['interruption_style'] = 'continuous'
        elif re.search(r'with\s*pauses?|interrupted|text\s*overlays?', prompt, re.IGNORECASE):
            parameters['interruption_style'] = 'pause'
        
        # Extract max duration
        duration_match = re.search(r'(\d+)\s*seconds?|(\d+)s\b', prompt, re.IGNORECASE)
        if duration_match:
            duration = int(duration_match.group(1) or duration_match.group(2))
            parameters['max_duration'] = min(duration, 60)  # Cap at 60 seconds
        
        return parameters
    
    def _analyze_with_llm(self, prompt: str, transcript_length: int) -> Dict[str, Any]:
        """Use LLM to analyze the prompt and determine optimal parameters."""
        # Calculate approximate duration from transcript length
        transcript_length_seconds = transcript_length / 20  # Simple heuristic
        
        # Format the system prompt using our template
        system_prompt = INPUT_ANALYZER_PROMPT.format(
            user_input=prompt,
            transcript_length_seconds=transcript_length_seconds
        )
        
        try:
            response = self.llm_client.chat_completion(
                system_prompt=system_prompt,
                user_messages=f"Analyze this prompt: {prompt}",
                temperature=0.3,
                json_response=True
            )
            
            if "error" in response:
                logger.error(f"LLM analysis error: {response['error']}")
                return self._get_default_parameters(prompt, transcript_length)
            
            # Parse LLM response
            content = response.get("content", "{}")
            parameters = json.loads(content)
            
            return parameters
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return self._get_default_parameters(prompt, transcript_length)
    
    def _get_default_parameters(self, prompt: str, transcript_length: int) -> Dict[str, Any]:
        """Determine default parameters based on simple heuristics when LLM fails."""
        # Simple word count to estimate complexity
        word_count = len(prompt.split())
        
        # Estimate clip count based on prompt length and transcript size
        if word_count < 10:
            clip_count = 3  # Simple/short prompt
        elif word_count < 25:
            clip_count = 4  # Medium prompt
        else:
            clip_count = 5  # Complex/detailed prompt
            
        # Adjust for transcript length
        if transcript_length > 3000:
            clip_count += 1
            
        # Cap at reasonable limits
        clip_count = min(clip_count, 8)
        
        # Determine interruption style - default to pause mode
        interruption_style = 'pause'
        
        # For longer prompts with storytelling, continuous might be better
        storytelling_indicators = ['story', 'journey', 'narrative', 'flow', 'sequence']
        if any(indicator in prompt.lower() for indicator in storytelling_indicators):
            interruption_style = 'continuous'
        
        return {
            'clip_count': clip_count,
            'interruption_style': interruption_style,
            'interruption_frequency': clip_count - 1 if interruption_style == 'pause' else 0,
            'max_duration': min(30 + (clip_count * 5), 60)  # Scale with clip count, max 60s
        }
    
    def _validate_parameters(self, parameters: Dict[str, Any], transcript_length: int) -> None:
        """Apply validations and constraints to the parameters."""
        # Ensure clip count is reasonable
        if parameters['clip_count'] is None:
            parameters['clip_count'] = 4  # Default
        else:
            parameters['clip_count'] = max(2, min(parameters['clip_count'], 8))
        
        # Default interruption style if not set
        if parameters['interruption_style'] is None:
            parameters['interruption_style'] = 'pause'
        
        # Set appropriate interruption frequency
        if parameters['interruption_frequency'] is None:
            if parameters['interruption_style'] == 'pause':
                parameters['interruption_frequency'] = parameters['clip_count'] - 1
            else:
                parameters['interruption_frequency'] = 0
        
        # Ensure max duration is reasonable
        if parameters['max_duration'] is None:
            # Estimate based on clip count: ~7s per clip, ~2.5s per interruption
            clip_time = parameters['clip_count'] * 7
            interruption_time = parameters['interruption_frequency'] * 2.5
            parameters['max_duration'] = min(clip_time + interruption_time, 60)
        else:
            parameters['max_duration'] = min(parameters['max_duration'], 60)


def analyze_user_prompt(prompt: str, transcript_length: int, llm_client=None) -> Result:
    """
    Convenience function to analyze a user prompt.
    
    Args:
        prompt: The user's prompt text
        transcript_length: Length of the source video transcript
        llm_client: Optional LLM client instance
        
    Returns:
        Result object with video parameters or error
    """
    analyzer = InputAnalyzer(llm_client)
    return analyzer.analyze_prompt(prompt, transcript_length)
