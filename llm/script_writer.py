"""
Script Writer for AI YouTube Shorts Generator.

This module generates compelling text for pause screens between video clips,
ensuring they tease the next clip's value and maintain user engagement.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

from core.error_handler import Result
from llm.llm_client import get_llm_client

# Configure logging
logger = logging.getLogger(__name__)

class ScriptWriter:
    """
    Generates pause screen text for narrative plans based on clip contexts.
    """
    
    def __init__(self, llm_client=None):
        """Initialize the script writer."""
        self.llm_client = llm_client
    
    def write_pause_texts(self, narrative_plan: Dict[str, Any], 
                         clip_contexts: List[Dict[str, Any]],
                         tone: str = "professional") -> Result:
        """
        Generate text for pause screens between clips.
        
        Args:
            narrative_plan: The narrative plan with action and pause segments
            clip_contexts: List of clip contexts with transcripts and metadata
            tone: Tone of the texts
            
        Returns:
            Result object with pause texts or error
        """
        # Validate inputs
        if not isinstance(narrative_plan, dict) or "segments" not in narrative_plan:
            return Result.failure("Invalid narrative plan format")
            
        if not isinstance(clip_contexts, list) or not clip_contexts:
            return Result.failure("Clip contexts must be a non-empty list")
        
        # Extract pause segments from narrative plan
        pause_segments = [s for s in narrative_plan["segments"] if s["type"] == "pause"]
        
        if not pause_segments:
            return Result.failure("No pause segments found in narrative plan")
        
        # Try to use LLM API if available
        if self.llm_client and self.llm_client.is_available():
            try:
                return self._write_texts_with_llm(pause_segments, clip_contexts, tone)
            except Exception as e:
                logger.error(f"LLM text generation failed: {str(e)}")
                logger.warning("Falling back to default text generation")
        
        # Fallback to basic text generation
        return self._write_fallback_texts(pause_segments, clip_contexts, tone)
    
    def _write_texts_with_llm(self, pause_segments: List[Dict[str, Any]],
                            clip_contexts: List[Dict[str, Any]], 
                            tone: str) -> Result:
        """Generate pause texts using LLM."""
        system_prompt = f'''
        Write pause screen text for:
        {json.dumps(clip_contexts, indent=2)}
        
        Rules:
        1. Match tone: {tone}
        2. Tease next clip's value
        3. Use numbers/statistics when possible
        4. 12-15 words max
        
        Example:
        "3 costly mistakes 90% founders make →"
        '''
        
        try:
            # Make API call using generic LLM client
            response = self.llm_client.chat_completion(
                system_prompt=system_prompt,
                user_messages="Generate pause screen texts based on the provided context",
                temperature=0.7,
                json_response=True
            )
            
            # Check for errors
            if "error" in response:
                return Result.failure(f"LLM API error: {response['error']}")
            
            # Parse response
            response_text = response["content"].strip()
            texts_data = json.loads(response_text)
            
            # Handle different JSON formats returned by different LLM providers
            # DeepSeek might not return data with the expected 'texts' key format
            if "texts" not in texts_data and isinstance(texts_data, list):
                # If we got a list of text entries, wrap in the expected format
                texts_data = {"texts": texts_data}
            elif "pause_screens" in texts_data:
                # Handle DeepSeek's specific format with 'pause_screens'
                logger.info("Converting DeepSeek 'pause_screens' format to standard format")
                if isinstance(texts_data["pause_screens"], list):
                    texts_data = {"texts": texts_data["pause_screens"]}
                else:
                    # If it's a dict with numbered keys like "1", "2", etc.
                    texts = []
                    for _, value in sorted(texts_data["pause_screens"].items()):
                        if isinstance(value, str):
                            texts.append({"text": value, "duration": 2.5, "position": "bottom_center"})
                        elif isinstance(value, dict) and "text" in value:
                            texts.append(value)
                    texts_data = {"texts": texts}
            elif "texts" not in texts_data and isinstance(texts_data, dict) and any(key.startswith("text") for key in texts_data.keys()):
                # If we have text entries directly in the dict with keys like "text1", "text2", etc.
                texts = []
                for key, value in texts_data.items():
                    if isinstance(value, str):
                        texts.append({"text": value, "duration": 2.5, "position": "bottom_center"})
                    elif isinstance(value, dict) and "text" in value:
                        texts.append(value)
                texts_data = {"texts": texts}
            
            # Log the received format for debugging
            logger.info(f"Received text format: {texts_data.keys() if isinstance(texts_data, dict) else 'list'}")
            
            # Validate the texts
            validation_result = self._validate_texts_contract(texts_data)
            if not validation_result.is_success:
                logger.error(f"LLM-generated texts failed validation: {str(validation_result.error)}")
                return validation_result
            
            return Result.success(texts_data)
            
        except Exception as e:
            logger.error(f"Error in LLM text generation: {str(e)}")
            return Result.failure(f"LLM text generation error: {str(e)}")
    
    def _write_fallback_texts(self, pause_segments: List[Dict[str, Any]],
                            clip_contexts: List[Dict[str, Any]], 
                            tone: str) -> Result:
        """
        Generate fallback texts when LLM is not available.
        
        This creates simple, generic texts that match the tone but
        won't be as engaging as LLM-generated content.
        """
        # Standard phrases by tone
        tone_phrases = {
            "professional": [
                "Key insight from industry experts →",
                "Critical factor for success →",
                "Essential strategy revealed →",
                "Professional tip: optimize this →",
                "Industry best practice →"
            ],
            "casual": [
                "Check this out! →",
                "You won't believe what's next →",
                "Here's the cool part →",
                "This changed everything →",
                "The secret sauce? →"
            ],
            "fun": [
                "Mind = blown! 🤯 →",
                "Wait for it... →",
                "The plot twist →",
                "This is wild! →",
                "Bet you didn't know this →"
            ],
            "dramatic": [
                "The shocking truth →",
                "Everything changed when →",
                "The critical moment →",
                "What happened next? →",
                "The ultimate revelation →"
            ]
        }
        
        # Use professional as default if tone not found
        phrases = tone_phrases.get(tone.lower(), tone_phrases["professional"])
        
        # Generate texts
        texts = []
        for i, segment in enumerate(pause_segments):
            # Pick a phrase based on position
            phrase_index = i % len(phrases)
            text = phrases[phrase_index]
            
            # Use existing text if available
            if "text" in segment and segment["text"]:
                text = segment["text"]
            
            # Set default duration if not specified
            duration = segment.get("duration", 2.5)
            
            texts.append({
                "text": text,
                "duration": duration,
                "position": "bottom_center"
            })
        
        return Result.success({"texts": texts})
    
    def _validate_texts_contract(self, texts_data: Dict[str, Any]) -> Result:
        """Validate that the texts data follows the contract."""
        # Check basic structure
        if not isinstance(texts_data, dict):
            return Result.failure("Texts data must be a dictionary")
        
        if "texts" not in texts_data:
            return Result.failure("Texts data must contain 'texts' key")
        
        texts = texts_data.get("texts", [])
        if not texts or not isinstance(texts, list):
            return Result.failure("Texts must be a non-empty list")
        
        # Validate each text entry
        for i, text_entry in enumerate(texts):
            if not isinstance(text_entry, dict):
                return Result.failure(f"Text entry {i} must be a dictionary")
            
            # Check required fields
            if "text" not in text_entry:
                return Result.failure(f"Text entry {i} missing 'text' content")
                
            text = text_entry["text"]
            if not isinstance(text, str):
                return Result.failure(f"Text entry {i} content must be a string")
            
            word_count = len(text.split())
            if word_count < 3:
                return Result.failure(f"Text entry {i} too short ({word_count} words)")
            
            if word_count > 15:
                logger.warning(f"Text entry {i} exceeds recommended 15 words ({word_count} words): {text}")
                
            # Check duration
            if "duration" not in text_entry:
                text_entry["duration"] = 2.5  # Default duration
            else:
                duration = text_entry["duration"]
                if not isinstance(duration, (int, float)):
                    return Result.failure(f"Text entry {i} duration must be a number")
                
                if duration < 2.0 or duration > 3.0:
                    logger.warning(f"Text entry {i} duration ({duration}s) outside recommended range (2-3s)")
            
            # Check position
            if "position" not in text_entry:
                text_entry["position"] = "bottom_center"  # Default position
            else:
                position = text_entry["position"]
                if position not in ["bottom_center", "top_center", "center"]:
                    logger.warning(f"Text entry {i} has unknown position: {position}, defaulting to 'bottom_center'")
                    text_entry["position"] = "bottom_center"
        
        return Result.success(texts_data)


# Singleton instance
_script_writer = None

def get_script_writer(llm_client=None) -> ScriptWriter:
    """Get the singleton script writer instance."""
    global _script_writer
    if _script_writer is None:
        _script_writer = ScriptWriter(llm_client)
    return _script_writer

def write_pause_texts(narrative_plan: Dict[str, Any], 
                     clip_contexts: List[Dict[str, Any]],
                     tone: str = "professional",
                     llm_provider: str = None,
                     api_key: str = None,
                     model: str = None) -> Result:
    """
    Convenience function to generate pause screen texts.
    
    Args:
        narrative_plan: The narrative plan with action and pause segments
        clip_contexts: List of clip contexts with transcripts and metadata
        tone: Tone of the texts
        llm_provider: Optional LLM provider name (openai, deepseek, anthropic, local)
        api_key: Optional API key for the LLM provider
        model: Optional model name to use
        
    Returns:
        Result object with pause texts or error
    """
    # Get LLM client with specified provider, if any
    llm_client = get_llm_client(
        provider=llm_provider,
        api_key=api_key,
        model=model
    )
    
    writer = get_script_writer(llm_client)
    return writer.write_pause_texts(narrative_plan, clip_contexts, tone)
