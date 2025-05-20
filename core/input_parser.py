"""
Input Parser for AI YouTube Shorts Generator

This module handles parsing of user input to extract mode, tone, and parameters
for video generation.
"""
import re
from typing import Dict, Optional, Tuple
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NarrativeMode(str, Enum):
    """Supported narrative modes for video generation."""
    HIGHLIGHT = "highlight"
    TUTORIAL = "tutorial"
    STORY = "story"
    PROMOTIONAL = "promotional"
    EDUCATIONAL = "educational"

class Tone(str, Enum):
    """Supported tone options for video generation."""
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FUN = "fun"
    DRAMATIC = "dramatic"
    INSPIRATIONAL = "inspirational"

class InputParser:
    """
    Parses user input to extract video generation parameters.
    
    The parser supports both command-line style flags and natural language input.
    """
    
    # Regular expressions for parameter extraction
    MODE_PATTERNS = {
        NarrativeMode.HIGHLIGHT: r'(?:--mode\s+highlight|highlight|best moments|top clips)',
        NarrativeMode.TUTORIAL: r'(?:--mode\s+tutorial|tutorial|how to|guide|step by step)',
        NarrativeMode.STORY: r'(?:--mode\s+story|story|narrative|tale)',
        NarrativeMode.PROMOTIONAL: r'(?:--mode\s+promotional|promo|promotional|advertisement)',
        NarrativeMode.EDUCATIONAL: r'(?:--mode\s+educational|educational|learn|teach|explain)'
    }
    
    TONE_PATTERNS = {
        Tone.PROFESSIONAL: r'(?:--tone\s+professional|professional|formal|business)',
        Tone.CASUAL: r'(?:--tone\s+casual|casual|conversational|informal)',
        Tone.FUN: r'(?:--tone\s+fun|fun|funny|humorous|entertaining)',
        Tone.DRAMATIC: r'(?:--tone\s+dramatic|dramatic|intense|emotional)',
        Tone.INSPIRATIONAL: r'(?:--tone\s+inspirational|inspirational|motivational|uplifting)'
    }
    
    PARAM_PATTERNS = {
        'duration': r'(?:--duration\s+(\d+)|(\d+)[-\s]*(?:seconds?|secs?|s|second))',
        'topic': r'(?:--topic\s+["\']?([^\n"\']+)["\']?|about\s+["\']?([^\n"\']+)["\']?)',
        'style': r'(?:--style\s+["\']?([^\n"\']+)["\']?|in\s+(?:a\s+)?["\']?([^\n"\']+)["\']?\s+style)',
        'target_audience': r'(?:for\s+([^\n]+?)(?:\s+audience|\s+viewers|$))',
    }
    
    def __init__(self):
        self.mode: Optional[NarrativeMode] = None
        self.tone: Optional[Tone] = None
        self.params: Dict[str, str] = {}
        
    def parse(self, text: str) -> Dict:
        """
        Parse the input text to extract mode, tone, and parameters.
        
        Args:
            text: Raw input text from the user
            
        Returns:
            Dict containing the parsed parameters
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty")
            
        text = text.lower().strip()
        self._extract_mode(text)
        self._extract_tone(text)
        self._extract_params(text)
        
        return {
            'mode': self.mode.value if self.mode else None,
            'tone': self.tone.value if self.tone else None,
            'params': self.params
        }
    
    def _extract_mode(self, text: str) -> None:
        """Extract the narrative mode from the input text."""
        for mode, pattern in self.MODE_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                self.mode = mode
                logger.info(f"Detected mode: {mode.value}")
                return
        
        # Default to highlight mode if none specified
        self.mode = NarrativeMode.HIGHLIGHT
        logger.info(f"No mode specified, defaulting to: {self.mode.value}")
    
    def _extract_tone(self, text: str) -> None:
        """Extract the tone from the input text."""
        for tone, pattern in self.TONE_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                self.tone = tone
                logger.info(f"Detected tone: {tone.value}")
                return
        
        # Default to professional tone if none specified
        self.tone = Tone.PROFESSIONAL
        logger.info(f"No tone specified, defaulting to: {self.tone.value}")
    
    def _extract_params(self, text: str) -> None:
        """Extract additional parameters from the input text."""
        self.params = {}
        
        # Extract duration
        duration_match = re.search(self.PARAM_PATTERNS['duration'], text, re.IGNORECASE)
        if duration_match:
            duration = next((d for d in duration_match.groups() if d is not None), None)
            if duration and duration.isdigit():
                self.params['duration'] = min(int(duration), 58)  # Cap at 58 seconds for Shorts
                logger.info(f"Detected duration: {self.params['duration']}s")
        
        # Extract topic
        topic_match = re.search(self.PARAM_PATTERNS['topic'], text, re.IGNORECASE)
        if topic_match:
            topic = next((t for t in topic_match.groups() if t is not None), "")
            if topic.strip():
                self.params['topic'] = topic.strip()
                logger.info(f"Detected topic: {self.params['topic']}")
        
        # Extract style - but skip if it matches a tone pattern to avoid duplication
        style_match = re.search(self.PARAM_PATTERNS['style'], text, re.IGNORECASE)
        if style_match:
            style = next((s for s in style_match.groups() if s is not None), "")
            if style.strip():
                # Check if this style word is already captured as a tone
                is_tone_word = False
                for tone_name, tone_pattern in self.TONE_PATTERNS.items():
                    if re.search(rf"\b{re.escape(style.strip())}\b", tone_pattern, re.IGNORECASE):
                        is_tone_word = True
                        break
                
                if not is_tone_word:
                    self.params['style'] = style.strip()
                    logger.info(f"Detected style: {self.params['style']}")
        
        # Extract target audience
        audience_match = re.search(self.PARAM_PATTERNS['target_audience'], text, re.IGNORECASE)
        if audience_match and audience_match.group(1):
            self.params['target_audience'] = audience_match.group(1).strip()
            logger.info(f"Detected target audience: {self.params['target_audience']}")

def parse_user_input(text: str) -> Dict:
    """
    Parse user input and return structured parameters.
    
    Args:
        text: Raw input text from the user
        
    Returns:
        Dict containing the parsed parameters
        
    Example:
        >>> parse_user_input("Create a highlight video about machine learning in a fun way for beginners")
        {
            'mode': 'highlight',
            'tone': 'fun',
            'params': {
                'topic': 'machine learning',
                'target_audience': 'beginners'
            }
        }
    """
    parser = InputParser()
    return parser.parse(text)


if __name__ == "__main__":
    # Example usage
    test_inputs = [
        "Create a highlight video about machine learning in a fun way for beginners",
        "--mode tutorial --tone professional --duration 45 --topic 'How to use Python' --style modern",
        "Make an inspirational story video about climate change",
        "I need a promotional video for my startup, keep it under 30 seconds"
    ]
    
    for test in test_inputs:
        print(f"\nInput: {test}")
        try:
            result = parse_user_input(test)
            print("Parsed:", result)
        except Exception as e:
            print(f"Error: {str(e)}")
