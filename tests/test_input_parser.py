"""
Tests for the input parser module.
"""
import pytest
import sys
import os
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.input_parser import parse_user_input, InputParser, NarrativeMode, Tone

class TestInputParser:
    """Test cases for the input parser."""
    
    def test_parse_highlight_mode(self):
        """Test parsing highlight mode from input."""
        text = "Create a highlight video about machine learning"
        result = parse_user_input(text)
        assert result['mode'] == NarrativeMode.HIGHLIGHT.value
        
    def test_parse_tutorial_mode(self):
        """Test parsing tutorial mode from input."""
        text = "--mode tutorial about python programming"
        result = parse_user_input(text)
        assert result['mode'] == NarrativeMode.TUTORIAL.value
        
    def test_parse_story_mode(self):
        """Test parsing story mode from input."""
        text = "I need a story video about my vacation"
        result = parse_user_input(text)
        assert result['mode'] == NarrativeMode.STORY.value
        
    def test_parse_promotional_mode(self):
        """Test parsing promotional mode from input."""
        text = "Make a promotional video for our new product"
        result = parse_user_input(text)
        assert result['mode'] == NarrativeMode.PROMOTIONAL.value
        
    def test_parse_educational_mode(self):
        """Test parsing educational mode from input."""
        text = "Create an educational video about the solar system"
        result = parse_user_input(text)
        assert result['mode'] == NarrativeMode.EDUCATIONAL.value
        
    def test_parse_tone_professional(self):
        """Test parsing professional tone."""
        text = "Make a professional video about finance"
        result = parse_user_input(text)
        assert result['tone'] == Tone.PROFESSIONAL.value
        
    def test_parse_tone_casual(self):
        """Test parsing casual tone."""
        text = "Create a casual video about daily life"
        result = parse_user_input(text)
        assert result['tone'] == Tone.CASUAL.value
        
    def test_parse_tone_fun(self):
        """Test parsing fun tone."""
        text = "Make a fun video about cats"
        result = parse_user_input(text)
        assert result['tone'] == Tone.FUN.value
        
    def test_parse_duration(self):
        """Test parsing duration parameter."""
        text = "Create a 30-second video about travel"
        result = parse_user_input(text)
        assert 'duration' in result['params']
        assert result['params']['duration'] == 30
        
    def test_parse_topic(self):
        """Test parsing topic parameter."""
        text = "Make a video about 'machine learning algorithms'"
        result = parse_user_input(text)
        assert 'topic' in result['params']
        assert result['params']['topic'] == 'machine learning algorithms'
        
    def test_parse_style(self):
        """Test parsing style parameter."""
        text = "Create a video in a modern style"
        result = parse_user_input(text)
        assert 'style' in result['params']
        assert result['params']['style'] == 'modern'
        
    def test_parse_target_audience(self):
        """Test parsing target audience parameter."""
        text = "Make a video for beginners who want to learn Python"
        result = parse_user_input(text)
        assert 'target_audience' in result['params']
        assert 'beginners' in result['params']['target_audience']
        
    def test_parse_multiple_parameters(self):
        """Test parsing multiple parameters at once."""
        text = """
        Create a 45-second tutorial video about 'Python Decorators' 
        in a professional style for intermediate developers
        """
        result = parse_user_input(text)
        
        assert result['mode'] == NarrativeMode.TUTORIAL.value
        assert result['tone'] == Tone.PROFESSIONAL.value
        assert result['params']['duration'] == 45
        assert result['params']['topic'] == 'python decorators'
        assert 'intermediate developers' in result['params']['target_audience']
        assert 'style' not in result['params']  # 'professional' is mapped to tone, not style
        
    def test_parse_empty_input(self):
        """Test parsing empty input."""
        with pytest.raises(ValueError):
            parse_user_input("")
            
    def test_parse_invalid_mode(self):
        """Test parsing with an invalid mode."""
        # The parser should default to HIGHLIGHT mode for unknown modes
        text = "--mode invalid_mode some content"
        result = parse_user_input(text)
        assert result['mode'] == NarrativeMode.HIGHLIGHT.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
