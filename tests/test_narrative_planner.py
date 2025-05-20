"""
Tests for the narrative planner module.
"""
import pytest
import sys
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.narrative_planner import NarrativePlanner, generate_narrative_plan, NarrativeMode

class TestNarrativePlanner:
    """Test cases for the narrative planner."""
    
    @pytest.fixture
    def sample_transcript(self):
        """Return a sample transcript for testing."""
        return """
        In this tutorial, I'll show you how to create a simple web application using Python and Flask. 
        First, we'll set up our development environment and install the necessary packages. 
        Then, we'll create a basic Flask application with a few routes. 
        Finally, we'll add some styling with CSS and deploy our application to a cloud platform.
        """
    
    @pytest.fixture
    def mock_llm_response(self):
        """Return a mock LLM response."""
        return """
        ```json
        {
          "segments": [
            {
              "type": "introduction",
              "start_time": 0.0,
              "end_time": 8.0,
              "description": "Introduction to the tutorial",
              "text": "Welcome to this tutorial on creating a web app with Flask.",
              "mood": "friendly"
            },
            {
              "type": "step_1",
              "start_time": 8.0,
              "end_time": 20.0,
              "description": "Setting up the environment",
              "text": "First, let's set up our development environment and install Flask.",
              "mood": "instructive"
            },
            {
              "type": "step_2",
              "start_time": 20.0,
              "end_time": 40.0,
              "description": "Creating the Flask app",
              "text": "Now, we'll create a basic Flask application with some routes.",
              "mood": "focused"
            },
            {
              "type": "conclusion",
              "start_time": 40.0,
              "end_time": 50.0,
              "description": "Wrapping up",
              "text": "That's it! You've created a simple Flask web application.",
              "mood": "satisfied"
            }
          ],
          "total_duration": 50.0,
          "summary": "A tutorial on creating a Flask web application from scratch."
        }
        ```
        """
    
    def test_init_default_mode(self):
        """Test initialization with default mode."""
        planner = NarrativePlanner()
        assert planner.mode == NarrativeMode.HIGHLIGHT
        assert planner.tone == 'professional'
        
    def test_init_custom_mode(self):
        """Test initialization with custom mode and tone."""
        planner = NarrativePlanner(mode='tutorial', tone='casual')
        assert planner.mode == NarrativeMode.TUTORIAL
        assert planner.tone == 'casual'
        
    def test_init_invalid_mode(self):
        """Test initialization with invalid mode falls back to default."""
        planner = NarrativePlanner(mode='invalid_mode')
        assert planner.mode == NarrativeMode.HIGHLIGHT  # Default fallback
        
    @patch('llm.narrative_planner.NarrativePlanner._call_llm')
    def test_generate_plan_success(self, mock_call_llm, sample_transcript, mock_llm_response):
        """Test successful generation of a narrative plan."""
        # Setup mock
        mock_call_llm.return_value = mock_llm_response
        
        # Test
        planner = NarrativePlanner(mode='tutorial')
        user_params = {
            'topic': 'Flask Web Development',
            'target_audience': 'beginners',
            'style': 'clear and concise'
        }
        
        plan = planner.generate_plan(sample_transcript, user_params)
        
        # Verify the structure of the returned plan
        assert 'segments' in plan
        assert 'total_duration' in plan
        assert 'summary' in plan
        assert 'metadata' in plan
        
        # Verify segments
        assert len(plan['segments']) > 0
        for segment in plan['segments']:
            assert 'type' in segment
            assert 'start_time' in segment
            assert 'end_time' in segment
            assert 'description' in segment
            assert 'text' in segment
            assert 'mood' in segment
        
        # Verify metadata
        assert plan['metadata']['mode'] == 'tutorial'
        assert plan['metadata']['tone'] == 'professional'
        assert 'parameters' in plan['metadata']
        
    @patch('llm.narrative_planner.NarrativePlanner._call_llm')
    def test_generate_plan_llm_failure(self, mock_call_llm, sample_transcript):
        """Test fallback behavior when LLM call fails."""
        # Setup mock to raise an exception
        mock_call_llm.side_effect = Exception("API error")
        
        # Test
        planner = NarrativePlanner()
        plan = planner.generate_plan(sample_transcript, {})
        
        # Should still return a valid plan (fallback)
        assert 'segments' in plan
        assert plan['metadata']['is_fallback'] is True
        
    def test_validate_plan_valid(self, mock_llm_response):
        """Test validation of a valid plan."""
        # Parse the mock response to create a plan
        import re
        import json
        
        # Extract the JSON part from the mock response - make the pattern more robust
        if '```json' in mock_llm_response:
            json_str = re.search(r'```json\s*(.*?)\s*```', mock_llm_response, re.DOTALL).group(1)
        else:
            # For tests where the response might not be in a code block
            json_str = mock_llm_response.strip()
            
        plan = json.loads(json_str)
        
        # Test validation
        planner = NarrativePlanner()
        validated_plan = planner._validate_plan(plan)
        
        # Should return the same plan if valid
        assert validated_plan == plan
        
    def test_validate_plan_invalid(self):
        """Test validation of an invalid plan."""
        planner = NarrativePlanner()
        
        # Missing required fields
        with pytest.raises(ValueError):
            planner._validate_plan({})
            
        # Empty segments
        with pytest.raises(ValueError):
            planner._validate_plan({
                'segments': [],
                'total_duration': 50,
                'summary': 'Test'
            })
    
    def test_generate_fallback_plan(self, sample_transcript):
        """Test generation of a fallback plan."""
        planner = NarrativePlanner()
        user_params = {'duration': 30}
        
        plan = planner._generate_fallback_plan(sample_transcript, user_params)
        
        assert len(plan['segments']) > 0
        assert plan['metadata']['is_fallback'] is True
        assert plan['total_duration'] <= 30


class TestGenerateNarrativePlan:
    """Test cases for the generate_narrative_plan function."""
    
    @patch('llm.narrative_planner.NarrativePlanner.generate_plan')
    def test_generate_narrative_plan(self, mock_generate_plan):
        """Test the convenience function."""
        # Setup mock
        mock_plan = {'segments': [], 'total_duration': 50, 'summary': 'Test'}
        mock_generate_plan.return_value = mock_plan
        
        # Test
        result = generate_narrative_plan(
            transcript="Test transcript",
            mode="tutorial",
            tone="professional",
            topic="Test Topic",
            target_audience="beginners"
        )
        
        assert result == mock_plan
        mock_generate_plan.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
