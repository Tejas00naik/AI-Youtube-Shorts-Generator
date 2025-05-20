"""
Tests for the pause narration mode of the media processor.
"""
import os
import pytest
import tempfile
from pathlib import Path

from media.media_processor import process_pause_narration
from core.script_validator import validate_script, get_total_duration

# Mock data for tests
SAMPLE_SCRIPT = [
    {"type": "clip", "start": 0, "end": 7},
    {"type": "narration", "text": "Startups fail when they ignore this!", "duration": 2.5},
    {"type": "clip", "start": 7, "end": 15},
    {"type": "narration", "text": "Listen to what experts say.", "duration": 2.0},
    {"type": "clip", "start": 15, "end": 22}
]

@pytest.fixture
def sample_video_path():
    """Return path to a sample video file for testing."""
    # This is a placeholder - for real tests, you'd need an actual video file
    # In a test environment, you could download a sample video or create one
    
    # For now, we'll return None and skip tests that need a real file
    return None

def test_validate_script_valid():
    """Test validation of a valid script."""
    result = validate_script(SAMPLE_SCRIPT)
    assert result.is_success
    assert "total_duration" in result.value
    # Total: 7 + 2.5 + 8 + 2.0 + 7 = 26.5 seconds
    assert result.value["total_duration"] == 26.5

def test_validate_script_too_long():
    """Test validation of a script that's too long."""
    long_script = [
        {"type": "clip", "start": 0, "end": 30},
        {"type": "narration", "text": "Mid point insight", "duration": 2.0},
        {"type": "clip", "start": 30, "end": 60}
    ]
    result = validate_script(long_script)
    assert not result.is_success
    assert "exceeds 60s limit" in result.error.message

def test_validate_script_invalid_format():
    """Test validation with invalid script format."""
    invalid_scripts = [
        [],  # Empty script
        [{"no_type_field": True}],  # Missing type
        [{"type": "clip"}],  # Missing start/end
        [{"type": "narration"}],  # Missing text/duration
        [{"type": "unknown", "field": "value"}]  # Invalid type
    ]
    
    for script in invalid_scripts:
        result = validate_script(script)
        assert not result.is_success

def test_get_total_duration():
    """Test calculation of total script duration."""
    duration = get_total_duration(SAMPLE_SCRIPT)
    assert duration == 26.5

@pytest.mark.skipif(not os.getenv("RUN_VIDEO_TESTS"), 
                   reason="Video processing tests require RUN_VIDEO_TESTS=1")
def test_basic_pause_narration(sample_video_path):
    """Test basic pause narration functionality."""
    if not sample_video_path:
        pytest.skip("No sample video available")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_path = os.path.join(tmp_dir, "output.mp4")
        result_path = process_pause_narration(sample_video_path, SAMPLE_SCRIPT, output_path)
        
        assert os.path.exists(result_path)
        # In a real test, you'd want to check the duration and other properties
        # of the output video to ensure it was processed correctly

if __name__ == "__main__":
    pytest.main(["-v", __file__])
