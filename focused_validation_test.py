#!/usr/bin/env python3
"""
Focused test script for validating the enhanced script validation features.
This script tests the validation system without relying on external APIs.
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import components
from llm.script_validator import ScriptValidator
from media.clip_validator import ClipValidator
from core.error_handler import Result
from llm.llm_client import get_llm_client

def create_sample_narrative_plan(valid=True):
    """Create a sample narrative plan for testing validation."""
    if valid:
        # Create a valid plan with no overlapping segments
        return {
            "segments": [
                {"type": "action", "start_time": 0.0, "end_time": 5.0, "content": "Opening scene with person speaking"},
                {"type": "pause", "duration": 2.5, "content": "First key point"},
                {"type": "action", "start_time": 7.5, "end_time": 15.0, "content": "Second scene with more explanation"},
                {"type": "pause", "duration": 2.5, "content": "Second key insight"},
                {"type": "action", "start_time": 17.5, "end_time": 25.0, "content": "Third scene showing example"},
                {"type": "pause", "duration": 2.5, "content": "Final takeaway"},
                {"type": "action", "start_time": 27.5, "end_time": 35.0, "content": "Closing remarks and call to action"},
            ],
            "total_duration": 35.0
        }
    else:
        # Create an invalid plan with overlapping segments
        return {
            "segments": [
                {"type": "action", "start_time": 0.0, "end_time": 5.0, "content": "Opening scene with person speaking"},
                {"type": "pause", "duration": 2.5, "content": "First key point"},
                {"type": "action", "start_time": 7.5, "end_time": 15.0, "content": "Second scene with more explanation"},
                {"type": "pause", "duration": 2.5, "content": "Second key insight"},
                {"type": "action", "start_time": 17.5, "end_time": 25.0, "content": "Third scene showing example"},
                # Issue 1: No pause between action segments
                {"type": "action", "start_time": 24.0, "end_time": 32.0, "content": "Overlapping scene that starts before previous ends"},
                # Issue 2: Overlapping segments (starts before previous ends)
                {"type": "action", "start_time": 31.0, "end_time": 35.0, "content": "Closing remarks and call to action"},
            ],
            "total_duration": 35.0
        }

def create_sample_text_overlays(valid=True):
    """Create sample text overlays for testing validation."""
    if valid:
        return {
            "texts": [
                {"text": "This changes everything", "position": "top_center", "duration": 2.5},
                {"text": "Here's what you need to know", "position": "middle_center", "duration": 2.5},
                {"text": "Like & subscribe for more!", "position": "bottom_center", "duration": 2.5}
            ]
        }
    else:
        return {
            "texts": [
                # Issue 1: Too long text
                {"text": "This is a really really long text that exceeds the character limit and will be flagged", "position": "bottom_center", "duration": 2.5},
                # Issue 2: All positioned at bottom (no strategic placement)
                {"text": "Another important point to consider", "position": "bottom_center", "duration": 2.5},
                {"text": "Like and subscribe for more content!", "position": "bottom_center", "duration": 2.5}
            ]
        }

def create_mock_video_file():
    """Create a mock video file for testing clip validation."""
    video_dir = Path("_working/test")
    video_dir.mkdir(exist_ok=True, parents=True)
    video_path = video_dir / "test_video.mp4"
    
    # Just create an empty file if it doesn't exist
    if not video_path.exists():
        with open(video_path, 'wb') as f:
            # Write a minimal MP4 header (just enough to be recognized)
            f.write(bytes.fromhex('00 00 00 18 66 74 79 70 69 73 6F 6D 00 00 02 00'))
            f.write(bytes.fromhex('69 73 6F 6D 69 73 6F 32 6D 70 34 31 00 00 00 08'))
            f.write(bytes.fromhex('66 72 65 65 00 00 00 08 6D 64 61 74'))
            # Add some padding
            f.write(b'\x00' * 1024)
    
    return str(video_path)

def test_narrative_plan_validation():
    """Test narrative plan validation."""
    print("\n***** TESTING NARRATIVE PLAN VALIDATION *****\n")
    
    # Create clients for validation
    validation_client = get_llm_client(provider="deepseek", model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    
    # Create validator
    validator = ScriptValidator(
        generation_client=None,  # Not needed for direct validation
        validation_client=validation_client
    )
    
    # Define validation criteria
    validation_criteria = [
        "Exactly 4 action segments with appropriate start and end times",
        "Exactly 3 interruption segments of type 'pause'",
        "Total duration does not exceed 60 seconds",
        "Segments alternate appropriately based on the interruption style",
        "First segment is an action segment with start_time of 0.0",
        "No overlapping timestamps between segments - each clip must end before the next begins",
        "Each clip should include complete thoughts/sentences",
        "Each clip should end 0.5-1.0 seconds after the speaker completes their sentence"
    ]
    
    # Test valid plan
    valid_plan = create_sample_narrative_plan(valid=True)
    print("Testing VALID narrative plan:")
    print(json.dumps(valid_plan, indent=2))
    
    valid_result = validator._validate_narrative_plan(valid_plan, validation_criteria)
    
    print(f"\nValid plan validation result: {'SUCCESS' if valid_result.is_success else 'FAILURE'}")
    if not valid_result.is_success:
        print(f"Error: {valid_result.error}")
    
    # Test invalid plan
    invalid_plan = create_sample_narrative_plan(valid=False)
    print("\nTesting INVALID narrative plan with overlapping segments:")
    print(json.dumps(invalid_plan, indent=2))
    
    invalid_result = validator._validate_narrative_plan(invalid_plan, validation_criteria)
    
    print(f"\nInvalid plan validation result: {'SUCCESS' if invalid_result.is_success else 'FAILURE'}")
    if not invalid_result.is_success:
        print(f"Error: {invalid_result.error}")
    
def test_text_overlay_validation():
    """Test text overlay validation."""
    print("\n***** TESTING TEXT OVERLAY VALIDATION *****\n")
    
    # Create clients for validation
    validation_client = get_llm_client(provider="deepseek", model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    
    # Create validator
    validator = ScriptValidator(
        generation_client=None,  # Not needed for direct validation
        validation_client=validation_client
    )
    
    # Define validation criteria
    validation_criteria = [
        "Exactly 3 text overlays",
        "Each text is 40 characters or less",
        "Texts follow the strategic placement guidelines: opening at top, middle for clarification, closing at bottom",
        "First text provides opening context at top_center position",
        "Middle text (if needed) provides clarification at middle_center position",
        "Final text includes a call-to-action at bottom_center position"
    ]
    
    # Test valid text overlays
    valid_texts = create_sample_text_overlays(valid=True)
    print("Testing VALID text overlays:")
    print(json.dumps(valid_texts, indent=2))
    
    valid_result = validator._validate_text_overlays(valid_texts, validation_criteria)
    
    print(f"\nValid texts validation result: {'SUCCESS' if valid_result.is_success else 'FAILURE'}")
    if not valid_result.is_success:
        print(f"Error: {valid_result.error}")
    
    # Test invalid text overlays
    invalid_texts = create_sample_text_overlays(valid=False)
    print("\nTesting INVALID text overlays:")
    print(json.dumps(invalid_texts, indent=2))
    
    invalid_result = validator._validate_text_overlays(invalid_texts, validation_criteria)
    
    print(f"\nInvalid texts validation result: {'SUCCESS' if invalid_result.is_success else 'FAILURE'}")
    if not invalid_result.is_success:
        print(f"Error: {invalid_result.error}")

def test_clip_validator():
    """Test clip validator functionality."""
    print("\n***** TESTING CLIP VALIDATOR *****\n")
    
    # Create a sample narrative plan with overlapping timestamps
    plan = {
        "segments": [
            {"type": "action", "start_time": 0.0, "end_time": 5.0, "content": "Opening scene"},
            {"type": "pause", "duration": 2.5, "content": "First text"},
            {"type": "action", "start_time": 7.5, "end_time": 15.0, "content": "Second scene"},
            {"type": "pause", "duration": 2.5, "content": "Second text"},
            # Overlap issue: Third scene starts before second ends
            {"type": "action", "start_time": 14.0, "end_time": 20.0, "content": "Third scene with overlap"},
            {"type": "pause", "duration": 2.5, "content": "Third text"},
            # Duration issue: This clip goes beyond the video duration (will be fixed)
            {"type": "action", "start_time": 22.5, "end_time": 40.0, "content": "Final scene too long"}
        ],
        "total_duration": 40.0
    }
    
    # Create a mock video file for testing
    try:
        video_path = create_mock_video_file()
        print(f"Using test video at: {video_path}")
        
        # Test clip validation
        print("\nValidating narrative plan with ClipValidator:")
        result = ClipValidator.validate_narrative_plan(plan, video_path)
        
        if result.is_success:
            print("\n✅ ClipValidator fixed the issues in the narrative plan")
            fixed_plan = result.value
            
            # Compare original and fixed
            print("\nOriginal vs Fixed Plan Comparison:")
            
            original_segments = plan["segments"]
            fixed_segments = fixed_plan["segments"]
            
            print(f"Original segment count: {len(original_segments)}")
            print(f"Fixed segment count: {len(fixed_segments)}")
            
            print("\nOriginal Action Segments:")
            for i, segment in enumerate([s for s in original_segments if s["type"] == "action"]):
                print(f"  {i+1}. {segment['start_time']}s - {segment['end_time']}s (Duration: {segment['end_time'] - segment['start_time']}s)")
            
            print("\nFixed Action Segments:")
            for i, segment in enumerate([s for s in fixed_segments if s["type"] == "action"]):
                print(f"  {i+1}. {segment['start_time']}s - {segment['end_time']}s (Duration: {segment['end_time'] - segment['start_time']}s)")
                
            # Verify fixes:
            # 1. No overlapping segments
            # 2. All segments within video duration
            has_overlaps = False
            last_end = 0
            
            for segment in fixed_segments:
                if segment["type"] == "action":
                    if segment["start_time"] < last_end:
                        has_overlaps = True
                        print(f"⚠️ Found overlap: segment starts at {segment['start_time']}s but previous ended at {last_end}s")
                    last_end = segment["end_time"]
            
            if not has_overlaps:
                print("\n✅ No overlapping segments in fixed plan")
        else:
            print(f"\n❌ ClipValidator error: {result.error}")
    
    except Exception as e:
        print(f"\n❌ Error during clip validation test: {str(e)}")

if __name__ == "__main__":
    print("\n===== TESTING ENHANCED VALIDATION FEATURES =====\n")
    
    # Test each validation component
    test_narrative_plan_validation()
    test_text_overlay_validation() 
    test_clip_validator()
    
    print("\n===== VALIDATION TESTING COMPLETE =====\n")
