"""
Test script to verify the implementation of default clip count and narration pause settings.

This script tests:
1. Default clip count of 2
2. Maximum of 2 narration pauses (with outro being optional)
3. Validation of narrative plans against these constraints
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any

from llm.narrative_planner_v2 import generate_narrative_plan
from core.error_handler import Result

# Since we're just testing narrative planning defaults, we don't need the actual ClipValidator

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample transcript (simplified for testing)
SAMPLE_TRANSCRIPT = """
The key to understanding this concept is to think about it from the user's perspective.
When designing interfaces, we always need to prioritize clarity and simplicity.
Too many options can overwhelm users and lead to decision paralysis.
That's why we recommend focusing on the most essential functions first.
Once users understand the core functionality, then you can introduce more advanced features.
This approach has been proven to increase user retention and satisfaction.
"""

def test_default_clip_count():
    """Test that the narrative planner defaults to 2 clips when none specified."""
    logger.info("Testing default clip count...")
    
    user_directions = "Create a short about UI/UX design principles"
    
    # Generate the narrative plan with no clip_count specified
    result = generate_narrative_plan(
        transcript=SAMPLE_TRANSCRIPT,
        user_directions=user_directions,
        tone="professional"
    )
    
    if not result.success:
        logger.error(f"Failed to generate narrative plan: {result.error}")
        return False
    
    plan = result.value
    
    # Count action segments
    action_segments = [s for s in plan.get('segments', []) if s.get('type') == 'action']
    logger.info(f"Generated plan with {len(action_segments)} action segments (default expected: 2)")
    
    # Check if the default is 2 as expected
    return len(action_segments) == 2

def test_interruption_frequency():
    """Test that the narrative planner respects the maximum of 2 narration pauses."""
    logger.info("Testing interruption frequency...")
    
    user_directions = "Create a short about UI/UX design principles"
    
    # Generate plan with a higher clip count to see if it still limits interruptions
    result = generate_narrative_plan(
        transcript=SAMPLE_TRANSCRIPT,
        user_directions=user_directions,
        clip_count=4,  # Request more clips
        tone="professional"
    )
    
    if not result.success:
        logger.error(f"Failed to generate narrative plan: {result.error}")
        return False
    
    plan = result.value
    
    # Count pause segments
    pause_segments = [s for s in plan.get('segments', []) if s.get('type') == 'pause']
    logger.info(f"Generated plan with {len(pause_segments)} pause segments (maximum expected: 2)")
    
    # Check if the count is 2 or less as expected
    return len(pause_segments) <= 2

def test_outro_optional():
    """Test that the narrative planner makes the outro optional."""
    logger.info("Testing optional outro...")
    
    user_directions = "Create a short about UI/UX design principles"
    
    # Generate with 2 clips (default) to see if outro is included
    result = generate_narrative_plan(
        transcript=SAMPLE_TRANSCRIPT,
        user_directions=user_directions,
        tone="professional"
    )
    
    if not result.success:
        logger.error(f"Failed to generate narrative plan: {result.error}")
        return False
    
    plan = result.value
    segments = plan.get('segments', [])
    
    # Check segment types
    segment_types = [s.get('type') for s in segments]
    logger.info(f"Generated plan with segment types: {segment_types}")
    
    # We expect the pattern to be: action, pause, action
    # This means only 1 pause (intro hook) for 2 action clips
    expected_pattern = ['action', 'pause', 'action']
    pattern_matches = (len(segment_types) == 3 and 
                      segment_types[0] == 'action' and 
                      segment_types[1] == 'pause' and 
                      segment_types[2] == 'action')
    
    return pattern_matches

def print_plan_details(plan):
    """Print details of a narrative plan for inspection."""
    logger.info("Plan details:")
    logger.info(f"Total duration: {plan.get('total_duration', 'unknown')}s")
    
    segments = plan.get('segments', [])
    logger.info(f"Total segments: {len(segments)}")
    
    for i, segment in enumerate(segments):
        segment_type = segment.get('type', 'unknown')
        
        if segment_type == 'action':
            start = segment.get('start_time', 'unknown')
            end = segment.get('end_time', 'unknown')
            logger.info(f"Segment {i+1}: {segment_type} ({start}s - {end}s)")
        elif segment_type == 'pause':
            duration = segment.get('duration', 'unknown')
            text = segment.get('text', 'unknown')
            logger.info(f"Segment {i+1}: {segment_type} ({duration}s) - '{text}'")

def main():
    """Run all tests."""
    logger.info("Starting tests for default clip count and narration pauses...")
    
    # Run tests
    default_clip_test = test_default_clip_count()
    interruption_test = test_interruption_frequency()
    outro_optional_test = test_outro_optional()
    
    # Report results
    logger.info("\nTest Results:")
    logger.info(f"Default Clip Count (should be 2): {'PASS' if default_clip_test else 'FAIL'}")
    logger.info(f"Maximum 2 Narration Pauses: {'PASS' if interruption_test else 'FAIL'}")
    logger.info(f"Optional Outro: {'PASS' if outro_optional_test else 'FAIL'}")
    
    # Overall status
    overall_status = all([default_clip_test, interruption_test, outro_optional_test])
    logger.info(f"\nOverall Test Status: {'PASS' if overall_status else 'FAIL'}")
    
    logger.info("\nTests completed.")

if __name__ == "__main__":
    main()
