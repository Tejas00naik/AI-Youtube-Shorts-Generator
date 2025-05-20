#!/usr/bin/env python3
"""
Test script for the new script validation system using DeepSeek-R1 and DeepSeek Chat.
This demonstrates how validation improves script and text overlay quality.
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
from component_tester import test_components
from llm.script_validator import ScriptValidator
from llm.narrative_planner_v2 import NarrativePlannerV2
from llm.script_writer import ScriptWriter
from llm.llm_client import get_llm_client

if __name__ == "__main__":
    # Test the ScriptValidator component directly
    print("\n==== TESTING SCRIPT VALIDATION SYSTEM ====\n")
    
    # Create clients for generation and validation
    generation_client = get_llm_client(provider="deepseek", model=os.getenv("DEEPSEEK_R1_MODEL", "deepseek-coder"))
    validation_client = get_llm_client(provider="deepseek", model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    
    # Create script validator
    validator = ScriptValidator(
        generation_client=generation_client,
        validation_client=validation_client
    )
    
    # Test narrative validation with enhanced criteria
    print("\n1. TESTING NARRATIVE PLAN VALIDATION:\n")
    
    narrative_criteria = [
        "Exactly 5 action segments with appropriate start and end times",
        "Exactly 3 interruption segments of type 'pause'",
        "Total duration does not exceed 60 seconds",
        "Segments alternate appropriately",
        "First segment is an action segment with start_time of 0.0",
        "No overlapping timestamps between segments - each clip must end before the next begins",
        "Each clip should include complete thoughts/sentences",
        "Each clip should end 0.5-1.0 seconds after the speaker completes their sentence"
    ]
    
    system_prompt = """
    Create a narrative plan for a short video with these requirements:
    
    1. 5 action segments showing footage
    2. 3 pause segments with text overlays
    3. First segment must start at 0.0 seconds
    4. Each action segment should be 5-10 seconds
    5. Each pause segment should be 2-3 seconds
    6. Total duration must not exceed 60 seconds
    7. CRITICAL: Each clip must END 0.5-1.0s AFTER speaker completes their sentence
    8. CRITICAL: NO OVERLAPPING CLIPS - each clip must end before the next begins
    9. CRITICAL: Do not cut off speakers mid-sentence
    
    Return ONLY valid JSON with this structure:
    {
      "segments": [
        {"type": "action", "start_time": float, "end_time": float, "content": "Sample content"},
        {"type": "pause", "duration": float, "content": "Text overlay content"}
      ],
      "total_duration": float
    }
    """
    
    print(f"Generating and validating narrative plan...")
    validation_result = validator.generate_and_validate_narrative_plan(
        system_prompt=system_prompt,
        user_messages="Generate a narrative plan as specified",
        validation_criteria=narrative_criteria,
        max_attempts=3
    )
    
    if validation_result.is_success:
        print(f"✅ Successfully generated and validated narrative plan")
        # Print a subset of the plan for demonstration
        plan = validation_result.value
        action_segments = [s for s in plan["segments"] if s["type"] == "action"]
        pause_segments = [s for s in plan["segments"] if s["type"] == "pause"]
        
        print(f"\nNarrative Plan Summary:")
        print(f"- Total duration: {plan.get('total_duration', 0)} seconds")
        print(f"- Action segments: {len(action_segments)}")
        print(f"- Pause segments: {len(pause_segments)}")
        
        if action_segments:
            print(f"\nSample action segment:")
            print(json.dumps(action_segments[0], indent=2))
        
        if pause_segments:
            print(f"\nSample pause segment:")
            print(json.dumps(pause_segments[0], indent=2))
    else:
        print(f"❌ Failed to validate narrative plan: {validation_result.error}")
    
    # Test text overlay validation
    print("\n2. TESTING TEXT OVERLAY VALIDATION:\n")
    
    text_criteria = [
        "Exactly 3 text overlays",
        "Each text is 40 characters or less",
        "Texts follow the strategic placement guidelines: opening at top, middle for clarification, closing at bottom",
        "First text provides opening context at top_center position",
        "Middle text (if needed) provides clarification at middle_center position",
        "Final text includes a call-to-action at bottom_center position",
        "JSON format is correct with required fields"
    ]
    
    system_prompt = """
    Create 3 text overlays for a YouTube Short with these requirements:
    
    1. First text must be an opening hook/context positioned at top_center
    2. Middle text provides clarification positioned at middle_center
    3. Final text must be a call-to-action positioned at bottom_center
    4. Each text must be 40 characters or less
    5. Each text should have a duration of 2.5s
    6. Use varied positions for better engagement (top, middle, bottom)
    
    Return ONLY valid JSON with this structure:
    {
      "texts": [
        {"text": "Your text here", "position": "bottom_center", "duration": 2.5},
        {"text": "Second text", "position": "bottom_center", "duration": 2.5},
        {"text": "Final CTA", "position": "bottom_center", "duration": 2.5}
      ]
    }
    """
    
    print(f"Generating and validating text overlays...")
    text_result = validator.generate_and_validate_text_overlays(
        system_prompt=system_prompt,
        user_messages="Generate 3 text overlays for a dramatic video",
        validation_criteria=text_criteria,
        max_attempts=3
    )
    
    if text_result.is_success:
        print(f"✅ Successfully generated and validated text overlays")
        # Print the text overlays
        texts = text_result.value
        if "texts" in texts and texts["texts"]:
            print(f"\nText Overlays:")
            for i, text in enumerate(texts["texts"]):
                print(f"{i+1}. \"{text['text']}\" ({text.get('position', 'unknown')}, {text.get('duration', 0)}s)")
    else:
        print(f"❌ Failed to validate text overlays: {text_result.error}")
    
    # Now run the full component test with our improved validation system
    print("\n3. TESTING IMPROVED VALIDATION SYSTEM WITH CLIP VALIDATION:\n")
    print("Creating YouTube Short with the enhanced validation system...")
    
    # Define the video URL - using a more reliable video with a proper transcript
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    # Create a prompt that will benefit from our enhanced validation
    prompt = """
    Create a short video with minimal interruptions. Use an engaging opening
    and a clear conclusion, with at most one clarification in the middle.
    Focus on longer clips that show complete thoughts and minimize text
    overlays that break the flow. Make sure each clip ends naturally when
    the speaker finishes their sentence, not in the middle of a thought.
    """
    
    print("\nUsing the following test prompt:")
    print(prompt)
    print("\nThis will test our new validation features:")
    print("1. Prevention of overlapping clips")
    print("2. Adding speech completion buffer to clips")
    print("3. Feedback-driven validation with best attempt fallback")
    print("4. Strategic text overlay positioning")
    
    # Run the component tester with our improved validation approach
    test_components(
        youtube_url=video_url,
        user_prompt=prompt,
        tone="informative"
    )
