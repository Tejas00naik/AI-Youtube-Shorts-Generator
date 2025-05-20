#!/usr/bin/env python3
"""
Script to create a YouTube short with strategic text placement.
This implements the updated approach with minimal interruptions.
"""

import sys
import os
from pathlib import Path
import subprocess

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Import component tester
from component_tester import test_components

if __name__ == "__main__":
    # Define the video URL - using the Pakistan video you specified
    video_id = "pB7QBJXL9P8"
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # The prompt now emphasizes minimal interruptions with commentary primarily at beginning/end
    prompt = """
    Create a reel exposing Pakistan's contradictions using mostly uninterrupted clips. 
    Let the speaker's own words tell the story with minimal interruptions.
    Only add context at beginning and conclusion with at most one clarification 
    in the middle if absolutely necessary. Use longer clip segments that show 
    complete thoughts.
    """
    
    # Run the component tester with our updated approach
    test_components(
        youtube_url=video_url,
        user_prompt=prompt,
        tone="dramatic"
    )
