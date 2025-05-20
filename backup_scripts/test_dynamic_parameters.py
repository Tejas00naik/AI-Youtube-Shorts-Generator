#!/usr/bin/env python3
"""
Test script for the AI YouTube Shorts Generator with dynamic parameters.
This script demonstrates how the system now dynamically determines the optimal
number of clips and interruption style based on the user prompt.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

# Import component tester
from component_tester import test_components

if __name__ == "__main__":
    # Test case 1: Prompt suggesting continuous style
    print("\n===== TEST CASE 1: CONTINUOUS STYLE PROMPT =====")
    continuous_prompt = "Create a flowing, uninterrupted narrative about chess strategy without any pauses or text interruptions."
    test_components(
        youtube_url="https://youtu.be/t2laAqRtAq0?si=rM50co1i_4GMVx-X",
        user_prompt=continuous_prompt
    )
    
    # Test case 2: Prompt suggesting pause style with many quotes
    print("\n===== TEST CASE 2: PAUSE STYLE WITH MANY QUOTES =====")
    pause_prompt = "Create a reel with shocking text overlays showing the 3 biggest mistakes Gukesh made in this game."
    test_components(
        youtube_url="https://youtu.be/t2laAqRtAq0?si=rM50co1i_4GMVx-X",
        user_prompt=pause_prompt
    )
    
    # Test case 3: Prompt with explicit parameters
    print("\n===== TEST CASE 3: EXPLICIT PARAMETERS IN PROMPT =====")
    explicit_prompt = "Create a 25-second reel with 5 clips showing Gukesh's most dramatic moments."
    test_components(
        youtube_url="https://youtu.be/t2laAqRtAq0?si=rM50co1i_4GMVx-X",
        user_prompt=explicit_prompt
    )
