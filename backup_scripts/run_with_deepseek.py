#!/usr/bin/env python3
"""
Script to run the AI YouTube Shorts Generator with DeepSeek API.
This script properly sets the environment variables before running the component test.
"""

import os
import sys
import subprocess
from pathlib import Path

# Set environment variables properly
os.environ["LLM_PROVIDER"] = "deepseek"  # This matches what the code is looking for
os.environ["DEEPSEEK_API_KEY"] = ""  # Empty as per user's instruction
os.environ["DEEPSEEK_MODEL"] = ""  # Empty as per user's instruction

# Import after setting environment variables
from component_tester import test_components

if __name__ == "__main__":
    print("\n=== Running AI YouTube Shorts Generator with DeepSeek API ===\n")
    print("Environment variables set:")
    print(f"LLM_PROVIDER: {os.environ.get('LLM_PROVIDER')}")
    print(f"DEEPSEEK_API_KEY: {'Set' if os.environ.get('DEEPSEEK_API_KEY') else 'Not set'}")
    print(f"DEEPSEEK_MODEL: {os.environ.get('DEEPSEEK_MODEL') or 'Not set'}")
    
    # Chess video and prompt
    youtube_url = "https://youtu.be/t2laAqRtAq0?si=rM50co1i_4GMVx-X"
    user_prompt = "create a reel with hook saying Did you really thaugh that gukesh is the real owrkld chess champion. And then fill up appropriate clips from this video and some narrations text."
    
    # Run the component test with DeepSeek
    print("\nStarting component test with DeepSeek API...\n")
    test_components(youtube_url, user_prompt, "dramatic", 4)
