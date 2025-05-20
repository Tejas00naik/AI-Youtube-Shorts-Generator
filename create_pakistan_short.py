#!/usr/bin/env python3
"""
Script to create a YouTube short with a specific video ID.
This ensures the exact video requested is used.
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
    # Define the exact video ID
    video_id = "pB7QBJXL9P8"
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Download video first to ensure we have the correct one
    working_dir = Path("_working")
    working_dir.mkdir(exist_ok=True)
    video_path = working_dir / f"{video_id}.mp4"
    
    if not video_path.exists():
        print(f"\nDownloading video {video_id} specifically...")
        try:
            from pytubefix import YouTube
            yt = YouTube(video_url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            if not stream:
                print("No suitable video stream found")
                sys.exit(1)
                
            # Download the video
            downloaded_path = stream.download(output_path=str(working_dir), 
                                            filename=f"{video_id}.mp4")
            print(f"Downloaded video to {downloaded_path}")
            
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            sys.exit(1)
    else:
        print(f"\nVideo already exists at {video_path}")

    # Define the prompt
    prompt = """
    Create a provocative reel mocking Pakistan's empty claims using maximum 
    video clips and minimum commentary. Focus on the most damning contradictions 
    and let the speaker's own words expose the truth. Include brief text overlays 
    highlighting key points.
    """
    
    # Run the component tester with the pre-downloaded video
    test_components(
        youtube_url=video_url,
        user_prompt=prompt,
        tone="dramatic"
    )
