#!/usr/bin/env python3
"""
Script to test just the narrative generation component with a local fallback.
This will use a local model instead of DeepSeek since we don't have a key.
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables to use local model
os.environ["LLM_PROVIDER"] = "local"

# Import component functions
from llm.narrative_planner_v2 import generate_narrative_plan
from youtube_transcript_api import YouTubeTranscriptApi

def extract_video_id(youtube_url):
    """Extract the video ID from a YouTube URL."""
    if "youtu.be" in youtube_url:
        parts = youtube_url.split("/")
        for part in parts:
            if "?" in part:
                return part.split("?")[0]
        return parts[-1]
    
    if "v=" in youtube_url:
        return youtube_url.split("v=")[1].split("&")[0]
    return youtube_url

def test_narrative_planning():
    """Test narrative planning with a local model."""
    print("\n=== Testing Narrative Planning Component ===\n")
    
    # Set up logs directory
    logs_dir = Path("narrative_test")
    logs_dir.mkdir(exist_ok=True)
    
    # Chess video and prompt
    youtube_url = "https://youtu.be/t2laAqRtAq0?si=rM50co1i_4GMVx-X"
    user_prompt = "create a reel with hook saying Did you really thaugh that gukesh is the real owrkld chess champion. And then fill up appropriate clips from this video and some narrations text."
    tone = "dramatic"
    clip_count = 4
    
    print(f"YouTube URL: {youtube_url}")
    print(f"User Prompt: {user_prompt}")
    print(f"Tone: {tone}")
    print(f"Clip Count: {clip_count}")
    
    # Extract video ID
    video_id = extract_video_id(youtube_url)
    print(f"Extracted Video ID: {video_id}")
    
    # Extract transcript directly using YouTube Transcript API
    print("\n[Step 1] Extracting transcript...")
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = ' '.join([item['text'] for item in transcript_list])
        
        print(f"Successfully extracted transcript ({len(transcript)} characters)")
        print(f"Transcript sample: {transcript[:150]}...")
        
        # Save transcript for reference
        with open(logs_dir / f"{video_id}_transcript.txt", "w") as f:
            f.write(transcript)
    except Exception as e:
        print(f"Error extracting transcript: {str(e)}")
        return
    
    # Generate narrative plan
    print("\n[Step 2] Generating narrative plan...")
    
    # Call narrative planner directly
    plan_result = generate_narrative_plan(
        transcript=transcript,
        user_directions=user_prompt,
        clip_count=clip_count,
        tone=tone
    )
    
    if not plan_result.is_success:
        print(f"Error generating narrative plan: {str(plan_result.error)}")
        return
    
    narrative_plan = plan_result.value
    
    # Save narrative plan to file
    with open(logs_dir / f"{video_id}_narrative_plan.json", "w") as f:
        json.dump(narrative_plan, f, indent=2)
    
    # Print summary
    action_segments = [s for s in narrative_plan['segments'] if s['type'] == 'action']
    pause_segments = [s for s in narrative_plan['segments'] if s['type'] == 'pause']
    
    print(f"\nNarrative Plan Summary:")
    print(f"- Total duration: {narrative_plan['total_duration']} seconds")
    print(f"- Action segments: {len(action_segments)}")
    print(f"- Pause segments: {len(pause_segments)}")
    print(f"- Is fallback: {narrative_plan.get('is_fallback', False)}")
    
    if pause_segments:
        print("\nPause Segments:")
        for i, segment in enumerate(pause_segments):
            print(f"Pause {i+1}: \"{segment.get('text', 'No text')}\"")
    
    print(f"\nNarrative plan saved to: {logs_dir / f'{video_id}_narrative_plan.json'}")

if __name__ == "__main__":
    test_narrative_planning()
