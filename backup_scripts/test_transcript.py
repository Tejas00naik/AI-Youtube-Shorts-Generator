#!/usr/bin/env python3
"""Simple script to test YouTube transcript extraction."""

from youtube_transcript_api import YouTubeTranscriptApi

def extract_transcript(video_id):
    """Extract transcript directly using YouTube Transcript API."""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        print(f"Successfully extracted transcript ({len(transcript_text)} characters)")
        print(f"Transcript sample: {transcript_text[:150]}...")
        return transcript_text
    except Exception as e:
        print(f"Error extracting transcript: {str(e)}")
        return None

if __name__ == "__main__":
    # Chess video
    video_id = "t2laAqRtAq0"
    print(f"Testing transcript extraction for video ID: {video_id}")
    extract_transcript(video_id)
    
    # Also try a known working video as backup
    backup_video_id = "jNQXAC9IVRw"  # "Me at the zoo" (first YouTube video)
    print(f"\nTesting backup video ID: {backup_video_id}")
    extract_transcript(backup_video_id)
