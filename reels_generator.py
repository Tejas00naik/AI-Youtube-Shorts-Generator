#!/usr/bin/env python3
"""
Reels Generator Agent

This script converts YouTube videos into engaging reels/shorts based on user directions.
The main inputs are a YouTube URL and user directions for narrative creation.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, Optional, List
from pathlib import Path

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import core modules
from core.orchestrator import process_input, get_orchestrator
from core.config import init_config
from core.error_handler import Result, ErrorCode

def print_segment(segment: Dict[str, Any], index: int) -> None:
    """Print a formatted segment for better readability."""
    print(f"\n{'-' * 50}")
    print(f"SEGMENT {index}: {segment['type'].upper()}")
    print(f"{'-' * 50}")
    print(f"⏱️  {segment['start_time']}s - {segment['end_time']}s")
    print(f"📝 Description: {segment['description']}")
    print(f"🗣️  Text: \"{segment['text']}\"")
    print(f"😀 Mood: {segment['mood']}")
    print(f"{'-' * 50}")

def print_narrative_plan(plan: Dict[str, Any]) -> None:
    """Print the narrative plan in a human-readable format."""
    if not plan:
        print("No plan was generated.")
        return
    
    print("\n" + "=" * 80)
    print(f"📽️  REELS NARRATIVE PLAN - Total Duration: {plan.get('total_duration', 0)}s")
    print("=" * 80)
    
    print(f"\n📋 Summary: {plan.get('summary', 'No summary available')}")
    
    # Print metadata if available
    metadata = plan.get('metadata', {})
    if metadata:
        print("\n📊 METADATA:")
        print(f"   Mode: {metadata.get('mode', 'Not specified')}")
        print(f"   Tone: {metadata.get('tone', 'Not specified')}")
        print(f"   Is Fallback: {'Yes' if metadata.get('is_fallback', False) else 'No'}")
        
        # Print user directions if available
        if 'parameters' in metadata and 'user_directions' in metadata['parameters']:
            print(f"\n🎯 USER DIRECTIONS: \n   {metadata['parameters']['user_directions']}")
    
    # Print each segment
    segments = plan.get('segments', [])
    print(f"\n🎬 SEGMENTS ({len(segments)}):")
    
    for i, segment in enumerate(segments, 1):
        print_segment(segment, i)
    
    print("\n" + "=" * 80)

def extract_youtube_transcript(youtube_url: str) -> Result:
    """
    Extract the transcript from a YouTube video URL.
    
    Args:
        youtube_url: The URL of the YouTube video
        
    Returns:
        Result object containing either the transcript or an error
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Extract video ID from URL
        if 'youtu.be' in youtube_url:
            video_id = youtube_url.split('/')[-1].split('?')[0]
        elif 'youtube.com/watch' in youtube_url:
            import urllib.parse
            parsed_url = urllib.parse.urlparse(youtube_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            video_id = query_params.get('v', [''])[0]
        else:
            return Result.failure(f"Invalid YouTube URL format: {youtube_url}", 
                                 code=ErrorCode.INVALID_INPUT)
        
        if not video_id:
            return Result.failure(f"Could not extract video ID from URL: {youtube_url}", 
                                 code=ErrorCode.INVALID_INPUT)
                                 
        # Get transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine all transcript parts into a single string
        transcript_text = ' '.join([item['text'] for item in transcript_list])
        
        # Also return the video ID for reference
        return Result.success({
            'transcript': transcript_text,
            'video_id': video_id
        })
        
    except ImportError:
        logger.error("youtube_transcript_api not installed. Run: pip install youtube_transcript_api")
        return Result.failure("Required package 'youtube_transcript_api' not installed", 
                            code=ErrorCode.DEPENDENCY_MISSING)
    except Exception as e:
        logger.error(f"Error extracting transcript: {str(e)}")
        return Result.failure(f"Failed to extract transcript: {str(e)}", 
                            code=ErrorCode.EXTERNAL_API_ERROR)

def create_user_query_from_directions(user_directions: str, duration: int = 30) -> str:
    """
    Create a structured user query from the user's directions.
    
    Args:
        user_directions: The directions from the user
        duration: Target duration in seconds
        
    Returns:
        A formatted query for the narrative planner
    """
    return f"Create a {duration}-second video that {user_directions}"

def generate_reels(youtube_url: str, user_directions: str, duration: int = 30, 
                  tone: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a reels narrative plan from a YouTube URL and user directions.
    
    Args:
        youtube_url: URL of YouTube video to extract transcript from
        user_directions: User's directions for how to build the narrative
        duration: Target duration in seconds
        tone: Optional tone override (fun, professional, casual, etc.)
        
    Returns:
        The generated narrative plan or None if an error occurred
    """
    # Initialize the config
    init_config()
    
    logger.info(f"Processing YouTube URL: {youtube_url}")
    logger.info(f"User directions: {user_directions}")
    
    # Get transcript from URL
    if not youtube_url:
        print("\n❌ Error: YouTube URL is required")
        return None
        
    logger.info(f"Extracting transcript from URL: {youtube_url}")
    transcript_result = extract_youtube_transcript(youtube_url)
    
    if not transcript_result.is_success:
        print(f"\n❌ Error extracting transcript: {transcript_result.error.message}")
        return None
        
    transcript_data = transcript_result.value
    transcript = transcript_data['transcript']
    video_id = transcript_data['video_id']
    
    logger.info(f"Successfully extracted transcript from video {video_id} ({len(transcript)} characters)")
    
    # Create a structured user query from the directions
    user_query = create_user_query_from_directions(user_directions, duration)
    
    # Add explicit tone if provided
    if tone:
        user_query += f" in a {tone} tone"
    
    logger.info(f"Generated user query: {user_query}")
    
    # Get the orchestrator instance and add the user directions as a parameter
    orchestrator = get_orchestrator()
    
    # Process the input through the pipeline
    result = process_input(user_query, transcript)
    
    # Add the original user directions to the metadata if successful
    if result.is_success and 'metadata' in result.value:
        if 'parameters' not in result.value['metadata']:
            result.value['metadata']['parameters'] = {}
        result.value['metadata']['parameters']['user_directions'] = user_directions
        result.value['metadata']['parameters']['youtube_video_id'] = video_id
    
    if result.is_success:
        print("\n✅ Successfully generated reels narrative plan!")
        print_narrative_plan(result.value)
        
        # Save the plan to a JSON file
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"reels_plan_{video_id}.json"
        with open(output_file, 'w') as f:
            json.dump(result.value, f, indent=2)
        print(f"\n💾 Saved reels narrative plan to {output_file}")
        
        return result.value
    else:
        print("\n❌ Error generating reels narrative plan:")
        print(f"Error: {result.error.message}")
        print(f"Code: {result.error.code}")
        print(f"Severity: {result.error.severity}")
        return None

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate Reels from YouTube videos')
    parser.add_argument('--url', '-u', type=str, required=True,
                        help='YouTube video URL to extract transcript from')
    parser.add_argument('--directions', '-d', type=str, required=True, 
                        help='Directions for how to build the narrative')
    parser.add_argument('--duration', type=int, default=30,
                        help='Target duration in seconds (default: 30)')
    parser.add_argument('--tone', '-t', type=str, choices=['fun', 'professional', 'casual', 'dramatic', 'inspirational'],
                        help='Tone of the narrative voice')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Check if arguments were provided
    if len(sys.argv) > 1:
        args = parse_arguments()
        
        # Generate reels
        generate_reels(
            youtube_url=args.url,
            user_directions=args.directions,
            duration=args.duration,
            tone=args.tone
        )
    else:
        # Display usage examples
        print("\nReels Generator Agent")
        print("====================\n")
        print("Examples:")
        print("  python reels_generator.py --url https://www.youtube.com/watch?v=VIDEO_ID --directions \"explains the key points about machine learning in an engaging way\" --duration 30 --tone fun")
        print("  python reels_generator.py -u https://youtu.be/VIDEO_ID -d \"shows the most important cooking techniques\" -t professional\n")
        print("Use --help for more information.")
