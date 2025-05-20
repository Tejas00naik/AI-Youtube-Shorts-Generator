#!/usr/bin/env python3
"""
Complete YouTube Shorts Generator Script.

This script takes a user query and a YouTube video URL, extracts the transcript,
generates a narrative plan, and outputs the plan for video creation.
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, Optional
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
from core.orchestrator import process_input
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
    print(f"📽️  NARRATIVE PLAN - Total Duration: {plan.get('total_duration', 0)}s")
    print("=" * 80)
    
    print(f"\n📋 Summary: {plan.get('summary', 'No summary available')}")
    
    # Print metadata if available
    metadata = plan.get('metadata', {})
    if metadata:
        print("\n📊 METADATA:")
        print(f"   Mode: {metadata.get('mode', 'Not specified')}")
        print(f"   Tone: {metadata.get('tone', 'Not specified')}")
        print(f"   Is Fallback: {'Yes' if metadata.get('is_fallback', False) else 'No'}")
    
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
        
        return Result.success(transcript_text)
        
    except ImportError:
        logger.error("youtube_transcript_api not installed. Run: pip install youtube_transcript_api")
        return Result.failure("Required package 'youtube_transcript_api' not installed", 
                            code=ErrorCode.DEPENDENCY_MISSING)
    except Exception as e:
        logger.error(f"Error extracting transcript: {str(e)}")
        return Result.failure(f"Failed to extract transcript: {str(e)}", 
                            code=ErrorCode.EXTERNAL_API_ERROR)

def generate_shorts(user_query: str, youtube_url: Optional[str] = None, transcript: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate a YouTube shorts narrative plan from a user query and either a YouTube URL or transcript.
    
    Args:
        user_query: The user's query describing the desired video
        youtube_url: URL of YouTube video to extract transcript from (optional)
        transcript: Directly provided transcript (optional)
        
    Returns:
        The generated narrative plan or None if an error occurred
    """
    # Initialize the config
    init_config()
    
    logger.info(f"Processing user query: {user_query}")
    
    # Get transcript either from URL or use provided transcript
    if transcript is None and youtube_url:
        logger.info(f"Extracting transcript from URL: {youtube_url}")
        transcript_result = extract_youtube_transcript(youtube_url)
        
        if not transcript_result.is_success:
            print(f"\n❌ Error extracting transcript: {transcript_result.error.message}")
            return None
            
        transcript = transcript_result.value
        logger.info(f"Successfully extracted transcript ({len(transcript)} characters)")
    elif transcript is None:
        print("\n❌ Error: Either a YouTube URL or transcript must be provided")
        return None
    
    # Process the input through the pipeline
    result = process_input(user_query, transcript)
    
    if result.is_success:
        print("\n✅ Successfully generated narrative plan!")
        print_narrative_plan(result.value)
        
        # Save the plan to a JSON file
        output_file = f"narrative_plan_{hash(user_query)}.json"
        with open(output_file, 'w') as f:
            json.dump(result.value, f, indent=2)
        print(f"\n💾 Saved narrative plan to {output_file}")
        
        return result.value
    else:
        print("\n❌ Error generating narrative plan:")
        print(f"Error: {result.error.message}")
        print(f"Code: {result.error.code}")
        print(f"Severity: {result.error.severity}")
        return None

def sample_transcript() -> str:
    """Return a sample transcript for demo purposes when no YouTube URL is provided."""
    return """
    Hello everyone! Today, I'm going to show you how to make a delicious chocolate cake from scratch.
    
    First, let's go through the ingredients you'll need:
    - 2 cups of all-purpose flour
    - 2 cups of sugar
    - 3/4 cup of unsweetened cocoa powder
    - 2 teaspoons of baking soda
    - 1 teaspoon of baking powder
    - 1 teaspoon of salt
    - 2 eggs
    - 1 cup of buttermilk
    - 1/2 cup of vegetable oil
    - 2 teaspoons of vanilla extract
    - 1 cup of hot coffee
    
    Now, let's start by preheating the oven to 350 degrees Fahrenheit or 175 degrees Celsius.
    
    While the oven is heating up, we'll prepare our dry ingredients. In a large bowl, whisk together the flour, sugar, cocoa powder, baking soda, baking powder, and salt.
    
    In another bowl, we'll mix our wet ingredients. Beat the eggs, then add the buttermilk, vegetable oil, and vanilla extract.
    
    Now, we'll combine the wet and dry ingredients together. Mix them until they're just combined - don't overmix!
    
    The last step is to add the hot coffee. This will make the batter quite thin, but don't worry, that's exactly what we want. The coffee enhances the chocolate flavor without making the cake taste like coffee.
    
    Pour the batter into two 9-inch round cake pans that have been greased and floured.
    
    Bake for about 30-35 minutes, or until a toothpick inserted in the center comes out clean.
    
    Let the cakes cool in the pans for about 10 minutes, then remove them and let them cool completely on a wire rack.
    
    Now for the frosting! You can use a simple chocolate buttercream, or get creative with different flavors.
    
    Once the cakes are completely cool, frost them and enjoy your homemade chocolate cake!
    
    This cake is perfect for birthdays, celebrations, or just when you're craving something sweet. Thanks for watching, and happy baking!
    """

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate YouTube Shorts narrative plan')
    parser.add_argument('--query', '-q', type=str, 
                        help='User query describing the desired video')
    parser.add_argument('--url', '-u', type=str, 
                        help='YouTube video URL to extract transcript from')
    parser.add_argument('--transcript-file', '-t', type=str, 
                        help='File containing transcript text')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Use command line arguments if provided, otherwise use defaults
    if args.query:
        user_query = args.query
    else:
        user_query = "Create a fun 30-second highlight video about chocolate cake baking for beginners"
    
    transcript = None
    
    # Read transcript from file if provided
    if args.transcript_file and os.path.exists(args.transcript_file):
        with open(args.transcript_file, 'r') as f:
            transcript = f.read()
            logger.info(f"Read transcript from file: {args.transcript_file}")
    
    # Generate shorts
    generate_shorts(
        user_query=user_query,
        youtube_url=args.url,
        transcript=transcript or sample_transcript()
    )
