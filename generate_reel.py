#!/usr/bin/env python3
"""
Generate Reel Script for AI YouTube Shorts Generator.

This script provides an end-to-end workflow for generating pause-and-narration
style reels from YouTube videos based on user directions.
"""

import os
import sys
import json
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modules
from core.config import init_config
from core.error_handler import Result
from core.script_validator import validate_script
from media.media_processor import process_pause_narration
from youtube_shorts_generator import extract_youtube_transcript

# For LLM functionality when OpenAI_API_KEY is available
try:
    import openai
    from jinja2 import Template
    OPENAI_AVAILABLE = True
except ImportError:
    logger.warning("OpenAI/Jinja2 not available. Will use fallback script generation.")
    OPENAI_AVAILABLE = False

def load_prompt_template(template_name: str) -> Optional[str]:
    """Load a prompt template from the prompts directory."""
    template_path = Path(__file__).parent / "prompts" / f"{template_name}.jinja"
    if not template_path.exists():
        logger.error(f"Template not found: {template_path}")
        return None
        
    return template_path.read_text()

def generate_script_with_llm(user_input: str, transcript: str) -> Result:
    """
    Generate a pause-narration script using OpenAI.
    
    Args:
        user_input: User's directions for the video
        transcript: Transcript of the source video
        
    Returns:
        Result containing the generated script or an error
    """
    if not OPENAI_AVAILABLE:
        logger.warning("OpenAI not available, using fallback script generation")
        return Result.failure("OpenAI not available")
        
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in environment")
        return Result.failure("OpenAI API key not found")
        
    # Load and render the template
    template_text = load_prompt_template("pause_narration")
    if not template_text:
        return Result.failure("Failed to load prompt template")
        
    # Truncate transcript if too long
    max_transcript_length = 4000  # Adjust based on context window size
    if len(transcript) > max_transcript_length:
        truncated = transcript[:max_transcript_length] + "... [truncated]"
        logger.warning(f"Truncated transcript from {len(transcript)} to {len(truncated)} characters")
        transcript = truncated
        
    # Render the Jinja template
    template = Template(template_text)
    prompt = template.render(user_input=user_input, transcript=transcript)
    
    try:
        # Make API call to OpenAI
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        
        # Extract and parse the response
        response_text = response.choices[0].message.content.strip()
        
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            logger.error(f"No JSON found in response: {response_text}")
            return Result.failure("Invalid response format from OpenAI")
            
        json_str = json_match.group(0)
        script_data = json.loads(json_str)
        
        if "script" not in script_data:
            logger.error(f"Invalid script format - missing 'script' key: {script_data}")
            return Result.failure("Invalid script format in LLM response")
            
        return Result.success(script_data["script"])
        
    except Exception as e:
        logger.error(f"Error generating script with OpenAI: {str(e)}")
        return Result.failure(f"OpenAI error: {str(e)}")

def generate_fallback_script(transcript: str) -> List[Dict[str, Any]]:
    """
    Generate a simple fallback script when LLM is not available.
    
    Args:
        transcript: Video transcript
        
    Returns:
        A simple alternating clip/narration script
    """
    lines = transcript.split('.')
    total_lines = len(lines)
    
    # Create a simple alternating script
    script = []
    current_pos = 0
    segment_length = 10  # 10-second clips
    
    for i in range(3):  # Create 3 segments
        if current_pos >= 30:  # Stop at 30 seconds
            break
            
        # Add a clip segment
        script.append({
            "type": "clip",
            "start": current_pos,
            "end": min(current_pos + segment_length, 30)
        })
        current_pos += segment_length
        
        # Add a narration segment after each clip except the last one
        if i < 2 and current_pos < 30:
            # Create a simple narration from the line
            narration_text = f"Point #{i+1}: important insight!"
            script.append({
                "type": "narration",
                "text": narration_text,
                "duration": 2.5
            })
    
    return script

def download_youtube_video(video_id: str, output_path: Optional[str] = None) -> Result:
    """
    Download a YouTube video for processing.
    
    Args:
        video_id: YouTube video ID
        output_path: Optional path to save the video
        
    Returns:
        Result with the path to the downloaded video
    """
    try:
        from pytube import YouTube
    except ImportError:
        logger.error("pytube not installed. Run: pip install pytube")
        return Result.failure("Required package 'pytube' not installed")
    
    if output_path is None:
        output_dir = Path(__file__).parent / "downloads"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{video_id}.mp4"
    
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        logger.info(f"Downloading video: {url}")
        
        yt = YouTube(url)
        # Get the highest resolution stream
        stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        
        if not stream:
            logger.error("No suitable video stream found")
            return Result.failure("No suitable video stream found")
        
        # Download the video
        downloaded_path = stream.download(output_path=os.path.dirname(output_path), 
                                         filename=os.path.basename(output_path))
        logger.info(f"Downloaded video to {downloaded_path}")
        
        return Result.success(downloaded_path)
        
    except Exception as e:
        logger.error(f"Error downloading video: {str(e)}")
        return Result.failure(f"Failed to download video: {str(e)}")

def generate_reel(youtube_url: str, user_directions: str, output_path: Optional[str] = None) -> Result:
    """
    Generate a complete reel from a YouTube URL and user directions.
    
    Args:
        youtube_url: YouTube video URL
        user_directions: User's creative directions
        output_path: Optional path to save the output video
        
    Returns:
        Result with the path to the generated reel
    """
    # Initialize configuration
    init_config()
    
    # 1. Extract transcript from YouTube
    logger.info(f"Extracting transcript from: {youtube_url}")
    transcript_result = extract_youtube_transcript(youtube_url)
    
    if not transcript_result.is_success:
        return Result.failure(f"Failed to extract transcript: {transcript_result.error.message}")
    
    transcript_data = transcript_result.value
    transcript = transcript_data['transcript']
    video_id = transcript_data['video_id']
    
    # 2. Generate script either with LLM or fallback
    logger.info("Generating script based on transcript and directions")
    script_result = generate_script_with_llm(user_directions, transcript)
    
    if not script_result.is_success:
        logger.warning(f"LLM script generation failed: {script_result.error.message}")
        logger.info("Using fallback script generation")
        script = generate_fallback_script(transcript)
    else:
        script = script_result.value
    
    # 3. Validate the script
    logger.info("Validating script")
    validation_result = validate_script(script)
    
    if not validation_result.is_success:
        return Result.failure(f"Script validation failed: {validation_result.error.message}")
    
    # 4. Download the video
    logger.info(f"Downloading video with ID: {video_id}")
    download_result = download_youtube_video(video_id)
    
    if not download_result.is_success:
        return Result.failure(f"Failed to download video: {download_result.error.message}")
    
    video_path = download_result.value
    
    # 5. Process the video with pause narration
    logger.info("Processing video with pause-narration format")
    try:
        if output_path is None:
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"{video_id}_reel.mp4"
        
        processed_path = process_pause_narration(video_path, script, output_path)
        logger.info(f"Successfully generated reel at: {processed_path}")
        
        # 6. Return the result
        return Result.success({
            "video_path": processed_path,
            "script": script,
            "duration": validation_result.value["total_duration"]
        })
    
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return Result.failure(f"Failed to process video: {str(e)}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate YouTube Shorts with pause-narration')
    parser.add_argument('--url', '-u', type=str, required=True,
                        help='YouTube video URL')
    parser.add_argument('--directions', '-d', type=str, required=True,
                        help='Directions for how to create the reel')
    parser.add_argument('--output', '-o', type=str,
                        help='Output path for the generated video')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import moviepy.editor
    except ImportError:
        print("Error: moviepy package is required but not installed.")
        print("Please install it with: pip install moviepy")
        sys.exit(1)
    
    # Parse arguments
    if len(sys.argv) > 1:
        args = parse_arguments()
        
        # Generate the reel
        result = generate_reel(args.url, args.directions, args.output)
        
        if result.is_success:
            print(f"\n✅ Successfully generated reel at: {result.value['video_path']}")
            print(f"Duration: {result.value['duration']:.2f} seconds")
        else:
            print(f"\n❌ Error generating reel: {result.error.message}")
            sys.exit(1)
    else:
        # Display usage examples
        print("\nYouTube Shorts Generator - Pause Narration Mode")
        print("=============================================\n")
        print("Examples:")
        print('  python generate_reel.py --url "https://www.youtube.com/watch?v=VIDEO_ID" \\ ')
        print('                         --directions "Create a video highlighting the key insights about startup failure"')
        print("\nRequired packages:")
        print("  - moviepy: for video processing")
        print("  - pytube: for downloading YouTube videos")
        print("  - openai: for script generation using GPT-4 (optional)")
        print("\nUse --help for more information.")
