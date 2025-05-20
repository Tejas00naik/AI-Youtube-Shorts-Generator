#!/usr/bin/env python3
"""
Shorts Pipeline - Main Integration Module

This script implements the contract-based pipeline for generating AI YouTube Shorts,
integrating all components with strict validation between stages.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import components
from core.error_handler import Result, ErrorCode
from core.config import init_config, get_config
from youtube_shorts_generator import extract_youtube_transcript  # Reusing existing function
from llm.narrative_planner_v2 import generate_narrative_plan
from media.clip_selector import select_best_clips
from llm.script_writer import write_pause_texts
from media.director_agent import assemble_video
from llm.llm_client import get_llm_client, LLMProvider

# Make sure required modules are available
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("Requests module not available. API calls may not work.")

class ShortsGenerator:
    """
    Main pipeline orchestrator for AI YouTube Shorts generation.
    Implements the contract-based approach with staged validation.
    """
    
    def __init__(self, llm_provider=None, api_key=None, model=None):
        """Initialize the shorts generator."""
        self.config = get_config()
        
        # Get LLM provider from args or environment
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "openai")
        self.api_key = api_key
        self.model = model
        
        # If API key not explicitly provided, look for it in environment
        if not self.api_key:
            if self.llm_provider.lower() == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.llm_provider.lower() == "deepseek":
                self.api_key = os.getenv("DEEPSEEK_API_KEY")
            elif self.llm_provider.lower() == "anthropic":
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Initialize LLM client
        try:
            self.llm_client = get_llm_client(
                provider=self.llm_provider,
                api_key=self.api_key,
                model=self.model
            )
            if self.llm_client.is_available():
                logger.info(f"{self.llm_provider.title()} client initialized successfully with model {self.llm_client.model}")
            else:
                logger.warning(f"Could not initialize {self.llm_provider} client. Using fallback methods.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            self.llm_client = None
        
        # Initialize working directory
        self.working_dir = Path("_working")
        self.working_dir.mkdir(exist_ok=True)
        
        # Initialize output directory
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)

    def download_youtube_video(self, video_id: str) -> Result:
        """Download a YouTube video for processing."""
        try:
            from pytube import YouTube
        except ImportError:
            logger.error("pytube not installed. Run: pip install pytube")
            return Result.failure("Required package 'pytube' not installed", 
                                code=ErrorCode.DEPENDENCY_MISSING)
        
        output_path = self.working_dir / f"{video_id}.mp4"
        
        try:
            url = f"https://www.youtube.com/watch?v={video_id}"
            logger.info(f"Downloading video: {url}")
            
            yt = YouTube(url)
            # Get the highest resolution stream
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            if not stream:
                logger.error("No suitable video stream found")
                return Result.failure("No suitable video stream found", 
                                    code=ErrorCode.EXTERNAL_API_ERROR)
            
            # Download the video
            downloaded_path = stream.download(output_path=str(self.working_dir), 
                                            filename=f"{video_id}.mp4")
            logger.info(f"Downloaded video to {downloaded_path}")
            
            return Result.success(str(output_path))
            
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            return Result.failure(f"Failed to download video: {str(e)}", 
                                code=ErrorCode.EXTERNAL_API_ERROR)

    def generate_shorts(self, youtube_url: str, user_directions: str,
                      tone: str = "professional", clip_count: Optional[int] = None,
                      output_filename: Optional[str] = None) -> Result:
        """
        Generate a YouTube Short from a YouTube URL and user directions.
        
        Args:
            youtube_url: URL of the YouTube video
            user_directions: User's directions for the narrative
            tone: Tone for the narrative (professional, casual, fun, etc.)
            clip_count: Optional number of clips to include
            output_filename: Optional filename for the output video
            
        Returns:
            Result object with output information or error
        """
        # Start pipeline process
        # Step 1: Extract video ID and transcript
        logger.info(f"Starting shorts generation pipeline for: {youtube_url}")
        logger.info(f"User directions: {user_directions}")
        logger.info(f"Using tone: {tone}")
        
        # Extract video ID and transcript
        transcript_result = extract_youtube_transcript(youtube_url)
        if not transcript_result.is_success:
            return transcript_result
            
        transcript_data = transcript_result.value
        transcript = transcript_data['transcript']
        video_id = transcript_data['video_id']
        
        logger.info(f"Extracted transcript ({len(transcript)} chars) from video {video_id}")
        
        # Save extracted transcript for reference
        transcript_path = self.working_dir / f"{video_id}_transcript.txt"
        with open(transcript_path, 'w') as f:
            f.write(transcript)
            
        # Step 2: Generate narrative plan using NarrativePlanner
        logger.info("Generating narrative plan...")
        plan_result = generate_narrative_plan(
            transcript, 
            user_directions, 
            clip_count=clip_count,
            tone=tone,
            llm_provider=self.llm_provider,
            api_key=self.api_key,
            model=self.model
        )
        
        if not plan_result.is_success:
            return plan_result
            
        narrative_plan = plan_result.value
        logger.info(f"Generated narrative plan with {len(narrative_plan['segments'])} segments")
        
        # Save narrative plan for reference
        plan_path = self.working_dir / f"{video_id}_narrative_plan.json"
        with open(plan_path, 'w') as f:
            json.dump(narrative_plan, f, indent=2)
            
        # Step 3: Download the video if not already done
        video_path = self.working_dir / f"{video_id}.mp4"
        if not video_path.exists():
            logger.info(f"Downloading video {video_id}...")
            download_result = self.download_youtube_video(video_id)
            if not download_result.is_success:
                return download_result
            video_path = Path(download_result.value)
        else:
            logger.info(f"Video already downloaded at {video_path}")
            
        # Step 4: Select best clips using ClipSelector
        logger.info("Selecting best clips...")
        clips_result = select_best_clips(
            str(video_path),
            narrative_plan,
            llm_provider=self.llm_provider,
            api_key=self.api_key,
            model=self.model
        )
        
        if not clips_result.is_success:
            return clips_result
            
        clips_data = clips_result.value
        logger.info(f"Selected {len(clips_data['clips'])} clips")
        
        # Save clips data for reference
        clips_path = self.working_dir / f"{video_id}_clips.json"
        with open(clips_path, 'w') as f:
            json.dump(clips_data, f, indent=2)
            
        # Step 5: Generate pause texts using ScriptWriter
        logger.info("Generating pause texts...")
        
        # Create clip contexts for script writer
        # This extracts transcript snippets for each selected clip
        clip_contexts = []
        for i, clip in enumerate(clips_data["clips"]):
            start = clip["start"]
            end = clip["end"]
            
            # Extract transcript for this clip time range
            # This is a simplistic approach - in reality you'd need
            # to match transcript timestamps with clip times
            words = transcript.split()
            
            # Estimate which portion of transcript corresponds to this clip
            # This is very rough and should be improved in a real implementation
            total_duration = narrative_plan["total_duration"]
            word_start = int(len(words) * (start / total_duration))
            word_end = int(len(words) * (end / total_duration))
            
            # Take a word window, ensuring it doesn't go out of bounds
            word_start = max(0, word_start)
            word_end = min(len(words), word_end)
            clip_transcript = " ".join(words[word_start:word_end])
            
            clip_contexts.append({
                "clip_index": i,
                "start_time": start,
                "end_time": end,
                "transcript": clip_transcript[:200] + "..." if len(clip_transcript) > 200 else clip_transcript,
                "focus_point": clip.get("focus_point", "speaker_face")
            })
        
        texts_result = write_pause_texts(
            narrative_plan,
            clip_contexts,
            tone=tone,
            llm_provider=self.llm_provider,
            api_key=self.api_key,
            model=self.model
        )
        
        if not texts_result.is_success:
            return texts_result
            
        texts_data = texts_result.value
        logger.info(f"Generated {len(texts_data['texts'])} pause texts")
        
        # Save texts data for reference
        texts_path = self.working_dir / f"{video_id}_texts.json"
        with open(texts_path, 'w') as f:
            json.dump(texts_data, f, indent=2)
            
        # Step 6: Assemble the final video using DirectorAgent
        logger.info("Assembling final video...")
        
        # Set output filename if not provided
        if output_filename is None:
            output_filename = f"{video_id}_shorts.mp4"
            
        output_path = self.output_dir / output_filename
        
        assembly_result = assemble_video(
            str(video_path),
            clips_data,
            texts_data,
            str(output_path),
            llm_provider=self.llm_provider,
            api_key=self.api_key,
            model=self.model
        )
        
        if not assembly_result.is_success:
            return assembly_result
            
        logger.info(f"Successfully generated video at {output_path}")
        
        # Return the final result with all intermediate data
        return Result.success({
            "output_path": str(output_path),
            "video_id": video_id,
            "narrative_plan": narrative_plan,
            "clips_data": clips_data,
            "texts_data": texts_data,
            "duration": narrative_plan["total_duration"]
        })

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate AI YouTube Shorts')
    parser.add_argument('--url', '-u', type=str, required=True,
                       help='YouTube video URL')
    parser.add_argument('--directions', '-d', type=str, required=True,
                       help="User's directions for the narrative")
    parser.add_argument('--tone', '-t', type=str, default="professional",
                       choices=["professional", "casual", "fun", "dramatic", "inspirational"],
                       help="Tone for the narrative")
    parser.add_argument('--clips', '-c', type=int,
                       help="Optional number of clips to include")
    parser.add_argument('--output', '-o', type=str,
                       help="Output filename for the generated video")
    parser.add_argument('--provider', '-p', type=str, default=os.getenv("LLM_PROVIDER", "openai"),
                       choices=["openai", "deepseek", "anthropic", "local"],
                       help="LLM provider to use")
    parser.add_argument('--api-key', '-k', type=str,
                       help="API key for the LLM provider (will use environment variable if not specified)")
    parser.add_argument('--model', '-m', type=str,
                       help="Model name to use with the LLM provider")
    
    return parser.parse_args()

if __name__ == "__main__":
    # Initialize configuration
    init_config()
    
    # Parse arguments
    if len(sys.argv) > 1:
        args = parse_arguments()
        
        # Create generator and run pipeline with specified LLM provider
        generator = ShortsGenerator(
            llm_provider=args.provider,
            api_key=args.api_key,
            model=args.model
        )
        result = generator.generate_shorts(
            args.url,
            args.directions,
            tone=args.tone,
            clip_count=args.clips,
            output_filename=args.output
        )
        
        if result.is_success:
            print(f"\n✅ Successfully generated shorts video:")
            print(f"  Output: {result.value['output_path']}")
            print(f"  Duration: {result.value['duration']:.1f} seconds")
            print(f"  Segments: {len(result.value['narrative_plan']['segments'])}")
        else:
            print(f"\n❌ Error generating shorts video:")
            print(f"  Error: {result.error.message}")
            print(f"  Code: {result.error.code}")
            sys.exit(1)
    else:
        # Display usage example
        print("\nAI YouTube Shorts Generator - Contract-based Pipeline")
        print("==================================================\n")
        print("Example usage:")
        print('  python shorts_pipeline.py --url "https://www.youtube.com/watch?v=VIDEO_ID" \\')
        print('                           --directions "Create a video highlighting the key insights" \\')
        print('                           --tone fun --clips 3 --provider deepseek\n')
        print("Required packages:")
        print("  - pytube: for downloading YouTube videos")
        print("  - moviepy: for video processing")
        print("  - requests: for API communication")
        print("\nSupported LLM providers:")
        print("  - openai: Uses OpenAI API (set OPENAI_API_KEY environment variable)")
        print("  - deepseek: Uses DeepSeek API (set DEEPSEEK_API_KEY environment variable)")
        print("  - anthropic: Uses Anthropic API (set ANTHROPIC_API_KEY environment variable)")
        print("  - local: Uses a local LLM API endpoint (set LOCAL_API_URL environment variable)")
        print("\nUse --help for more information.")
