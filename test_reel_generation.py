#!/usr/bin/env python3
"""
Universal test script for the AI YouTube Shorts Generator.

This script allows testing the entire pipeline with any YouTube URL and user prompt,
outputting the full narrative plan and generating the final video.

Usage:
python test_reel_generation.py --url "https://www.youtube.com/watch?v=VIDEO_ID" \
                              --prompt "Your creative directions here" \
                              --tone "dramatic" --clips 4
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from shorts_pipeline import ShortsGenerator
from youtube_shorts_generator import extract_youtube_transcript
from core.error_handler import Result

def extract_video_id(url):
    """Extract the video ID from a YouTube URL."""
    if "youtu.be" in url:
        parts = url.split("/")
        for part in parts:
            if "?" in part:
                return part.split("?")[0]
        return parts[-1]
    
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    return url

def test_reel_generation(youtube_url, user_prompt, tone="professional", clip_count=3):
    """
    Test generating a reel with the specified YouTube URL and user prompt.
    
    Args:
        youtube_url: The YouTube video URL
        user_prompt: The user's creative directions
        tone: The tone for the narrative (professional, casual, fun, dramatic)
        clip_count: Number of clips to include
        
    Returns:
        Result object with output path or error
    """
    print("\n=== AI YouTube Shorts Generator Test ===\n")
    
    print(f"YouTube URL: {youtube_url}")
    print(f"User Prompt: {user_prompt}")
    print(f"Tone: {tone}")
    print(f"Clip Count: {clip_count}")
    
    log_dir = Path("logs/component_test")
    log_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n-- Starting Generation Process --\n")
    
    try:
        # Extract video ID from URL
        video_id = extract_video_id(youtube_url)
        print(f"Extracted Video ID: {video_id}")
        
        # Create working and output directories
        working_dir = Path("_working")
        output_dir = Path("output")
        working_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        
        # Step 1: Extract Transcript
        print("\n[Step 1] Extracting transcript...")
        transcript_result = extract_youtube_transcript(youtube_url)
        if not transcript_result.is_success:
            print(f"Error extracting transcript: {transcript_result.error.message}")
            return Result.failure(transcript_result.error.message)
        
        # The extract_youtube_transcript function returns transcript text directly
        transcript = transcript_result.value
        
        # Save transcript for reference
        transcript_path = working_dir / f"{video_id}_transcript.txt"
        with open(transcript_path, 'w') as f:
            f.write(transcript)
        
        # Print snippet of transcript
        print(f"\nTranscript snippet (first 200 chars):")
        print(f"{transcript[:200]}...\n")
        
        # Create a generator instance 
        generator = ShortsGenerator(llm_provider="local")
        
        # Set output filename
        output_filename = f"{video_id}_reel.mp4"
        output_path = str(output_dir / output_filename)
        
        # Need to patch the existing shorts_pipeline.py implementation to match what the generate_shorts expects
        # We'll modify the ShortsGenerator's extract_youtube_transcript method to make it compatible
        
        # Create wrapper functions to patch the implementation mismatches with detailed logging
        
        # 1. Patch the extract_youtube_transcript function
        def patched_extract_youtube_transcript(url):
            result = extract_youtube_transcript(url)
            if result.is_success:
                # The shorts_pipeline.py expects a dictionary with 'transcript' and 'video_id'
                # But our extract_youtube_transcript only returns the text
                # So we need to wrap the result in the expected format
                return Result.success({
                    'transcript': result.value,
                    'video_id': extract_video_id(url)
                })
            return result
        
        # 2. Patch the download_youtube_video method to use pytubefix instead of pytube
        original_download_method = ShortsGenerator.download_youtube_video
        
        def patched_download_youtube_video(self, video_id):
            try:
                import pytubefix as pytube_module
                from pytubefix import YouTube
            except ImportError:
                logger.error("pytubefix not installed. Run: pip install pytubefix")
                return Result.failure("Required package 'pytubefix' not installed")
            
            output_path = self.working_dir / f"{video_id}.mp4"
            
            try:
                url = f"https://www.youtube.com/watch?v={video_id}"
                print(f"Downloading video: {url}")
                
                yt = YouTube(url)
                # Get the highest resolution stream
                stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                
                if not stream:
                    print("No suitable video stream found")
                    return Result.failure("No suitable video stream found")
                
                # Download the video
                downloaded_path = stream.download(output_path=str(self.working_dir), 
                                                filename=f"{video_id}.mp4")
                print(f"Downloaded video to {downloaded_path}")
                
                return Result.success(str(output_path))
                
            except Exception as e:
                print(f"Error downloading video: {str(e)}")
                return Result.failure(f"Failed to download video: {str(e)}")
                
        # 3. Patch the generate_narrative_plan function to accept the extra parameters and log details
        from llm.narrative_planner_v2 import generate_narrative_plan as original_narrative_plan
        
        def patched_generate_narrative_plan(transcript, user_directions, clip_count=None, tone="professional", 
                                           llm_provider=None, api_key=None, model=None):
            # Log inputs to the narrative planner
            narrative_input = {
                "transcript_length": len(transcript),
                "transcript_snippet": transcript[:100] + "...",
                "user_directions": user_directions,
                "clip_count": clip_count,
                "tone": tone
            }
            
            narrative_input_path = log_dir / "narrative_planner_input.json"
            with open(narrative_input_path, "w") as f:
                json.dump(narrative_input, f, indent=2)
                
            print("\n[COMPONENT 1: NARRATIVE PLANNING]")
            print("Generating the full narrative plan based on transcript and user prompt")
            print(f"Logged narrative planner input to: {narrative_input_path}")
            
            # Call the original function with appropriate parameters
            result = original_narrative_plan(transcript, user_directions, clip_count, tone)
            
            # Log the narrative plan output
            if result.is_success:
                narrative_plan = result.value
                narrative_output_path = log_dir / "narrative_plan_output.json"
                with open(narrative_output_path, "w") as f:
                    json.dump(narrative_plan, f, indent=2)
                print(f"Generated narrative plan with {len(narrative_plan['segments'])} segments")
                print(f"Logged narrative plan output to: {narrative_output_path}")
                
                # Print a summary of the narrative plan
                action_segments = [s for s in narrative_plan['segments'] if s['type'] == 'action']
                pause_segments = [s for s in narrative_plan['segments'] if s['type'] == 'pause']
                
                print(f"\nNarrative Plan Summary:")
                print(f"- Total duration: {narrative_plan['total_duration']} seconds")
                print(f"- Action segments: {len(action_segments)}")
                print(f"- Pause segments: {len(pause_segments)}")
                
                if pause_segments:
                    print("\nSample pause text:")
                    print(f"'{pause_segments[0].get('text', 'No text found')}'")                
            else:
                print(f"Narrative planning failed: {result.error.message}")
            
            return result
            
        # 4. Patch the select_best_clips function to log details
        from media.clip_selector import select_best_clips as original_select_clips
        
        def patched_select_best_clips(video_path, narrative_plan, llm_provider=None, api_key=None, model=None):
            # Log inputs to the clip selector
            clip_selector_input = {
                "video_path": video_path,
                "narrative_plan": narrative_plan
            }
            
            clip_selector_input_path = log_dir / "clip_selector_input.json"
            with open(clip_selector_input_path, "w") as f:
                json.dump(clip_selector_input, f, indent=2)
                
            print("\n[COMPONENT 2: CLIP SELECTION]")
            print("Selecting the best video clips that match the narrative plan")
            print(f"Logged clip selector input to: {clip_selector_input_path}")
            
            # Call the original function
            result = original_select_clips(video_path, narrative_plan)
            
            # Log the selected clips output
            if result.is_success:
                clips_data = result.value
                clips_output_path = log_dir / "selected_clips_output.json"
                with open(clips_output_path, "w") as f:
                    json.dump(clips_data, f, indent=2)
                print(f"Selected {len(clips_data['clips'])} clips from the video")
                print(f"Logged clip selection output to: {clips_output_path}")
                
                # Print a summary of the selected clips
                print("\nSelected Clips Summary:")
                for i, clip in enumerate(clips_data['clips']):
                    print(f"Clip {i+1}: {clip['start']:.1f}s to {clip['end']:.1f}s ({clip['end']-clip['start']:.1f}s duration)")
            else:
                print(f"Clip selection failed: {result.error.message}")
            
            return result
            
        # 5. Patch the write_pause_texts function to log details
        from llm.script_writer import write_pause_texts as original_write_pause_texts
        
        def patched_write_pause_texts(narrative_plan, clip_contexts, tone="professional", 
                                     llm_provider=None, api_key=None, model=None):
            # Log inputs to the script writer
            script_writer_input = {
                "narrative_plan": narrative_plan,
                "clip_contexts": clip_contexts,
                "tone": tone
            }
            
            script_writer_input_path = log_dir / "script_writer_input.json"
            with open(script_writer_input_path, "w") as f:
                json.dump(script_writer_input, f, indent=2)
                
            print("\n[COMPONENT 3: TEXT GENERATION]")
            print("Generating text overlays for pause segments")
            print(f"Logged script writer input to: {script_writer_input_path}")
            
            # Call the original function
            result = original_write_pause_texts(narrative_plan, clip_contexts, tone)
            
            # Log the generated texts output
            if result.is_success:
                texts_data = result.value
                texts_output_path = log_dir / "generated_texts_output.json"
                with open(texts_output_path, "w") as f:
                    json.dump(texts_data, f, indent=2)
                print(f"Generated {len(texts_data['texts'])} text overlays")
                print(f"Logged text generation output to: {texts_output_path}")
                
                # Print a summary of the generated texts
                print("\nGenerated Texts Summary:")
                for i, text in enumerate(texts_data['texts']):
                    print(f"Text {i+1}: '{text['text']}'")
                    print(f"  Duration: {text['duration']}s, Position: {text['position']}")
            else:
                print(f"Text generation failed: {result.error.message}")
            
            return result
        
        # Apply the patches
        from shorts_pipeline import extract_youtube_transcript as original_func
        import shorts_pipeline
        import llm.narrative_planner_v2
        import media.clip_selector
        import llm.script_writer
        
        # Apply all our patches
        shorts_pipeline.extract_youtube_transcript = patched_extract_youtube_transcript
        ShortsGenerator.download_youtube_video = patched_download_youtube_video
        shorts_pipeline.generate_narrative_plan = patched_generate_narrative_plan
        shorts_pipeline.select_best_clips = patched_select_best_clips
        shorts_pipeline.write_pause_texts = patched_write_pause_texts
        
        # Run the full pipeline - this is the main function that handles everything
        print("\n[Step 2] Running full generation pipeline...")
        result = generator.generate_shorts(
            youtube_url=youtube_url,
            user_directions=user_prompt,
            tone=tone,
            clip_count=clip_count,
            output_filename=output_filename
        )
        
        # Restore the original function when done
        shorts_pipeline.extract_youtube_transcript = original_func
        
        if not result.is_success:
            print(f"\n❌ Error: {result.error.message}")
            if hasattr(result.error, 'code'):
                print(f"Error code: {result.error.code}")
            return result
        
        # Output the narrative plan for review
        if result.is_success and "narrative_plan" in result.value:
            narrative_plan = result.value["narrative_plan"]
            clips_data = result.value.get("clips_data", {})
            texts_data = result.value.get("texts_data", {})
            
            print("\n=== NARRATIVE PLAN ===")
            print(json.dumps(narrative_plan, indent=2))
            print("=== END OF NARRATIVE PLAN ===\n")
            
            print("\n=== SELECTED CLIPS ===")
            print(json.dumps(clips_data, indent=2))
            print("=== END OF SELECTED CLIPS ===\n")
            
            print("\n=== PAUSE TEXTS ===")
            print(json.dumps(texts_data, indent=2))
            print("=== END OF PAUSE TEXTS ===\n")
            
            print(f"\n✅ Success! Video created at {result.value['output_path']}")
            print(f"Duration: {result.value['duration']:.1f} seconds")
        
        return result
    
    except Exception as e:
        print(f"Error during reel generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return Result.failure(f"Unexpected error: {str(e)}")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test AI YouTube Shorts Generator')
    parser.add_argument('--url', '-u', type=str, required=True,
                       help='YouTube video URL')
    parser.add_argument('--prompt', '-p', type=str, required=True,
                       help="User's creative directions")
    parser.add_argument('--tone', '-t', type=str, default="professional",
                       choices=["professional", "casual", "fun", "dramatic", "inspirational"],
                       help="Tone for the narrative")
    parser.add_argument('--clips', '-c', type=int, default=3,
                       help="Number of clips to include")
    
    return parser.parse_args()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Parse command line arguments
        args = parse_arguments()
        test_reel_generation(args.url, args.prompt, args.tone, args.clips)
    else:
        # Default test with chess video if no arguments provided
        youtube_url = "https://youtu.be/t2laAqRtAq0?si=rM50co1i_4GMVx-X"
        user_prompt = "create a reel with hook saying Did you really thaugh that gukesh is the real owrkld chess champion. And then fill up appropriate clips from this video and some narrations text."
        test_reel_generation(youtube_url, user_prompt, "dramatic", 4)
