#!/usr/bin/env python3
"""
Verbose test script for AI YouTube Shorts Generator that logs all LLM interactions.
This script shows the detailed prompts, responses, and data contracts at each step.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import necessary modules
from shorts_pipeline import ShortsGenerator
from core.error_handler import Result, ErrorCode
from youtube_shorts_generator import extract_youtube_transcript
from media.clip_selector import select_best_clips
from llm.script_writer import write_pause_texts
from llm.narrative_planner_v2 import generate_narrative_plan
from media.director_agent import assemble_video

class VerboseLogger:
    """Simple logger to capture all steps in the AI YouTube Shorts generation process."""
    
    def __init__(self, log_dir="logs"):
        """Initialize logger with timestamp-based session directory."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped session folder
        self.session_id = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.log_dir / self.session_id
        self.session_dir.mkdir(exist_ok=True)
        
        # Initialize logging
        self.log_file = self.session_dir / "session.log"
        with open(self.log_file, "w") as f:
            f.write(f"=== AI YouTube Shorts Generator Session: {self.session_id} ===\n")
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    def log(self, message, level="INFO"):
        """Log a simple message to the log file."""
        with open(self.log_file, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {level}: {message}\n")
        print(f"[{level}] {message}")
    
    def log_step(self, step_name, description=None):
        """Log a major step in the pipeline."""
        with open(self.log_file, "a") as f:
            f.write(f"\n{'=' * 50}\n")
            f.write(f"STEP: {step_name}\n")
            if description:
                f.write(f"{description}\n")
            f.write(f"{'=' * 50}\n\n")
        print(f"\n--- STEP: {step_name} ---")
        if description:
            print(description)
    
    def log_data(self, name, data, format_json=True):
        """Log data to a separate file and reference it in the main log."""
        # Create a file for this data
        data_file = self.session_dir / f"{name.lower().replace(' ', '_')}.json"
        
        if format_json and isinstance(data, (dict, list)):
            with open(data_file, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(data_file, "w") as f:
                f.write(str(data))
        
        # Reference in main log
        with open(self.log_file, "a") as f:
            f.write(f"[DATA] {name} saved to {data_file.name}\n")
        
        print(f"[DATA] {name} saved to {data_file.name}")
        return data_file
    
    def log_prompt(self, component, prompt, response):
        """Log an LLM prompt and response."""
        prompt_dir = self.session_dir / "prompts"
        prompt_dir.mkdir(exist_ok=True)
        
        # Create unique filename
        timestamp = time.strftime("%H%M%S")
        prompt_file = prompt_dir / f"{component}_{timestamp}.txt"
        
        with open(prompt_file, "w") as f:
            f.write(f"=== PROMPT for {component} ===\n\n")
            f.write(str(prompt))
            f.write("\n\n=== RESPONSE ===\n\n")
            f.write(str(response))
        
        # Reference in main log
        with open(self.log_file, "a") as f:
            f.write(f"[PROMPT] {component} prompt and response saved to {prompt_file.name}\n")
        
        print(f"[PROMPT] {component} prompt and response saved")
        return prompt_file
    
    def log_contract(self, component, input_data, output_data):
        """Log data contracts between pipeline components."""
        contract_dir = self.session_dir / "contracts"
        contract_dir.mkdir(exist_ok=True)
        
        contract_file = contract_dir / f"{component}_contract.json"
        
        contract_data = {
            "component": component,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "input": input_data,
            "output": output_data
        }
        
        with open(contract_file, "w") as f:
            json.dump(contract_data, f, indent=2)
        
        # Reference in main log
        with open(self.log_file, "a") as f:
            f.write(f"[CONTRACT] {component} contract saved to {contract_file.name}\n")
        
        print(f"[CONTRACT] {component} I/O contract saved")
        return contract_file
    
    def finish(self):
        """Finalize the logging session."""
        with open(self.log_file, "a") as f:
            f.write(f"\n\nSession completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"\nLogging session completed. All logs saved to {self.session_dir}")
        return str(self.session_dir)

def monkey_patch_llm_functions(logger):
    """
    Monkey patch key LLM functions to add logging.
    This will intercept calls to the LLM and log prompts and responses.
    """
    # Patch narrative planner
    original_narrative_plan = generate_narrative_plan
    
    def logged_narrative_plan(transcript, user_directions, clip_count=None, tone="professional", **kwargs):
        logger.log_step("Narrative Planning", 
                      "Generating a structured plan for the short video based on transcript and user directions")
        
        # Log input contract
        input_contract = {
            "transcript": transcript[:100] + "..." if len(transcript) > 100 else transcript,
            "user_directions": user_directions,
            "clip_count": clip_count,
            "tone": tone
        }
        logger.log_data("Narrative Plan Input", input_contract)
        
        # Generate the actual plan
        result = original_narrative_plan(transcript, user_directions, clip_count, tone)
        
        # Log the result
        if result.is_success:
            logger.log_data("Narrative Plan Result", result.value)
            # Try to extract the actual prompt used (this may not work if the implementation doesn't expose it)
            # This is a best-effort approach
            try:
                from llm.narrative_planner_v2 import NarrativePlanner
                planner = NarrativePlanner()
                system_prompt = planner.system_prompt
                user_prompt = planner._build_prompt(transcript, user_directions, clip_count, tone)
                
                logger.log_prompt("Narrative Planner", 
                                f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}", 
                                json.dumps(result.value, indent=2))
            except Exception as e:
                logger.log(f"Could not extract narrative planner prompt: {str(e)}", "WARNING")
        else:
            logger.log(f"Narrative planning failed: {result.error.message}", "ERROR")
        
        return result
    
    # Patch clip selector
    original_select_clips = select_best_clips
    
    def logged_select_clips(video_path, narrative_plan, **kwargs):
        logger.log_step("Clip Selection", 
                      "Selecting the best video clips based on the narrative plan")
        
        # Log input contract
        input_contract = {
            "video_path": video_path,
            "narrative_plan": narrative_plan
        }
        logger.log_data("Clip Selection Input", input_contract)
        
        # Select clips
        result = original_select_clips(video_path, narrative_plan)
        
        # Log the result
        if result.is_success:
            logger.log_data("Selected Clips", result.value)
            # Try to extract prompt (best effort)
            try:
                from media.clip_selector import ClipSelector
                selector = ClipSelector()
                system_prompt = selector.system_prompt
                # This is approximate and depends on implementation details
                logger.log_prompt("Clip Selector", 
                                f"SYSTEM:\n{system_prompt}\n\nNarrative Plan:\n{json.dumps(narrative_plan, indent=2)}", 
                                json.dumps(result.value, indent=2))
            except Exception as e:
                logger.log(f"Could not extract clip selector prompt: {str(e)}", "WARNING")
        else:
            logger.log(f"Clip selection failed: {result.error.message}", "ERROR")
        
        return result
    
    # Patch script writer
    original_write_texts = write_pause_texts
    
    def logged_write_texts(narrative_plan, clip_contexts, tone="professional", **kwargs):
        logger.log_step("Script Writing", 
                      "Generating pause texts and captions for the video")
        
        # Log input contract
        input_contract = {
            "narrative_plan": narrative_plan,
            "clip_contexts": clip_contexts,
            "tone": tone
        }
        logger.log_data("Script Writer Input", input_contract)
        
        # Generate scripts
        result = original_write_texts(narrative_plan, clip_contexts, tone)
        
        # Log the result
        if result.is_success:
            logger.log_data("Generated Texts", result.value)
            # Try to extract prompt (best effort)
            try:
                from llm.script_writer import ScriptWriter
                writer = ScriptWriter()
                system_prompt = writer.system_prompt
                logger.log_prompt("Script Writer", 
                                f"SYSTEM:\n{system_prompt}\n\nINPUT:\nTone: {tone}\nClip Contexts: {json.dumps(clip_contexts, indent=2)}", 
                                json.dumps(result.value, indent=2))
            except Exception as e:
                logger.log(f"Could not extract script writer prompt: {str(e)}", "WARNING")
        else:
            logger.log(f"Script writing failed: {result.error.message}", "ERROR")
        
        return result
    
    # Apply all patches
    globals()['generate_narrative_plan'] = logged_narrative_plan
    globals()['select_best_clips'] = logged_select_clips
    globals()['write_pause_texts'] = logged_write_texts
    
    return (original_narrative_plan, original_select_clips, original_write_texts)

def extract_youtube_video_id(youtube_url):
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

def verbose_test(youtube_url, user_prompt, tone="dramatic", clip_count=4):
    """Run a test of the AI YouTube Shorts Generator with detailed logging."""
    # Initialize logger
    logger = VerboseLogger()
    logger.log_step("Initialization", "Setting up test environment")
    
    try:
        # Log test parameters
        test_params = {
            "youtube_url": youtube_url,
            "user_prompt": user_prompt,
            "tone": tone,
            "clip_count": clip_count,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        logger.log_data("Test Parameters", test_params)
        
        # Extract video ID
        video_id = extract_youtube_video_id(youtube_url)
        logger.log(f"Extracted video ID: {video_id}")
        
        # Create working directories
        working_dir = Path("_working")
        output_dir = Path("output")
        working_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        
        # Setup output paths
        output_filename = f"{video_id}_verbose_reel.mp4"
        output_path = str(output_dir / output_filename)
        
        # Monkey patch functions to add logging
        logger.log("Applying logging patches to LLM functions")
        original_functions = monkey_patch_llm_functions(logger)
        
        # Step 1: Extract transcript
        logger.log_step("Transcript Extraction", "Extracting transcript from YouTube video")
        transcript_result = extract_youtube_transcript(youtube_url)
        
        if not transcript_result.is_success:
            logger.log(f"Error extracting transcript: {transcript_result.error.message}", "ERROR")
            return
        
        transcript = transcript_result.value
        logger.log(f"Successfully extracted transcript ({len(transcript)} characters)")
        logger.log(f"Transcript sample: {transcript[:150]}...")
        logger.log_data("Full Transcript", transcript)
        
        # Save transcript for reference
        transcript_path = working_dir / f"{video_id}_transcript.txt"
        with open(transcript_path, "w") as f:
            f.write(transcript)
        
        # Initialize ShortsGenerator
        logger.log_step("Pipeline Initialization", "Setting up the ShortsGenerator with local LLM")
        generator = ShortsGenerator(llm_provider="local")
        
        # Patch the generator functions to use our logged versions
        generator_patches = {}
        generator_patches['download_youtube_video'] = generator.download_youtube_video
        
        def logged_download(self, video_id):
            """Log the download process."""
            logger.log_step("Video Download", f"Downloading YouTube video {video_id}")
            
            try:
                import pytubefix as pytube_module
                from pytubefix import YouTube
            except ImportError:
                logger.log("pytubefix not installed. Run: pip install pytubefix", "ERROR")
                return Result.failure("Required package 'pytubefix' not installed")
            
            output_path = self.working_dir / f"{video_id}.mp4"
            
            try:
                url = f"https://www.youtube.com/watch?v={video_id}"
                logger.log(f"Downloading video: {url}")
                
                yt = YouTube(url)
                # Get the highest resolution stream
                stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
                
                if not stream:
                    logger.log("No suitable video stream found", "ERROR")
                    return Result.failure("No suitable video stream found")
                
                # Download the video
                downloaded_path = stream.download(output_path=str(self.working_dir), 
                                                filename=f"{video_id}.mp4")
                logger.log(f"Downloaded video to {downloaded_path}")
                
                return Result.success(str(output_path))
                
            except Exception as e:
                logger.log(f"Error downloading video: {str(e)}", "ERROR")
                return Result.failure(f"Failed to download video: {str(e)}")
        
        # Apply download patch
        ShortsGenerator.download_youtube_video = logged_download
        
        # Create a wrapper for generate_shorts that logs all steps
        original_generate_shorts = generator.generate_shorts
        
        def patched_generate_shorts(self, youtube_url, user_directions, 
                                   tone="professional", clip_count=None,
                                   output_filename=None):
            """Patched version of generate_shorts that creates narrative plan with logging."""
            # This function is intentionally abbreviated since we'll handle each step manually
            logger.log("Using verbose mode - will process pipeline steps individually")
            return Result.failure("Verbose mode active - using manual pipeline instead")
        
        # Apply patch to skip the normal function
        generator.generate_shorts = patched_generate_shorts
        
        # Step 2: Generate narrative plan using our logged version
        logger.log_step("Narrative Planning", "Generating the narrative plan")
        plan_result = generate_narrative_plan(
            transcript=transcript,
            user_directions=user_prompt,
            clip_count=clip_count,
            tone=tone
        )
        
        if not plan_result.is_success:
            logger.log(f"Error generating narrative plan: {plan_result.error.message}", "ERROR")
            return
        
        narrative_plan = plan_result.value
        logger.log(f"Generated narrative plan with {len(narrative_plan['segments'])} segments")
        
        # Save narrative plan for reference
        plan_path = working_dir / f"{video_id}_narrative_plan.json"
        with open(plan_path, "w") as f:
            json.dump(narrative_plan, f, indent=2)
        
        # Step 3: Download the video if not already done
        video_path = working_dir / f"{video_id}.mp4"
        if not video_path.exists():
            logger.log(f"Video not found, downloading {video_id}...")
            download_result = generator.download_youtube_video(video_id)
            if not download_result.is_success:
                logger.log(f"Error downloading video: {download_result.error.message}", "ERROR")
                return
            video_path = Path(download_result.value)
        else:
            logger.log(f"Video already exists at {video_path}")
        
        # Step 4: Select best clips using our logged version
        logger.log_step("Clip Selection", "Selecting the best clips from the video")
        clips_result = select_best_clips(
            str(video_path),
            narrative_plan
        )
        
        if not clips_result.is_success:
            logger.log(f"Error selecting clips: {clips_result.error.message}", "ERROR")
            return
        
        clips_data = clips_result.value
        logger.log(f"Selected {len(clips_data['clips'])} clips")
        
        # Save clips data for reference
        clips_path = working_dir / f"{video_id}_clips.json"
        with open(clips_path, "w") as f:
            json.dump(clips_data, f, indent=2)
        
        # Step 5: Create clip contexts for script writer
        logger.log_step("Clip Context Creation", "Creating context objects for each clip")
        
        clip_contexts = []
        for i, clip in enumerate(clips_data["clips"]):
            start = clip["start"]
            end = clip["end"]
            
            # Extract approximate transcript for this clip
            words = transcript.split()
            total_duration = narrative_plan["total_duration"]
            word_start = int(len(words) * (start / total_duration))
            word_end = int(len(words) * (end / total_duration))
            
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
        
        logger.log(f"Created {len(clip_contexts)} clip contexts")
        logger.log_data("Clip Contexts", clip_contexts)
        
        # Step 6: Generate pause texts using our logged version
        logger.log_step("Script Writing", "Generating pause texts with captions")
        texts_result = write_pause_texts(
            narrative_plan,
            clip_contexts,
            tone=tone
        )
        
        if not texts_result.is_success:
            logger.log(f"Error generating pause texts: {texts_result.error.message}", "ERROR")
            return
        
        texts_data = texts_result.value
        logger.log(f"Generated {len(texts_data['texts'])} pause texts")
        
        # Save texts data for reference
        texts_path = working_dir / f"{video_id}_texts.json"
        with open(texts_path, "w") as f:
            json.dump(texts_data, f, indent=2)
        
        # Step 7: Assemble the video
        logger.log_step("Video Assembly", "Assembling final video with selected clips and text overlays")
        assembly_result = assemble_video(
            str(video_path),
            clips_data,
            texts_data,
            output_path
        )
        
        if not assembly_result.is_success:
            logger.log(f"Error assembling video: {assembly_result.error.message}", "ERROR")
            return
        
        logger.log(f"\n✅ Success! Video created at {output_path}")
        logger.log(f"Duration: {narrative_plan['total_duration']} seconds")
        
        # Restore original functions
        globals()['generate_narrative_plan'] = original_functions[0]
        globals()['select_best_clips'] = original_functions[1]
        globals()['write_pause_texts'] = original_functions[2]
        ShortsGenerator.download_youtube_video = generator_patches['download_youtube_video']
        
        # Finish logging
        log_dir = logger.finish()
        print(f"\nAll logs and LLM interactions saved to: {log_dir}")
        print(f"To review the logs, open: {logger.log_file}")
        
        return {
            "output_path": output_path,
            "log_dir": log_dir,
            "narrative_plan": narrative_plan,
            "clips_data": clips_data,
            "texts_data": texts_data
        }
        
    except Exception as e:
        import traceback
        logger.log(f"Error during verbose test: {str(e)}", "ERROR")
        logger.log(traceback.format_exc(), "ERROR")
        logger.finish()

if __name__ == "__main__":
    # Chess video and prompt
    youtube_url = "https://youtu.be/t2laAqRtAq0?si=rM50co1i_4GMVx-X"
    user_prompt = "create a reel with hook saying Did you really thaugh that gukesh is the real owrkld chess champion. And then fill up appropriate clips from this video and some narrations text."
    
    # Run the test with verbose logging
    print("\n=== Verbose AI YouTube Shorts Generator Test ===\n")
    print(f"YouTube URL: {youtube_url}")
    print(f"User Prompt: {user_prompt}")
    print(f"Tone: dramatic")
    print(f"Clip Count: 4\n")
    
    verbose_test(youtube_url, user_prompt, "dramatic", 4)
