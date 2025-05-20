#!/usr/bin/env python3
"""
Component tester for AI YouTube Shorts Generator.

This script tests each component individually and logs its inputs and outputs:
1. Narrative Planning
2. Clip Selection
3. Text Generation

No patches or monkeypatching is used - just direct calls to component functions.
"""

import os
import sys
import json
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import the existing YouTube downloader
from Components.YoutubeDownloader import download_youtube_video

from shorts_pipeline import ShortsGenerator
from core.error_handler import Result
from youtube_shorts_generator import extract_youtube_transcript
from llm.narrative_planner_v2 import generate_narrative_plan
from media.clip_selector import select_best_clips
from llm.script_writer import write_pause_texts
from media.director_agent import assemble_video
from media.clip_validator import ClipValidator


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


def test_components(youtube_url, user_prompt, tone="dramatic", clip_count=None):
    """Test each component individually and log details."""
    print("\n=== AI YouTube Shorts Generator Component Test ===\n")
    
    # Set up logging directory
    log_dir = Path("logs/components")  # Standard logs directory
    log_dir.mkdir(exist_ok=True, parents=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_dir = log_dir / timestamp
    test_dir.mkdir(exist_ok=True)
    
    # Initial test parameters
    params = {
        "youtube_url": youtube_url,
        "user_prompt": user_prompt,
        "tone": tone,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Will be updated after prompt analysis
    with open(test_dir / "test_parameters.json", "w") as f:
        json.dump(params, f, indent=2)
    
    print(f"YouTube URL: {youtube_url}")
    print(f"User Prompt: {user_prompt}")
    print(f"Tone: {tone}")
    print(f"Logs will be saved to: {test_dir}")
    
    # Create working directories
    working_dir = Path("_working")
    output_dir = Path("output")
    working_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Extract video ID from URL
        video_id = extract_video_id(youtube_url)
        print(f"\nExtracted Video ID: {video_id}")
        
        # Step 2: Extract transcript
        print("\n[COMPONENT 0: TRANSCRIPT EXTRACTION]")
        print("Extracting transcript from YouTube video...")
        
        # Use direct YouTube API to avoid any compatibility issues
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            transcript = ' '.join([item['text'] for item in transcript_list])
            
            with open(test_dir / "transcript.txt", "w") as f:
                f.write(transcript)
            
            print(f"Successfully extracted transcript ({len(transcript)} characters)")
            print(f"Transcript saved to: {test_dir / 'transcript.txt'}")
            print(f"Transcript sample: {transcript[:150]}...\n")
        except Exception as e:
            print(f"Error extracting transcript: {str(e)}")
            return
        
        # Step 3: Analyze user prompt to determine optimal parameters
        print("\n[COMPONENT 1: PROMPT ANALYSIS]")
        print("Analyzing user prompt to determine optimal video parameters...")
        
        # Import the input analyzer
        from llm.input_analyzer import analyze_user_prompt
        
        # Analyze the prompt
        analysis_result = analyze_user_prompt(user_prompt, len(transcript))
        
        if not analysis_result.is_success:
            print(f"Error analyzing prompt: {str(analysis_result.error)}")
            # Fall back to default parameters
            video_params = {
                "clip_count": clip_count or 4,
                "interruption_style": "pause",
                "interruption_frequency": 3,
                "max_duration": 35
            }
        else:
            video_params = analysis_result.value
            
        # Override with explicit clip_count if provided
        if clip_count is not None:
            video_params["clip_count"] = clip_count
        
        # Update and log test parameters
        params.update(video_params)
        with open(test_dir / "test_parameters.json", "w") as f:
            json.dump(params, f, indent=2)
        
        # Display the determined parameters
        print(f"Analysis complete. Determined parameters:")
        print(f"- Clip count: {video_params['clip_count']}")
        print(f"- Interruption style: {video_params['interruption_style']}")
        print(f"- Interruption frequency: {video_params['interruption_frequency']}")
        print(f"- Target duration: {video_params['max_duration']} seconds")
        
        # Step 4: Generate narrative plan
        print("\n[COMPONENT 2: NARRATIVE PLANNING]")
        print("Generating the full narrative plan based on transcript and user prompt...")
        
        # Log inputs to the narrative planner
        narrative_input = {
            "transcript_length": len(transcript),
            "transcript_snippet": transcript[:100] + "...",
            "user_directions": user_prompt,
            "clip_count": video_params["clip_count"],
            "interruption_style": video_params["interruption_style"],
            "interruption_frequency": video_params["interruption_frequency"],
            "max_duration": video_params["max_duration"],
            "tone": tone
        }
        
        with open(test_dir / "narrative_planner_input.json", "w") as f:
            json.dump(narrative_input, f, indent=2)
        
        # Generate narrative plan
        plan_result = generate_narrative_plan(
            transcript=transcript,
            user_directions=user_prompt,
            clip_count=video_params["clip_count"],
            interruption_style=video_params["interruption_style"],
            interruption_frequency=video_params["interruption_frequency"],
            max_duration=video_params["max_duration"],
            tone=tone
        )
        
        if not plan_result.is_success:
            print(f"Error generating narrative plan: {str(plan_result.error)}")
            return
        
        narrative_plan = plan_result.value
        with open(test_dir / "narrative_plan_output.json", "w") as f:
            json.dump(narrative_plan, f, indent=2)
        
        print(f"Generated narrative plan with {len(narrative_plan['segments'])} segments")
        print(f"Narrative plan saved to: {test_dir / 'narrative_plan_output.json'}")
        
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
        
        # Step 5: Download video if needed (for clip selection)
        video_path = working_dir / f"{video_id}.mp4"
        if not video_path.exists():
            print("\n[VIDEO DOWNLOAD]")
            print(f"Downloading video {video_id}...")
            
            try:
                from pytube import YouTube
                yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
                stream = yt.streams.filter(progressive=True, file_extension='mp4').get_highest_resolution()
                stream.download(output_path=str(working_dir), filename=f"{video_id}.mp4")
                print(f"Downloaded video to {video_path}")
                
            except Exception as e:
                print(f"Error downloading video: {str(e)}")
                return
        else:
            print(f"\nVideo already exists at {video_path}")
        
        # Step 5: Select optimal clips from the video
        print("\n[COMPONENT 3: CLIP SELECTION]")
        print("Selecting the best video clips that match the narrative plan...")
        
        # Ensure the video is downloaded locally
        if video_id:
            video_path = working_dir / f"{video_id}.mp4"  # _working/VIDEO_ID.mp4
            
            # Check if video already exists
            if not video_path.exists():
                # We need to download it
                try:
                    # Use the existing download function from Components.YoutubeDownloader
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    downloaded_path = download_youtube_video(video_url)
                    
                    # Move the file to the working directory with the expected name
                    import shutil
                    shutil.copy(downloaded_path, str(video_path))
                    print(f"Downloaded video to {video_path}")
                except Exception as e:
                    print(f"Error downloading video: {str(e)}")
                    return
            else:
                print(f"Video already exists at {video_path}")
        else:
            print("No video ID provided, skipping clip selection")
            return
            
        # First validate the narrative plan against video constraints
        print("Validating narrative plan against video constraints...")
        validation_result = ClipValidator.validate_narrative_plan(narrative_plan, str(video_path))
        
        if not validation_result.is_success:
            print(f"Error validating narrative plan: {validation_result.error}")
            return
            
        # Use the validated plan
        validated_plan = validation_result.value
        print("Narrative plan successfully validated and adjusted to fit video constraints")
            
        # Save clip selector input with validated plan
        clip_selector_input = {
            "narrative_plan": validated_plan,
            "video_path": str(video_path) 
        }
        with open(test_dir / "clip_selector_input.json", "w") as f:
            json.dump(clip_selector_input, f, indent=2)
        
        # Select clips using validated plan
        clips_result = select_best_clips(str(video_path), validated_plan)
        
        if not clips_result.is_success:
            print(f"Error selecting clips: {str(clips_result.error)}")
            return
            
        clips_data = clips_result.value
        with open(test_dir / "selected_clips_output.json", "w") as f:
            json.dump(clips_data, f, indent=2)
        
        print(f"Selected {len(clips_data['clips'])} clips from the video")
        print(f"Selected clips saved to: {test_dir / 'selected_clips_output.json'}")
        
        # Print a summary of the clips
        print("\nSelected Clips Summary:")
        for i, clip in enumerate(clips_data['clips']):
            print(f"Clip {i+1}: {clip['start_time']:.1f}s to {clip['end_time']:.1f}s ({clip['duration']:.1f}s duration)")
        
        # Step 6: Create clip contexts for the script writer
        clip_contexts = []
        for i, clip in enumerate(clips_data["clips"]):
            start = clip["start"]
            end = clip["end"]
            
            # Extract transcript for this clip time range (approximate)
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
        
        # Step 7: Generate pause texts
        print("\n[COMPONENT 4: TEXT GENERATION]")
        print("Generating text overlays for pause segments...")
        
        # Save script writer input
        script_writer_input = {
            "narrative_plan": narrative_plan,
            "clip_contexts": clip_contexts,
            "tone": tone
        }
        with open(test_dir / "script_writer_input.json", "w") as f:
            json.dump(script_writer_input, f, indent=2)
        
        # Generate texts
        texts_result = write_pause_texts(
            narrative_plan,
            clip_contexts,
            tone=tone
        )
        
        if not texts_result.is_success:
            print(f"Error generating texts: {str(texts_result.error)}")
            return
        
        texts_data = texts_result.value
        with open(test_dir / "generated_texts_output.json", "w") as f:
            json.dump(texts_data, f, indent=2)
        
        print(f"Generated {len(texts_data['texts'])} text overlays")
        print(f"Generated texts saved to: {test_dir / 'generated_texts_output.json'}")
        
        # Print a summary of the generated texts
        print("\nGenerated Texts Summary:")
        for i, text in enumerate(texts_data['texts']):
            print(f"Text {i+1}: '{text['text']}'")
            print(f"  Duration: {text['duration']}s, Position: {text['position']}")
        
        # Step 8: Assemble final video
        print("\n[COMPONENT 5: VIDEO ASSEMBLY]")
        print("Assembling final video with selected clips and text overlays...")
        
        output_filename = f"{video_id}_component_test.mp4"
        output_path = str(output_dir / output_filename)
        
        assembly_result = assemble_video(
            str(video_path),
            clips_data,
            texts_data,
            output_path
        )
        
        if not assembly_result.is_success:
            print(f"Error assembling video: {str(assembly_result.error)}")
        else:
            print(f"\n✅ Success! Video created at {output_path}")
            print(f"Duration: {narrative_plan['total_duration']} seconds")
        
        # Save complete test summary
        summary = {
            "status": "success",
            "video_id": video_id,
            "output_path": output_path,
            "components": {
                "narrative_plan": {
                    "segments": len(narrative_plan['segments']),
                    "duration": narrative_plan['total_duration']
                },
                "clip_selection": {
                    "clips": len(clips_data['clips'])
                },
                "text_generation": {
                    "texts": len(texts_data['texts'])
                }
            }
        }
        with open(test_dir / "test_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nComplete test logs saved to: {test_dir}")
        return summary
        
    except Exception as e:
        import traceback
        print(f"\nError during component test: {str(e)}")
        print(traceback.format_exc())
        
        # Save error information
        with open(test_dir / "error.txt", "w") as f:
            f.write(f"Error: {str(e)}\n\n")
            f.write(traceback.format_exc())


if __name__ == "__main__":
    # Chess video and prompt
    youtube_url = "https://youtu.be/t2laAqRtAq0?si=rM50co1i_4GMVx-X"
    user_prompt = "create a reel with hook saying Did you really thaugh that gukesh is the real owrkld chess champion. And then fill up appropriate clips from this video and some narrations text."
    
    # Run the component test
    test_components(youtube_url, user_prompt, "dramatic", 4)
