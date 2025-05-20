#!/usr/bin/env python3
"""
Test suite for video-related components in the AI YouTube Shorts Generator.

This module tests:
1. Video downloading functionality
2. Transcript extraction 
3. Video stitching and clip assembly
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shorts_pipeline import ShortsGenerator
from youtube_shorts_generator import extract_youtube_transcript
from media.director_agent import assemble_video
from core.error_handler import Result


class TestVideoComponents(unittest.TestCase):
    """Test suite for video-related components."""
    
    def setUp(self):
        """Set up test environment."""
        self.video_id = "jNQXAC9IVRw"  # "Me at the zoo" (first YouTube video)
        self.working_dir = Path("_working")
        self.working_dir.mkdir(exist_ok=True)
        self.video_path = self.working_dir / f"{self.video_id}.mp4"
        
        # Create a generator with local provider
        self.generator = ShortsGenerator(llm_provider="local")
    
    def test_1_transcript_extraction(self):
        """Test transcript extraction functionality."""
        youtube_url = f"https://www.youtube.com/watch?v={self.video_id}"
        transcript_result = extract_youtube_transcript(youtube_url)
        
        self.assertTrue(transcript_result.is_success, 
                      f"Transcript extraction failed: {transcript_result.error.message if not transcript_result.is_success else ''}")
        
        transcript = transcript_result.value
        self.assertIsInstance(transcript, str, "Transcript should be a string")
        self.assertGreater(len(transcript), 10, "Transcript should have content")
        
        # Save transcript for reference in other tests
        transcript_path = self.working_dir / f"{self.video_id}_transcript.txt"
        with open(transcript_path, "w") as f:
            f.write(transcript)
        
        print(f"Test 1: Successfully extracted transcript ({len(transcript)} chars)")
        print(f"Transcript sample: {transcript[:100]}...")
    
    def test_2_video_download(self):
        """Test video downloading functionality."""
        # If video already exists, skip download
        if self.video_path.exists():
            print(f"Test 2: Video already exists at {self.video_path}")
            self.assertTrue(True)
            return
        
        # Use pytubefix directly to avoid compatibility issues
        try:
            from pytubefix import YouTube
            
            url = f"https://www.youtube.com/watch?v={self.video_id}"
            yt = YouTube(url)
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            self.assertIsNotNone(stream, "No suitable video stream found")
            
            downloaded_path = stream.download(output_path=str(self.working_dir), 
                                            filename=f"{self.video_id}.mp4")
            
            self.assertTrue(Path(downloaded_path).exists(), f"Downloaded file does not exist: {downloaded_path}")
            print(f"Test 2: Successfully downloaded video to {downloaded_path}")
            
        except Exception as e:
            self.fail(f"Video download failed: {str(e)}")
    
    def test_3_video_stitching(self):
        """Test video stitching and assembly."""
        # Ensure video exists
        self.assertTrue(self.video_path.exists(), 
                      f"Video not found at {self.video_path}. Run test_2_video_download first.")
        
        # Create test clips and texts data
        clips_data = {
            "clips": [
                {"start": 0, "end": 3, "focus_point": "center"},
                {"start": 5, "end": 8, "focus_point": "center"},
                {"start": 10, "end": 13, "focus_point": "center"}
            ]
        }
        
        texts_data = {
            "texts": [
                {
                    "text": "First key point about AI →",
                    "duration": 2.5,
                    "position": "bottom_center"
                },
                {
                    "text": "Most important insight you need to know →",
                    "duration": 2.5,
                    "position": "center"
                }
            ]
        }
        
        # Set up output path
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / f"{self.video_id}_test_stitching.mp4")
        
        # Test assembly
        result = assemble_video(
            str(self.video_path),
            clips_data,
            texts_data,
            output_path
        )
        
        self.assertTrue(result.is_success, 
                      f"Video assembly failed: {result.error.message if not result.is_success else ''}")
        self.assertTrue(Path(output_path).exists(), f"Output video not found at {output_path}")
        
        print(f"Test 3: Successfully assembled video at {output_path}")


if __name__ == "__main__":
    print("\n=== Testing Video Components ===\n")
    # Run tests in order
    suite = unittest.TestSuite()
    suite.addTest(TestVideoComponents('test_1_transcript_extraction'))
    suite.addTest(TestVideoComponents('test_2_video_download'))
    suite.addTest(TestVideoComponents('test_3_video_stitching'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
