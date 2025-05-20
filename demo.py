#!/usr/bin/env python3
"""
Demo script for AI YouTube Shorts Generator.

This script demonstrates the core functionality of the AI YouTube Shorts Generator,
taking a user query and a YouTube transcript to generate a narrative plan.
"""

import os
import sys
import json
import logging
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

def print_segment(segment, index):
    """Print a formatted segment for better readability."""
    print(f"\n{'-' * 50}")
    print(f"SEGMENT {index}: {segment['type'].upper()}")
    print(f"{'-' * 50}")
    print(f"⏱️  {segment['start_time']}s - {segment['end_time']}s")
    print(f"📝 Description: {segment['description']}")
    print(f"🗣️  Text: \"{segment['text']}\"")
    print(f"😀 Mood: {segment['mood']}")
    print(f"{'-' * 50}")

def print_narrative_plan(plan):
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

def demo(user_query, transcript):
    """Run the AI YouTube Shorts Generator pipeline with the given inputs."""
    # Initialize the config
    init_config()
    
    logger.info(f"Processing user query: {user_query}")
    logger.info(f"Transcript length: {len(transcript)} characters")
    
    # Process the input through the pipeline
    result = process_input(user_query, transcript)
    
    if result.is_success:
        print("\n✅ Successfully generated narrative plan!")
        print_narrative_plan(result.value)
        return result.value
    else:
        print("\n❌ Error generating narrative plan:")
        print(f"Error: {result.error.message}")
        print(f"Code: {result.error.code}")
        print(f"Severity: {result.error.severity}")
        return None

def sample_transcript():
    """Return a sample transcript for demo purposes."""
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

if __name__ == "__main__":
    # Check if command line arguments were provided
    if len(sys.argv) > 1:
        user_query = sys.argv[1]
        transcript_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        # Read transcript from file if provided
        if transcript_file and os.path.exists(transcript_file):
            with open(transcript_file, 'r') as f:
                transcript = f.read()
        else:
            transcript = sample_transcript()
    else:
        # Use default inputs
        user_query = "Create a fun 30-second highlight video about chocolate cake baking for beginners"
        transcript = sample_transcript()
    
    # Run the demo
    demo(user_query, transcript)
