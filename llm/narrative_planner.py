"""
Narrative Planner for AI YouTube Shorts Generator

This module generates a structured narrative plan using LLM based on user input
and video transcript.
"""
import json
import logging
from typing import Dict, List, Optional, Any
from enum import Enum
import openai
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure OpenAI
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')

class NarrativeMode(str, Enum):
    """Supported narrative modes for video generation."""
    HIGHLIGHT = "highlight"
    TUTORIAL = "tutorial"
    STORY = "story"
    PROMOTIONAL = "promotional"
    EDUCATIONAL = "educational"

class NarrativePlanner:
    """
    Generates a narrative plan using LLM based on user input and transcript.
    """
    
    # Default parameters for different narrative modes
    MODE_PARAMETERS = {
        NarrativeMode.HIGHLIGHT: {
            'max_segments': 5,
            'max_segment_duration': 15,  # seconds
            'min_segment_duration': 3,    # seconds
            'target_duration': 58,        # Max duration for YouTube Shorts
            'segment_types': ['hook', 'highlight_1', 'highlight_2', 'highlight_3', 'conclusion']
        },
        NarrativeMode.TUTORIAL: {
            'max_segments': 4,
            'max_segment_duration': 20,
            'min_segment_duration': 5,
            'target_duration': 58,
            'segment_types': ['introduction', 'step_1', 'step_2', 'conclusion']
        },
        NarrativeMode.STORY: {
            'max_segments': 6,
            'max_segment_duration': 15,
            'min_segment_duration': 5,
            'target_duration': 58,
            'segment_types': ['setup', 'conflict', 'rising_action', 'climax', 'falling_action', 'resolution']
        },
        NarrativeMode.PROMOTIONAL: {
            'max_segments': 4,
            'max_segment_duration': 15,
            'min_segment_duration': 3,
            'target_duration': 30,  # Shorter for promotional content
            'segment_types': ['hook', 'problem', 'solution', 'call_to_action']
        },
        NarrativeMode.EDUCATIONAL: {
            'max_segments': 5,
            'max_segment_duration': 15,
            'min_segment_duration': 5,
            'target_duration': 58,
            'segment_types': ['introduction', 'concept_1', 'example', 'concept_2', 'summary']
        }
    }
    
    def __init__(self, mode: str = NarrativeMode.HIGHLIGHT, tone: str = 'professional'):
        """
        Initialize the NarrativePlanner.
        
        Args:
            mode: Narrative mode (e.g., 'highlight', 'tutorial', 'story')
            tone: Tone of the narrative (e.g., 'professional', 'casual', 'fun')
        """
        try:
            self.mode = NarrativeMode(mode.lower())
        except ValueError:
            logger.warning(f"Invalid mode '{mode}', defaulting to 'highlight'")
            self.mode = NarrativeMode.HIGHLIGHT
            
        self.tone = tone.lower()
        self.parameters = self.MODE_PARAMETERS.get(self.mode, self.MODE_PARAMETERS[NarrativeMode.HIGHLIGHT])
        
    def generate_plan(self, transcript: str, user_params: Dict[str, Any]) -> Dict:
        """
        Generate a narrative plan based on the transcript and user parameters.
        
        Args:
            transcript: The full transcript of the video
            user_params: Additional parameters from the user input
            
        Returns:
            Dict containing the structured narrative plan
        """
        # Prepare the prompt for the LLM
        prompt = self._build_prompt(transcript, user_params)
        
        try:
            # Call the LLM to generate the narrative plan
            response = self._call_llm(prompt)
            
            # Parse and validate the response
            plan = self._parse_llm_response(response)
            
            # Add metadata and validate the plan
            plan['metadata'] = {
                'mode': self.mode.value,
                'tone': self.tone,
                'parameters': {**self.parameters, **user_params}
            }
            
            return self._validate_plan(plan)
            
        except Exception as e:
            logger.error(f"Error generating narrative plan: {str(e)}")
            return self._generate_fallback_plan(transcript, user_params)
    
    def _build_prompt(self, transcript: str, user_params: Dict[str, Any]) -> str:
        """Build the prompt for the LLM based on the transcript and parameters."""
        # Extract key information from user parameters
        topic = user_params.get('topic', 'the content')
        target_audience = user_params.get('target_audience', 'general audience')
        style = user_params.get('style', 'engaging')
        
        # Build the prompt
        prompt = f"""You are an expert video editor creating a {self.mode.value} video with a {self.tone} tone.
        
Video Details:
- Topic: {topic}
- Target Audience: {target_audience}
- Style: {style}
- Maximum Duration: {self.parameters['target_duration']} seconds
- Maximum Segments: {self.parameters['max_segments']}
- Maximum Segment Duration: {self.parameters['max_segment_duration']} seconds

Transcript:
"""
        prompt += f"{transcript[:8000]}"  # Limit transcript length to avoid token limits
        
        prompt += f"""

Based on the above transcript, create a narrative plan with the following segments:
"""
        
        # Add segment descriptions based on mode
        if self.mode == NarrativeMode.HIGHLIGHT:
            prompt += """- Hook: A captivating 3-5 second intro that grabs attention
- Highlight 1: First key moment (5-15s)
- Highlight 2: Second key moment (5-15s)
- Highlight 3: Third key moment (5-15s)
- Conclusion: A satisfying wrap-up (3-5s)"""
        elif self.mode == NarrativeMode.TUTORIAL:
            prompt += """- Introduction: Briefly introduce what will be taught (5-10s)
- Step 1: First step in the tutorial (10-20s)
- Step 2: Second step in the tutorial (10-20s)
- Conclusion: Summary and call to action (5-10s)"""
        # Add other modes as needed...
        
        prompt += """

Return the plan as a JSON object with the following structure:
{
  "segments": [
    {
      "type": "string",  // Segment type (e.g., 'hook', 'highlight_1')
      "start_time": float,  // Start time in seconds
      "end_time": float,    // End time in seconds
      "description": "string",  // Description of the segment
      "text": "string",        // Text to display/read for this segment
      "mood": "string"         // Suggested mood/emotion
    }
  ],
  "total_duration": float,  // Total duration in seconds
  "summary": "string"       // Brief summary of the narrative
}

Ensure the total duration does not exceed 58 seconds and each segment is meaningful and engaging.
"""
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt and return the response."""
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
            
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates structured video narrative plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            raise
    
    def _parse_llm_response(self, response: str) -> Dict:
        """Parse the LLM response into a structured format."""
        try:
            # Extract JSON from the response (handling potential markdown code blocks)
            json_str = response.strip()
            if '```json' in json_str:
                json_str = json_str.split('```json')[1].split('```')[0]
            elif '```' in json_str:
                json_str = json_str.split('```')[1].split('```')[0]
                
            plan = json.loads(json_str)
            return plan
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            logger.debug(f"Response content: {response}")
            raise ValueError("Failed to generate a valid narrative plan. Please try again.")
    
    def _validate_plan(self, plan: Dict) -> Dict:
        """Validate the generated narrative plan."""
        required_keys = ['segments', 'total_duration', 'summary']
        for key in required_keys:
            if key not in plan:
                raise ValueError(f"Invalid plan: Missing required key '{key}'")
        
        # Validate segments
        if not isinstance(plan['segments'], list) or len(plan['segments']) == 0:
            raise ValueError("Invalid plan: No segments found")
            
        # Validate total duration
        max_duration = self.parameters['target_duration']
        if plan['total_duration'] > max_duration + 2:  # Allow 2-second tolerance
            logger.warning(f"Plan duration ({plan['total_duration']}s) exceeds maximum ({max_duration}s)")
            # Optionally: Trim the plan to fit duration
            
        return plan
    
    def _generate_fallback_plan(self, transcript: str, user_params: Dict[str, Any]) -> Dict:
        """Generate a simple fallback plan if LLM generation fails."""
        logger.warning("Generating fallback narrative plan")
        
        # Simple logic to split transcript into chunks
        words = transcript.split()
        chunk_size = max(10, len(words) // 3)  # Split into 3 segments
        
        segments = []
        start_time = 0
        duration_per_segment = min(15, (user_params.get('duration') or 30) // 3)
        
        for i in range(3):
            if i * chunk_size >= len(words):
                break
                
            segment_text = ' '.join(words[i*chunk_size:(i+1)*chunk_size])
            end_time = start_time + duration_per_segment
            
            segments.append({
                'type': f'segment_{i+1}',
                'start_time': start_time,
                'end_time': end_time,
                'description': f'Part {i+1} of the content',
                'text': segment_text,
                'mood': 'neutral'
            })
            
            start_time = end_time
        
        return {
            'segments': segments,
            'total_duration': sum(s['end_time'] - s['start_time'] for s in segments),
            'summary': 'Automatically generated fallback plan',
            'metadata': {
                'mode': self.mode.value,
                'tone': self.tone,
                'parameters': {**self.parameters, **user_params},
                'is_fallback': True
            }
        }


def generate_narrative_plan(
    transcript: str,
    mode: str = 'highlight',
    tone: str = 'professional',
    **user_params
) -> Dict:
    """
    Generate a narrative plan for the given transcript and parameters.
    
    Args:
        transcript: The full transcript of the video
        mode: Narrative mode (e.g., 'highlight', 'tutorial', 'story')
        tone: Tone of the narrative (e.g., 'professional', 'casual', 'fun')
        **user_params: Additional parameters for the narrative
        
    Returns:
        Dict containing the structured narrative plan
    """
    planner = NarrativePlanner(mode=mode, tone=tone)
    return planner.generate_plan(transcript, user_params)


if __name__ == "__main__":
    # Example usage
    test_transcript = """
    In today's video, I'll show you how to create amazing content with AI. 
    First, we'll start with the basics of content creation. Then, we'll move on to 
    more advanced techniques. Finally, I'll share some pro tips to make your 
    content stand out from the crowd. Let's get started!"""
    
    try:
        plan = generate_narrative_plan(
            transcript=test_transcript,
            mode="tutorial",
            tone="professional",
            topic="AI Content Creation",
            target_audience="beginners",
            style="clear and concise"
        )
        print(json.dumps(plan, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
