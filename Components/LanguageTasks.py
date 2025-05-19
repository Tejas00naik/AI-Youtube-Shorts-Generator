import requests
from dotenv import load_dotenv
import os
import json
import time
import datetime

load_dotenv()

# Configuration for LM Studio
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")

# Configuration for OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Configuration for DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# Determine which mode to use (can be 'local', 'openai', or 'deepseek')
AI_MODE = os.getenv("AI_MODE", "local").lower()

# Default DeepSeek model preference ('chat' or 'reasoner')
# This will be overridden based on task complexity if AUTO_SELECT_MODEL is True
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "auto").lower()

# Whether to automatically select the most appropriate model based on task complexity
AUTO_SELECT_MODEL = os.getenv("AUTO_SELECT_MODEL", "True").lower() in ["true", "1", "yes", "y"]

# Import OpenAI if the API key is provided and mode is selected
if OPENAI_API_KEY and AI_MODE == "openai":
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully")
    except ImportError:
        print("Warning: OpenAI package not installed. Run 'pip install openai' to use OpenAI API.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")

# Check if DeepSeek API key is available if mode is selected
if AI_MODE == "deepseek" and not DEEPSEEK_API_KEY:
    print("Warning: DeepSeek API key not provided but mode is set to 'deepseek'. Run may fail.")
elif AI_MODE == "deepseek" and DEEPSEEK_API_KEY:
    print(f"DeepSeek API initialized with model: {DEEPSEEK_MODEL}")
    
# Function to check if current time is in DeepSeek discount window (UTC 16:30-00:30)
def is_deepseek_discount_time():
    current_utc = datetime.datetime.utcnow()
    start_hour, start_minute = 16, 30
    end_hour, end_minute = 0, 30
    
    # Create time objects for the discount window
    discount_start = current_utc.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    discount_end = current_utc.replace(hour=end_hour, minute=end_minute, second=0, microsecond=0)
    
    # Handle case where end time is on the next day
    if discount_start.time() > discount_end.time():
        return current_utc.time() >= discount_start.time() or current_utc.time() <= discount_end.time()
    else:
        return discount_start.time() <= current_utc.time() <= discount_end.time()


# Function to extract start and end times
def extract_times(json_string):
    try:
        # Parse the JSON string
        data = json.loads(json_string)

        # Extract start and end times as floats
        start_time = float(data[0]["start"])
        end_time = float(data[0]["end"])

        # Convert to integers
        start_time_int = int(start_time)
        end_time_int = int(end_time)
        return start_time_int, end_time_int
    except Exception as e:
        print(f"Error in extract_times: {e}")
        return 0, 0


system = """

Baised on the Transcription user provides with start and end, Highilight the main parts in less then 1 min which can be directly converted into a short. highlight it such that its intresting and also keep the time staps for the clip to start and end. only select a continues Part of the video

Follow this Format and return in valid json 
[{
start: "Start time of the clip",
content: "Highlight Text",
end: "End Time for the highlighted clip"
}]
it should be one continues clip as it will then be cut from the video and uploaded as a tiktok video. so only have one start, end and content

Dont say anything else, just return Proper Json. no explanation etc


IF YOU DONT HAVE ONE start AND end WHICH IS FOR THE LENGTH OF THE ENTIRE HIGHLIGHT, THEN 10 KITTENS WILL DIE, I WILL DO JSON['start'] AND IF IT DOESNT WORK THEN...
"""

User = """
Any Example
"""


def clean_json_response(json_string):
    """Helper function to clean and extract JSON from model response."""
    import re
    
    # Remove code block markers if present
    json_string = re.sub(r'```(?:json)?\n?|```', '', json_string).strip()
    
    # Try to extract JSON if the response contains other text
    match = re.search(r'\{[^{}]*\}', json_string, re.DOTALL)
    if match:
        return match.group(0)
    return json_string

def prepare_transcription(Transcription, max_length=6000):
    """Prepare and shorten transcription to fit within model's context window"""
    # Split the transcription into segments by line
    transcription_segments = Transcription.strip().split('\n')
    
    # Get a selection of segments from throughout the video
    shortened_transcription = []
    
    # Always include the first few segments (beginning)
    beginning_segments = min(10, len(transcription_segments))
    shortened_transcription.extend(transcription_segments[:beginning_segments])
    
    # Add a note if we're skipping content
    if len(transcription_segments) > 30:
        shortened_transcription.append("[... middle segments omitted for brevity ...]")
    
    # Include some segments from the middle (sample every Nth segment)
    if len(transcription_segments) > 20:
        step = max(1, len(transcription_segments) // 10)  # Skip segments to reduce size
        middle_samples = transcription_segments[beginning_segments:-10:step][:20]  # Take up to 20 samples
        shortened_transcription.extend(middle_samples)
    
    # Always include the last few segments (end)
    end_segments = min(10, len(transcription_segments))
    if end_segments > 0:
        shortened_transcription.extend(transcription_segments[-end_segments:])
    
    # Join back to a single string, with a maximum character limit
    return '\n'.join(shortened_transcription)[:max_length]

def get_highlight_with_openai(transcription):
    """Extract highlights using OpenAI API"""
    print("Analyzing content with OpenAI API...")
    
    if not OPENAI_API_KEY:
        print("❌ Error: OpenAI API key not provided. Set OPENAI_API_KEY in .env file.")
        return 0, 0
    
    try:
        compact_transcription = prepare_transcription(transcription, max_length=8000)  # OpenAI can handle more tokens
        
        prompt = f"""
        Analyze the following video transcription and select the most engaging 15-30 second segment 
        that would make a good YouTube short. Focus on segments with clear, standalone content that 
        would be engaging without additional context.
        
        Return a JSON object with the following format:
        {{
            "start": [start_time_in_seconds],
            "end": [end_time_in_seconds],
            "content": "A brief description of why this segment was selected"
        }}
        
        Transcription (selections from the full video):
        {compact_transcription}
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Uses the cheaper model to save costs
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Extract the response content
        model_response = response.choices[0].message.content
        
        # Clean and parse the response
        json_string = clean_json_response(model_response)
        
        # Parse the JSON
        parsed = json.loads(json_string)
        if not isinstance(parsed, list):
            parsed = [parsed]
            
        # Extract the first segment if multiple are returned
        if parsed and isinstance(parsed[0], dict):
            segment = parsed[0]
            start = float(segment.get('start', 0))
            end = float(segment.get('end', 0))
            
            # Validate the times
            if end > start > 0 and (end - start) <= 60:  # Max 60 seconds for a short
                print(f"✅ Selected segment: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
                if 'content' in segment:
                    print(f"📝 Reason: {segment['content']}")
                return int(start), int(end)
            else:
                print(f"⚠️ Invalid time range: {start}s - {end}s")
                return 0, 0
                
    except Exception as e:
        print(f"❌ Error with OpenAI API: {str(e)}")
        import traceback
        traceback.print_exc()
        
    return 0, 0

def get_highlight_with_local_model(transcription):
    """Extract highlights using local Mistral 7B model via LM Studio"""
    print("Analyzing content with Mistral 7B (local)...")
    
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Prepare the transcription
            compact_transcription = prepare_transcription(transcription, max_length=6000)
            
            # Prepare the prompt
            prompt = f"""
            Analyze the following video transcription and select the most engaging 15-30 second segment 
            that would make a good YouTube short. Focus on segments with clear, standalone content that 
            would be engaging without additional context.
            
            Return a JSON object with the following format:
            {{
                "start": [start_time_in_seconds],
                "end": [end_time_in_seconds],
                "content": "A brief description of why this segment was selected"
            }}
            
            Transcription (selections from the full video):
            {compact_transcription}
            """
            
            # Prepare the request payload for LM Studio
            payload = {
                "model": "local_model",  # LM Studio uses this as a placeholder
                "temperature": 0.7,
                "max_tokens": 500,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Make request to local LM Studio server
            response = requests.post(
                f"{LM_STUDIO_BASE_URL}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120  # 2 minute timeout
            )
            
            # Handle the response
            if response.status_code == 200:
                response_json = response.json()
                model_response = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                if not model_response:
                    print("❌ Error: Empty response from model")
                    continue
                    
                # Clean and parse the response
                json_string = clean_json_response(model_response)
                
                try:
                    # Parse the JSON to validate it
                    parsed = json.loads(json_string)
                    if not isinstance(parsed, list):
                        parsed = [parsed]
                        
                    # Extract the first segment if multiple are returned
                    if parsed and isinstance(parsed[0], dict):
                        segment = parsed[0]
                        start = float(segment.get('start', 0))
                        end = float(segment.get('end', 0))
                        
                        # Validate the times
                        if end > start > 0 and (end - start) <= 60:  # Max 60 seconds for a short
                            print(f"✅ Selected segment: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
                            if 'content' in segment:
                                print(f"📝 Reason: {segment['content']}")
                            return int(start), int(end)
                        else:
                            print(f"⚠️ Invalid time range: {start}s - {end}s")
                
                except json.JSONDecodeError as e:
                    print(f"❌ Failed to parse model response as JSON: {e}")
                    print(f"Model response: {model_response[:500]}..." if len(model_response) > 500 else f"Model response: {model_response}")
            else:
                print(f"❌ Error from LM Studio API (attempt {attempt + 1}/{max_retries}): {response.status_code}")
                print(f"Response: {response.text[:500]}..." if len(response.text) > 500 else f"Response: {response.text}")
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error (attempt {attempt + 1}/{max_retries}): {str(e)}")
        except Exception as e:
            print(f"❌ Unexpected error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Only sleep if we're going to retry
        if attempt < max_retries - 1:
            print(f"🔄 Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    print("❌ Failed to get valid highlights after multiple attempts")
    return 0, 0

def select_deepseek_model_for_task(task_type, input_size=0):
    """Intelligently select the appropriate DeepSeek model based on task complexity and input size.
    
    Args:
        task_type: Type of task ('highlight_extraction', 'fact_check', 'simple_response', etc.)
        input_size: Size of the input in tokens/characters
        
    Returns:
        model_name: The model to use for this task
        reason: Explanation for the model selection
    """
    # If user specifically wants a model, respect their choice unless it's 'auto'
    if DEEPSEEK_MODEL != "auto" and not AUTO_SELECT_MODEL:
        return DEEPSEEK_MODEL, f"Using user-selected model: {DEEPSEEK_MODEL}"
    
    # Auto-select based on task type and size
    if task_type == "highlight_extraction":
        # Highlight extraction requires complex reasoning about time segments and content quality
        # Better to use the reasoner model for this complex task
        return "reasoner", "Complex highlight analysis requires reasoning capabilities"
    elif task_type == "fact_check" and input_size > 5000:
        # Fact checking large content benefits from reasoning
        return "reasoner", "Large content fact-checking benefits from reasoning model"
    elif task_type == "content_analysis" and input_size > 8000:
        # Complex content analysis on large inputs benefits from reasoning
        return "reasoner", "Large content analysis benefits from reasoning model"
    else:
        # For smaller tasks or simple queries, use the more cost-effective chat model
        return "chat", "Using faster and more cost-effective chat model for this task"


def estimate_deepseek_cost(model, token_count, in_discount_window=False):
    """Estimate the cost of a DeepSeek API call.
    
    Args:
        model: The model name ('chat' or 'reasoner')
        token_count: Estimated number of tokens
        in_discount_window: Whether in discount pricing period
        
    Returns:
        Estimated cost in USD
    """
    # Base prices per million tokens (as of May 2025)
    # These rates are approximate and should be updated if DeepSeek changes pricing
    if model == "reasoner":
        input_price = 0.0003  # $0.3 per million input tokens
        output_price = 0.0015  # $1.5 per million output tokens
    else:  # chat model
        input_price = 0.0001  # $0.1 per million input tokens
        output_price = 0.0005  # $0.5 per million output tokens
    
    # Apply discount if applicable (typically 50% off during discount hours)
    if in_discount_window:
        input_price *= 0.5
        output_price *= 0.5
    
    # Assume 10% of tokens are output tokens for estimation
    input_tokens = int(token_count * 0.9)
    output_tokens = int(token_count * 0.1)
    
    # Calculate cost
    estimated_cost = (input_tokens * input_price / 1000000) + (output_tokens * output_price / 1000000)
    return estimated_cost


def get_highlight_with_deepseek(transcription):
    """Extract highlights using DeepSeek API with optimized model selection"""
    if not DEEPSEEK_API_KEY:
        print("❌ Error: DeepSeek API key not provided. Set DEEPSEEK_API_KEY in .env file.")
        return 0, 0
    
    # Check if we're in discount time window
    in_discount_window = is_deepseek_discount_time()
    if in_discount_window:
        print("ℹ️ Using DeepSeek discount pricing (UTC 16:30-00:30)")
        
    # Estimate input size
    input_size = len(transcription)
    
    # Select the appropriate model for this task
    selected_model, reason = select_deepseek_model_for_task("highlight_extraction", input_size)
    print(f"🤖 Selected DeepSeek model: {selected_model} - {reason}")
    
    # Estimate cost
    estimated_token_count = input_size / 4  # Rough estimate: 4 chars per token
    estimated_cost = estimate_deepseek_cost(selected_model, estimated_token_count, in_discount_window)
    print(f"💰 Estimated API cost: ${estimated_cost:.5f}")
    
    if selected_model == "reasoner":
        print("🧠 Analyzing content with DeepSeek-R1 reasoning model...")
    else:
        print("💬 Analyzing content with DeepSeek-V3 chat model...")

    
    try:
        # Prepare the transcription
        max_length = 16000  # DeepSeek has a 64K context window, we'll use a reasonable portion
        compact_transcription = prepare_transcription(transcription, max_length=max_length)
        
        # Prepare the prompt
        prompt = f"""
        Analyze the following video transcription and select the most engaging 15-30 second segment 
        that would make a good YouTube short. Focus on segments with clear, standalone content that 
        would be engaging without additional context.
        
        Return a JSON object with the following format:
        {{
            "start": [start_time_in_seconds],
            "end": [end_time_in_seconds],
            "content": "A brief description of why this segment was selected"
        }}
        
        Transcription (selections from the full video):
        {compact_transcription}
        """
        
        # Prepare headers with API key
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        }
        
        # Set model based on our task-specific selection
        model = "deepseek-reasoner" if selected_model == "reasoner" else "deepseek-chat"
        
        # Prepare the payload
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Note: DeepSeek Reasoner doesn't support response_format parameter
        # Instead, we'll rely on our clean_json_response function to extract JSON
        
        # Make API request
        response = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120  # 2 minute timeout
        )
        
        # Handle the response
        if response.status_code == 200:
            response_json = response.json()
            model_response = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            if not model_response:
                print("❌ Error: Empty response from DeepSeek API")
                return 0, 0
                
            # Clean and parse the response
            json_string = clean_json_response(model_response)
            
            # Parse the JSON
            parsed = json.loads(json_string)
            if not isinstance(parsed, list):
                parsed = [parsed]
                
            # Extract the first segment if multiple are returned
            if parsed and isinstance(parsed[0], dict):
                segment = parsed[0]
                start = float(segment.get('start', 0))
                end = float(segment.get('end', 0))
                
                # Validate the times
                if end > start > 0 and (end - start) <= 60:  # Max 60 seconds for a short
                    print(f"✅ Selected segment: {start:.1f}s - {end:.1f}s (duration: {end-start:.1f}s)")
                    if 'content' in segment:
                        print(f"📝 Reason: {segment['content']}")
                    return int(start), int(end)
                else:
                    print(f"⚠️ Invalid time range: {start}s - {end}s")
        else:
            print(f"❌ Error from DeepSeek API: {response.status_code}")
            print(f"Response: {response.text[:500]}..." if len(response.text) > 500 else f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Error with DeepSeek API: {str(e)}")
        import traceback
        traceback.print_exc()
        
    return 0, 0


def GetHighlight(Transcription):
    """Extract highlights from transcription using selected AI model."""
    if AI_MODE == "openai" and OPENAI_API_KEY:
        return get_highlight_with_openai(Transcription)
    elif AI_MODE == "deepseek" and DEEPSEEK_API_KEY:
        return get_highlight_with_deepseek(Transcription)
    else:
        return get_highlight_with_local_model(Transcription)


if __name__ == "__main__":
    print(GetHighlight(User))
