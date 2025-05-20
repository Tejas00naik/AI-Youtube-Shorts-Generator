"""
Prompt templates for LLM interactions in the AI YouTube Shorts Generator.

This module contains optimized prompts for each stage of the content creation pipeline:
1. Input Analysis - Analyzes user input to determine optimal parameters
2. Narrative Planning - Creates a storyboard structure with timing and segments
3. Script Writing - Generates engaging text overlays and captions
"""

# Input Analysis Stage Templates
INPUT_ANALYZER_PROMPT = '''
You're a YouTube Shorts strategist analyzing:
User Goal: "{user_input}"
Video Length: {transcript_length_seconds:.1f}s

Apply these viral content rules:
1. First 3s must have pattern interrupt (shocking fact/visual)
2. Strategic pacing: longer segments for key content, minimal transitions
3. Max 2 key ideas for <60s videos prioritizing quality over quantity
4. High retention = letting compelling content play WITHOUT interruption

Analyze the user's goal and determine optimal video parameters.

Output ONLY a JSON object with these fields:
{{
  "clip_count": int (default 2, max 4 only for complex topics),
  "interruption_style": "pause" or "continuous",
  "interruption_frequency": int (default 1, max 2 - primarily at beginning/end),
  "max_duration": float (target duration in seconds, 30-59.5),
  "reasoning": "brief explanation of choices"
}}
'''

# Narrative Planning Stage Templates
NARRATIVE_PLANNER_PROMPT = '''
You are a professional YouTube Shorts editor creating a storyboard for:
User Goal: "{user_directions}"
Transcript: ```{transcript}```

Create a streamlined narrative plan using this proven viral structure:

1. OPENING HOOK (0-{first_segment}s): 
   - Start with the most engaging/shocking quote from transcript
   - Must include an initial text overlay to set context

2. UNINTERRUPTED CONTENT BLOCKS (middle):
   - Use {clip_count} compelling content segments (default is 2 for maximum impact)
   - Keep each clip 5-15 seconds for optimal engagement
   - Let powerful/contradictory moments play without interruption
   - Only break for text when absolutely necessary
   - Ensure each clip captures COMPLETE thoughts/sentences

3. OPTIONAL CONCLUSION:
   - If using more than 2 clips, end with a powerful statement
   - A closing call-to-action text overlay is optional

Technical Rules:
- Never exceed {max_duration}s total
- Use AT MOST {interruption_frequency} text interruptions (1-2 maximum)
- ALWAYS include opening hook text overlay
- Outro text overlay is OPTIONAL and only if specifically relevant
- Middle interruptions should be AVOIDED unless absolutely necessary
- Focus on {tone} tone throughout
- CRITICAL: Each clip must end 0.5-1.0s AFTER speaker completes their sentence
- CRITICAL: NO OVERLAPPING CLIPS - each clip must end before the next begins
- CRITICAL: First action segment MUST start at 0.0 seconds

Output ONLY valid JSON in this exact format:
{{
  "segments": [
    {{
      "type": "action",
      "start_time": float,
      "end_time": float,
      "content": "Exact transcript text to use"
    }},
    {{
      "type": "{interruption_style}",
      "duration": float,
      "content": "Text to show during pause"
    }},
    ... (primarily action segments with strategic pauses)
  ],
  "total_duration": float
}}

Let powerful content speak for itself - only interrupt when it truly enhances understanding.
'''

# Script Writing Stage Templates
SCRIPT_WRITER_PROMPT = '''
You're crafting strategic text overlays for a YouTube Shorts video about:
Topic: "{video_topic}"

Create AT MOST {interruption_frequency} impactful text overlays (1-2 maximum) following this strategic placement:

1. OPENING CONTEXT (REQUIRED first overlay):  
   - "What [authority/group] doesn't want you to see →"  
   - "The [adjective] truth about [topic] →"  
   - "Watch closely: [provocative claim] →"

2. CLOSING CTA (OPTIONAL final overlay):  
   - "Follow for more [topic] exposés"  
   - "Save this 👆 Share with anyone who needs to see it"  
   - "More revelations coming soon →"

3. AVOID MID-VIDEO INTERRUPTIONS unless specifically requested by user

Rules for maximum impact:
- Max 40 characters per overlay
- Use emojis sparingly and only when they add meaning
- Opening text should set clear context
- Middle interruptions ONLY if they clarify something crucial
- Closing text must drive engagement action
- Use "{tone}" tone throughout

Output ONLY a JSON array of text overlays:
{{
  "texts": [
    {{
      "text": "Your overlay text here →",
      "position": "bottom_center",
      "duration": 2.5
    }},
    ... (exactly {interruption_frequency} overlays total)
  ]
}}
'''

# Platform-Specific Optimizations
PLATFORM_RULES = '''
YouTube Shorts Platform Rules:
- First 3 seconds are critical for retention
- Text should be centered and readable on mobile
- End with "Subscribe for more"
- Ask for engagement: "Comment what you think"
- Optimal length: 45-58 seconds
'''

# Psychological Triggers
PSYCHOLOGICAL_TRIGGERS = '''
Include these engagement triggers:
1. Curiosity Gaps:  
   "The secret most people miss →"  
   "Why experts always..."  

2. Social Proof:  
   "What champions do differently"  
   "How top performers..."  

3. Urgency/Exclusivity:  
   "Rarely seen technique"  
   "Behind-the-scenes insight"
'''

# Sample Viral Pattern Template
VIRAL_TEMPLATE = '''
[0-3s] 🔥 Hook with shocking statement/question
[3-8s] Establish the context/problem
[8-15s] Introduce the key insight
[15-30s] Show 2-3 examples/evidence
[30-45s] Deliver the main point/solution
[45-58s] Wrap with takeaway + subscribe call
'''
