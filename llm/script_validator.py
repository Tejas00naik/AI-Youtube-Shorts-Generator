"""
Script Validator for AI YouTube Shorts Generator.

This module provides validation for generated scripts and narrative plans,
allowing the system to verify the quality and correctness of AI-generated content
before proceeding to the next stage in the pipeline.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union, Tuple

from core.error_handler import Result
from llm.llm_client import LLMClient, get_llm_client

# Configure logging
logger = logging.getLogger(__name__)

class ScriptValidator:
    """
    Validates scripts and narrative plans using a secondary LLM call.
    Uses DeepSeek-R1 for initial generation and regular DeepSeek for validation.
    """
    
    def __init__(self, generation_client=None, validation_client=None):
        """
        Initialize the script validator with separate clients for generation and validation.
        
        Args:
            generation_client: LLM client for script generation (DeepSeek-R1 recommended)
            validation_client: LLM client for script validation (regular DeepSeek)
        """
        # For script generation: prefer DeepSeek-R1 if available
        self.generation_client = generation_client or get_llm_client(
            provider="deepseek",
            model=os.getenv("DEEPSEEK_R1_MODEL", "deepseek-coder")
        )
        
        # For validation: use standard DeepSeek model
        self.validation_client = validation_client or get_llm_client(
            provider="deepseek",
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        )
    
    def _perform_basic_validation(self, narrative_plan: Dict[str, Any], validation_criteria: List[str]) -> Result:
        """
        Perform basic validation checks on the narrative plan before LLM validation.
        This ensures critical constraints like clip count and pause limits are enforced.
        
        Args:
            narrative_plan: The narrative plan to validate
            validation_criteria: List of criteria to check against
            
        Returns:
            Result object with success or validation failures
        """
        segments = narrative_plan.get('segments', [])
        if not segments:
            return Result.failure("Narrative plan has no segments")
        
        # Count action and pause segments
        action_segments = [s for s in segments if s.get('type') == 'action']
        pause_segments = [s for s in segments if s.get('type') == 'pause']
        
        issues = []
        
        # Check clip count (should be 2-4 with 2 being preferred)
        # Extract expected clip count from validation criteria if available
        expected_clip_count = 2  # Default to 2
        for criterion in validation_criteria:
            if "action segments" in criterion.lower():
                # Parse the expected count from criteria like "Exactly 4 action segments"
                try:
                    import re
                    match = re.search(r'Exactly (\d+) action segments', criterion)
                    if match:
                        expected_clip_count = int(match.group(1))
                except:
                    pass
        
        # Check if we have the right number of action segments
        if len(action_segments) != expected_clip_count:
            issues.append(f"Incorrect number of action segments: found {len(action_segments)}, expected {expected_clip_count}")
        
        # Check if we have 1-2 pause segments max
        max_pause_segments = 2
        if len(pause_segments) > max_pause_segments:
            issues.append(f"Too many pause segments: found {len(pause_segments)}, maximum allowed is {max_pause_segments}")
        
        # Check if first segment is an action segment starting at 0.0
        if not segments or segments[0].get('type') != 'action':
            issues.append("First segment must be an action segment")
        elif segments[0].get('type') == 'action' and segments[0].get('start_time', -1) != 0.0:
            issues.append(f"First action segment must start at 0.0, found {segments[0].get('start_time')}")
        
        # Check for overlapping segments
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]
            
            if current.get('type') == 'action' and next_seg.get('type') == 'action':
                if current.get('end_time', 0) > next_seg.get('start_time', 0):
                    issues.append(f"Overlapping segments found: segment {i} ends at {current.get('end_time')}, but segment {i+1} starts at {next_seg.get('start_time')}")
        
        if issues:
            return Result.failure("Basic validation failed:\n- " + "\n- ".join(issues))
        
        return Result.success(True)
    
    def generate_and_validate_narrative_plan(self, 
                                           system_prompt: str,
                                           user_messages: Union[str, List[Dict[str, str]]],
                                           validation_criteria: List[str],
                                           max_attempts: int = 3) -> Result:
        """
        Generate a narrative plan and validate it against criteria.
        
        Args:
            system_prompt: System prompt for generation
            user_messages: User messages for generation
            validation_criteria: List of criteria the plan must meet
            max_attempts: Maximum number of generation attempts
            
        Returns:
            Result object with validated narrative plan or error
        """
        for attempt in range(max_attempts):
            try:
                logger.info(f"Generating narrative plan (attempt {attempt+1}/{max_attempts})")
                
                # Generate initial plan
                generation_result = self.generation_client.chat_completion(
                    system_prompt=system_prompt,
                    user_messages=user_messages,
                    temperature=0.7,
                    json_response=True
                )
                
                if "error" in generation_result:
                    logger.error(f"Generation error: {generation_result['error']}")
                    if attempt == max_attempts - 1:
                        return Result.failure(f"Failed to generate narrative plan: {generation_result['error']}")
                    continue
                
                # Parse the generated plan
                try:
                    plan_content = generation_result.get("content", "{}")
                    narrative_plan = json.loads(plan_content)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in generation response: {str(e)}")
                    if attempt == max_attempts - 1:
                        return Result.failure(f"Invalid JSON in generation response: {str(e)}")
                    continue
                
                # Validate the plan
                validation_result = self._validate_narrative_plan(narrative_plan, validation_criteria)
                
                if validation_result.is_success:
                    logger.info(f"Narrative plan validated successfully on attempt {attempt+1}")
                    return Result.success(narrative_plan)
                else:
                    logger.warning(f"Validation failed on attempt {attempt+1}: {validation_result.error}")
                    
                    # Store this attempt as best so far if it's the first one or better than previous
                    if not hasattr(self, '_best_attempt') or not self._best_attempt:
                        self._best_attempt = {'plan': narrative_plan, 'error': validation_result.error}
                    
                    # If this is the last attempt, either return the best attempt or the failure
                    if attempt == max_attempts - 1:
                        logger.warning(f"Returning best attempt despite validation issues: {validation_result.error}")
                        
                        # Perform a final fix on critical issues before returning
                        try:
                            fixed_plan = self._fix_critical_issues(narrative_plan, validation_criteria)
                            logger.info("Applied fixes to critical issues in narrative plan")
                            return Result.success(fixed_plan)
                        except Exception as fix_error:
                            logger.warning(f"Could not fix critical issues: {str(fix_error)}")
                            return Result.success(narrative_plan)
                
                # Update system prompt with validation feedback for next attempt
                system_prompt += f"\n\nPrevious attempt failed validation: {validation_result.error}" 
                system_prompt += "\nPlease fix these issues in your next attempt."
                system_prompt += "\n\nCRITICAL REQUIREMENTS:"  
                system_prompt += "\n1. Use exactly 2 action clips (or the specified number in criteria)"  
                system_prompt += "\n2. Maximum 2 pause segments"  
                system_prompt += "\n3. First segment must be an action segment starting at 0.0"  
                system_prompt += "\n4. Segments must alternate appropriately"
                system_prompt += "\n\nAdditional guidance:"
                system_prompt += "\n1. Ensure clips don't overlap - each clip should end before the next one starts"
                system_prompt += "\n2. Allow a small buffer at the end of clips to avoid cutting off speech"
                system_prompt += "\n3. Check that segment counts exactly match the requirements"
                system_prompt += "\n4. Verify that the total duration is reasonable and doesn't exceed limits"
            
            except Exception as e:
                logger.error(f"Error in plan generation/validation: {str(e)}")
                if attempt == max_attempts - 1:
                    return Result.failure(f"Error in plan generation/validation: {str(e)}")
        
        # Should not reach here, but just in case
        return Result.failure("Failed to generate and validate narrative plan after maximum attempts")
    
    def _validate_narrative_plan(self, narrative_plan: Dict[str, Any], validation_criteria: List[str]) -> Result:
        """
        Validate a narrative plan against specific criteria.
        
        Args:
            narrative_plan: The narrative plan to validate
            validation_criteria: List of criteria the plan must meet
            
        Returns:
            Result object with success or validation failures
        """
        # First perform basic structure validation ourselves to ensure clip count and pause constraints
        basic_validation_result = self._perform_basic_validation(narrative_plan, validation_criteria)
        if not basic_validation_result.is_success:
            return basic_validation_result
        
        # Build validation prompt for the LLM-based validation
        validation_prompt = f"""
        You are a quality control expert for YouTube Shorts scripts. 
        
        Evaluate this narrative plan against the following criteria:
        {json.dumps(validation_criteria, indent=2)}
        
        Narrative plan to validate:
        {json.dumps(narrative_plan, indent=2)}
        
        SPECIAL INSTRUCTIONS:
        1. CRITICAL: The narrative plan MUST have no more than 2-4 action clips (2 is preferred)
        2. CRITICAL: The narrative plan MUST have a maximum of 2 pause segments
        3. The first segment MUST be an action segment starting at 0.0
        4. Segments MUST alternate appropriately based on interruption style
        5. Each segment MUST have complete thoughts/sentences
        6. NO OVERLAPPING TIMESTAMPS - clips must end before next begins
        
        Instructions:
        1. Analyze the narrative plan objectively
        2. Verify that ALL validation criteria are met
        3. For each criterion, mark it as PASS or FAIL
        4. Provide specific feedback for any failed criteria
        
        Respond in JSON format with:
        {{"validation": "pass" or "fail", "issues": [list of specific issues], "feedback": "overall feedback"}}
        """
        
        try:
            # Run validation with secondary LLM
            validation_response = self.validation_client.chat_completion(
                system_prompt=validation_prompt,
                user_messages="Validate this narrative plan against the criteria",
                temperature=0.3,
                json_response=True
            )
            
            if "error" in validation_response:
                return Result.failure(f"Validation failed: {validation_response['error']}")
            
            # Parse validation results
            try:
                validation_content = validation_response.get("content", "{}")
                validation_results = json.loads(validation_content)
                
                # Check if validation passed
                if validation_results.get("validation", "").lower() == "pass":
                    return Result.success(True)
                else:
                    # Collect specific issues
                    issues = validation_results.get("issues", [])
                    feedback = validation_results.get("feedback", "Did not meet validation criteria")
                    
                    if issues:
                        issues_str = "\n- " + "\n- ".join(issues)
                        return Result.failure(f"{feedback}{issues_str}")
                    else:
                        return Result.failure(feedback)
                
            except json.JSONDecodeError as e:
                return Result.failure(f"Invalid JSON in validation response: {str(e)}")
                
        except Exception as e:
            return Result.failure(f"Error during validation: {str(e)}")
    
    def generate_and_validate_text_overlays(self,
                                          system_prompt: str,
                                          user_messages: Union[str, List[Dict[str, str]]],
                                          validation_criteria: List[str],
                                          max_attempts: int = 3) -> Result:
        """
        Generate text overlays and validate them against criteria.
        
        Args:
            system_prompt: System prompt for generation
            user_messages: User messages for generation
            validation_criteria: List of criteria the texts must meet
            max_attempts: Maximum number of generation attempts
            
        Returns:
            Result object with validated text overlays or error
        """
        # Similar implementation to generate_and_validate_narrative_plan but for text overlays
        for attempt in range(max_attempts):
            try:
                logger.info(f"Generating text overlays (attempt {attempt+1}/{max_attempts})")
                
                # Generate initial texts
                generation_result = self.generation_client.chat_completion(
                    system_prompt=system_prompt,
                    user_messages=user_messages,
                    temperature=0.7,
                    json_response=True
                )
                
                if "error" in generation_result:
                    logger.error(f"Generation error: {generation_result['error']}")
                    if attempt == max_attempts - 1:
                        return Result.failure(f"Failed to generate text overlays: {generation_result['error']}")
                    continue
                
                # Parse the generated texts
                try:
                    texts_content = generation_result.get("content", "{}")
                    text_overlays = json.loads(texts_content)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in generation response: {str(e)}")
                    if attempt == max_attempts - 1:
                        return Result.failure(f"Invalid JSON in generation response: {str(e)}")
                    continue
                
                # Validate the texts
                validation_result = self._validate_text_overlays(text_overlays, validation_criteria)
                
                if validation_result.is_success:
                    logger.info(f"Text overlays validated successfully on attempt {attempt+1}")
                    return Result.success(text_overlays)
                else:
                    logger.warning(f"Validation failed on attempt {attempt+1}: {validation_result.error}")
                    
                    # Store this attempt as best so far if it's the first one or better than previous
                    if not hasattr(self, '_best_text_attempt') or not self._best_text_attempt:
                        self._best_text_attempt = {'texts': text_overlays, 'error': validation_result.error}
                    
                    # If this is the last attempt, either return the best attempt or the failure
                    if attempt == max_attempts - 1:
                        if hasattr(self, '_best_text_attempt') and self._best_text_attempt:
                            logger.info(f"Returning best text attempt despite validation issues: {self._best_text_attempt['error']}")
                            # Clear best attempt for next validation run
                            best_texts = self._best_text_attempt['texts']
                            self._best_text_attempt = None
                            return Result.success(best_texts)
                        return validation_result
                    
                    # Otherwise, append detailed validation feedback to the system prompt
                    system_prompt += f"\n\nPrevious attempt failed validation: {validation_result.error}" 
                    system_prompt += "\nPlease fix these issues in your next attempt."
                    # Add specific guidance for common issues
                    system_prompt += "\n\nAdditional guidance:"
                    system_prompt += "\n1. Keep text overlays under 40 characters"
                    system_prompt += "\n2. Follow strategic placement: opening context, minimal mid-video, closing CTA"
                    system_prompt += "\n3. Ensure exactly the requested number of text overlays"
                    system_prompt += "\n4. Use position parameters as specified"
            
            except Exception as e:
                logger.error(f"Error in text generation/validation: {str(e)}")
                if attempt == max_attempts - 1:
                    return Result.failure(f"Error in text generation/validation: {str(e)}")
        
        # Should not reach here, but just in case
        return Result.failure("Failed to generate and validate text overlays after maximum attempts")
    
    def _fix_critical_issues(self, narrative_plan: Dict[str, Any], validation_criteria: List[str]) -> Dict[str, Any]:
        """
        Fix critical issues in a narrative plan when validation has failed after max attempts.
        This ensures the plan meets basic requirements like clip count and pause limits.
        
        Args:
            narrative_plan: The narrative plan to fix
            validation_criteria: List of criteria to check against
            
        Returns:
            Fixed narrative plan
        """
        fixed_plan = narrative_plan.copy()
        segments = fixed_plan.get('segments', [])
        
        # Get expected clip count from validation criteria
        expected_clip_count = 2  # Default to 2
        for criterion in validation_criteria:
            if "action segments" in criterion.lower():
                try:
                    import re
                    match = re.search(r'Exactly (\d+) action segments', criterion)
                    if match:
                        expected_clip_count = int(match.group(1))
                except:
                    pass
        
        # Fix the segments list
        action_segments = [s for s in segments if s.get('type') == 'action']
        pause_segments = [s for s in segments if s.get('type') == 'pause']
        
        # 1. Fix number of action segments
        if len(action_segments) > expected_clip_count:
            # Remove excess action segments (keep the best ones)
            action_segments.sort(key=lambda s: s.get('end_time', 0) - s.get('start_time', 0), reverse=True)
            action_segments = action_segments[:expected_clip_count]
        elif len(action_segments) < expected_clip_count:
            # We don't have enough action segments, so duplicate the existing ones
            while len(action_segments) < expected_clip_count:
                if not action_segments:
                    # Create a default action segment if none exist
                    action_segments.append({
                        'type': 'action',
                        'start_time': 0.0,
                        'end_time': 8.0,
                        'content': 'Default action segment'
                    })
                else:
                    # Duplicate the last segment and adjust its timestamps
                    last_segment = action_segments[-1].copy()
                    last_end = last_segment.get('end_time', 10.0)
                    last_segment['start_time'] = last_end + 1.0
                    last_segment['end_time'] = last_end + 8.0
                    action_segments.append(last_segment)
        
        # 2. Fix number of pause segments (max 2)
        if len(pause_segments) > 2:
            # Keep only the first two pause segments
            pause_segments = pause_segments[:2]
        
        # 3. Ensure action segments have proper timestamps
        for i, segment in enumerate(action_segments):
            if i == 0:
                # First segment always starts at 0.0
                segment['start_time'] = 0.0
                segment['end_time'] = max(8.0, segment.get('end_time', 8.0))
            else:
                # Subsequent segments start after the previous one
                prev_end = action_segments[i-1].get('end_time', 0.0)
                segment['start_time'] = prev_end + 2.0  # 2 seconds gap
                segment['end_time'] = segment['start_time'] + 8.0  # 8 seconds duration
        
        # 4. Create alternating sequence
        fixed_segments = []
        
        # Always start with an action segment
        fixed_segments.append(action_segments[0])
        
        # Add a pause after the first action segment if we have any pause segments
        if pause_segments:
            fixed_segments.append(pause_segments[0])
        
        # Add the final action segment
        if len(action_segments) > 1:
            fixed_segments.append(action_segments[1])
        
        # Only add a second pause if we have more than 2 action segments (which is unlikely with our new defaults)
        # This makes the outro optional as requested
        if len(action_segments) > 2 and len(pause_segments) > 1:
            fixed_segments.append(pause_segments[1])
            
            # Add any remaining action segments
            for i in range(2, len(action_segments)):
                fixed_segments.append(action_segments[i])
        
        # Update the fixed plan
        fixed_plan['segments'] = fixed_segments
        
        # Recalculate total duration
        total_duration = 0.0
        for segment in fixed_segments:
            if segment.get('type') == 'action':
                total_duration += segment.get('end_time', 0.0) - segment.get('start_time', 0.0)
            else:  # pause segment
                total_duration += segment.get('duration', 2.0)
        
        fixed_plan['total_duration'] = total_duration
        
        logger.info(f"Fixed narrative plan: {len(action_segments)} action segments, {len(pause_segments)} pause segments")
        return fixed_plan
    
    def _validate_text_overlays(self, text_overlays: Dict[str, Any], validation_criteria: List[str]) -> Result:
        """
        Validate text overlays against specific criteria.
        
        Args:
            text_overlays: The text overlays to validate
            validation_criteria: List of criteria the texts must meet
            
        Returns:
            Result object with success or validation failures
        """
        # Build validation prompt
        validation_prompt = f"""
        You are a quality control expert for YouTube Shorts text overlays. 
        
        Evaluate these text overlays against the following criteria:
        {json.dumps(validation_criteria, indent=2)}
        
        Text overlays to validate:
        {json.dumps(text_overlays, indent=2)}
        
        Instructions:
        1. Check if texts follow the requested format and structure
        2. Verify that ALL validation criteria are met
        3. For each criterion, mark it as PASS or FAIL
        4. Provide specific feedback for any failed criteria
        
        Respond in JSON format with:
        {{"validation": "pass" or "fail", "issues": [list of specific issues], "feedback": "overall feedback"}}
        """
        
        try:
            # Run validation with secondary LLM
            validation_response = self.validation_client.chat_completion(
                system_prompt=validation_prompt,
                user_messages="Validate these text overlays against the criteria",
                temperature=0.3,
                json_response=True
            )
            
            if "error" in validation_response:
                return Result.failure(f"Validation failed: {validation_response['error']}")
            
            # Parse validation results
            try:
                validation_content = validation_response.get("content", "{}")
                validation_results = json.loads(validation_content)
                
                # Check if validation passed
                if validation_results.get("validation", "").lower() == "pass":
                    return Result.success(True)
                else:
                    # Collect specific issues
                    issues = validation_results.get("issues", [])
                    feedback = validation_results.get("feedback", "Did not meet validation criteria")
                    
                    if issues:
                        issues_str = "\n- " + "\n- ".join(issues)
                        return Result.failure(f"{feedback}{issues_str}")
                    else:
                        return Result.failure(feedback)
                
            except json.JSONDecodeError as e:
                return Result.failure(f"Invalid JSON in validation response: {str(e)}")
                
        except Exception as e:
            return Result.failure(f"Error during validation: {str(e)}")
