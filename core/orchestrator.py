"""
Orchestrator for AI YouTube Shorts Generator.

This module coordinates the overall pipeline, managing the flow between
input parsing, narrative planning, and validation, while handling errors
and fallbacks.
"""
import logging
from typing import Dict, Any, Optional, List, Tuple
import json

from core.input_parser import parse_user_input, NarrativeMode, Tone
from core.validator import validate_phase1_output, validate_transcript
from core.config import get_config
from core.error_handler import (
    ErrorHandler, get_error_handler, ErrorCode, ErrorSeverity,
    PipelineError, Result, with_error_handling, safe_execute
)
from llm.narrative_planner import generate_narrative_plan

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineStage:
    """Enum-like class for pipeline stages."""
    INPUT_PARSING = "input_parsing"
    NARRATIVE_PLANNING = "narrative_planning"
    VALIDATION = "validation"
    CLIP_SELECTION = "clip_selection"
    SCRIPT_GENERATION = "script_generation"
    MEDIA_PROCESSING = "media_processing"
    EXPORT = "export"


class Orchestrator:
    """
    Main orchestrator for the video generation pipeline.
    
    This class manages the entire process from user input to final video generation,
    handling each stage of the pipeline and providing error recovery.
    """
    
    def __init__(self):
        """Initialize the orchestrator."""
        self.config = get_config()
        self.error_handler = get_error_handler()
        self.pipeline_state: Dict[str, Any] = {
            "current_stage": None,
            "input_params": {},
            "transcript": "",
            "narrative_plan": {},
            "clip_selections": [],
            "script": {},
            "processing_options": {},
            "output_path": "",
            "start_time": None,
            "end_time": None
        }
        
        # Register fallback handlers
        self._register_fallbacks()
    
    def _register_fallbacks(self) -> None:
        """Register fallback handlers for different error types."""
        # Fallback for LLM API errors
        self.error_handler.register_fallback(
            ErrorCode.LLM_API_ERROR,
            self._llm_api_fallback
        )
        
        # Fallback for validation errors
        self.error_handler.register_fallback(
            ErrorCode.VALIDATION_ERROR,
            self._validation_fallback
        )
        
        # Fallback for input parsing errors
        self.error_handler.register_fallback(
            ErrorCode.INVALID_INPUT,
            self._input_fallback
        )
    
    def _llm_api_fallback(self, error: PipelineError) -> Dict[str, Any]:
        """
        Fallback handler for LLM API errors.
        
        Args:
            error: The pipeline error
            
        Returns:
            A simplified narrative plan
        """
        logger.warning(f"Using fallback for LLM API error: {error}")
        
        transcript = self.pipeline_state.get("transcript", "")
        params = self.pipeline_state.get("input_params", {})
        
        # Create a simple fallback narrative plan
        words = transcript.split()
        total_words = len(words)
        
        # If transcript is too short, return a minimal plan
        if total_words < 20:
            return {
                "segments": [
                    {
                        "type": "content",
                        "start_time": 0.0,
                        "end_time": min(total_words * 0.5, 58.0),  # Rough estimate
                        "description": "Content from transcript",
                        "text": transcript,
                        "mood": "neutral"
                    }
                ],
                "total_duration": min(total_words * 0.5, 58.0),
                "summary": "Fallback narrative plan",
                "metadata": {
                    "is_fallback": True,
                    "mode": params.get("mode", "highlight"),
                    "tone": params.get("tone", "professional")
                }
            }
            
        # For longer transcripts, create a simple 3-part structure
        segment_count = min(3, max(1, total_words // 50))
        chunk_size = total_words // segment_count
        
        segments = []
        duration_per_segment = min(15, 58 // segment_count)
        total_duration = 0
        
        for i in range(segment_count):
            start_idx = i * chunk_size
            end_idx = min(total_words, (i + 1) * chunk_size)
            
            segment_text = " ".join(words[start_idx:end_idx])
            segment_duration = min(duration_per_segment, 58 - total_duration)
            
            segments.append({
                "type": f"segment_{i+1}",
                "start_time": total_duration,
                "end_time": total_duration + segment_duration,
                "description": f"Part {i+1} of content",
                "text": segment_text,
                "mood": "neutral"
            })
            
            total_duration += segment_duration
        
        return {
            "segments": segments,
            "total_duration": total_duration,
            "summary": "Fallback narrative plan due to LLM API error",
            "metadata": {
                "is_fallback": True,
                "mode": params.get("mode", "highlight"),
                "tone": params.get("tone", "professional"),
                "error": str(error)
            }
        }
    
    def _validation_fallback(self, error: PipelineError) -> Dict[str, Any]:
        """
        Fallback handler for validation errors.
        
        Args:
            error: The pipeline error
            
        Returns:
            Fixed version of the invalid data
        """
        logger.warning(f"Using fallback for validation error: {error}")
        
        # Get the context to determine what was being validated
        context = error.context or {}
        stage = context.get("stage")
        
        if stage == PipelineStage.NARRATIVE_PLANNING:
            # Fix the narrative plan
            plan = context.get("data", {})
            
            # If we have no segments, create a simple one
            if not plan.get("segments"):
                return self._llm_api_fallback(error)
            
            # Fix common issues with the narrative plan
            fixed_plan = plan.copy()
            
            # Ensure we have required keys
            if "total_duration" not in fixed_plan:
                # Calculate from segments
                total_duration = sum(
                    segment["end_time"] - segment["start_time"]
                    for segment in fixed_plan.get("segments", [])
                )
                fixed_plan["total_duration"] = total_duration
                
            if "summary" not in fixed_plan:
                fixed_plan["summary"] = "Automatically generated summary"
                
            # Fix segment times to ensure no overlaps and proper sequence
            segments = fixed_plan.get("segments", [])
            current_time = 0.0
            max_time = 58.0  # Max duration for YouTube Shorts
            
            for i, segment in enumerate(segments):
                segment_duration = min(
                    segment.get("end_time", current_time + 10) - segment.get("start_time", current_time),
                    max_time - current_time
                )
                
                if segment_duration <= 0:
                    segment_duration = min(5.0, max_time - current_time)
                
                segment["start_time"] = current_time
                segment["end_time"] = current_time + segment_duration
                
                # Ensure required fields exist
                if "type" not in segment:
                    segment["type"] = f"segment_{i+1}"
                    
                if "description" not in segment:
                    segment["description"] = f"Segment {i+1}"
                    
                if "text" not in segment:
                    segment["text"] = f"Content for segment {i+1}"
                    
                if "mood" not in segment:
                    segment["mood"] = "neutral"
                
                current_time += segment_duration
                
                # Break if we've reached the maximum duration
                if current_time >= max_time:
                    break
            
            # Truncate segments if necessary
            if current_time >= max_time:
                fixed_plan["segments"] = segments[:i+1]
                
            # Update total duration
            fixed_plan["total_duration"] = min(current_time, max_time)
            
            # Add metadata about the fallback
            if "metadata" not in fixed_plan:
                fixed_plan["metadata"] = {}
            
            fixed_plan["metadata"]["fixed_by_validation_fallback"] = True
            fixed_plan["metadata"]["original_error"] = str(error)
            
            return fixed_plan
            
        # Default fallback for other validation errors
        return {
            "fallback_data": True,
            "error": str(error),
            "message": "Used generic validation fallback"
        }
    
    def _input_fallback(self, error: PipelineError) -> Dict[str, Any]:
        """
        Fallback handler for input parsing errors.
        
        Args:
            error: The pipeline error
            
        Returns:
            Default input parameters
        """
        logger.warning(f"Using fallback for input parsing error: {error}")
        
        # Return default parameters
        return {
            "mode": "highlight",
            "tone": "professional",
            "params": {
                "duration": 58  # Maximum for YouTube Shorts
            }
        }
    
    @with_error_handling(error_code=ErrorCode.INVALID_INPUT)
    def parse_input(self, user_input: str) -> Dict[str, Any]:
        """
        Parse user input to extract parameters.
        
        Args:
            user_input: Raw input text from the user
            
        Returns:
            Dictionary of parsed parameters
        """
        self.pipeline_state["current_stage"] = PipelineStage.INPUT_PARSING
        
        try:
            params = parse_user_input(user_input)
            self.pipeline_state["input_params"] = params
            logger.info(f"Successfully parsed input: {json.dumps(params)}")
            return params
        except Exception as e:
            raise PipelineError(
                message=f"Failed to parse input: {str(e)}",
                code=ErrorCode.INVALID_INPUT,
                severity=ErrorSeverity.HIGH,
                original_error=e,
                context={"user_input": user_input}
            )
    
    @with_error_handling(error_code=ErrorCode.LLM_API_ERROR)
    def generate_narrative(self, transcript: str) -> Dict[str, Any]:
        """
        Generate a narrative plan based on the transcript and user parameters.
        
        Args:
            transcript: Full transcript of the video
            
        Returns:
            Dictionary containing the narrative plan
        """
        self.pipeline_state["current_stage"] = PipelineStage.NARRATIVE_PLANNING
        self.pipeline_state["transcript"] = transcript
        
        params = self.pipeline_state.get("input_params", {})
        mode = params.get("mode", "highlight")
        tone = params.get("tone", "professional")
        user_params = params.get("params", {})
        
        try:
            plan = generate_narrative_plan(
                transcript=transcript,
                mode=mode,
                tone=tone,
                **user_params
            )
            
            self.pipeline_state["narrative_plan"] = plan
            logger.info(f"Generated narrative plan with {len(plan.get('segments', []))} segments")
            return plan
        except Exception as e:
            raise PipelineError(
                message=f"Failed to generate narrative plan: {str(e)}",
                code=ErrorCode.LLM_API_ERROR,
                severity=ErrorSeverity.HIGH,
                original_error=e,
                context={
                    "mode": mode,
                    "tone": tone,
                    "transcript_length": len(transcript)
                }
            )
    
    @with_error_handling(error_code=ErrorCode.VALIDATION_ERROR)
    def validate(self, stage: str, data: Any) -> Result[bool]:
        """
        Validate data from a specific pipeline stage.
        
        Args:
            stage: Pipeline stage to validate
            data: Data to validate
            
        Returns:
            Result containing True if valid, or an error
        """
        self.pipeline_state["current_stage"] = PipelineStage.VALIDATION
        
        try:
            if stage == PipelineStage.NARRATIVE_PLANNING:
                result = validate_phase1_output(data)
                
                if not result:
                    raise PipelineError(
                        message=f"Narrative plan validation failed: {result}",
                        code=ErrorCode.VALIDATION_ERROR,
                        severity=ErrorSeverity.MEDIUM,
                        context={
                            "stage": stage,
                            "data": data,
                            "validation_issues": [issue for issue in result.issues]
                        }
                    )
                
                logger.info("Narrative plan validation successful")
                return Result.success(True)
                
            elif stage == PipelineStage.INPUT_PARSING:
                # Validate input parameters if needed
                return Result.success(True)
                
            else:
                logger.warning(f"No validation implemented for stage: {stage}")
                return Result.success(True)
                
        except PipelineError as e:
            # Re-raise PipelineErrors
            raise e
        except Exception as e:
            # Wrap other exceptions
            raise PipelineError(
                message=f"Validation error in stage {stage}: {str(e)}",
                code=ErrorCode.VALIDATION_ERROR,
                severity=ErrorSeverity.MEDIUM,
                original_error=e,
                context={"stage": stage}
            )
    
    def process_user_input(self, user_input: str, transcript: str) -> Result[Dict[str, Any]]:
        """
        Process user input and transcript to generate a narrative plan.
        
        This is the main entry point for Phase 1 of the pipeline.
        
        Args:
            user_input: Raw input text from the user
            transcript: Full transcript of the video
            
        Returns:
            Result containing the narrative plan or an error
        """
        # Step 1: Parse user input
        input_result = self.parse_input(user_input)
        if input_result.is_error:
            # Try to recover with fallback
            fallback_params = self.error_handler.handle_error(input_result.error)
            if fallback_params:
                self.pipeline_state["input_params"] = fallback_params
                logger.info("Recovered from input parsing error with fallback parameters")
            else:
                return input_result
        
        # Step 2: Validate the transcript
        transcript_validation = validate_transcript(transcript)
        if not transcript_validation:
            logger.warning(f"Transcript validation issues: {transcript_validation}")
            # Continue anyway as these are just warnings
        
        # Step 3: Generate narrative plan
        narrative_result = self.generate_narrative(transcript)
        if narrative_result.is_error:
            # Try to recover with fallback
            fallback_plan = self.error_handler.handle_error(narrative_result.error)
            if fallback_plan:
                self.pipeline_state["narrative_plan"] = fallback_plan
                logger.info("Recovered from narrative planning error with fallback plan")
            else:
                return narrative_result
        
        # Step 4: Validate the narrative plan
        plan = self.pipeline_state.get("narrative_plan", {})
        validation_result = self.validate(PipelineStage.NARRATIVE_PLANNING, plan)
        if validation_result.is_error:
            # Try to recover with fallback
            fixed_plan = self.error_handler.handle_error(validation_result.error)
            if fixed_plan:
                self.pipeline_state["narrative_plan"] = fixed_plan
                logger.info("Recovered from validation error with fixed plan")
            else:
                return validation_result
        
        # Return the final narrative plan
        return Result.success(self.pipeline_state["narrative_plan"])
    
    def get_pipeline_state(self) -> Dict[str, Any]:
        """
        Get the current state of the pipeline.
        
        Returns:
            Dictionary with the current pipeline state
        """
        return self.pipeline_state.copy()
    
    def reset_pipeline(self) -> None:
        """Reset the pipeline state."""
        self.pipeline_state = {
            "current_stage": None,
            "input_params": {},
            "transcript": "",
            "narrative_plan": {},
            "clip_selections": [],
            "script": {},
            "processing_options": {},
            "output_path": "",
            "start_time": None,
            "end_time": None
        }
        
        # Reset the error handler
        self.error_handler.reset()


# Create a singleton instance
orchestrator = Orchestrator()


def get_orchestrator() -> Orchestrator:
    """
    Get the global orchestrator instance.
    
    Returns:
        The Orchestrator instance
    """
    return orchestrator


def process_input(user_input: str, transcript: str) -> Result[Dict[str, Any]]:
    """
    Process user input and transcript to generate a narrative plan.
    
    This is a convenience function that uses the global orchestrator.
    
    Args:
        user_input: Raw input text from the user
        transcript: Full transcript of the video
        
    Returns:
        Result containing the narrative plan or an error
    """
    return orchestrator.process_user_input(user_input, transcript)


if __name__ == "__main__":
    # Example usage
    test_input = "Create a highlight video about machine learning in a fun way for beginners"
    test_transcript = """
    In this tutorial, I'll show you how to get started with machine learning.
    First, we'll talk about what machine learning actually is.
    Machine learning is a subset of artificial intelligence that allows computers to learn from data.
    There are several types of machine learning, including supervised learning, unsupervised learning, and reinforcement learning.
    Let's start by installing the necessary libraries like scikit-learn, TensorFlow, and PyTorch.
    Once we have our environment set up, we can load a sample dataset to work with.
    The iris dataset is a good starting point for beginners.
    Now let's split our data into training and testing sets.
    Next, we'll train a simple classifier on our data.
    As you can see, our model achieved 95% accuracy on the test set!
    That's it for this introduction to machine learning. In the next video, we'll dive deeper into neural networks.
    """
    
    result = process_input(test_input, test_transcript)
    
    if result.is_success:
        print("Successfully generated narrative plan:")
        print(json.dumps(result.value, indent=2))
    else:
        print(f"Error: {result.error}")
