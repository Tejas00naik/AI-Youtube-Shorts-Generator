"""
Validator module for AI YouTube Shorts Generator.

This module provides validation functions to ensure that the outputs
from different stages of the pipeline meet the required constraints.
"""
import logging
from typing import Dict, List, Any, Tuple, Optional
import json
import jsonschema
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation severity levels."""
    ERROR = "error"  # Critical issues that must be fixed
    WARNING = "warning"  # Issues that should be addressed but won't block
    INFO = "info"  # Informational messages


class ValidationResult:
    """Result of a validation operation."""
    
    def __init__(self, is_valid: bool = True):
        """
        Initialize a validation result.
        
        Args:
            is_valid: Whether the validation passed or not
        """
        self.is_valid = is_valid
        self.issues: List[Dict[str, Any]] = []
    
    def add_issue(
        self,
        message: str,
        level: ValidationLevel = ValidationLevel.ERROR,
        field: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an issue to the validation result.
        
        Args:
            message: Description of the issue
            level: Severity level (ERROR, WARNING, INFO)
            field: Field that has the issue
            context: Additional context for the issue
        """
        self.issues.append({
            "message": message,
            "level": level.value,
            "field": field,
            "context": context or {}
        })
        
        # Mark as invalid for ERROR level issues
        if level == ValidationLevel.ERROR:
            self.is_valid = False
    
    def has_errors(self) -> bool:
        """Check if there are any ERROR level issues."""
        return any(issue["level"] == ValidationLevel.ERROR.value for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if there are any WARNING level issues."""
        return any(issue["level"] == ValidationLevel.WARNING.value for issue in self.issues)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the validation result to a dictionary."""
        return {
            "is_valid": self.is_valid,
            "issues": self.issues
        }
    
    def __bool__(self) -> bool:
        """Boolean representation of validation result."""
        return self.is_valid
    
    def __str__(self) -> str:
        """String representation of validation result."""
        if self.is_valid and not self.issues:
            return "Validation passed successfully."
        
        result = []
        if not self.is_valid:
            result.append("Validation failed with the following issues:")
        else:
            result.append("Validation passed with the following warnings/info:")
        
        for issue in self.issues:
            field_info = f" in {issue['field']}" if issue.get('field') else ""
            result.append(f"  [{issue['level'].upper()}]{field_info}: {issue['message']}")
        
        return "\n".join(result)


class Validator:
    """Base validator class."""
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate the data.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult object
        """
        raise NotImplementedError("Subclasses must implement validate()")


class JsonSchemaValidator(Validator):
    """Validator using JSON Schema."""
    
    def __init__(self, schema: Dict[str, Any]):
        """
        Initialize with a JSON schema.
        
        Args:
            schema: JSON schema to validate against
        """
        self.schema = schema
    
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate data against the JSON schema.
        
        Args:
            data: Data to validate
            
        Returns:
            ValidationResult object
        """
        result = ValidationResult()
        
        try:
            jsonschema.validate(data, self.schema)
        except jsonschema.exceptions.ValidationError as e:
            # Extract the path to the field with the error
            path = ".".join(str(x) for x in e.path) if e.path else None
            result.add_issue(
                message=e.message,
                field=path,
                level=ValidationLevel.ERROR,
                context={"schema_path": e.schema_path}
            )
        except Exception as e:
            result.add_issue(
                message=str(e),
                level=ValidationLevel.ERROR,
                context={"exception_type": type(e).__name__}
            )
        
        return result


class NarrativePlanValidator(Validator):
    """Validator for narrative plans."""
    
    def __init__(self, max_duration: int = 58, min_segments: int = 1, max_segments: int = 10):
        """
        Initialize with validation parameters.
        
        Args:
            max_duration: Maximum total duration in seconds
            min_segments: Minimum number of segments
            max_segments: Maximum number of segments
        """
        self.max_duration = max_duration
        self.min_segments = min_segments
        self.max_segments = max_segments
        
        # Define the schema for a narrative plan
        self.schema = {
            "type": "object",
            "required": ["segments", "total_duration", "summary"],
            "properties": {
                "segments": {
                    "type": "array",
                    "minItems": self.min_segments,
                    "maxItems": self.max_segments,
                    "items": {
                        "type": "object",
                        "required": ["type", "start_time", "end_time", "description", "text"],
                        "properties": {
                            "type": {"type": "string"},
                            "start_time": {"type": "number", "minimum": 0},
                            "end_time": {"type": "number", "minimum": 0},
                            "description": {"type": "string"},
                            "text": {"type": "string"},
                            "mood": {"type": "string"}
                        }
                    }
                },
                "total_duration": {"type": "number", "minimum": 0},
                "summary": {"type": "string"},
                "metadata": {"type": "object"}
            }
        }
        self.schema_validator = JsonSchemaValidator(self.schema)
    
    def validate(self, plan: Dict[str, Any]) -> ValidationResult:
        """
        Validate a narrative plan.
        
        Args:
            plan: Narrative plan to validate
            
        Returns:
            ValidationResult object
        """
        # First, validate against the schema
        result = self.schema_validator.validate(plan)
        if not result:
            return result
        
        # Additional custom validations
        self._validate_duration(plan, result)
        self._validate_segments(plan, result)
        self._validate_overlaps(plan, result)
        self._validate_content(plan, result)
        
        return result
    
    def _validate_duration(self, plan: Dict[str, Any], result: ValidationResult) -> None:
        """Validate the total duration."""
        total_duration = plan.get("total_duration", 0)
        
        # Check if the duration exceeds the maximum
        if total_duration > self.max_duration + 0.5:  # Add 0.5s tolerance
            result.add_issue(
                message=f"Total duration ({total_duration:.2f}s) exceeds maximum ({self.max_duration}s)",
                level=ValidationLevel.ERROR,
                field="total_duration"
            )
        
        # Verify that the sum of segment durations matches the total duration
        segment_sum = sum(
            segment["end_time"] - segment["start_time"]
            for segment in plan.get("segments", [])
        )
        
        if abs(total_duration - segment_sum) > 0.5:  # Allow 0.5s tolerance
            result.add_issue(
                message=f"Sum of segment durations ({segment_sum:.2f}s) does not match total duration ({total_duration:.2f}s)",
                level=ValidationLevel.WARNING,
                field="segments",
                context={"segment_sum": segment_sum, "total_duration": total_duration}
            )
    
    def _validate_segments(self, plan: Dict[str, Any], result: ValidationResult) -> None:
        """Validate individual segments."""
        segments = plan.get("segments", [])
        
        for i, segment in enumerate(segments):
            # Check if end_time is after start_time
            if segment["end_time"] <= segment["start_time"]:
                result.add_issue(
                    message=f"Segment {i} has end time ({segment['end_time']:.2f}s) <= start time ({segment['start_time']:.2f}s)",
                    level=ValidationLevel.ERROR,
                    field=f"segments[{i}]"
                )
            
            # Check if the segment is too short (less than 1 second)
            duration = segment["end_time"] - segment["start_time"]
            if duration < 1.0:
                result.add_issue(
                    message=f"Segment {i} is too short ({duration:.2f}s < 1.0s)",
                    level=ValidationLevel.WARNING,
                    field=f"segments[{i}]"
                )
            
            # Check if description and text are not empty
            if not segment.get("description", "").strip():
                result.add_issue(
                    message=f"Segment {i} has an empty description",
                    level=ValidationLevel.WARNING,
                    field=f"segments[{i}].description"
                )
            
            if not segment.get("text", "").strip():
                result.add_issue(
                    message=f"Segment {i} has empty text",
                    level=ValidationLevel.WARNING,
                    field=f"segments[{i}].text"
                )
    
    def _validate_overlaps(self, plan: Dict[str, Any], result: ValidationResult) -> None:
        """Validate that segments don't overlap."""
        segments = sorted(
            plan.get("segments", []),
            key=lambda s: s["start_time"]
        )
        
        for i in range(1, len(segments)):
            prev_end = segments[i-1]["end_time"]
            curr_start = segments[i]["start_time"]
            
            if curr_start < prev_end:
                result.add_issue(
                    message=f"Segment {i} (start: {curr_start:.2f}s) overlaps with segment {i-1} (end: {prev_end:.2f}s)",
                    level=ValidationLevel.ERROR,
                    field=f"segments[{i}]",
                    context={"prev_segment": i-1, "overlap": prev_end - curr_start}
                )
    
    def _validate_content(self, plan: Dict[str, Any], result: ValidationResult) -> None:
        """Validate the content of the plan."""
        # Check if summary is not too short
        summary = plan.get("summary", "")
        if len(summary.split()) < 5:
            result.add_issue(
                message="Summary is too short",
                level=ValidationLevel.WARNING,
                field="summary",
                context={"word_count": len(summary.split())}
            )
        
        # Check content for each segment based on its type
        segments = plan.get("segments", [])
        segment_types = {segment["type"] for segment in segments}
        
        # Ensure there's a hook or introduction
        if not any(segment["type"] in ["hook", "introduction"] for segment in segments):
            result.add_issue(
                message="Plan is missing a hook or introduction segment",
                level=ValidationLevel.WARNING,
                field="segments"
            )
        
        # Ensure there's a conclusion or call_to_action
        if not any(segment["type"] in ["conclusion", "call_to_action"] for segment in segments):
            result.add_issue(
                message="Plan is missing a conclusion or call to action segment",
                level=ValidationLevel.WARNING,
                field="segments"
            )


class TranscriptValidator(Validator):
    """Validator for transcripts."""
    
    def __init__(self, min_length: int = 10, min_words: int = 5):
        """
        Initialize with validation parameters.
        
        Args:
            min_length: Minimum transcript length in characters
            min_words: Minimum number of words in the transcript
        """
        self.min_length = min_length
        self.min_words = min_words
    
    def validate(self, transcript: str) -> ValidationResult:
        """
        Validate a transcript.
        
        Args:
            transcript: Transcript to validate
            
        Returns:
            ValidationResult object
        """
        result = ValidationResult()
        
        # Check if transcript is empty or None
        if not transcript:
            result.add_issue(
                message="Transcript is empty",
                level=ValidationLevel.ERROR
            )
            return result
        
        # Check minimum length
        if len(transcript) < self.min_length:
            result.add_issue(
                message=f"Transcript is too short ({len(transcript)} chars < {self.min_length} chars)",
                level=ValidationLevel.WARNING,
                context={"length": len(transcript)}
            )
        
        # Check minimum words
        words = transcript.split()
        if len(words) < self.min_words:
            result.add_issue(
                message=f"Transcript has too few words ({len(words)} words < {self.min_words} words)",
                level=ValidationLevel.WARNING,
                context={"word_count": len(words)}
            )
        
        return result


# Convenience functions
def validate_phase1_output(output: Dict[str, Any]) -> ValidationResult:
    """
    Validate the output of Phase 1 (Narrative Planning).
    
    Args:
        output: Phase 1 output to validate
        
    Returns:
        ValidationResult object
    """
    validator = NarrativePlanValidator()
    return validator.validate(output)


def validate_transcript(transcript: str) -> ValidationResult:
    """
    Validate a transcript.
    
    Args:
        transcript: Transcript to validate
        
    Returns:
        ValidationResult object
    """
    validator = TranscriptValidator()
    return validator.validate(transcript)


def validate_schema(data: Any, schema: Dict[str, Any]) -> ValidationResult:
    """
    Validate data against a JSON schema.
    
    Args:
        data: Data to validate
        schema: JSON schema to validate against
        
    Returns:
        ValidationResult object
    """
    validator = JsonSchemaValidator(schema)
    return validator.validate(data)


if __name__ == "__main__":
    # Example usage
    sample_plan = {
        "segments": [
            {
                "type": "hook",
                "start_time": 0.0,
                "end_time": 5.0,
                "description": "Introduction to the topic",
                "text": "Let's learn about Python decorators!",
                "mood": "excited"
            },
            {
                "type": "highlight_1",
                "start_time": 5.0,
                "end_time": 20.0,
                "description": "Explaining basic decorator syntax",
                "text": "A decorator is a function that takes another function as an argument.",
                "mood": "instructive"
            },
            {
                "type": "conclusion",
                "start_time": 20.0,
                "end_time": 25.0,
                "description": "Summarizing the key points",
                "text": "That's it for our quick intro to decorators!",
                "mood": "satisfied"
            }
        ],
        "total_duration": 25.0,
        "summary": "A brief introduction to Python decorators, covering basic syntax and usage."
    }
    
    # Validate the sample plan
    result = validate_phase1_output(sample_plan)
    print(result)
    
    if result.issues:
        print("\nIssues found:")
        for issue in result.issues:
            print(f"  - {issue['level'].upper()}: {issue['message']}")
