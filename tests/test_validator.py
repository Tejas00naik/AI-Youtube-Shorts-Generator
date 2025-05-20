"""
Tests for the validator module.
"""
import pytest
import json
from typing import Dict, Any

from core.validator import (
    ValidationResult, ValidationLevel, Validator, JsonSchemaValidator,
    NarrativePlanValidator, TranscriptValidator,
    validate_phase1_output, validate_transcript, validate_schema
)


class TestValidationResult:
    """Tests for the ValidationResult class."""
    
    def test_init_valid(self):
        """Test initialization with valid status."""
        result = ValidationResult()
        assert result.is_valid
        assert not result.issues
        
    def test_add_error(self):
        """Test adding an error level issue."""
        result = ValidationResult()
        result.add_issue("Test error", ValidationLevel.ERROR)
        
        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0]["level"] == "error"
        assert result.issues[0]["message"] == "Test error"
        
    def test_add_warning(self):
        """Test adding a warning level issue."""
        result = ValidationResult()
        result.add_issue("Test warning", ValidationLevel.WARNING)
        
        assert result.is_valid  # Warnings don't make it invalid
        assert len(result.issues) == 1
        assert result.issues[0]["level"] == "warning"
        
    def test_add_info(self):
        """Test adding an info level issue."""
        result = ValidationResult()
        result.add_issue("Test info", ValidationLevel.INFO)
        
        assert result.is_valid  # Info doesn't make it invalid
        assert len(result.issues) == 1
        assert result.issues[0]["level"] == "info"
        
    def test_has_errors(self):
        """Test has_errors method."""
        result = ValidationResult()
        assert not result.has_errors()
        
        result.add_issue("Test error", ValidationLevel.ERROR)
        assert result.has_errors()
        
    def test_has_warnings(self):
        """Test has_warnings method."""
        result = ValidationResult()
        assert not result.has_warnings()
        
        result.add_issue("Test warning", ValidationLevel.WARNING)
        assert result.has_warnings()
        
    def test_boolean_representation(self):
        """Test boolean representation."""
        result = ValidationResult()
        assert bool(result) is True
        
        result.add_issue("Test error", ValidationLevel.ERROR)
        assert bool(result) is False
        
    def test_string_representation(self):
        """Test string representation."""
        result = ValidationResult()
        assert "Validation passed" in str(result)
        
        result.add_issue("Test error", ValidationLevel.ERROR)
        assert "Validation failed" in str(result)
        assert "ERROR" in str(result)
        assert "Test error" in str(result)


class TestJsonSchemaValidator:
    """Tests for the JsonSchemaValidator class."""
    
    def test_valid_schema(self):
        """Test validation with a valid schema."""
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number", "minimum": 0}
            }
        }
        
        data = {"name": "John", "age": 30}
        validator = JsonSchemaValidator(schema)
        result = validator.validate(data)
        
        assert result.is_valid
        assert not result.issues
        
    def test_invalid_schema(self):
        """Test validation with an invalid schema."""
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number", "minimum": 0}
            }
        }
        
        # Missing required field 'age'
        data = {"name": "John"}
        validator = JsonSchemaValidator(schema)
        result = validator.validate(data)
        
        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0]["level"] == "error"
        assert "required" in result.issues[0]["message"]
        
    def test_type_error(self):
        """Test validation with type error."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "number"}
            }
        }
        
        # String instead of number
        data = {"age": "thirty"}
        validator = JsonSchemaValidator(schema)
        result = validator.validate(data)
        
        assert not result.is_valid
        assert len(result.issues) == 1
        assert result.issues[0]["level"] == "error"
        assert "type" in result.issues[0]["message"]


class TestNarrativePlanValidator:
    """Tests for the NarrativePlanValidator class."""
    
    @pytest.fixture
    def valid_plan(self) -> Dict[str, Any]:
        """Return a valid narrative plan for testing."""
        return {
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
    
    def test_valid_plan(self, valid_plan):
        """Test validation with a valid plan."""
        validator = NarrativePlanValidator()
        result = validator.validate(valid_plan)
        
        assert result.is_valid
        assert not result.has_errors()
        
    def test_duration_too_long(self, valid_plan):
        """Test validation when duration is too long."""
        # Modify the total duration to exceed the maximum
        valid_plan["total_duration"] = 60.0
        
        validator = NarrativePlanValidator(max_duration=58)
        result = validator.validate(valid_plan)
        
        assert not result.is_valid
        assert "duration" in str(result)
        
    def test_segments_too_many(self, valid_plan):
        """Test validation when there are too many segments."""
        # Add extra segments to exceed the maximum
        for i in range(10):
            valid_plan["segments"].append({
                "type": f"extra_{i}",
                "start_time": 25.0 + i,
                "end_time": 26.0 + i,
                "description": f"Extra segment {i}",
                "text": f"Extra text {i}",
                "mood": "neutral"
            })
        
        validator = NarrativePlanValidator(max_segments=5)
        result = validator.validate(valid_plan)
        
        assert not result.is_valid
        assert any("segments" in issue["field"] for issue in result.issues)
        
    def test_segment_end_before_start(self, valid_plan):
        """Test validation when segment end time is before start time."""
        # Modify a segment to have end_time before start_time
        valid_plan["segments"][1]["end_time"] = 3.0  # Now before its start_time of 5.0
        
        validator = NarrativePlanValidator()
        result = validator.validate(valid_plan)
        
        assert not result.is_valid
        assert any("end time" in issue["message"] and "start time" in issue["message"] 
                for issue in result.issues)
        
    def test_overlapping_segments(self, valid_plan):
        """Test validation when segments overlap."""
        # Modify segment to overlap with the next one
        valid_plan["segments"][0]["end_time"] = 7.0  # Now overlaps with segment 1
        
        validator = NarrativePlanValidator()
        result = validator.validate(valid_plan)
        
        assert not result.is_valid
        assert any("overlaps" in issue["message"] for issue in result.issues)
        
    def test_missing_hook_warning(self, valid_plan):
        """Test validation when plan is missing a hook segment."""
        # Change the first segment type from 'hook' to something else
        valid_plan["segments"][0]["type"] = "not_a_hook"
        
        validator = NarrativePlanValidator()
        result = validator.validate(valid_plan)
        
        assert result.is_valid  # This is a warning, not an error
        assert result.has_warnings()
        assert any("missing a hook" in issue["message"] for issue in result.issues)
        
    def test_missing_conclusion_warning(self, valid_plan):
        """Test validation when plan is missing a conclusion segment."""
        # Change the last segment type from 'conclusion' to something else
        valid_plan["segments"][2]["type"] = "not_a_conclusion"
        
        validator = NarrativePlanValidator()
        result = validator.validate(valid_plan)
        
        assert result.is_valid  # This is a warning, not an error
        assert result.has_warnings()
        assert any("missing a conclusion" in issue["message"] for issue in result.issues)


class TestTranscriptValidator:
    """Tests for the TranscriptValidator class."""
    
    def test_valid_transcript(self):
        """Test validation with a valid transcript."""
        transcript = "This is a valid transcript with sufficient length and words."
        
        validator = TranscriptValidator()
        result = validator.validate(transcript)
        
        assert result.is_valid
        assert not result.issues
        
    def test_empty_transcript(self):
        """Test validation with an empty transcript."""
        transcript = ""
        
        validator = TranscriptValidator()
        result = validator.validate(transcript)
        
        assert not result.is_valid
        assert any("empty" in issue["message"] for issue in result.issues)
        
    def test_short_transcript(self):
        """Test validation with a short transcript."""
        transcript = "Too short"
        
        validator = TranscriptValidator(min_length=20)
        result = validator.validate(transcript)
        
        assert result.is_valid  # This is a warning, not an error
        assert result.has_warnings()
        assert any("too short" in issue["message"] for issue in result.issues)
        
    def test_few_words(self):
        """Test validation with too few words."""
        transcript = "One two three four."
        
        validator = TranscriptValidator(min_words=5)
        result = validator.validate(transcript)
        
        assert result.is_valid  # This is a warning, not an error
        assert result.has_warnings()
        assert any("too few words" in issue["message"] for issue in result.issues)


class TestConvenienceFunctions:
    """Tests for the convenience functions."""
    
    def test_validate_phase1_output(self):
        """Test validate_phase1_output function."""
        # Create a sample valid plan
        valid_plan = {
            "segments": [
                {
                    "type": "hook",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "description": "Introduction to the topic",
                    "text": "Let me show you what Python decorators can do!",
                    "mood": "excited"
                },
                {
                    "type": "highlight_1",
                    "start_time": 5.0,
                    "end_time": 15.0,
                    "description": "Basic decorator syntax",
                    "text": "Decorators let you wrap functions to add functionality.",
                    "mood": "instructive"
                },
                {
                    "type": "conclusion",
                    "start_time": 15.0,
                    "end_time": 20.0,
                    "description": "Summary and call to action",
                    "text": "Now you know the basics of decorators! Like and subscribe for more Python tips.",
                    "mood": "excited"
                }
            ],
            "total_duration": 20.0,
            "summary": "A brief introduction to Python decorators, covering basic syntax and usage."
        }
        
        result = validate_phase1_output(valid_plan)
        assert result.is_valid
        
    def test_validate_transcript(self):
        """Test validate_transcript function."""
        result = validate_transcript("This is a valid transcript.")
        assert result.is_valid
        
    def test_validate_schema(self):
        """Test validate_schema function."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        data = {"name": "John"}
        
        result = validate_schema(data, schema)
        assert result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
