"""
Error handler for AI YouTube Shorts Generator.

This module provides error handling, logging, and fallback
mechanisms for different stages of the video generation pipeline.
"""
import logging
import traceback
import json
from enum import Enum
from typing import Dict, Any, Optional, Callable, TypeVar, Generic, Union, List, Tuple
import uuid
from functools import wraps
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')


class ErrorCode(str, Enum):
    """Error codes for different types of errors in the pipeline."""
    # Input errors
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    
    # Processing errors
    LLM_API_ERROR = "LLM_API_ERROR"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    
    # Media errors
    VIDEO_PROCESSING_ERROR = "VIDEO_PROCESSING_ERROR"
    AUDIO_PROCESSING_ERROR = "AUDIO_PROCESSING_ERROR"
    FACE_DETECTION_ERROR = "FACE_DETECTION_ERROR"
    TRANSCRIPTION_ERROR = "TRANSCRIPTION_ERROR"
    
    # System errors
    FILESYSTEM_ERROR = "FILESYSTEM_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    DEPENDENCY_ERROR = "DEPENDENCY_ERROR"
    
    # Unknown errors
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    CRITICAL = "CRITICAL"  # System cannot continue, must abort
    HIGH = "HIGH"          # Feature cannot be completed, but system can continue
    MEDIUM = "MEDIUM"      # Feature degraded but functional
    LOW = "LOW"            # Minor issue, can be ignored


class PipelineError(Exception):
    """Custom exception for pipeline errors with context."""
    
    def __init__(
        self,
        message: str,
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize a pipeline error.
        
        Args:
            message: Error message
            code: Error code
            severity: Error severity level
            context: Additional context for the error
            original_error: Original exception if this is a wrapper
        """
        self.code = code
        self.severity = severity
        self.context = context or {}
        self.original_error = original_error
        self.error_id = str(uuid.uuid4())
        self.timestamp = time.time()
        
        # Format the message with context if available
        full_message = message
        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            full_message = f"{message} [{context_str}]"
        
        super().__init__(full_message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary."""
        result = {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "message": str(self),
            "code": self.code.value,
            "severity": self.severity.value,
            "context": self.context
        }
        
        if self.original_error:
            result["original_error"] = {
                "type": type(self.original_error).__name__,
                "message": str(self.original_error)
            }
            
            # Add traceback if available
            if hasattr(self.original_error, "__traceback__"):
                result["stacktrace"] = traceback.format_exception(
                    type(self.original_error),
                    self.original_error,
                    self.original_error.__traceback__
                )
        
        return result


class Result(Generic[T]):
    """A result type that can either contain a value or an error."""
    
    def __init__(
        self,
        value: Optional[T] = None,
        error: Optional[PipelineError] = None
    ):
        """
        Initialize a result.
        
        Args:
            value: The success value (if successful)
            error: The error (if failed)
        """
        self._value = value
        self._error = error
        
        # Validate that exactly one of value or error is set
        if (value is None and error is None) or (value is not None and error is not None):
            raise ValueError("Result must have either a value or an error, not both or neither")
    
    @property
    def is_success(self) -> bool:
        """Check if the result is successful."""
        return self._error is None
    
    @property
    def is_error(self) -> bool:
        """Check if the result is an error."""
        return self._error is not None
    
    @property
    def value(self) -> T:
        """Get the success value."""
        if self._error:
            raise ValueError(f"Cannot access value of an error result: {self._error}")
        return self._value
    
    @property
    def error(self) -> PipelineError:
        """Get the error."""
        if self._error is None:
            raise ValueError("Cannot access error of a success result")
        return self._error
    
    @staticmethod
    def success(value: T) -> 'Result[T]':
        """Create a success result."""
        return Result(value=value)
    
    @staticmethod
    def failure(
        error: Union[PipelineError, str],
        code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_error: Optional[Exception] = None
    ) -> 'Result[T]':
        """Create a failure result."""
        if isinstance(error, PipelineError):
            return Result(error=error)
        else:
            return Result(error=PipelineError(
                message=error,
                code=code,
                severity=severity,
                context=context,
                original_error=original_error
            ))
    
    def map(self, fn: Callable[[T], R]) -> 'Result[R]':
        """
        Map a function over the success value.
        
        If this is a success result, applies the function to the value.
        If this is an error result, returns a new error result with the same error.
        
        Args:
            fn: Function to apply to the success value
            
        Returns:
            A new result with the mapped value or the same error
        """
        if self.is_success:
            try:
                new_value = fn(self.value)
                return Result.success(new_value)
            except Exception as e:
                return Result.failure(
                    "Error mapping result",
                    code=ErrorCode.UNKNOWN_ERROR,
                    original_error=e
                )
        else:
            return Result(error=self.error)
    
    def flat_map(self, fn: Callable[[T], 'Result[R]']) -> 'Result[R]':
        """
        Apply a function that returns a Result to the success value.
        
        Args:
            fn: Function that takes a value and returns a Result
            
        Returns:
            The Result returned by the function or an error
        """
        if self.is_success:
            try:
                return fn(self.value)
            except Exception as e:
                return Result.failure(
                    "Error flat-mapping result",
                    code=ErrorCode.UNKNOWN_ERROR,
                    original_error=e
                )
        else:
            return Result(error=self.error)
    
    def recover(self, fn: Callable[[PipelineError], T]) -> 'Result[T]':
        """
        Recover from an error by applying a function to it.
        
        Args:
            fn: Function that takes an error and returns a value
            
        Returns:
            A success result with the recovered value or the original success
        """
        if self.is_error:
            try:
                return Result.success(fn(self.error))
            except Exception as e:
                return Result.failure(
                    "Error recovering from failure",
                    code=ErrorCode.UNKNOWN_ERROR,
                    original_error=e
                )
        else:
            return self
    
    def __str__(self) -> str:
        """String representation of the result."""
        if self.is_success:
            return f"Success({self._value})"
        else:
            return f"Failure({self._error})"


class ErrorHandler:
    """
    Main error handler class for the pipeline.
    
    Tracks errors, provides fallback mechanisms, and handles recovery strategies.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.errors: List[PipelineError] = []
        self.fallback_handlers: Dict[ErrorCode, Callable] = {}
    
    def log_error(self, error: PipelineError) -> None:
        """
        Log an error and add it to the error history.
        
        Args:
            error: The error to log
        """
        self.errors.append(error)
        
        # Log based on severity
        error_dict = error.to_dict()
        
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(json.dumps(error_dict))
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(json.dumps(error_dict))
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(json.dumps(error_dict))
        else:  # LOW
            logger.info(json.dumps(error_dict))
    
    def register_fallback(
        self,
        error_code: ErrorCode,
        handler: Callable
    ) -> None:
        """
        Register a fallback handler for a specific error code.
        
        Args:
            error_code: The error code to handle
            handler: Function to call when the error occurs
        """
        self.fallback_handlers[error_code] = handler
    
    def get_fallback(self, error_code: ErrorCode) -> Optional[Callable]:
        """
        Get the fallback handler for an error code.
        
        Args:
            error_code: The error code to get a handler for
            
        Returns:
            The handler function or None if no handler is registered
        """
        return self.fallback_handlers.get(error_code)
    
    def handle_error(self, error: PipelineError) -> Optional[Any]:
        """
        Handle an error by logging it and calling the appropriate fallback.
        
        Args:
            error: The error to handle
            
        Returns:
            The result of the fallback handler or None if no handler is available
        """
        self.log_error(error)
        
        # Check if we have a fallback for this error code
        handler = self.get_fallback(error.code)
        if handler:
            try:
                return handler(error)
            except Exception as e:
                # Log fallback failure but don't try another fallback
                fallback_error = PipelineError(
                    message=f"Fallback handler for {error.code} failed",
                    code=ErrorCode.UNKNOWN_ERROR,
                    severity=ErrorSeverity.HIGH,
                    original_error=e,
                    context={"original_error_code": error.code.value}
                )
                self.log_error(fallback_error)
                return None
        
        return None
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all errors.
        
        Returns:
            Dictionary with error statistics
        """
        if not self.errors:
            return {"error_count": 0, "errors_by_severity": {}, "errors_by_code": {}}
        
        # Count errors by severity and code
        errors_by_severity: Dict[str, int] = {}
        errors_by_code: Dict[str, int] = {}
        
        for error in self.errors:
            sev = error.severity.value
            code = error.code.value
            
            errors_by_severity[sev] = errors_by_severity.get(sev, 0) + 1
            errors_by_code[code] = errors_by_code.get(code, 0) + 1
        
        # Get latest errors
        latest_errors = [e.to_dict() for e in self.errors[-5:]]
        
        return {
            "error_count": len(self.errors),
            "errors_by_severity": errors_by_severity,
            "errors_by_code": errors_by_code,
            "latest_errors": latest_errors
        }
    
    def reset(self) -> None:
        """Reset the error handler state (clear errors)."""
        self.errors = []


# Create a singleton instance
error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """
    Get the global error handler instance.
    
    Returns:
        The ErrorHandler instance
    """
    return error_handler


def with_error_handling(
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
) -> Callable:
    """
    Decorator to automatically handle errors in a function.
    
    Args:
        error_code: Default error code to use for exceptions
        severity: Default severity level for exceptions
        
    Returns:
        Decorator function that adds error handling
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Result]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Result:
            try:
                result = func(*args, **kwargs)
                
                # If the function already returns a Result, return it directly
                if isinstance(result, Result):
                    return result
                
                # Otherwise wrap the result in a success Result
                return Result.success(result)
            except PipelineError as e:
                # Already a PipelineError, just wrap it
                error_handler.log_error(e)
                return Result.failure(e)
            except Exception as e:
                # Wrap other exceptions
                pipeline_error = PipelineError(
                    message=f"Error in {func.__name__}: {str(e)}",
                    code=error_code,
                    severity=severity,
                    original_error=e,
                    context={"function": func.__name__}
                )
                error_handler.log_error(pipeline_error)
                return Result.failure(pipeline_error)
        
        return wrapper
    
    return decorator


def safe_execute(
    func: Callable[..., T],
    *args,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Result[T]:
    """
    Execute a function safely and return a Result.
    
    Args:
        func: Function to execute
        *args: Arguments for the function
        error_code: Error code to use if the function raises an exception
        severity: Severity level for exceptions
        context: Additional context for errors
        **kwargs: Keyword arguments for the function
        
    Returns:
        A Result containing either the function's return value or an error
    """
    try:
        result = func(*args, **kwargs)
        return Result.success(result)
    except Exception as e:
        pipeline_error = PipelineError(
            message=f"Error executing {func.__name__}: {str(e)}",
            code=error_code,
            severity=severity,
            original_error=e,
            context=context or {"function": func.__name__}
        )
        error_handler.log_error(pipeline_error)
        return Result.failure(pipeline_error)


if __name__ == "__main__":
    # Example usage
    
    # Register a fallback for LLM API errors
    def llm_api_fallback(error):
        print(f"LLM API fallback triggered for: {error}")
        return {"fallback_response": "This is a fallback response when the LLM API fails"}
    
    handler = get_error_handler()
    handler.register_fallback(ErrorCode.LLM_API_ERROR, llm_api_fallback)
    
    # Example function with error handling
    @with_error_handling(error_code=ErrorCode.LLM_API_ERROR)
    def generate_text(prompt: str) -> str:
        # Simulate an API error
        if "error" in prompt:
            raise Exception("API timeout")
        return f"Generated text for: {prompt}"
    
    # Test success case
    result = generate_text("Hello world")
    if result.is_success:
        print(f"Success: {result.value}")
    
    # Test error case
    result = generate_text("Trigger error")
    if result.is_error:
        print(f"Error: {result.error}")
        
        # Try the fallback
        fallback = handler.handle_error(result.error)
        if fallback:
            print(f"Fallback: {fallback}")
