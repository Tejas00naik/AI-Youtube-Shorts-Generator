"""Test script for error handler."""
from core.error_handler import safe_execute, ErrorCode

def divide(a, b):
    return a / b

result = safe_execute(divide, 10, 0, error_code=ErrorCode.PROCESSING_TIMEOUT)
print(f'Is success: {result.is_success}')
print(f'Error code: {result.error.code}' if result.is_error else 'No error')
