"""
Core functionality for the AI YouTube Shorts Generator.

This package contains the core components for processing videos, handling user input,
and managing the overall pipeline for generating YouTube Shorts.
"""
from .input_parser import parse_user_input, InputParser, NarrativeMode, Tone
from .config import config, get_config, init_config, get_setting

__all__ = [
    'config',
    'get_config',
    'init_config',
    'get_setting',
    'parse_user_input',
    'InputParser',
    'NarrativeMode',
    'Tone',
]
