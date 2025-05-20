"""
Configuration file for pytest.

This file provides fixtures and configuration for all tests.
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent
os.environ['PYTHONPATH'] = str(PROJECT_ROOT)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create and clean up a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    try:
        yield Path(temp_dir)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_audio_file(temp_dir: Path) -> Path:
    """Create a sample audio file for testing."""
    # This is a minimal WAV file header for testing
    audio_data = (
        b'RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00'  # noqa: E501
        b'\x80\xbb\x00\x00\x00\x77\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00'
    )
    
    audio_path = temp_dir / 'test_audio.wav'
    with open(audio_path, 'wb') as f:
        f.write(audio_data)
    
    return audio_path


@pytest.fixture
def sample_video_file(temp_dir: Path) -> Path:
    """Create a sample video file for testing."""
    # This is a minimal MP4 header for testing
    video_data = (
        b'\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2avc1mp41\x00\x00\x00\x08'
        b'free\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    )
    
    video_path = temp_dir / 'test_video.mp4'
    with open(video_path, 'wb') as f:
        f.write(video_data)
    
    return video_path


@pytest.fixture
def mock_openai_response() -> dict:
    """Return a mock OpenAI API response."""
    return {
        'choices': [{
            'message': {
                'content': 'Mock response from OpenAI',
                'role': 'assistant'
            },
            'finish_reason': 'stop',
            'index': 0
        }],
        'created': 1234567890,
        'id': 'chatcmpl-123',
        'model': 'gpt-4',
        'object': 'chat.completion',
        'usage': {
            'completion_tokens': 10,
            'prompt_tokens': 20,
            'total_tokens': 30
        }
    }


@pytest.fixture
def mock_whisper_response() -> dict:
    """Return a mock Whisper API response."""
    return {
        'text': 'This is a test transcription.',
        'segments': [
            {
                'id': 0,
                'start': 0.0,
                'end': 2.0,
                'text': 'This is a test',
                'words': [
                    {'word': 'This', 'start': 0.0, 'end': 0.5, 'probability': 0.99},
                    {'word': 'is', 'start': 0.5, 'end': 0.8, 'probability': 0.98},
                    {'word': 'a', 'start': 0.8, 'end': 0.9, 'probability': 0.97},
                    {'word': 'test', 'start': 0.9, 'end': 1.2, 'probability': 0.99}
                ]
            },
            {
                'id': 1,
                'start': 2.0,
                'end': 3.5,
                'text': 'transcription.',
                'words': [
                    {'word': 'transcription', 'start': 2.0, 'end': 3.0, 'probability': 0.99},
                    {'word': '.', 'start': 3.0, 'end': 3.5, 'probability': 0.99}
                ]
            }
        ],
        'language': 'en'
    }
