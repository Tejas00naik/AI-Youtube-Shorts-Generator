"""
Configuration management for the AI YouTube Shorts Generator.

This module handles loading and validating environment variables,
providing default values, and managing configuration settings.
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

class Config:
    """Configuration class for the application."""
    
    # Default configuration values
    DEFAULTS = {
        # OpenAI Configuration
        'OPENAI_API_KEY': '',
        'OPENAI_MODEL': 'gpt-4',
        
        # Video Generation Settings
        'DEFAULT_VIDEO_WIDTH': 1080,
        'DEFAULT_VIDEO_HEIGHT': 1920,
        'MAX_VIDEO_DURATION': 58,  # seconds
        
        # Audio Settings
        'SAMPLE_RATE': 44100,
        'BITRATE': '192k',
        'AUDIO_CHANNELS': 2,
        
        # Whisper Settings
        'WHISPER_MODEL': 'base',  # base, small, medium, large, large-v2
        'WHISPER_DEVICE': 'auto',  # cuda, cpu, or auto
        
        # Face Detection Settings
        'FACE_DETECTION_CONFIDENCE': 0.7,
        'MIN_FACE_SIZE': 50,  # pixels
        
        # Debug Settings
        'DEBUG': False,
        'LOG_LEVEL': 'INFO',
        
        # Output Directories
        'OUTPUT_DIR': 'output',
        'TEMP_DIR': 'temp',
    }
    
    def __init__(self):
        """Initialize the configuration with environment variables."""
        self._config = {}
        self._load_environment()
        self._setup_logging()
        self._create_directories()
    
    def _load_environment(self) -> None:
        """Load configuration from environment variables."""
        # Load all default values first
        self._config = self.DEFAULTS.copy()
        
        # Override with environment variables
        for key in self.DEFAULTS.keys():
            env_value = os.getenv(key)
            if env_value is not None:
                # Convert to the same type as the default value
                default_value = self.DEFAULTS[key]
                if isinstance(default_value, bool):
                    self._config[key] = env_value.lower() in ('true', '1', 't', 'y', 'yes')
                elif isinstance(default_value, int):
                    try:
                        self._config[key] = int(env_value)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid value for {key}, using default: {default_value}")
                elif isinstance(default_value, float):
                    try:
                        self._config[key] = float(env_value)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid value for {key}, using default: {default_value}")
                else:
                    self._config[key] = env_value
        
        # Validate required settings
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate the configuration."""
        # Check for required settings
        if not self.get('OPENAI_API_KEY'):
            logger.warning("OPENAI_API_KEY is not set. Some features may not work.")
        
        # Validate video dimensions
        if self.get('DEFAULT_VIDEO_WIDTH') <= 0 or self.get('DEFAULT_VIDEO_HEIGHT') <= 0:
            raise ValueError("Video dimensions must be positive integers")
        
        # Validate max duration
        if self.get('MAX_VIDEO_DURATION') <= 0:
            raise ValueError("Maximum video duration must be a positive number")
    
    def _setup_logging(self) -> None:
        """Configure logging based on the configuration."""
        log_level = self.get('LOG_LEVEL', 'INFO').upper()
        numeric_level = getattr(logging, log_level, None)
        
        if not isinstance(numeric_level, int):
            logger.warning(f"Invalid log level: {log_level}, defaulting to INFO")
            numeric_level = logging.INFO
        
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        
        if self.get('DEBUG'):
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug mode enabled")
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_key in ['OUTPUT_DIR', 'TEMP_DIR']:
            dir_path = Path(self.get(dir_key))
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        
        Args:
            key: Configuration key
            default: Default value to return if key is not found
            
        Returns:
            The configuration value or default if not found
        """
        return self._config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get a configuration value using dict-like access."""
        if key not in self._config:
            raise KeyError(f"Configuration key not found: {key}")
        return self._config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if a configuration key exists."""
        return key in self._config
    
    def to_dict(self) -> Dict[str, Any]:
        """Return the configuration as a dictionary."""
        return self._config.copy()


# Create a singleton instance
config = Config()


def get_config() -> Config:
    """
    Get the application configuration.
    
    Returns:
        The Config instance
    """
    return config


def init_config(env_file: Optional[Union[str, Path]] = None) -> None:
    """
    Initialize the configuration with an optional environment file.
    
    Args:
        env_file: Path to the environment file to load
    """
    if env_file:
        load_dotenv(env_file)
    global config
    config = Config()


# For backward compatibility
def get_setting(key: str, default: Any = None) -> Any:
    """
    Get a configuration setting (legacy function).
    
    Args:
        key: Configuration key
        default: Default value to return if key is not found
        
    Returns:
        The configuration value or default if not found
    """
    return config.get(key, default)


if __name__ == "__main__":
    # Print the current configuration
    import json
    print("Current configuration:")
    print(json.dumps(config.to_dict(), indent=2, default=str))
