"""
Tests for the configuration module.
"""
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from core.config import Config, get_config, init_config, get_setting


class TestConfig:
    """Test cases for the Config class."""
    
    def test_default_values(self):
        """Test that default values are set correctly."""
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.get('OPENAI_MODEL') == 'gpt-4'
            assert config.get('DEFAULT_VIDEO_WIDTH') == 1080
            assert config.get('DEFAULT_VIDEO_HEIGHT') == 1920
            assert config.get('MAX_VIDEO_DURATION') == 58
    
    def test_environment_override(self):
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {
            'OPENAI_MODEL': 'gpt-3.5-turbo',
            'DEFAULT_VIDEO_WIDTH': '720',
            'MAX_VIDEO_DURATION': '45'
        }, clear=True):
            config = Config()
            assert config.get('OPENAI_MODEL') == 'gpt-3.5-turbo'
            assert config.get('DEFAULT_VIDEO_WIDTH') == 720
            assert config.get('MAX_VIDEO_DURATION') == 45
    
    def test_type_conversion(self):
        """Test that values are converted to the correct types."""
        with patch.dict(os.environ, {
            'DEFAULT_VIDEO_WIDTH': '800',  # int
            'FACE_DETECTION_CONFIDENCE': '0.8',  # float
            'DEBUG': 'true'  # bool
        }, clear=True):
            config = Config()
            assert isinstance(config.get('DEFAULT_VIDEO_WIDTH'), int)
            assert isinstance(config.get('FACE_DETECTION_CONFIDENCE'), float)
            assert isinstance(config.get('DEBUG'), bool)
            assert config.get('DEBUG') is True
    
    def test_required_directories_created(self, tmp_path):
        """Test that required directories are created."""
        with patch.dict(os.environ, {
            'OUTPUT_DIR': str(tmp_path / 'output'),
            'TEMP_DIR': str(tmp_path / 'temp')
        }, clear=True):
            config = Config()
            
            # Check that directories were created
            assert (tmp_path / 'output').exists()
            assert (tmp_path / 'temp').exists()
    
    def test_validation(self):
        """Test configuration validation."""
        with patch.dict(os.environ, {
            'DEFAULT_VIDEO_WIDTH': '0',  # Invalid
            'MAX_VIDEO_DURATION': '-10'  # Invalid
        }, clear=True):
            with pytest.raises(ValueError):
                Config()
    
    def test_get_nonexistent_key(self):
        """Test getting a non-existent key with a default value."""
        config = Config()
        assert config.get('NON_EXISTENT_KEY', 'default') == 'default'
    
    def test_getitem_nonexistent_key(self):
        """Test that __getitem__ raises KeyError for non-existent keys."""
        config = Config()
        with pytest.raises(KeyError):
            _ = config['NON_EXISTENT_KEY']


class TestConfigModule:
    """Test cases for the config module functions."""
    
    def test_get_config_singleton(self):
        """Test that get_config returns a singleton instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2
    
    def test_init_config(self, tmp_path):
        """Test initializing config with a custom environment file."""
        # Create a temporary .env file
        env_file = tmp_path / '.env.test'
        env_file.write_text('''
        OPENAI_MODEL=custom-model
        DEFAULT_VIDEO_WIDTH=999
        ''')
        
        # Initialize config with the custom env file
        init_config(env_file)
        config = get_config()
        
        # Check that values from the custom env file are loaded
        assert config.get('OPENAI_MODEL') == 'custom-model'
        assert config.get('DEFAULT_VIDEO_WIDTH') == 999
    
    def test_get_setting(self):
        """Test the get_setting convenience function."""
        # Use a fresh environment with known values
        with patch.dict(os.environ, {
            'OPENAI_MODEL': 'gpt-4-test'
        }, clear=True):
            # Reset to config with our environment
            init_config()
            
            # Test getting a setting
            assert get_setting('OPENAI_MODEL') == 'gpt-4-test'
            assert get_setting('NON_EXISTENT', 'default') == 'default'


class TestConfigLogging:
    """Test cases for logging configuration."""
    
    def test_log_level_setting(self, caplog):
        """Test that log level is set correctly."""
        with patch.dict(os.environ, {'LOG_LEVEL': 'DEBUG'}, clear=True):
            import logging
            # Instead of trying to reload the module, let's just reinitialize the config
            from core.config import init_config
            
            # Initialize with debug log level
            init_config()
            
            logger = logging.getLogger('test_logger')
            logger.debug('This is a debug message')
            
            # Check that debug messages are captured
            assert 'This is a debug message' in caplog.text
    
    def test_debug_mode(self, caplog):
        """Test that debug mode enables debug logging."""
        with patch.dict(os.environ, {'DEBUG': 'true'}, clear=True):
            import logging
            # Instead of trying to reload the module, let's just reinitialize the config
            from core.config import init_config
            
            # Initialize with debug mode enabled
            init_config()
            
            logger = logging.getLogger('test_logger')
            logger.debug('Debug message in debug mode')
            
            # Check that debug messages are captured
            assert 'Debug message in debug mode' in caplog.text
