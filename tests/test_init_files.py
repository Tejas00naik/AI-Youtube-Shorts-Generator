"""
Tests for __init__.py files to ensure proper API exposure.
"""
import importlib
import pytest

# List of modules to test with their expected public API
MODULES_TO_TEST = [
    ('core', [
        'config',
        'get_config',
        'init_config',
        'get_setting',
        'parse_user_input',
        'InputParser',
        'NarrativeMode',
        'Tone',
    ]),
    ('llm', [
        'NarrativePlanner',
        'generate_narrative_plan',
        'NarrativeMode',
    ])
]

@pytest.mark.parametrize("module_name,expected_attrs", MODULES_TO_TEST)
def test_module_exports(module_name, expected_attrs):
    """Test that modules expose the expected public API."""
    # Import the module
    module = importlib.import_module(module_name)
    
    # Check that __all__ is defined
    assert hasattr(module, '__all__'), f"{module_name} should define __all__"
    
    # Get the actual exported symbols
    actual_attrs = module.__all__
    
    # Check that all expected attributes are exported
    for attr in expected_attrs:
        assert attr in actual_attrs, f"{module_name} should export '{attr}'"
        assert hasattr(module, attr), f"{module_name} should have attribute '{attr}'"
    
    # Check that there are no unexpected exports
    unexpected = set(actual_attrs) - set(expected_attrs)
    assert not unexpected, f"{module_name} has unexpected exports: {unexpected}"

def test_core_init_imports():
    """Test that core/__init__.py imports all necessary components."""
    from core import (
        config,
        get_config,
        init_config,
        get_setting,
        parse_user_input,
        InputParser,
        NarrativeMode,
        Tone,
    )
    
    # Just verify the imports work
    assert config is not None
    assert callable(get_config)
    assert callable(init_config)
    assert callable(get_setting)
    assert callable(parse_user_input)
    assert hasattr(InputParser, '__init__')
    assert isinstance(NarrativeMode.TUTORIAL, NarrativeMode)
    assert isinstance(Tone.PROFESSIONAL, Tone)

def test_llm_init_imports():
    """Test that llm/__init__.py imports all necessary components."""
    from llm import (
        NarrativePlanner,
        generate_narrative_plan,
        NarrativeMode,
    )
    
    # Just verify the imports work
    assert hasattr(NarrativePlanner, '__init__')
    assert callable(generate_narrative_plan)
    assert isinstance(NarrativeMode.HIGHLIGHT, NarrativeMode)
