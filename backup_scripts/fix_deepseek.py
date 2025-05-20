#!/usr/bin/env python3
"""
Script to fix DeepSeek API integration by setting proper environment variables
and using a valid model name.
"""

import os
import sys
import dotenv
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=== Fixing DeepSeek API Configuration ===")

# Create a copy of the .env file with updated settings
env_path = Path(".env")
backup_path = Path(".env.backup")

if env_path.exists():
    # Create backup
    with open(env_path, "r") as src, open(backup_path, "w") as dest:
        dest.write(src.read())
    print(f"Created backup of .env file at: {backup_path}")
    
    # Read the current .env content
    with open(env_path, "r") as f:
        env_content = f.read()
    
    # Update environment variables with proper values
    lines = env_content.splitlines()
    updated_lines = []
    found_llm_provider = False
    found_deepseek_model = False
    
    for line in lines:
        # Add LLM_PROVIDER if it doesn't exist
        if line.startswith("LLM_PROVIDER="):
            updated_lines.append("LLM_PROVIDER=deepseek")
            found_llm_provider = True
        # Update DeepSeek model name to a valid one
        elif line.startswith("DEEPSEEK_MODEL="):
            updated_lines.append("DEEPSEEK_MODEL=deepseek-chat")
            found_deepseek_model = True
        else:
            updated_lines.append(line)
    
    # Add missing variables if not found
    if not found_llm_provider:
        updated_lines.append("LLM_PROVIDER=deepseek")
    if not found_deepseek_model:
        updated_lines.append("DEEPSEEK_MODEL=deepseek-chat")
    
    # Write updated .env file
    with open(env_path, "w") as f:
        f.write("\n".join(updated_lines))
    
    print("Updated .env file with correct DeepSeek configuration")
    
    # Test if the changes fixed the issue
    print("\nTesting DeepSeek API with updated configuration...")
    
    # Reload environment
    dotenv.load_dotenv(override=True)
    
    # Set correct values for this session
    os.environ["LLM_PROVIDER"] = "deepseek"
    os.environ["DEEPSEEK_MODEL"] = "deepseek-chat"
    
    # Import and test LLM client
    from llm.llm_client import LLMClient, LLMProvider
    
    client = LLMClient(provider=LLMProvider.DEEPSEEK)
    print(f"Provider: {client.provider.value}")
    print(f"Model: {client.model}")
    
    # Test API call
    print("\nTesting API call...")
    try:
        response = client.chat_completion(
            system_prompt="You are a helpful assistant.",
            user_messages="Say hello world briefly.",
            temperature=0.7
        )
        
        if "error" in response:
            print(f"API call failed: {response['error']}")
        else:
            print("✅ API call successful!")
            print(f"Response: {response.get('content', 'No content')}")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    print("\n=== Next Steps ===")
    print("Now run your component test script to test the full pipeline:")
    print("python component_tester.py")
else:
    print("Error: .env file not found in the current directory.")
    print("Please create an .env file with your DeepSeek API settings.")
