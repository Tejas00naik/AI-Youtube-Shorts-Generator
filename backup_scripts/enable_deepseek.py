#!/usr/bin/env python3
"""
Script to test DeepSeek API integration and solve authentication issues.
"""

import os
import sys
import json
import dotenv
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env
dotenv.load_dotenv()

# Check current environment settings
print("\n=== Current DeepSeek API Configuration ===")
print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER', 'Not set')}")
print(f"AI_MODE: {os.getenv('AI_MODE', 'Not set')}")
print(f"DEEPSEEK_API_KEY: {'Present' if os.getenv('DEEPSEEK_API_KEY') else 'Not set'}")
print(f"DEEPSEEK_MODEL: {os.getenv('DEEPSEEK_MODEL', 'Not set')}")

# Fix environment variables
os.environ["LLM_PROVIDER"] = "deepseek"  # This is what the code actually uses
print("\n=== Updated Environment Variables ===")
print(f"LLM_PROVIDER: {os.environ['LLM_PROVIDER']}")

# Check if we can initialize a DeepSeek client
print("\n=== Testing DeepSeek Client Initialization ===")
from llm.llm_client import LLMClient, LLMProvider

client = LLMClient(provider=LLMProvider.DEEPSEEK)
print(f"Provider: {client.provider.value}")
print(f"Model: {client.model}")
print(f"API Key Format: {'Valid format' if client.api_key and client.api_key.startswith('sk-') else 'Invalid or missing'}")

# Fix API key if needed
if not client.api_key or not client.api_key.startswith('sk-'):
    print("\n⚠️ DeepSeek API key is missing or improperly formatted!")
    print("Please ensure your .env file has a valid DeepSeek API key in this format:")
    print("\nDEEPSEEK_API_KEY=sk-your-api-key-here")
    
    # Provide instructions for adding the API key
    print("\n=== To fix this issue ===")
    print("1. Open your .env file")
    print("2. Add your DeepSeek API key in this format: DEEPSEEK_API_KEY=sk-your-api-key-here")
    print("3. Make sure the API key starts with 'sk-'")
    print("4. Ensure there are no spaces around the equals sign")
    
    # Generate a template .env file
    template_env = """# AI Mode: Set to 'local' to use Mistral 7B in LM Studio or 'openai' to use OpenAI API
AI_MODE=deepseek

# LLM Provider (this is what the code actually uses)
LLM_PROVIDER=deepseek

# DeepSeek API Settings (add your key below)
DEEPSEEK_API_KEY=sk-your-api-key-here
DEEPSEEK_MODEL=reasoner
"""
    
    with open("template.env", "w") as f:
        f.write(template_env)
    
    print(f"\nA template .env file has been created at: {Path('template.env').absolute()}")
    print("Copy this file to .env and replace 'sk-your-api-key-here' with your actual DeepSeek API key.")
else:
    print("\n✅ DeepSeek API key format is valid!")
    
    # Test a simple API call if the key format looks valid
    print("\n=== Testing DeepSeek API Call ===")
    try:
        response = client.chat_completion(
            system_prompt="You are a helpful assistant.",
            user_messages="Say hello world!",
            temperature=0.7
        )
        
        if "error" in response:
            print(f"❌ API call failed: {response['error']}")
        else:
            print("✅ API call successful!")
            print(f"Response: {response.get('content', 'No content')[:100]}...")
            
            # Write updated environment settings to a file for reference
            with open("deepseek_test_results.json", "w") as f:
                json.dump({
                    "provider": client.provider.value,
                    "model": client.model,
                    "api_key_format_valid": bool(client.api_key and client.api_key.startswith('sk-')),
                    "api_call_successful": "error" not in response,
                    "response_sample": response.get('content', 'No content')[:100] + "..."
                }, f, indent=2)
            
            print(f"\nTest results saved to: {Path('deepseek_test_results.json').absolute()}")
    except Exception as e:
        print(f"❌ Error testing API: {str(e)}")
        
print("\n=== Next Steps ===")
print("1. Ensure your .env file has the correct DeepSeek API key")
print("2. Make sure both LLM_PROVIDER and AI_MODE are set to 'deepseek'")
print("3. Run the component_tester.py script again to test the full pipeline")
