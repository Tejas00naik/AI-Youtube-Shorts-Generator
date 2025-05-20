"""
Generic LLM Client Interface for AI YouTube Shorts Generator.

This module provides a unified interface for interacting with different LLM APIs,
allowing the application to be provider-agnostic.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Union
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # For local models via API

class LLMClient:
    """
    Generic LLM client that provides a unified interface for different LLM providers.
    """
    
    def __init__(self, provider: Union[str, LLMProvider] = None, api_key: Optional[str] = None,
                model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider to use (openai, deepseek, anthropic, local)
            api_key: API key for the provider
            model: Model name to use
            base_url: Base URL for the API (useful for local models)
        """
        # Set provider
        if provider is None:
            provider = os.getenv("LLM_PROVIDER", "openai")
        
        if isinstance(provider, str):
            try:
                self.provider = LLMProvider(provider.lower())
            except ValueError:
                logger.warning(f"Unknown provider: {provider}, defaulting to OpenAI")
                self.provider = LLMProvider.OPENAI
        else:
            self.provider = provider
            
        # Set API key
        self.api_key = api_key
        if self.api_key is None:
            if self.provider == LLMProvider.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == LLMProvider.DEEPSEEK:
                self.api_key = os.getenv("DEEPSEEK_API_KEY")
            elif self.provider == LLMProvider.ANTHROPIC:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == LLMProvider.LOCAL:
                # Local models might not need an API key
                self.api_key = os.getenv("LOCAL_API_KEY")
        
        # Set model
        self.model = model
        if self.model is None:
            if self.provider == LLMProvider.OPENAI:
                self.model = os.getenv("OPENAI_MODEL", "gpt-4")
            elif self.provider == LLMProvider.DEEPSEEK:
                self.model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            elif self.provider == LLMProvider.ANTHROPIC:
                self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus")
            elif self.provider == LLMProvider.LOCAL:
                self.model = os.getenv("LOCAL_MODEL", "mistral-7b")
        
        # Set base URL (useful for local models or custom endpoints)
        self.base_url = base_url
        if self.base_url is None:
            if self.provider == LLMProvider.LOCAL:
                self.base_url = os.getenv("LOCAL_API_URL", "http://localhost:8000")
            elif self.provider == LLMProvider.DEEPSEEK:
                self.base_url = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com")
        
        # Initialize client
        self.client = None
        self._init_client()
        
        logger.info(f"Initialized LLM client with provider: {self.provider.value}, model: {self.model}")
    
    def _init_client(self):
        """Initialize the specific LLM client based on provider."""
        try:
            if self.provider == LLMProvider.OPENAI:
                self._init_openai()
            elif self.provider == LLMProvider.DEEPSEEK:
                self._init_deepseek()
            elif self.provider == LLMProvider.ANTHROPIC:
                self._init_anthropic()
            elif self.provider == LLMProvider.LOCAL:
                self._init_local()
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider.value} client: {str(e)}")
            logger.warning(f"LLM client initialization failed. Some features may not work.")
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            if self.api_key:
                self.client = openai.OpenAI(api_key=self.api_key)
                if self.base_url:
                    self.client.base_url = self.base_url
            else:
                logger.warning("OpenAI API key not provided. OpenAI features will not work.")
        except ImportError:
            logger.warning("OpenAI module not installed. Run: pip install openai")
    
    def _init_deepseek(self):
        """Initialize DeepSeek client."""
        try:
            import requests
            self.client = {
                "api_key": self.api_key,
                "base_url": self.base_url or "https://api.deepseek.com"
            }
        except ImportError:
            logger.warning("Requests module not installed. Run: pip install requests")
    
    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            else:
                logger.warning("Anthropic API key not provided. Anthropic features will not work.")
        except ImportError:
            logger.warning("Anthropic module not installed. Run: pip install anthropic")
    
    def _init_local(self):
        """Initialize local model client."""
        try:
            import requests
            self.client = {
                "base_url": self.base_url or "http://localhost:8000"
            }
        except ImportError:
            logger.warning("Requests module not installed. Run: pip install requests")
    
    def is_available(self) -> bool:
        """Check if the LLM client is available."""
        return self.client is not None
    
    def chat_completion(self, 
                      system_prompt: str,
                      user_messages: Union[str, List[Dict[str, str]]],
                      temperature: float = 0.7,
                      json_response: bool = False,
                      max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generic chat completion method that works with different LLM providers.
        
        Args:
            system_prompt: System prompt to use
            user_messages: User message(s) - either a string or list of message dicts
            temperature: Temperature for generation
            json_response: Whether to request a JSON response
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with response content and metadata
        """
        if not self.client:
            return {"error": f"{self.provider.value} client not initialized"}
        
        # Format messages
        messages = [{"role": "system", "content": system_prompt}]
        
        if isinstance(user_messages, str):
            messages.append({"role": "user", "content": user_messages})
        else:
            for msg in user_messages:
                messages.append(msg)
        
        try:
            # Route to appropriate provider-specific method
            if self.provider == LLMProvider.OPENAI:
                return self._openai_chat_completion(messages, temperature, json_response, max_tokens)
            elif self.provider == LLMProvider.DEEPSEEK:
                return self._deepseek_chat_completion(messages, temperature, json_response, max_tokens)
            elif self.provider == LLMProvider.ANTHROPIC:
                return self._anthropic_chat_completion(messages, temperature, json_response, max_tokens)
            elif self.provider == LLMProvider.LOCAL:
                return self._local_chat_completion(messages, temperature, json_response, max_tokens)
            else:
                return {"error": f"Unsupported provider: {self.provider.value}"}
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            return {"error": str(e)}
    
    def _openai_chat_completion(self, messages, temperature, json_response, max_tokens):
        """Handle OpenAI-specific chat completion."""
        response_format = {"type": "json_object"} if json_response else None
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            response_format=response_format,
            max_tokens=max_tokens
        )
        
        return {
            "content": response.choices[0].message.content,
            "finish_reason": response.choices[0].finish_reason,
            "model": response.model,
            "id": response.id
        }
    
    def _deepseek_chat_completion(self, messages, temperature, json_response, max_tokens):
        """Handle DeepSeek-specific chat completion."""
        import requests
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.client['api_key']}"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        if json_response:
            data["response_format"] = {"type": "json_object"}
        
        # DeepSeek requires the word 'json' to be in the prompt when using json_object response format
        # Inject it into the system message if it's not already there
        if any(msg.get('role') == 'system' for msg in messages):
            for msg in messages:
                if msg.get('role') == 'system':
                    if 'json' not in msg.get('content', '').lower():
                        msg['content'] = f"{msg['content']}\n\nPlease format your response as a valid JSON."
        else:
            # Add a system message if none exists
            messages.insert(0, {"role": "system", "content": "Please format your response as a valid JSON."})
        
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        response = requests.post(
            f"{self.client['base_url']}/v1/chat/completions",
            headers=headers,
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "content": result["choices"][0]["message"]["content"],
                "finish_reason": result["choices"][0]["finish_reason"],
                "model": result["model"],
                "id": result["id"]
            }
        else:
            raise Exception(f"DeepSeek API error: {response.text}")
    
    def _anthropic_chat_completion(self, messages, temperature, json_response, max_tokens):
        """Handle Anthropic-specific chat completion."""
        # Convert messages to Anthropic format
        system = ""
        human_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                human_messages.append({
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["content"]
                })
        
        response = self.client.messages.create(
            model=self.model,
            system=system,
            messages=human_messages,
            temperature=temperature,
            max_tokens=max_tokens or 4096
        )
        
        return {
            "content": response.content[0].text,
            "finish_reason": response.stop_reason,
            "model": response.model,
            "id": response.id
        }
    
    def _local_chat_completion(self, messages, temperature, json_response, max_tokens):
        """Handle local model-specific chat completion."""
        import requests
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }
        
        if max_tokens:
            data["max_tokens"] = max_tokens
        
        response = requests.post(
            f"{self.client['base_url']}/v1/chat/completions",
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return {
                "content": result["choices"][0]["message"]["content"],
                "finish_reason": result["choices"][0]["finish_reason"],
                "model": result.get("model", self.model),
                "id": result.get("id", "local-completion")
            }
        else:
            raise Exception(f"Local API error: {response.text}")


# Helper function to get LLM client
def get_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None
) -> LLMClient:
    """
    Get an initialized LLM client.
    
    Args:
        provider: LLM provider to use (openai, deepseek, anthropic, local)
        api_key: API key for the provider
        model: Model name to use
        base_url: Base URL for the API (useful for local models)
        
    Returns:
        Initialized LLMClient
    """
    return LLMClient(provider=provider, api_key=api_key, model=model, base_url=base_url)
