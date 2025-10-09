"""
LLM Interface module supporting cloud (Groq, Gemini, Azure OpenAI) models.
Optimized for performance with timeout and context management.
SSL issues handled comprehensively.
"""

import requests
import json
import logging
import certifi
import ssl
import os
from typing import Optional
from abc import ABC, abstractmethod
import config
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Fix SSL certificate issues globally
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

class GroqLLM(BaseLLM):
    """Interface for Groq cloud LLM using direct REST API."""
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODELS[0]
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        if not self.api_key:
            logger.warning("Groq API key not configured")
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            return "Error: Groq not configured. Please set GROQ_API_KEY in .env"
        try:
            max_length = 4000
            if len(prompt) > max_length:
                prompt = prompt[:max_length] + "..."
                logger.warning(f"Truncated Groq prompt to {max_length} characters")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt[:500]})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 5000
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30,
                verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                choices = result.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"]["content"]
                return "Error: Groq returned empty response"
            else:
                logger.error(f"Groq error: {response.status_code} - {response.text}")
                if response.status_code == 401:
                    return "Error: Invalid Groq API key. Please check your .env file."
                elif response.status_code == 429:
                    return "Error: Groq API rate limit exceeded. Please try again later."
                else:
                    return f"Error: Unable to generate response (Status: {response.status_code})"
                    
        except requests.exceptions.Timeout:
            logger.error("Groq request timeout")
            return "Error: Request timeout. Please try again."
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Groq connection error: {str(e)}")
            return "Error: Cannot connect to Groq API. Check your internet connection."
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Groq generation error: {error_msg}")
            return f"Error: {error_msg}"

class GeminiLLM(BaseLLM):
    """Interface for Google Gemini cloud LLM."""
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key or config.GEMINI_API_KEY
        self.model = model or config.GEMINI_MODELS[0]
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
        if not self.api_key:
            logger.warning("Gemini API key not configured")
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            return "Error: Gemini not configured. Please set GEMINI_API_KEY in .env"
        try:
            # Gemini uses API key in URL parameter, not Authorization header
            url = f"{self.api_url}?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            
            # Build the proper Gemini request format
            parts = []
            if system_prompt:
                parts.append({"text": system_prompt[:500]})
            parts.append({"text": prompt})
            
            payload = {
                "contents": [{
                    "parts": parts
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 5000
                }
            }
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30,
                verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                candidates = result.get("candidates", [])
                if candidates and "content" in candidates[0]:
                    parts = candidates[0]["content"].get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
                return "Error: Gemini returned empty response"
            else:
                logger.error(f"Gemini error: {response.status_code} - {response.text}")
                if response.status_code == 400:
                    return "Error: Invalid Gemini API key or request format."
                elif response.status_code == 429:
                    return "Error: Gemini API rate limit exceeded. Please try again later."
                else:
                    return f"Error: Unable to generate response (Status: {response.status_code})"
                    
        except requests.exceptions.Timeout:
            logger.error("Gemini request timeout")
            return "Error: Request timeout. Please try again."
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Gemini connection error: {str(e)}")
            return "Error: Cannot connect to Gemini API. Check your internet connection."
        except Exception as e:
            logger.error(f"Gemini generation error: {str(e)}")
            return f"Error: {str(e)}"

class AzureOpenAILLM(BaseLLM):
    """Interface for Azure OpenAI cloud LLM."""
    def __init__(self, api_key: str = None, endpoint: str = None, model: str = None, api_version: str = None):
        self.api_key = api_key or config.AZURE_OPENAI_API_KEY
        self.endpoint = endpoint or config.AZURE_OPENAI_ENDPOINT
        self.model = model or config.AZURE_OPENAI_MODELS[0]
        self.api_version = api_version or config.AZURE_OPENAI_API_VERSION
        if not self.api_key or not self.endpoint:
            logger.warning("Azure OpenAI API key or endpoint not configured")
    
    def is_available(self) -> bool:
        return bool(self.api_key and self.endpoint)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key or not self.endpoint:
            return "Error: Azure OpenAI not configured. Please set API key and endpoint in .env"
        try:
            url = f"{self.endpoint}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}"
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt[:500]})
            messages.append({"role": "user", "content": prompt})
            
            payload = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 5000
            }
            
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=30,
                verify=False
            )
            
            if response.status_code == 200:
                result = response.json()
                choices = result.get("choices", [])
                if choices and "message" in choices[0]:
                    return choices[0]["message"]["content"]
                return "Error: Azure OpenAI returned empty response"
            else:
                logger.error(f"Azure OpenAI error: {response.status_code} - {response.text}")
                if response.status_code == 401:
                    return "Error: Invalid Azure OpenAI API key."
                elif response.status_code == 404:
                    return "Error: Azure OpenAI deployment not found. Check endpoint and model name."
                elif response.status_code == 429:
                    return "Error: Azure OpenAI rate limit exceeded. Please try again later."
                else:
                    return f"Error: Unable to generate response (Status: {response.status_code})"
                    
        except requests.exceptions.Timeout:
            logger.error("Azure OpenAI request timeout")
            return "Error: Request timeout. Please try again."
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Azure OpenAI connection error: {str(e)}")
            return "Error: Cannot connect to Azure OpenAI. Check your internet connection."
        except Exception as e:
            logger.error(f"Azure OpenAI generation error: {str(e)}")
            return f"Error: {str(e)}"

class LLMManager:
    """Manages multiple cloud LLM backends with fallback."""
    def __init__(self, provider=None, model_name=None):
        self.groq = GroqLLM()
        self.gemini = GeminiLLM()
        self.azure_openai = AzureOpenAILLM()
    
    def _select_active_llm(self):
        # If specific provider requested, use that
        if self.provider:
            if self.provider.lower() == "groq" and self.groq.is_available():
                logger.info(f"Using requested Groq LLM: {self.model_name}")
                if self.model_name:
                    self.groq.model = self.model_name
                return self.groq
            elif self.provider.lower() == "gemini" and self.gemini.is_available():
                logger.info(f"Using requested Gemini LLM: {self.model_name}")
                if self.model_name:
                    self.gemini.model = self.model_name
                return self.gemini
            elif self.provider.lower() == "azure_openai" and self.azure_openai.is_available():
                logger.info(f"Using requested Azure OpenAI LLM: {self.model_name}")
                if self.model_name:
                    self.azure_openai.model = self.model_name
                return self.azure_openai
        
        # Fallback to priority: Groq > Gemini > Azure OpenAI
        for llm in [self.groq, self.gemini, self.azure_openai]:
            if llm.is_available():
                logger.info(f"Using {llm.__class__.__name__} as primary LLM")
                return llm
        logger.warning("No LLM available!")
        return None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.active_llm:
            return "Error: No LLM available. Please configure API keys."
        
        result = self.active_llm.generate(prompt, system_prompt)
        
        # Fallback logic
        if result.startswith("Error:"):
            for llm in [self.groq, self.gemini, self.azure_openai]:
                if llm != self.active_llm and llm.is_available():
                    logger.warning(f"{self.active_llm.__class__.__name__} failed, falling back to {llm.__class__.__name__}")
                    result = llm.generate(prompt, system_prompt)
                    if not result.startswith("Error:"):
                        self.active_llm = llm
                        break
        return result
    
    def get_active_model(self) -> str:
        if self.active_llm:
            return f"{self.active_llm.__class__.__name__} ({getattr(self.active_llm, 'model', 'unknown')})"
        return "None"
    
    def is_working(self) -> bool:
        return self.active_llm is not None

# Example usage and testing
if __name__ == "__main__":
    print("Testing LLM Interface...\n")
    manager = LLMManager()
    print(f"Active model: {manager.get_active_model()}")
    if manager.is_working():
        response = manager.generate("Explain AI in one sentence.")
        print(f"Response: {response}")
    else:
        print("✗ No LLM working")
    print("\n✓ LLM Interface module ready!")
