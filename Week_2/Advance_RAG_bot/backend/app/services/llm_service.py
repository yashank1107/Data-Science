import google.generativeai as genai
from groq import Groq
import cohere
import base64
import requests
from typing import List, Optional, Dict, Any
import os
import logging
from config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Initialize clients only if API keys are available
        self.groq_client = None
        self.cohere_client = None
        
        if settings.GROQ_API_KEY:
            try:
                self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
                logger.info("Groq client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        
        if settings.COHERE_API_KEY:
            try:
                self.cohere_client = cohere.Client(settings.COHERE_API_KEY)
                logger.info("Cohere client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Cohere client: {e}")
        
        if settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                logger.info("Google Gemini client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Google Gemini client: {e}")
        else:
            logger.warning("Google API key not found")
    
    async def generate_response(
        self, 
        llm_choice: str,
        prompt: str,
        images: Optional[List[str]] = None,
        context: Optional[str] = None,
        document_context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        
        try:
            provider, model = llm_choice.split(":", 1) if ":" in llm_choice else (llm_choice, "")
            
            if provider == "gemini":
                return await self._generate_gemini_response(model, prompt, images, context, document_context)
            elif provider == "groq":
                return await self._generate_groq_response(model, prompt, context, document_context)
            elif provider == "cohere":
                return await self._generate_cohere_response(model, prompt, images, context, document_context)
            else:
                return {"content": "Unsupported LLM", "tokens_used": 0}
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return {"content": f"Error generating response: {str(e)}", "tokens_used": 0}
    
    async def _generate_gemini_response(self, model: str, prompt: str, images: List[str], context: str, document_context: List[str]) -> Dict[str, Any]:
        try:
            if not settings.GEMINI_API_KEY:
                return {"content": "Gemini API key not configured", "tokens_used": 0}
            
            # Use gemini-1.5-flash as default if model not specified
            if not model:
                model = "gemini-1.5-flash"
            
            full_prompt = self._build_prompt(prompt, context, document_context)
            
            # For Gemini models that support images
            if images and model in ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]:
                model_obj = genai.GenerativeModel(model)
                image_parts = []
                for img_data in images:
                    try:
                        image_part = {
                            "mime_type": "image/jpeg",
                            "data": base64.b64decode(img_data)
                        }
                        image_parts.append(image_part)
                    except Exception as e:
                        logger.warning(f"Failed to process image: {e}")
                
                response = model_obj.generate_content([full_prompt] + image_parts)
            else:
                model_obj = genai.GenerativeModel(model)
                response = model_obj.generate_content(full_prompt)
            
            return {
                "content": response.text,
                "tokens_used": len(response.text.split())  # Approximate
            }
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return {"content": f"Gemini API error: {str(e)}", "tokens_used": 0}
    
    async def _generate_groq_response(self, model: str, prompt: str, context: str, document_context: List[str]) -> Dict[str, Any]:
        try:
            if not self.groq_client:
                return {"content": "Groq client not initialized", "tokens_used": 0}
            
            full_prompt = self._build_prompt(prompt, context, document_context)
            
            # Map model names to Groq's model IDs
            model_map = {
                "llama-3.1-8b-instant": "llama3-8b-8192",
                "gemma2-9b-it": "gemma2-9b-it",
                "mixtral-8x7b-32768": "mixtral-8x7b-32768"
            }
            
            groq_model = model_map.get(model, "llama3-8b-8192")  # Default model
            
            response = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": full_prompt}],
                model=groq_model,
                temperature=0.7
            )
            
            return {
                "content": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens
            }
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            return {"content": f"Groq API error: {str(e)}", "tokens_used": 0}
    
    async def _generate_cohere_response(self, model: str, prompt: str, images: List[str], context: str, document_context: List[str]) -> Dict[str, Any]:
        try:
            if not self.cohere_client:
                return {"content": "Cohere client not initialized", "tokens_used": 0}
            
            full_prompt = self._build_prompt(prompt, context, document_context)
            
            # Cohere vision model handling
            if images and model == "command-a-vision-07-2025":
                image_docs = []
                for img_data in images:
                    image_docs.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_data
                        }
                    })
                
                response = self.cohere_client.chat(
                    message=full_prompt,
                    model=model,
                    documents=image_docs
                )
            else:
                # Use command-r-plus as default if model not specified
                if not model:
                    model = "command-r-plus"
                
                response = self.cohere_client.chat(
                    message=full_prompt,
                    model=model
                )
            
            return {
                "content": response.text,
                "tokens_used": getattr(response.meta.tokens.usage, 'total_tokens', 0) if hasattr(response, 'meta') else 0
            }
        except Exception as e:
            logger.error(f"Cohere API error: {e}")
            return {"content": f"Cohere API error: {str(e)}", "tokens_used": 0}
    
    def _build_prompt(self, prompt: str, context: str, document_context: List[str]) -> str:
        """Build enhanced prompt with context and document information"""
        prompt_parts = []
        
        if document_context:
            prompt_parts.append("DOCUMENT CONTEXT:")
            for i, doc in enumerate(document_context, 1):
                prompt_parts.append(f"Document {i}: {doc}")
            prompt_parts.append("")
        
        if context:
            prompt_parts.append(f"CONVERSATION CONTEXT:\n{context}")
            prompt_parts.append("")
        
        prompt_parts.append(f"USER QUESTION: {prompt}")
        prompt_parts.append("")
        prompt_parts.append("Please provide a helpful response based on the available context and documents.")
        
        return "\n".join(prompt_parts)


class InternetSearchService:
    def __init__(self):
        self.api_key = settings.SERPER_API_KEY
        if self.api_key:
            logger.info("Serper API key found - internet search enabled")
        else:
            logger.warning("Serper API key not found - internet search disabled")
    
    async def search(self, query: str) -> List[Dict[str, Any]]:
        """Perform internet search using Serper API"""
        if not self.api_key:
            logger.warning("Internet search disabled - no Serper API key")
            return []
        
        try:
            url = "https://google.serper.dev/search"
            payload = {
                "q": query,
                "num": 5  # Number of results
            }
            headers = {
                'X-API-KEY': self.api_key,
                'Content-Type': 'application/json'
            }
            
            logger.info(f"Performing internet search for: {query}")
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            results = response.json()
            
            search_results = []
            if 'organic' in results:
                for item in results['organic'][:5]:  # Limit to 5 results
                    search_results.append({
                        "title": item.get('title', ''),
                        "snippet": item.get('snippet', ''),
                        "link": item.get('link', ''),
                        "source": "internet",
                        "type": "search_result"
                    })
            
            logger.info(f"Internet search returned {len(search_results)} results")
            return search_results
            
        except requests.exceptions.Timeout:
            logger.error("Internet search timeout")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Internet search error: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in internet search: {e}")
            return []