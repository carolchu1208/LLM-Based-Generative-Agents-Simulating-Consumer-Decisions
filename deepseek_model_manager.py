# Standard library imports
import os
import json
import random
import threading
import time
import traceback
import logging
from collections import defaultdict, deque
from datetime import datetime
from typing import (
    Dict, List, Optional, Any, Tuple, Set, Union, 
    TYPE_CHECKING, TypeVar, Generic, Protocol
)

# Third-party imports
import requests


# Model Manager Interface
class ModelManagerInterface(Protocol):
    """Interface for model management functionality."""
    def generate(self, prompt: str) -> str:
        """Generate a response from the model."""
        ...
    
    def generate_with_context(self, prompt: str, context: str, **kwargs) -> str:
        """Generate a response with additional context."""
        ...
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Generate a structured response following the provided schema."""
        ...

class ModelManager(ModelManagerInterface):
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
            return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._initialize()
    
    def _initialize(self):
        """Initialize the model manager with settings."""
        # Set API key
        self.api_key = 'sk-4e401a02a7084517bcf02541fa87be78'
        self.use_fallback = False
        
        # Model configuration
        self.model_config = {
            "model": "deepseek-chat",
            "max_tokens": 2048,
            "temperature": 1.0,
            "top_p": 0.9
        }
        
        # Initialize fallback responses (only used if API calls fail)
        self.fallback_responses = {
            "daily_plan": """{
  "activities": [
    {"time": 7, "action": "go_to", "target": "Fried Chicken Shop", "description": "Travel to work"},
    {"time": 8, "action": "work", "target": "Fried Chicken Shop", "description": "Start work shift"},
    {"time": 9, "action": "work", "target": "Fried Chicken Shop", "description": "Continue work"},
    {"time": 10, "action": "work", "target": "Fried Chicken Shop", "description": "Continue work"},
    {"time": 11, "action": "work", "target": "Fried Chicken Shop", "description": "Continue work"},
    {"time": 12, "action": "eat", "target": "Local Diner", "description": "Lunch break"},
    {"time": 13, "action": "work", "target": "Fried Chicken Shop", "description": "Resume work"},
    {"time": 14, "action": "work", "target": "Fried Chicken Shop", "description": "Continue work"},
    {"time": 15, "action": "work", "target": "Fried Chicken Shop", "description": "Continue work"},
    {"time": 16, "action": "work", "target": "Fried Chicken Shop", "description": "Continue work"},
    {"time": 17, "action": "work", "target": "Fried Chicken Shop", "description": "End work shift"},
    {"time": 18, "action": "go_to", "target": "Maple Street Apartments", "description": "Travel home"},
    {"time": 19, "action": "eat", "target": "Maple Street Apartments", "description": "Dinner at home"},
    {"time": 20, "action": "idle", "target": "Maple Street Apartments", "description": "Relax at home"},
    {"time": 21, "action": "idle", "target": "Maple Street Apartments", "description": "Evening relaxation"},
    {"time": 22, "action": "idle", "target": "Maple Street Apartments", "description": "Wind down"},
    {"time": 23, "action": "rest", "target": "Maple Street Apartments", "description": "Prepare for sleep"}
  ]
}""",
            "conversation": "I understand. Let's continue our conversation.",
            "action": "I'll proceed with my current activity."
        }
        
        print("[DEBUG] ModelManager initialized with API key")
        print(f"[DEBUG] Using model: {self.model_config['model']}")
    
    def _make_api_call(self, prompt: str, prompt_type: str, max_retries=3, **kwargs) -> Optional[str]:
        """Process the prompt and generate appropriate response."""
        try:
            # Validate prompt type
            valid_types = ["daily_plan", "conversation", "action"]
            if prompt_type not in valid_types:
                raise ValueError(f"Invalid prompt type: {prompt_type}. Must be one of: {valid_types}")
                
            print(f"\n[DEBUG] Starting API call for prompt type: {prompt_type}")
            
            # Extract context if present
            context = {}
            if isinstance(prompt, dict):
                context = prompt.get('context', {})
                prompt = prompt.get('content', str(prompt))
                print(f"[DEBUG] Extracted context with keys: {list(context.keys())}")
            elif isinstance(prompt, list):
                prompt = ' '.join(str(p) for p in prompt)
            
            print(f"[DEBUG] Prompt length: {len(prompt)} characters")
            
            # Initialize retry variables
            retry_delay = 2  # Initial delay in seconds
            last_error = None
            
            # Make the actual API call to DeepSeek with retries
            for attempt in range(max_retries):
                try:
                    print(f"[DEBUG] Making API call attempt {attempt + 1}/{max_retries}")
                    print(f"[DEBUG] API endpoint: https://api.deepseek.com/v1/chat/completions")
                    
                    # Test internet connectivity first
                    try:
                        print("[DEBUG] Testing internet connectivity...")
                        requests.get("https://api.deepseek.com", timeout=(5, 5))  # (connect timeout, read timeout)
                        print("[DEBUG] Internet connection test successful")
                    except requests.exceptions.RequestException as e:
                        print(f"[DEBUG] Internet connection test failed: {str(e)}")
                        print("[DEBUG] Please check your internet connection and try again")
                        if attempt < max_retries - 1:
                            print(f"[DEBUG] Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Exponential backoff
                            continue
                        return self._get_fallback_response(prompt_type)
                    
                    # Set timeouts: (connect timeout, read timeout)
                    connect_timeout = 10  # 10 seconds to establish connection
                    read_timeout = 60    # 60 seconds to read response
                    print(f"[DEBUG] Using timeouts - Connect: {connect_timeout}s, Read: {read_timeout}s")
                    
                    # Prepare request payload
                    payload = {
                        **self.model_config,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                    
                    # Make the API call
                    response = requests.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json=payload,
                        timeout=(connect_timeout, read_timeout)
                    )
                    
                    print(f"[DEBUG] API Response status code: {response.status_code}")
                    
                    # Handle different response status codes
                    if response.status_code == 200:
                        try:
                            response_json = response.json()
                            if not response_json or "choices" not in response_json:
                                print("[DEBUG] Invalid response format from API")
                                last_error = "Invalid response format"
                                continue
                                
                            choices = response_json["choices"]
                            if not choices or len(choices) == 0:
                                print("[DEBUG] No choices in API response")
                                last_error = "No choices in response"
                                continue
                                
                            content = choices[0].get("message", {}).get("content")
                            if not content:
                                print("[DEBUG] No content in API response")
                                last_error = "No content in response"
                                continue
                                
                            print(f"[DEBUG] Successfully received response of length: {len(content)}")
                            return content
                            
                        except (json.JSONDecodeError, KeyError, IndexError) as e:
                            print(f"[DEBUG] Error parsing API response: {str(e)}")
                            last_error = f"Error parsing response: {str(e)}"
                            continue
                            
                    elif response.status_code == 429:  # Rate limit
                        retry_after = int(response.headers.get('Retry-After', 5))
                        print(f"[DEBUG] Rate limited. Waiting {retry_after} seconds...")
                        time.sleep(retry_after)
                        continue
                        
                    elif response.status_code >= 500:  # Server errors
                        print(f"[DEBUG] Server error: {response.status_code}")
                        last_error = f"Server error: {response.status_code}"
                        if attempt < max_retries - 1:
                            print(f"[DEBUG] Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                            
                    else:  # Other errors
                        error_msg = response.json().get("error", {}).get("message", "Unknown error")
                        print(f"[DEBUG] API error: {response.status_code} {error_msg}")
                        last_error = f"API error: {error_msg}"
                        if attempt < max_retries - 1:
                            print(f"[DEBUG] Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                            continue
                            
                except requests.exceptions.Timeout as e:
                    if isinstance(e, requests.exceptions.ConnectTimeout):
                        print(f"[DEBUG] Connection timed out on attempt {attempt + 1}")
                        last_error = "Connection timeout"
                    elif isinstance(e, requests.exceptions.ReadTimeout):
                        print(f"[DEBUG] Read timed out on attempt {attempt + 1}")
                        last_error = "Read timeout"
                    else:
                        print(f"[DEBUG] Request timed out on attempt {attempt + 1}")
                        last_error = "Request timeout"
                    
                    if attempt < max_retries - 1:
                        print(f"[DEBUG] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                        
                except requests.exceptions.RequestException as e:
                    print(f"[DEBUG] Request failed on attempt {attempt + 1}: {str(e)}")
                    last_error = f"Request failed: {str(e)}"
                    if attempt < max_retries - 1:
                        print(f"[DEBUG] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    
            print(f"[DEBUG] All retry attempts failed. Last error: {last_error}")
            return self._get_fallback_response(prompt_type)
                
        except Exception as e:
            print(f"[DEBUG] Error in _make_api_call: {e}")
            traceback.print_exc()
            return self._get_fallback_response(prompt_type)
    
    def _get_fallback_response(self, prompt_type: str) -> str:
        """Get an appropriate fallback response based on the prompt type."""
        valid_types = ["daily_plan", "conversation", "action"]
        if prompt_type not in valid_types:
            raise ValueError(f"Invalid prompt type: {prompt_type}. Must be one of: {valid_types}")
        return self.fallback_responses.get(prompt_type)
    
    def generate(self, prompt: str, prompt_type: str) -> str:
        """Generate a response from the model or use fallback if processing fails."""
        try:
            print(f"\n[DEBUG] Starting response generation...")
            print(f"[DEBUG] Prompt type: {prompt_type}")
            print(f"[DEBUG] Prompt length: {len(prompt)} characters")
            
            response = self._make_api_call(prompt, prompt_type)
            if response:
                # Log the full response for debugging
                print(f"[DEBUG] Full API response received:")
                print(f"[DEBUG] Response length: {len(response)} characters")
                return response
                
            print("[DEBUG] No response received from model, using fallback")
            return self._get_fallback_response(prompt_type)
        except Exception as e:
            print(f"[DEBUG] Error in generate method:")
            print(f"[DEBUG] Error type: {type(e).__name__}")
            print(f"[DEBUG] Error message: {str(e)}")
            print(f"[DEBUG] Error traceback:")
            traceback.print_exc()
            return self._get_fallback_response(prompt_type)
    
    def generate_with_context(self, prompt: str, context: str, prompt_type: str, **kwargs) -> str:
        """Generate a response with additional context."""
        combined_prompt = f"Context: {context}\n\nPrompt: {prompt}"
        return self.generate(combined_prompt, prompt_type)
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any], prompt_type: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Generate a structured response following the provided schema."""
        try:
            print(f"\n[DEBUG] Generating structured response with schema: {schema}")
            print(f"[DEBUG] Prompt type: {prompt_type}")
            print(f"[DEBUG] Prompt length: {len(prompt)} characters")
            
            # Get the raw response
            response = self.generate(prompt, prompt_type)
            if not response:
                print("[DEBUG] No response generated from model")
                return None
                
            # Log response preview safely
            response_preview = response[:200] if isinstance(response, str) else str(response)[:200]
            print(f"[DEBUG] Raw response received: {response_preview}...")
            
            # Try to parse JSON response
            try:
                if isinstance(response, str):
                    result = json.loads(response)
                else:
                    result = response
                    
                if not result:
                    print("[DEBUG] Empty result after parsing")
                    return None
                    
                print(f"[DEBUG] Successfully parsed response: {result}")
                return result
            except json.JSONDecodeError:
                print("[DEBUG] Response is not valid JSON, attempting to extract structured data")
            
            # If not JSON, try to extract structured data from text
            result = {}
            for key, value_type in schema.items():
                print(f"[DEBUG] Processing key: {key}")
                if isinstance(response, str) and key in response.lower():
                    try:
                        # Find the section after the key
                        key_index = response.lower().find(key.lower())
                        if key_index != -1:
                            # Get the text after the key
                            after_key = response[key_index + len(key):].strip()
                            # Find the next key or end of text
                            next_key_index = float('inf')
                            for other_key in schema.keys():
                                if other_key != key:
                                    other_index = after_key.lower().find(other_key.lower())
                                    if other_index != -1 and other_index < next_key_index:
                                        next_key_index = other_index
                            
                            # Extract the value
                            if next_key_index != float('inf'):
                                value = after_key[:next_key_index].strip()
                            else:
                                value = after_key.strip()
                            
                            print(f"[DEBUG] Extracted value for {key}: {value[:100]}...")
                            
                            # Convert to appropriate type
                            if value_type == str:
                                result[key] = value
                            elif value_type == int:
                                result[key] = int(value)
                            elif value_type == float:
                                result[key] = float(value)
                            elif value_type == bool:
                                result[key] = value.lower() in ['true', 'yes', '1']
                            else:
                                result[key] = str(value)
                    except Exception as e:
                        print(f"[DEBUG] Error parsing value for {key}: {str(e)}")
                        result[key] = str(value_type())  # Use default value for type
                else:
                    print(f"[DEBUG] Key '{key}' not found in response")
                    result[key] = str(value_type())  # Use default value for type

            print(f"[DEBUG] Final parsed result: {result}")
            return result

        except Exception as e:
            print(f"[DEBUG] Error in generate_structured: {str(e)}")
            traceback.print_exc()
            return None 