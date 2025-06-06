import threading
import time
from typing import Dict, Any, Optional
import re
import requests
import os
import traceback
import json

class ModelManager:
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
        self.fallback_responses = {
            "daily_plan": {
                "schedule": "I'll start my day at 7:00 AM with breakfast at home. At 8:30 AM, I'll head to work. During my lunch break at 12:00 PM, I'll eat at a nearby restaurant. I'll continue working until 5:00 PM, then have dinner at home at 6:00 PM. I'll spend the evening relaxing at home.",
                "reasoning": "This schedule maintains a good balance between work and personal time, with proper meal breaks and rest periods.",
                "energy_considerations": "I'll need to ensure I have enough energy for work by eating proper meals and taking breaks when needed."
            },
            "conversation": "I understand. Let's continue our conversation.",
            "contextual_action": "I'll proceed with my current activity.",
            "default": "I understand and will proceed accordingly."
        }
    
    def _make_api_call(self, prompt: str, max_retries=3, **kwargs) -> Optional[str]:
        """Process the prompt and generate appropriate response."""
        try:
            print(f"\n[DEBUG] Starting API call for prompt type: {self._identify_prompt_type(prompt)}")
            
            # Extract context if present
            context = {}
            if isinstance(prompt, dict):
                context = prompt.get('context', {})
                prompt = prompt.get('content', str(prompt))
                print(f"[DEBUG] Extracted context with keys: {list(context.keys())}")
            elif isinstance(prompt, list):
                prompt = ' '.join(str(p) for p in prompt)
            
            # Process the prompt based on its type
            prompt_type = self._identify_prompt_type(prompt)
            print(f"[DEBUG] Identified prompt type: {prompt_type}")
            print(f"[DEBUG] Prompt length: {len(prompt)} characters")
            
            # Make the actual API call to DeepSeek with retries
            for attempt in range(max_retries):
                try:
                    print(f"[DEBUG] Making API call attempt {attempt + 1}/{max_retries}")
                    print(f"[DEBUG] Request payload: {{'model': 'deepseek-chat', 'max_tokens': 512, 'temperature': 0.7}}")
                    
                    response = requests.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer sk-4e401a02a7084517bcf02541fa87be78",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "deepseek-chat",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 512,
                            "temperature": 0.7,
                            "top_p": 0.9
                        },
                        timeout=30
                    )
                    
                    print(f"[DEBUG] API Response status code: {response.status_code}")
                    print(f"[DEBUG] API Response headers: {dict(response.headers)}")
                    
                    if response.status_code == 200:
                        content = response.json()["choices"][0]["message"]["content"]
                        print(f"[DEBUG] Successfully received response of length: {len(content)}")
                        return content
                    elif response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', 5))
                        print(f"[DEBUG] Rate limited. Waiting {retry_after} seconds...")
                        time.sleep(retry_after)
                        continue
                    else:
                        error_msg = response.json().get("error", {}).get("message", "Unknown error")
                        print(f"[DEBUG] API error: {response.status_code} {error_msg}")
                        print(f"[DEBUG] Full error response: {response.text}")
                        if attempt < max_retries - 1:
                            wait_time = 2 ** attempt
                            print(f"[DEBUG] Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        print("[DEBUG] Using fallback response after all retries failed")
                        return self._get_fallback_response(prompt_type)
                        
                except requests.exceptions.Timeout:
                    print(f"[DEBUG] Request timed out on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"[DEBUG] Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    print("[DEBUG] Using fallback response after timeout")
                    return self._get_fallback_response(prompt_type)
                except requests.exceptions.RequestException as e:
                    print(f"[DEBUG] Request failed on attempt {attempt + 1}: {str(e)}")
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"[DEBUG] Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    print("[DEBUG] Using fallback response after request exception")
                    return self._get_fallback_response(prompt_type)
                    
            print("[DEBUG] Using fallback response after all retries exhausted")
            return self._get_fallback_response(prompt_type)
                
        except Exception as e:
            print(f"[DEBUG] Error in _make_api_call: {e}")
            traceback.print_exc()
            return self._get_fallback_response(prompt_type)
    
    def _get_fallback_response(self, prompt_type: str) -> str:
        """Get an appropriate fallback response based on the prompt type."""
        return self.fallback_responses.get(prompt_type, self.fallback_responses["default"])
    
    def generate(self, prompt: str) -> str:
        """Generate a response from the model or use fallback if processing fails."""
        try:
            response = self._make_api_call(prompt)
            if response:
                # Only log the response once
                print(f"[DEBUG] Raw response received: {response[:200]}...")
                return response
            return self._get_fallback_response("default")
        except Exception as e:
            print(f"[DeepSeek] Error generating response: {e}")
            return self._get_fallback_response("default")
    
    def _identify_prompt_type(self, prompt: str) -> str:
        """Identify the type of prompt to determine appropriate response generation."""
        prompt_lower = prompt.lower()
        
        if "starting my day" in prompt_lower or "daily plan" in prompt_lower:
            return "daily_plan"
        elif "conversation" in prompt_lower or "chat" in prompt_lower:
            return "conversation"
        elif "action" in prompt_lower or "what to do" in prompt_lower:
            return "contextual_action"
        else:
            return "default"
    
    def generate_with_context(self, prompt: str, context: str, **kwargs) -> str:
        """Generate a response with additional context."""
        combined_prompt = f"Context: {context}\n\nPrompt: {prompt}"
        return self.generate(combined_prompt)
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Generate a structured response following the provided schema."""
        try:
            print(f"\n[DEBUG] Generating structured response with schema: {schema}")
            print(f"[DEBUG] Prompt type: {self._identify_prompt_type(prompt)}")
            print(f"[DEBUG] Prompt length: {len(prompt)} characters")
            
            # Get the raw response
            response = self.generate(prompt)
            if not response:
                print("[DEBUG] No response generated from model")
                return None
                
            print(f"[DEBUG] Raw response received: {response[:200]}...")  # Log first 200 chars
            
            # Try to parse JSON response
            try:
                result = json.loads(response)
                print(f"[DEBUG] Successfully parsed JSON response: {result}")
                return result
            except json.JSONDecodeError:
                print("[DEBUG] Response is not valid JSON, attempting to extract structured data")
            
            # If not JSON, try to extract structured data from text
            result = {}
            for key, value_type in schema.items():
                print(f"[DEBUG] Processing key: {key}")
                if key in response.lower():
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
                            
                            print(f"[DEBUG] Extracted value for {key}: {value[:100]}...")  # Log first 100 chars
                            
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