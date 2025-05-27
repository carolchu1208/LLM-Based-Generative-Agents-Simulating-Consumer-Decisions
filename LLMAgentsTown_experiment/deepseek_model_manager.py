import requests
import json
import time
from typing import Optional, Dict, Any
import os

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the model manager with API settings"""
        self.api_key = "sk-21209fe70fe441789dc550409fb2ee7e"  # Directly set API key
        self.base_url = "https://api.deepseek.com/v1"
        self.default_model = "deepseek-chat"
        self.max_retries = 3
        self.retry_delay = 2
        self.timeout = 30
        
        # Fallback responses for different prompt types
        self.fallback_responses = {
            "daily_plan": """I'll start my day at 7 AM. First, I'll have breakfast at home and get ready for work. 
I plan to head to my workplace by 8:30 AM to arrive before 9. I'll work until lunch break at noon,
probably grab something from a nearby place. After work at 5 PM, I might need to stop by the grocery store
if needed, then head home. Evening will be for dinner and relaxation before getting ready for tomorrow.""",
            
            "conversation": "Having a pleasant conversation while being mindful of time.",
            
            "contextual_action": "Continuing with my current planned activity.",
            
            "default": "Proceeding with standard daily routine."
        }

    def _make_api_call(self, prompt: str, **kwargs) -> Optional[str]:
        """Make API call to DeepSeek API"""
        try:
            if not self.api_key:
                print("Warning: DEEPSEEK_API_KEY not set. Using fallback response.")
                return self._get_fallback_response(prompt)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": kwargs.get("model", self.default_model),
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"API Error: {response.status_code} - {response.text}")
                return self._get_fallback_response(prompt)

        except requests.exceptions.RequestException as e:
            print(f"Request error: {str(e)}")
            return self._get_fallback_response(prompt)
        except Exception as e:
            print(f"Error in API call: {str(e)}")
            return self._get_fallback_response(prompt)

    def _get_fallback_response(self, prompt: str) -> str:
        """Get an appropriate fallback response based on the prompt content"""
        prompt_lower = prompt.lower()
        
        if "starting my day" in prompt_lower or "daily plan" in prompt_lower:
            return self.fallback_responses["daily_plan"]
        elif "conversation" in prompt_lower or "chat" in prompt_lower:
            return self.fallback_responses["conversation"]
        elif "action" in prompt_lower or "what to do" in prompt_lower:
            return self.fallback_responses["contextual_action"]
        else:
            return self.fallback_responses["default"]

    def generate(self, prompt: str, model: Optional[str] = None, return_json: bool = False, **kwargs) -> str:
        """Generate a response using the DeepSeek API with retries"""
        if not prompt:
            return self._get_fallback_response("Empty prompt")

        model = model or self.default_model
        retries = 0
        
        while retries < self.max_retries:
            try:
                response = self._make_api_call(prompt, model=model, **kwargs)
                if response:
                    return response
                retries += 1
                time.sleep(self.retry_delay)
            except Exception as e:
                print(f"Error in generate (attempt {retries + 1}): {str(e)}")
                retries += 1
                if retries < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                break

        # If all retries failed, return fallback
        return self._get_fallback_response(prompt)

    def generate_with_context(self, prompt: str, context: str, **kwargs) -> str:
        """Generate with additional context"""
        full_prompt = f"Context: {context}\n\n{prompt}"
        return self.generate(full_prompt, **kwargs)
    
    def generate_structured(self, prompt: str, schema: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """Generate structured output matching a schema"""
        full_prompt = f"{prompt}\n\nRespond in JSON format matching this schema: {json.dumps(schema, indent=2)}"
        return self.generate(full_prompt, return_json=True, **kwargs) 