import requests
import time
import os
from modules.config import Config

class LLMClient:
    """Client for interacting with language models via Hugging Face's API"""
    
    def __init__(self, api_key=None, model_name=Config.DEFAULT_MODEL):
        """
        Initialize the LLM client
        
        Args:
            api_key (str): Hugging Face API key (defaults to Config.API_KEY)
            model_name (str): Name of the model to use
        """
        self.api_key = api_key or Config.API_KEY
        self.model_name = model_name
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.model_type = self._determine_model_type(model_name)
        
        print(f"Using model: {model_name} via Hugging Face API")
        print(f"Model type identified as: {self.model_type}")
        
    def _determine_model_type(self, model_name):
        """Determine the type of model to format prompts correctly"""
        model_name_lower = model_name.lower()
        
        if "t5" in model_name_lower:
            return "t5"
        elif "gpt" in model_name_lower:
            return "gpt"
        elif "llama" in model_name_lower:
            return "llama"
        elif "mistral" in model_name_lower:
            return "mistral"
        elif "bert" in model_name_lower:
            return "bert"
        else:
            return "generic"
    
    def _format_prompt(self, user_input):
        """Format the prompt based on the model type"""
        if self.model_type == "t5":
            # T5 models work best with simple, direct prompts
            return f"Answer this finance question: {user_input}"
        elif self.model_type == "gpt":
            # GPT-style models work well with chat-like prompts
            return f"You are a helpful financial assistant that specializes in stock markets and investment analysis.\nUser: {user_input}\nAssistant:"
        elif self.model_type in ["llama", "mistral"]:
            # Llama and Mistral use a specific chat format
            return f"<s>[INST] You are a financial advisor. {user_input} [/INST]"
        elif self.model_type == "bert":
            # BERT models are not ideal for generation but can answer simple queries
            return f"Finance question: {user_input}"
        else:
            # Generic format for other models
            return f"You are a helpful financial assistant that specializes in stock markets and investment analysis.\n\nUser: {user_input}\n\nAssistant:"
    
    def get_response(self, user_input):
        """
        Get a response from the LLM for the user's input
        
        Args:
            user_input (str): User's input/question
            
        Returns:
            str: Generated response from the model
        """
        if not self.api_key:
            return "Error: No API key provided. Please set your Hugging Face API key."
        
        # Format the prompt based on model type
        formatted_prompt = self._format_prompt(user_input)
        
        # Add safety guardrails to the prompt
        safe_prompt = f"{formatted_prompt}\nImportant: Provide only financial information and advice. Do not generate any content that is not related to finance or investing."
        
        # Query the API
        start_time = time.time()
        print("Querying Hugging Face API...")
        
        response = self.query_api(safe_prompt)
        
        # Post-process the response
        # Some models might return the whole prompt, so extract just the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[1].strip()
            
        # Safety check on the response
        unsafe_terms = ["sex", "porn", "xxx", "nude", "naked", "adult", "explicit"]
        if any(term in response.lower() for term in unsafe_terms):
            # Replace with a safe fallback response
            response = "I can only provide financial information. For your query about stocks or financial matters, please try rephrasing your question or asking specifically about financial topics."
        
        elapsed_time = time.time() - start_time
        print(f"Response received in {elapsed_time:.2f} seconds")
        
        return response
    
    def query_api(self, prompt):
        """
        Query the Hugging Face API
        
        Args:
            prompt (str): Prompt to send to the model
            
        Returns:
            str: Generated text from the model
        """
        if not self.api_key:
            return "Error: No API key provided. Please set your Hugging Face API key."
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Different payload structure based on the model type
        if self.model_type == "t5":
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 100,
                    "temperature": 0.7
                }
            }
        elif self.model_type == "bert":
            # BERT models are for classification/fill-mask, not text generation
            # Let's use a simple fill-mask approach for financial terms
            payload = {
                "inputs": prompt
            }
        else:
            # Generic payload for most models
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 100,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True,
                    "return_full_text": False
                }
            }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Handle different response formats
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict) and "generated_text" in result[0]:
                        return result[0]["generated_text"]
                    else:
                        return str(result[0])
                elif isinstance(result, dict) and "generated_text" in result:
                    return result["generated_text"]
                else:
                    return str(result)
            
            # Handle errors
            elif response.status_code == 503:
                return "The model is currently loading. Please try again in a few moments."
            elif response.status_code == 429:
                return "Too many requests. Please slow down or upgrade your Hugging Face subscription."
            elif response.status_code == 403:
                return f"API Error: The model {self.model_name} is too large for the free tier. Please try a smaller model like 'google/flan-t5-small'."
            else:
                return f"API Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error querying API: {str(e)}"
