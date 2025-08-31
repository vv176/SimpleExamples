import openai
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


class SimpleOpenAIClient:
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=api_key)
    
    def chat_completion(self, 
                       user_message: str, 
                       system_message: Optional[str] = None,
                       max_tokens: int = 170,
                       temperature: float = 0.4) -> str:
        messages = []
        # List of dictionaries.
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                n=2,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            print(response)
            print("--------------------------------")
            print(response.choices[0].message.content.strip())
            print("--------------------------------")
            print(response.choices[1].message.content.strip())
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error making API call: {str(e)}"


# Example usage and demo
if __name__ == "__main__":
    client = SimpleOpenAIClient()
    response = client.chat_completion(
        user_message="write me a poem on stars",
        system_message="You are a good personal assistant. Never respond in more than 100 words."
    )
    print(response)