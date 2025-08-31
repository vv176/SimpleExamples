import openai
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


class MultiOpenAIClient:
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=api_key)
    
    def chat_completion(self, 
                       user_message: str, 
                       system_message: Optional[str] = None,
                       max_tokens: int = 150,
                       temperature: float = 0.7) -> str:
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error making API call: {str(e)}"
    
    def start_conversation(self):
        """
        Start a continuous conversation loop.
        Note: This implementation has NO MEMORY - each response is independent.
        """
        print("🤖 Chatbot is ready to talk to you!")
        print("=" * 50)
        
        system_message = "You are a helpful AI assistant. Keep responses concise and engaging."
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 AI: Goodbye! Thanks for chatting!")
                    break
                
                if not user_input:
                    print("Please enter a message.")
                    continue
                
                # Send to LLM (without any conversation history)
                print("🔄 Processing...")
                response = self.chat_completion(
                    user_message=user_input,
                    system_message=system_message
                )
                
                # Display response
                print(f"🤖 AI: {response}")
                
            except KeyboardInterrupt:
                print("\n\n🤖 AI: Conversation interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


# Example usage and demo
if __name__ == "__main__":
    try:
        client = MultiOpenAIClient()
        client.start_conversation()
    except ValueError as e:
        print(f"Setup error: {e}")
        print("Please set your OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
    except Exception as e:
        print(f"Unexpected error: {e}")
