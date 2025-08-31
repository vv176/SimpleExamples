import openai
import os
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


class MultiOpenAIClientWithMemory:
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=api_key)
        # Initialize conversation history in memory
        self.conversation_history: List[Dict[str, str]] = []
    
    def chat_completion(self, 
                       user_message: str, 
                       system_message: Optional[str] = None,
                       max_tokens: int = 150,
                       temperature: float = 0.7) -> str:
        # Build messages list with full conversation history
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add all previous conversation messages
        messages.extend(self.conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Update conversation history with both user message and AI response
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            return ai_response
            
        except Exception as e:
            return f"Error making API call: {str(e)}"
    
    def start_conversation(self):
        """
        Start a continuous conversation loop with memory.
        This implementation maintains conversation history in memory.
        """
        print("🤖 Chatbot with Memory is ready to talk to you!")
        
        system_message = "You are a helpful AI assistant. Keep responses concise and engaging. You can reference previous parts of our conversation."
        
        while True:
            try:
                # Get user input
                user_input = input("\n👤 You: ").strip()
                
                # Check for exit commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n🤖 AI: Goodbye! Thanks for chatting!")
                    break
                
                # Check for special commands
                if user_input.lower() == 'history':
                    self.show_conversation_history()
                    continue
                
                if user_input.lower() == 'clear':
                    self.clear_conversation_history()
                    continue
                
                if not user_input:
                    print("Please enter a message.")
                    continue
                
                # Send to LLM (with full conversation history)
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
    
    def show_conversation_history(self):
        """Display the current conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history yet.")
            return
        
        print("\n📝 Conversation History:")
        print("-" * 30)
        for i, message in enumerate(self.conversation_history, 1):
            role_emoji = "👤" if message["role"] == "user" else "🤖"
            print(f"{i}. {role_emoji} {message['role'].title()}: {message['content']}")
        print("-" * 30)
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        print("🧹 Conversation history cleared!")


# Example usage and demo
if __name__ == "__main__":
    try:
        client = MultiOpenAIClientWithMemory()
        client.start_conversation()
    except ValueError as e:
        print(f"Setup error: {e}")
        print("Please set your OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
    except Exception as e:
        print(f"Unexpected error: {e}")
