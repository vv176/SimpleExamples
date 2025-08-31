import openai
import os
import json
import requests
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import quote

# Load environment variables from .env file
load_dotenv()


class OpenAIClientWithMemoryAndTools:
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=api_key)
        # Initialize conversation history in memory
        self.conversation_history: List[Dict[str, str]] = []
    
    def get_weather(self, city: str) -> str:
        """
        Simple weather tool that fetches current weather for a city.
        Uses wttr.in (free, no API key required).
        """
        try:            
            city_enc = quote(city.strip())
            url = f"https://wttr.in/{city_enc}?format=j1&m&lang=en"
            
            response = requests.get(url, timeout=6)
            response.raise_for_status()
            data = response.json()
            
            cur = data["current_condition"][0]
            temp_c = cur.get("temp_C", "N/A")
            condition = cur["weatherDesc"][0]["value"]
            humidity = cur.get("humidity", "N/A")
            wind_kmph = cur.get("windspeedKmph", "N/A")
            feels_like_c = cur.get("FeelsLikeC", "N/A")
            
            return f"🌤️ Weather in {city}: {temp_c}°C, {condition}, Humidity: {humidity}%, Wind: {wind_kmph} km/h, Feels like: {feels_like_c}°C"
                
        except Exception as e:
            return f"❌ Error fetching weather for {city}: {str(e)}"
    
    def chat_completion_with_tools(self, 
                                  user_message: str, 
                                  system_message: Optional[str] = None,
                                  max_tokens: int = 200,
                                  temperature: float = 0.7) -> str:
        """
        Make a chat completion request with tool calling capabilities.
        """
        # Build messages list with full conversation history
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add all previous conversation messages
        messages.extend(self.conversation_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Define available tools
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "The city name to get weather for"
                            }
                        },
                        "required": ["city"]
                    }
                }
            }
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=temperature
            )
            #print(response)
            response_message = response.choices[0].message
            
            # Check if the model wants to call a tool
            if response_message.tool_calls:
                # Handle tool calls
                for tool_call in response_message.tool_calls:
                    if tool_call.function.name == "get_weather":
                        # Parse the arguments
                        arguments = json.loads(tool_call.function.arguments)
                        city = arguments.get("city", "Unknown")
                        
                        # Call the tool
                        tool_result = self.get_weather(city)
                        
                        # Make another API call to get the final response
                        messages.append({
                            "role": "assistant", 
                            "content": response_message.content or "",
                            "tool_calls": [tool_call]
                        })
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })
                        
                        final_response = self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            max_tokens=max_tokens,
                            temperature=temperature
                        )
                        
                        ai_response = final_response.choices[0].message.content.strip()
                        
                        # Add user message and final AI response to history
                        self.conversation_history.append({"role": "user", "content": user_message})
                        # Add the tool call and result to conversation history
                        self.conversation_history.append({
                            "role": "assistant", 
                            "content": response_message.content or "",
                            "tool_calls": [tool_call]
                        })
                        
                        self.conversation_history.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })
                        self.conversation_history.append({"role": "assistant", "content": ai_response})
                        
                        return ai_response
            else:
                # No tool calls, normal response
                ai_response = response_message.content.strip()
                
                # Update conversation history
                self.conversation_history.append({"role": "user", "content": user_message})
                self.conversation_history.append({"role": "assistant", "content": ai_response})
                
                return ai_response
            
        except Exception as e:
            return f"Error making API call: {str(e)}"
    
    def start_conversation(self):
        """
        Start a continuous conversation loop with memory and tool calling.
        """
        print("🤖 Chatbot with Memory and Tools is ready!")
        print("=" * 60)
        print("Available tools: get_weather (try asking about weather!)")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'history' to see conversation history")
        print("Type 'clear' to clear conversation history")
        print("=" * 60)
        
        system_message = """You are a helpful AI assistant with access to tools. 
        You can get weather information for cities using the get_weather tool.
        Keep responses concise and engaging. You can reference previous parts of our conversation.
        When asked about weather, use the get_weather tool to provide accurate information."""
        
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
                
                # Send to LLM with tool calling capabilities
                print("🔄 Processing...")
                response = self.chat_completion_with_tools(
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
        print("-" * 40)
        for i, message in enumerate(self.conversation_history, 1):
            role_emoji = "👤" if message["role"] == "user" else "🤖"
            content = message.get("content", "")
            
            if message["role"] == "tool":
                role_emoji = "🔧"
                content = f"[Tool Result: {content}]"
            elif message["role"] == "assistant":
                # Check if this is a tool call message
                if "tool_calls" in message and message["tool_calls"]:
                    tool_calls = message["tool_calls"]
                    tool_info = []
                    for tool_call in tool_calls:
                        tool_info.append(f"🔧 Called {tool_call.function.name} with args: {tool_call.function.arguments}")
                    
                    if content:
                        content = f"{content} {' '.join(tool_info)}"
                    else:
                        content = " ".join(tool_info)
            
            print(f"{i}. {role_emoji} {message['role'].title()}: {content}")
        print("-" * 40)
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history.clear()
        print("🧹 Conversation history cleared!")


# Example usage and demo
if __name__ == "__main__":
    try:
        client = OpenAIClientWithMemoryAndTools()
        client.start_conversation()
    except ValueError as e:
        print(f"Setup error: {e}")
        print("Please set your OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nNote: Weather data is provided by wttr.in (free, no API key required)")
    except Exception as e:
        print(f"Unexpected error: {e}")
