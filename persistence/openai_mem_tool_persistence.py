import openai
import os
import json
import requests
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI
from urllib.parse import quote
from db_accessor import DatabaseAccessor

# Load environment variables from .env file
load_dotenv()


class OpenAIClientWithMemoryAndToolsPersistence:
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.model = model
        self.client = OpenAI(api_key=api_key)
        
        # Initialize database accessor
        self.db = DatabaseAccessor()
        
        # Define available tools
        self.tools = [
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
            
            weather_info = f"🌤️ Weather in {city}: {temp_c}°C, {condition}, Humidity: {humidity}%, Wind: {wind_kmph} km/h, Feels like: {feels_like_c}°C"
                        
            return weather_info
                
        except Exception as e:
            error_msg = f"❌ Error fetching weather for {city}: {str(e)}"
            # Store error in database
            self.db.insert_conversation("tool", f"Weather tool error for {city}: {str(e)}")
            return error_msg
    
    def _get_conversation_history_from_db(self) -> List[Dict[str, str]]:
        """Get conversation history from database and convert to OpenAI message format."""
        try:
            db_history = self.db.get_conversation_history()
            messages = []
            
            for entry in db_history:
                role = entry['role']
                content = entry['response']
                
                if role in ['user', 'assistant']:
                    messages.append({"role": role, "content": content})
                elif role == 'tool':
                    # Convert tool results to assistant messages for context
                    messages.append({"role": "assistant", "content": f"Tool result: {content}"})
            
            return messages
        except Exception as e:
            print(f"Warning: Could not load conversation history from database: {e}")
            return []
    
    def chat_completion_with_tools(self, 
                                  user_message: str, 
                                  system_message: Optional[str] = None,
                                  max_tokens: int = 200,
                                  temperature: float = 0.7) -> str:
        """
        Make a chat completion request with tool calling capabilities and database persistence.
        """
        
        # Build messages list with conversation history from database
        messages = []
        
        # Add system message if provided
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        # Add conversation history from database
        messages.extend(self._get_conversation_history_from_db())
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        # Store user message in database
        self.db.insert_conversation("user", user_message)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            response_message = response.choices[0].message
            
            # Check if the model wants to call a tool
            if response_message.tool_calls:
                print("Tool call is made")
                
                # Process all tool calls and collect results
                tool_results = []
                all_tool_calls = response_message.tool_calls
                tool_call_descriptions = []
                tool_result_descriptions = []
                
                # Add assistant message with all tool calls to messages (for final API call)
                messages.append({
                    "role": "assistant", 
                    "content": response_message.content or "",
                    "tool_calls": all_tool_calls
                })
                
                for tool_call in all_tool_calls:
                    if tool_call.function.name == "get_weather":
                        # Parse the arguments
                        arguments = json.loads(tool_call.function.arguments)
                        city = arguments.get("city", "Unknown")
                        
                        # Call the tool
                        tool_result = self.get_weather(city)
                        tool_results.append({
                            "tool_call_id": tool_call.id,
                            "content": tool_result
                        })
                        
                        # Collect tool call descriptions for database storage
                        tool_call_descriptions.append(f"get_weather called for: {arguments}")
                        # Collect tool result descriptions for database storage
                        tool_result_descriptions.append(f"Weather result for {city}: {tool_result}")
                
                # Add all tool results to messages for the final API call (after the loop)
                for result in tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result["tool_call_id"],
                        "content": result["content"]
                    })
                
                # Store grouped tool calls as single database entry
                grouped_tool_calls = " | ".join(tool_call_descriptions)
                self.db.insert_conversation("assistant", grouped_tool_calls)
                
                # Store grouped tool results as single database entry
                grouped_tool_results = " | ".join(tool_result_descriptions)
                self.db.insert_conversation("tool", grouped_tool_results)
                
                # Get final response from OpenAI
                final_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                ai_response = final_response.choices[0].message.content.strip()
                
                # Store final AI response in database
                self.db.insert_conversation("assistant", ai_response)
                
                return ai_response
            else:
                print("No Tool")
                # No tool calls, normal response
                ai_response = response_message.content.strip()
                
                # Store AI response in database
                self.db.insert_conversation("assistant", ai_response)
                
                return ai_response
            
        except Exception as e:
            error_msg = f"Error making API call: {str(e)}"
            # Store error in database
            self.db.insert_conversation("system", f"Error: {error_msg}")
            return error_msg
    
    def start_conversation(self):
        """
        Start a continuous conversation loop with memory, tool calling, and database persistence.
        """
        print("🤖 Chatbot with Memory, Tools, and Database Persistence is ready!")
        print("=" * 70)
        print("Available tools: get_weather (try asking about weather!)")
        print("All conversations are automatically saved to PostgreSQL database!")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'history' to see conversation history from database")
        print("Type 'clear' to clear conversation history from database")
        print("Type 'count' to see total conversation count")
        print("=" * 70)
        
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
                
                if user_input.lower() == 'count':
                    self.show_conversation_count()
                    continue
                
                if not user_input:
                    print("Please enter a message.")
                    continue
                
                # Send to LLM with tool calling capabilities and database persistence
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
        """Display the current conversation history from database."""
        try:
            history = self.db.get_conversation_history()
            if not history:
                print("📝 No conversation history in database yet.")
                return
            
            print(f"\n📝 Conversation History (Total: {len(history)} messages):")
            print("-" * 50)
            for i, entry in enumerate(history, 1):
                role_emoji = "👤" if entry["role"] == "user" else "🤖"
                if entry["role"] == "tool":
                    role_emoji = "🔧"
                elif entry["role"] == "system":
                    role_emoji = "⚙️"
                
                timestamp = entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                print(f"{i}. {role_emoji} {entry['role'].title()} ({timestamp}): {entry['response'][:100]}{'...' if len(entry['response']) > 100 else ''}")
            print("-" * 50)
        except Exception as e:
            print(f"❌ Error loading conversation history: {e}")
    
    def clear_conversation_history(self):
        """Clear the conversation history from database."""
        try:
            deleted_count = self.db.clear_conversation_history()
            print(f"🧹 Cleared {deleted_count} conversation entries from database!")
        except Exception as e:
            print(f"❌ Error clearing conversation history: {e}")
    
    def show_conversation_count(self):
        """Show the total number of conversation entries in database."""
        try:
            count = self.db.get_conversation_count()
            print(f"📊 Total conversation entries in database: {count}")
        except Exception as e:
            print(f"❌ Error getting conversation count: {e}")
    
    def close_database_connection(self):
        """Close the database connection."""
        try:
            self.db.close_connection()
            print("🔌 Database connection closed successfully!")
        except Exception as e:
            print(f"❌ Error closing database connection: {e}")


# Example usage and demo
if __name__ == "__main__":
    try:
        client = OpenAIClientWithMemoryAndToolsPersistence()
        client.start_conversation()
    except ValueError as e:
        print(f"Setup error: {e}")
        print("Please set your OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        print("\nNote: Weather data is provided by wttr.in (free, no API key required)")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Ensure database connection is closed
        if 'client' in locals():
            client.close_database_connection()
