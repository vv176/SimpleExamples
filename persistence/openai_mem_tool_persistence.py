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
            
            response = requests.get(url, timeout=15)
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
            i = 0
            
            while i < len(db_history):
                entry = db_history[i]
                role = entry['role']
                content = entry['response']
                
                # Check if this is an assistant message with tool_calls (stored as JSON)
                if role == 'assistant':
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and 'tool_calls' in parsed:
                            # Reconstruct assistant message with tool_calls in proper format
                            tool_calls_list = []
                            for tc in parsed.get("tool_calls", []):
                                # Convert to API format
                                tool_calls_list.append({
                                    "id": tc.get("id"),
                                    "type": "function",  # Default type
                                    "function": {
                                        "name": tc.get("function_name"),
                                        "arguments": tc.get("arguments")
                                    }
                                })
                            
                            assistant_msg = {
                                "role": "assistant",
                                "content": parsed.get("content", ""),
                                "tool_calls": tool_calls_list
                            }
                            messages.append(assistant_msg)
                            
                            # Check if next entry is a tool message (tool results)
                            if i + 1 < len(db_history) and db_history[i + 1]['role'] == 'tool':
                                i += 1  # Move to next entry
                                tool_entry = db_history[i]
                                tool_content = tool_entry['response']
                                
                                try:
                                    tool_parsed = json.loads(tool_content)
                                    if isinstance(tool_parsed, dict) and "tool_results" in tool_parsed:
                                        # Convert each tool result to proper tool message format
                                        for tool_result in tool_parsed.get("tool_results", []):
                                            tool_msg = {
                                                "role": "tool",
                                                "tool_call_id": tool_result.get("tool_call_id"),
                                                "content": tool_result.get("content", "")
                                            }
                                            messages.append(tool_msg)
                                except (json.JSONDecodeError, KeyError):
                                    # If parsing fails, skip this tool message
                                    pass
                            
                            i += 1
                            continue
                    except (json.JSONDecodeError, KeyError):
                        # Not JSON or doesn't have tool_calls, treat as normal message
                        pass
                
                # Regular message (user, assistant without tool_calls, system)
                # Skip tool messages that were already processed above
                if role != 'tool':
                    messages.append({"role": role, "content": content})
                
                i += 1
            
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
                
                # Store assistant message with tool_calls (serialize each tool call with id, function name, and args)
                tool_calls_data = {
                    "content": response_message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "function_name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                        for tc in all_tool_calls
                    ]
                }
                self.db.insert_conversation("assistant", json.dumps(tool_calls_data))
                
                # Store tool results (serialize with tool_call_id, function name, and content)
                tool_results_data = {
                    "tool_results": [
                        {
                            "tool_call_id": result["tool_call_id"],
                            "content": result["content"]
                        }
                        for result in tool_results
                    ]
                }
                self.db.insert_conversation("tool", json.dumps(tool_results_data))
                
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
                content = entry['response']
                
                if entry["role"] == "tool":
                    role_emoji = "🔧"
                    # Try to parse as JSON to display tool results
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "tool_results" in parsed:
                            tool_results = parsed["tool_results"]
                            result_contents = []
                            for result in tool_results:
                                tool_call_id = result.get("tool_call_id", "unknown")
                                result_content = result.get("content", "")
                                result_contents.append(f"[ID: {tool_call_id[:20]}... Result: {result_content}]")
                            content = " | ".join(result_contents)
                        else:
                            content = f"[Tool Result: {content}]"
                    except (json.JSONDecodeError, KeyError):
                        content = f"[Tool Result: {content}]"
                elif entry["role"] == "assistant":
                    # Check if this is an assistant message with tool_calls (stored as JSON)
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "tool_calls" in parsed:
                            tool_calls = parsed["tool_calls"]
                            tool_info = []
                            for tool_call in tool_calls:
                                tool_id = tool_call.get("id", "unknown")
                                func_name = tool_call.get("function_name", "unknown")
                                func_args = tool_call.get("arguments", "{}")
                                tool_info.append(f"🔧 ID: {tool_id[:20]}... | {func_name}({func_args})")
                            
                            base_content = parsed.get("content", "")
                            if base_content:
                                content = f"{base_content} {' | '.join(tool_info)}"
                            else:
                                content = " | ".join(tool_info)
                    except (json.JSONDecodeError, KeyError):
                        # Not JSON or doesn't have tool_calls, display as normal message
                        pass
                elif entry["role"] == "system":
                    role_emoji = "⚙️"
                
                timestamp = entry["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                display_content = content[:100] + ('...' if len(content) > 100 else '')
                print(f"{i}. {role_emoji} {entry['role'].title()} ({timestamp}): {display_content}")
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
