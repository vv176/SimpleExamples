import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI


# Load environment variables from .env file
load_dotenv()


class MovieRecommenderAgent:
    """
    A chat-based, multi-hop tool-calling agent for movie recommendations.

    Data is kept entirely in-memory for demo/teaching purposes:
    - Conversation history is stored in a Python list
    - User reviews are stored in a list of tuples: (user_id, movie_id, review)
    - Movie catalog is stored in a list of tuples: (movie_id, movie_name, [genres])

    Tools exposed to the model:
    1) fetch_past_reviews(user_id)
    2) getGenre(movie_ids)
    3) getMovies(genres, pastIds)
    4) sendResponse(response)

    The agent runs a multi-hop loop: execute tool calls → feed results → repeat,
    until the model calls sendResponse, at which point we print the final reply and stop.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in environment or pass api_key.")

        self.client = OpenAI(api_key=self.api_key)
        self.model = model

        # In-memory conversation history
        self.conversation_history: List[Dict[str, Any]] = []

        # In-memory data stores
        # Reviews: (user_id, movie_id, review)
        self.past_reviews: List[Tuple[int, int, str]] = [
            (101, 1, "Loved it, great sci-fi!"),
            (101, 2, "Okayish, too slow"),
            (101, 3, "Amazing visuals, would watch again"),
            (202, 2, "Fantastic drama"),
            (202, 4, "Not my type"),
        ]

        # Movies: (movie_id, movie_name, [genres])
        self.movies: List[Tuple[int, str, List[str]]] = [
            (1, "Interstellar", ["Sci-Fi", "Adventure", "Drama"]),
            (2, "The Irishman", ["Crime", "Drama"]),
            (3, "Blade Runner 2049", ["Sci-Fi", "Thriller"]) ,
            (4, "La La Land", ["Romance", "Musical", "Drama"]),
            (5, "Arrival", ["Sci-Fi", "Drama"]),
            (6, "Mad Max: Fury Road", ["Action", "Adventure", "Sci-Fi"]),
            (7, "Whiplash", ["Drama", "Music"]) ,
            (8, "Inception", ["Sci-Fi", "Action", "Thriller"]),
        ]

        # Tool definitions for the Chat Completions API
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "fetch_past_reviews",
                    "description": "Return the (movie_id, review) pairs for this user_id from in-memory data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "integer", "description": "User id to fetch reviews for"}
                        },
                        "required": ["user_id"]
                    }
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "getGenre",
                    "description": "Given a list of movie_ids, return the union of genres for those movies.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "movie_ids": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "List of movie ids"
                            }
                        },
                        "required": ["movie_ids"]
                    }
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "getMovies",
                    "description": "Return movies whose genres intersect the provided genres, excluding pastIds.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "genres": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Target genres to match"
                            },
                            "pastIds": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Movie ids the user has already watched to exclude"
                            }
                        },
                        "required": ["genres", "pastIds"]
                    }
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "sendResponse",
                    "description": "Signal the final response to the user. Ends the multi-hop loop.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "response": {"type": "string", "description": "Final message to the user"}
                        },
                        "required": ["response"]
                    }
                },
            },
        ]

    # =============================
    # Tool implementations
    # =============================
    def fetch_past_reviews(self, user_id: int) -> List[Tuple[int, str]]:
        results: List[Tuple[int, str]] = []
        for uid, movie_id, review in self.past_reviews:
            if uid == user_id:
                results.append((movie_id, review))
        return results

    def get_genre(self, movie_ids: List[int]) -> List[str]:
        genre_set = set()
        for mid, _name, genres in self.movies:
            if mid in movie_ids:
                for g in genres:
                    genre_set.add(g)
        return sorted(list(genre_set))

    def get_movies(self, genres: List[str], past_ids: List[int]) -> List[str]:
        results: List[str] = []
        genre_set = set([g.strip() for g in genres])
        past_set = set(past_ids)
        for mid, name, movie_genres in self.movies:
            if mid in past_set:
                continue
            if any(g in genre_set for g in movie_genres):
                results.append(name)
        return results

    def send_response(self, response: str) -> str:
        # The agent will stop when the model calls this tool; we just return the response.
        return response

    # =============================
    # Agent loop
    # =============================
    def run(self):
        print("🎬 Movie Recommender Agent (Multi-hop) is ready!")
        print("Ask for recommendations. First, I'll ask for your user id (e.g., 101). Type 'exit' to quit.")

        system_message = (
            "You are a helpful movie recommendation assistant with access to tools. "
            "STRICT ORDER OF OPERATIONS (no deviations): "
            "1) If user_id is unknown, ask for it. "
            "2) Once you have user_id, Call fetch_past_reviews(user_id). From those reviews, select ONLY positive ones (keywords: 'love', 'loved', 'great', 'amazing', 'fantastic', 'good', 'favorite') and extract their movie_ids. "
            "3) Next, we need to get the genres of the movieIDs obtained in the previous step. Call getGenre(movie_ids_of_positive_reviews). "
            "4) Now call getMovies for the genres obtained from last step i.e. getMovies(genres, pastIds=all previously watched movie_ids). Make sure to pass the pastIds as all previously watched movie_ids from fetch_past_reviews. This is important to ensure that you don't recommend movies that the user has already watched."
            "Finally, compose a friendly recommendation and call sendResponse(response). The response must include all the movie names obtained from getMovies. "
            "Keep responses concise and explain choices briefly. You may perform multiple tool calls across multiple steps (multi-hop) until you can send the final response."
        )

        while True:
            user_input = input("\n👤 You: ").strip()
            if not user_input:
                print("Please enter a message.")
                continue
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\n🤖 AI: Goodbye!")
                break

            # Build messages: system + conversation + new user
            messages: List[Dict[str, Any]] = []
            messages.append({"role": "system", "content": system_message})
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_input})

            # First model call for this user turn
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto"
            )

            # Multi-hop loop: keep going until model returns final answer (no tool_calls)
            # Record user message once per turn in history
            self.conversation_history.append({"role": "user", "content": user_input})
            response_message = response.choices[0].message

            # If there are no tool calls, just print and update history, then move to next user turn
            if not response_message.tool_calls:
                final_content = (response_message.content or "").strip()
                if final_content:
                    self.conversation_history.append({"role": "assistant", "content": final_content})
                    print(f"🤖 AI: {final_content}")
                continue

            while True:
                if response_message.tool_calls:
                    # Tool calls for this hop
                    all_tool_calls = response_message.tool_calls

                    # Add assistant tool_calls to messages (grouped for API correctness)
                    messages.append({
                        "role": "assistant",
                        "content": response_message.content or "",
                        "tool_calls": all_tool_calls
                    })

                    # Execute tools; record each tool call and result separately in history

                    for tool_call in all_tool_calls:
                        fname = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)

                        # Record assistant tool call (separate entry per tool call)
                        self.conversation_history.append({
                            "role": "assistant",
                            "content": response_message.content or "",
                            "tool_calls": [tool_call]
                        })
                        print(f"{fname}:::{json.dumps(args)}")
                        if fname == "fetch_past_reviews":
                            user_id = int(args.get("user_id"))
                            result = self.fetch_past_reviews(user_id)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result)
                            })
                            # Record tool result as a separate history entry
                            self.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result)
                            })
                            print(result)

                        elif fname == "getGenre":
                            movie_ids = list(map(int, args.get("movie_ids", [])))
                            result = self.get_genre(movie_ids)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result)
                            })
                            self.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result)
                            })
                            print(result)

                        elif fname == "getMovies":
                            genres = list(map(str, args.get("genres", [])))
                            past_ids = list(map(int, args.get("pastIds", [])))
                            result = self.get_movies(genres, past_ids)
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result)
                            })
                            self.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(result)
                            })
                            print(result)

                        elif fname == "sendResponse":
                            response_text = str(args.get("response", ""))
                            # Also append a matching tool message to satisfy API requirements
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": response_text
                            })
                            self.conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": response_text
                            })
                            print(f"🤖 AI: {response_text}")
                            # Also store the final assistant message in history for context
                            self.conversation_history.append({"role": "assistant", "content": response_text})
                            break

                        else:
                            # Unknown tool: return empty content
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps({"error": "Unknown tool"})
                            })
                            tool_results_for_history.append({"tool_call_id": tool_call.id, "content": json.dumps({"error": "Unknown tool"})})

                    # If we just handled sendResponse, end this user turn
                    if any(tc.function.name == "sendResponse" for tc in all_tool_calls):
                        break

                    # Make the next call (next hop)
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=self.tools,
                        tool_choice="auto"
                    )
                    response_message = response.choices[0].message
                    # Continue the while True loop for next hop
                    continue

                # No tool calls: final assistant content for this turn
                final_content = (response_message.content or "").strip()
                if final_content:
                    self.conversation_history.append({"role": "assistant", "content": final_content})
                    print(f"🤖 AI: {final_content}")
                break


if __name__ == "__main__":
    try:
        agent = MovieRecommenderAgent()
        agent.run()
    except ValueError as e:
        print(f"Setup error: {e}")
        print("Please set your OPENAI_API_KEY environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
    except Exception as e:
        print(f"Unexpected error: {e}")


