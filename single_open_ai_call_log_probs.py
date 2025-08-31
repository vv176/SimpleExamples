import openai
import os
import math
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
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,       # <- ask for log probabilities
                top_logprobs=5       # <- return the top-5 alternatives per token
            )
            # log(probability)
            choice = response.choices[0]
            print("\n== Token-level logprobs (first 20 tokens) ==")
            total_logprob = 0.0
            for i, t in enumerate(choice.logprobs.content[:20]):
               tok = t.token
               lp  = t.logprob            # natural log prob of chosen token
               total_logprob += lp
               print(f"{i:02d}: {tok!r:>12}  logprob={lp: .3f}")

               # Optional: show the top-5 alternatives for this position
               if t.top_logprobs:
                  alts = ", ".join(f"{a.token!r}:{a.logprob:.2f}" for a in t.top_logprobs)
                  print("     top:", alts)

            print("\nSequence log-likelihood (sum of token logprobs):", total_logprob)
            print("Sequence probability (approx):", math.exp(total_logprob))
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