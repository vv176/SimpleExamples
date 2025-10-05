# pip install "langchain>=0.2" langchain-openai langchain-google-genai langchain-anthropic python-dotenv
import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic  # type: ignore

load_dotenv()


class LLMFactory:
    """Factory for creating LangChain chat models by provider name."""

    @staticmethod
    def create(provider: str, model: Optional[str] = None, temperature: float = 0.0):
        name = (provider or "").strip().lower()

        if name in {"openai", "oai"}:
            # Default model aligned with existing examples
            llm_model = model or "gpt-4o"
            return ChatOpenAI(
                model=llm_model,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
            )

        if name in {"gemini", "google"}:
            # Default model per your preference
            llm_model = model or "gemini-2.5-flash-preview-05-20"
            return ChatGoogleGenerativeAI(
                model=llm_model,
                temperature=temperature,
                api_key=os.getenv("GEMINI_API_KEY"),
            )

        if name in {"claude", "anthropic"}:
            # Reasonable Claude default
            llm_model = model or "claude-3-5-sonnet-latest"
            return ChatAnthropic(model=llm_model, temperature=temperature, api_key=os.getenv("ANTHROPIC_API_KEY"))

        raise ValueError("Unknown provider. Choose one of: OpenAI, Gemini, Claude.")



def main():
    print("Select provider: openai, gemini, or claude")
    provider = input("Provider: ").strip()

    try:
        llm = LLMFactory.create(provider)
    except Exception as error:
        print(f"Error: {error}")
        return

    # Initialize messages with only a SystemMessage
    messages = [
        SystemMessage(content="You are a helpful assistant. Keep answers concise."),
    ]

    print("Type your message and press Enter. Type 'quit' to exit.\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in {"quit", "exit", ":q"}:
                # Before exiting, print the entire conversation history
                print("Goodbye!")
                break
            if not user_input:
                continue

            # Add user message
            messages.append(HumanMessage(content=user_input))

            # Invoke LLM with full message history
            ai_message = llm.invoke(messages)  # returns an AIMessage

            # Display and store AI response
            print(f"AI: {ai_message.content}")
            messages.append(ai_message)
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting...")


if __name__ == "__main__":
    main()


