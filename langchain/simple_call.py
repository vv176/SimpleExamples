# pip install "langchain>=0.2" langchain-openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

import os
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

messages = [
    SystemMessage(content="You are an amazing joke teller."),
    HumanMessage(content="Tell me a joke about a cat")
]

print(type(messages[0]))
print(messages[0])

print(type(messages[1]))
print(messages[1])

result = llm.invoke(messages)  # returns an AIMessage
print(type(result))
print(result)


print(result.content)
