# pip install "langchain>=0.2" langchain-openai
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Write in a {tone} tone."),
    ("human", "Summarize in one sentence:\n\n{text}")
])

chain = prompt | model | StrOutputParser()
print(chain.invoke({"tone": "concise", "text": "Transformers use self-attention to share information across tokens."}))


prompt_json = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Given a company name, return a JSON object containing its founder's name, date of establishment and headquarter location."),
    ("human", "Company name:\n\n{company_name}")
])

chain_normal = prompt_json | model | StrOutputParser()
res1 = chain_normal.invoke({"company_name": "Uber"})
print(res1)
print(type(res1))

chain_json = prompt_json | model | JsonOutputParser()
res2 = chain_json.invoke({"company_name": "Uber"})
print(res2)
print(type(res2))

print(res2.get("founder")) # changing to res1 errors out.

# Pydantic: a library for data validation and settings management using Python type hints.
# BaseModel: the core pydantic class; you subclass it to define validated schemas.
# Pydantic: Strict, configurable validation; detailed error messages; custom validators.

class Company(BaseModel):
    founder: str
    date_of_establishment: str
    headquarter_location: str
pydantic_output_parser = PydanticOutputParser(pydantic_object=Company)
chain_pydantic = prompt_json | model | pydantic_output_parser
res3 = chain_pydantic.invoke({"company_name": "Uber"})
print(res3)
print(type(res3))
print(res3.founder + " " + res3.date_of_establishment + " " + res3.headquarter_location)


