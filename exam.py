from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
llm = ChatOpenAI(openai_api_key=api_key)

output = llm.invoke("2024년 청년 지원 정책에 대하여 알려줘")
print(output)

from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 청년을 행복하게 하기 위한 정부정책 안내 컨설턴트야"),
    ("user", "{input}")
])

chain = prompt | llm 

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
chain = prompt | llm | output_parser
chain.invoke({"input": "2024년 청년 지원 정책에 대해 알려줘"})