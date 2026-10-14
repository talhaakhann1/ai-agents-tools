from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt= ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

llm=ChatMistralAI(
    model=""
)

parser=StrOutputParser()

chain=prompt|llm|parser

result=chain.invoke("what is deep learning")

print(result.content)