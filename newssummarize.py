
from dotenv import load_dotenv

load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

search_tool = TavilySearchResults(max_results=5)

llm = ChatGroq(
    model="openai/gpt-oss-20b"
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful assistant.

    Summarize the following news into clear bullet points:
    {news}
    """
)

chain = prompt | llm | StrOutputParser()

news_result = search_tool.invoke({"query": "Latest news of AI in 2026"})

result = chain.invoke({"news": news_result})

print(result)
