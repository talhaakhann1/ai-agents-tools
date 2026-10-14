from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda

# Components
model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

# Two different prompts
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

chain=RunnableParallel({
    "short":RunnableLambda(lambda x:x['short']|short_prompt|parser),
    "detailed":RunnableLambda(lambda x:x['detailed']|detailed_prompt|parser)
})

result=chain.invoke({
    'short':{'topic':''},
    'detailed':{'topic':''}
})

print(result['short'])
print(result['detailed'])