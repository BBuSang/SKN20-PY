import os
from dotenv import load_dotenv
from langchain_community.retrievers import TavilySearchAPIRetriever
load_dotenv()

retriever = TavilySearchAPIRetriever(k=3)
result = retriever.invoke('2025 멜론 뮤직 어워드 시상내역은 어떻게 되나요?')
context = '\n\n--\n\n'.join([doc.page_content for doc in result])

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 음원 분석 전문가입니다. 다음 수상내역을 참고로해서 올해 가장 유망하고 내년에도 유망한 가수를 추천해 주세요."),
    ("human", '질문 {question} 수상내역 : {context}')
])
llm = ChatOpenAI(temperature=0, model_name='gpt-4o-mini')
chain = prompt | llm | StrOutputParser()
question = '올해 가장 인기있는 가수는?'
result = chain.invoke({'context' : context, 'question': question})
print(result)