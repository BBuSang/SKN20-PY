# Langchain 프롬프트 템플릿
# LCEL 사용법
# RAG 체인 구성 및 실행
# 답변 품질 개선 전략

# 파이프라인
# [질문] -> [리트리버] -> [관련문서] -> [프롬프트] -> [LLM] -> [답변]

import os
import warnings
import pickle
from dotenv import load_dotenv

# 경고 메세지 삭제
warnings.filterwarnings("ignore")
load_dotenv()

# openapi key 확인
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")

# 필수 라이브러리 로드
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time

# vertorDB 로드
# 임베딩 모델 초기화
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
# 이전 단계에서 저장항 vectorDB 로드
persist_dir = './chroma_db_reg2'
contig_file = 'vectordb_config.pkl'
if os.path.exists(persist_dir):
    vectorstore = Chroma(
        persist_directory=persist_dir,
        collection_name="persistent_rag",
        embedding_function=embedding_model
    )
else :
    raise ValueError(f"지정된 경로에 벡터 DB가 존재하지 않습니다: {persist_dir}")

# 리트리버 설정
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

# LLM 모델 생성
llm = ChatOpenAI(
    model = 'gpt-4o-mini',
    temperature=0,
)

# 프롬프트 템플릿 생성
basic_prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 전문적인 지식 기반 Q&A 시스템입니다.

## 역할
제공된 문맥을 분석하여 사용자의 질문에 정확하게 답변합니다.

## 답변 규칙
1. **출처 기반**: 반드시 제공된 문맥의 정보만 사용합니다.
2. **정확성**: 문맥에 없는 내용은 추측하지 않습니다.
3. **명확성**: 답변은 이해하기 쉽게 구조화합니다.
4. **언어**: 한국어로 답변합니다.

## 답변 불가 시
문맥에서 정보를 찾을 수 없으면:
"제공된 문서에서 해당 정보를 찾을 수 없습니다. 다른 질문을 해주세요."
라고 답변합니다."""),
    ("human", """## 참조 문맥
{context}

## 질문
{question}

## 답변""")
])

# 문서 포메터 작성 : 
def format_docs(docs):
    '''검색된 문서들을 하나의 문자열로 포멧팅'''
    return '\n\n---\n\n'.join([doc.page_content for doc in docs])

def format_docs_with_source(docs):
    '''검색된 문서들을 하나의 문자열로 포멧팅 (출처 포함)'''
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', '알 수 없음')
        formatted_docs.append(f"[출처: {source}]\n{doc.page_content}")
    return '\n\n---\n\n'.join(formatted_docs)

# 테스트
test_docs = retriever.invoke("RAG에 대해 설명해줘.")
print("검색된 문서 포맷팅 결과:\n", format_docs_with_source(test_docs))

# RAG 체인 구성
# 기본 RAG 체인 (LECL 사용)
rag_chain = (
    {
        'context' : retriever | format_docs, 
        'question' : RunnablePassthrough()
        | basic_prompt
        | llm
        | StrOutputParser()
    }
)
print("RAG 체인 구성 완료.")
# 출처 포함 RAG 체인
rag_chain_wirh_source = (
    {
        'context' : retriever | format_docs_with_source, 
        'question' : RunnablePassthrough()
        | basic_prompt
        | llm
        | StrOutputParser()
    }
)
print("출처 포함 RAG 체인 구성 완료.")
'''
체인구조
질문 -> retriever -> 관련문서 검색
        format_docs-> 문자열로 변환

        prompt -> 답변생성

        strparser -> 문자열로 파싱
'''

print('RAG 체인 실행 테스트 시작...')
test_questions = [
    "RAG란 무엇이고 어떤 장점이 있나요?",
    "LangChain의 주요 구성 요소를 설명해주세요.",
    "VectorDB에는 어떤 종류가 있나요?",
]

for i, question in enumerate(test_questions):
    print(f"\n[질문 {i+1}]: {question}")
    start_time = time.time()
    answer = rag_chain.invoke(question)
    end_time = time.time()
    print(f"[답변 {i+1}]: {answer}")
    print(f"응답 시간: {end_time - start_time:.2f}초")

    # 참조문서
    print(f"\n[출처 포함 답변 {i+1}]:")
    start_time = time.time()
    answer_with_source = rag_chain_wirh_source.invoke(question)
    end_time = time.time()
    print(f"[출처 포함 답변 {i+1}]: {answer_with_source}")
    print(f"응답 시간: {end_time - start_time:.2f}초")

# 고급 RAG 사용
print('RAG 성능향상을 위한 고급 패턴')

print('query transformaton ')
query_transform_prompt = ChatPromptTemplate.from_template(
    '''다음 질문을 검색에 더 적합한 형태로 변환해주세요.
    키워드 중심으로 명확하게 바꿔주세요

    원본질문:{question}

    변환된 검색어 (한 줄로):'''
)
query_chain = query_transform_prompt | llm | StrOutputParser()

orginal_question = 'RAG가 뭔지 좀 알려주세요'
transformed = query_chain.invoke({'question':orginal_question })
print(f'    원본 : {orginal_question}')
print(f'    변환 : {transformed}')

