import os
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')
load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key:
    raise ValueError('OPENAI_API_KEY not set')

# 필수 라이브러리 로드
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 문서로드
path = 'C:/SKN20/LLM3/sample_docs'
loader = DirectoryLoader(
    path,
    glob = '**/*.txt',
    loader_cls=TextLoader,
    loader_kwargs={'encoding':'utf-8'}
)
docs = loader.load()

# 청크
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=['\n\n', '\n', '.', ' ', '']
)
chunk_docs = splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
# 벡터
vectorstore = Chroma.from_documents(
    documents=chunk_docs,
    collection_name='basic_rag_collection',
    embedding=embedding_model
)

# 리트리버
retriever = vectorstore.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3}
)

llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
# self-RAG (자기 보정 RAG)
print(f'3. self-RAG')
print(f'검색된 문서의 관련성을 평가함여 필터링합니다.\n')

# 프롬프트
check_prompt = ChatPromptTemplate.from_template("""
다음 문서가 질문에 관련이 있는지 평가하세요
                                 'yes' 또는 'no'로 대답하세요.

                                 문서 : {document}
                                 질문 : {question}
                                 관련성:
""")
# LCEL 체인 구성
check_prompt_chain = check_prompt | llm | StrOutputParser()

def filler_relevant_docs(docs, question):
    '''관련 문서만 필터링'''
    relevant_docs = []
    for doc in docs:
        result = check_prompt_chain.invoke({
            'document': doc.page_content,
            'question': question
        })
        if 'yes' in result.lower():
            relevant_docs.append(doc)
    return relevant_docs


# 출력
question = 'RAG의 장점은 무엇인가요?'
docs = retriever.invoke(question)
result = check_prompt_chain.invoke({
    'document': docs[0].page_content,
    'question': question
})
print(f'평가결과 : {result}')