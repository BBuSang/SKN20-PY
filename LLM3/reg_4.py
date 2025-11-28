import warnings
import os
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()
api_key = os.environ.get('OPENAI_API_KEY')
if not api_key :
    raise ValueError('OPENAI_API_KEY not set')

# 필수 라이브러리 로드
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

script_dir = os.path.dirname(os.path.abspath(__file__))
docs_path = os.path.join(script_dir, 'sample_docs')
print(f"docs path: {docs_path}")

loader = DirectoryLoader(
    docs_path,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
document = loader.load()
print('읽은 문서의 수 : ', len(document))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
)
# 스플릿 = 청킹
doc_splits = text_splitter.split_documents(document)
print(f'청킹개수 : {len(doc_splits)}')

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma.from_documents(
    documents=doc_splits,
    embedding=embedding_model,
    collection_name='basic-rag-collection',
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

prompt_template = ChatPromptTemplate.from_messages([
    ('system', 'Answer questions based on the context below. If the question cannot be answered using the context, say "I don\'t know".\n\nContext:\n{context}'),
    ('human', 'Context:\n{context}\n\nQuestion: {question}\nAnswer: ')
])

def format_docs(docs):
    return '\n\n---\n\n'.join([ doc.page_content for doc in docs ])

# LCEL 방식 Runnable 객체 실행 invoke → 파이프라인
llm = ChatOpenAI(model = 'gpt-4o-mini', temperature = 0)
rag_chain = (
    {"context" : retriever | format_docs, 'question' : RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

test_question = [
    'RAG란 무엇인가요?'
    , 'LangGraph의 핵심 개념을 알려주세요.'
    , '프롬프트 엔지니어링 기법에는 어떤 것들이 있나요?'
]

def ask_question(question):
    '''질문에 대한 답변생성'''
    answer = rag_chain.invoke(question)
    retrievered_docs = retriever.invoke(question)
    sources = [os.path.basename(doc.metadata.get('source', 'unknown')) for doc in retrievered_docs]
    return answer, sources

# 각 질문에 대한 답변 생성

for i, question in enumerate(test_question, 1):
    print(f'question_{i} : {question}')
    answer, sources = ask_question(question)
    print(f'answer : {answer}')
    print(f'sources : {sources}')