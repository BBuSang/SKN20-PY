import os
import warnings
warnings.filterwarnings("ignore")

from typing import List, Literal
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangChain 관련 임포트
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# LangGraph 관련 임포트
from langgraph.graph import StateGraph, START, END

# 환경설정
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")

# state 정의
# TypeDict 상태 스키마 정의
class RAGState(TypedDict):
    query: str
    context: str
    answer: str
    docs: List[Document]

initial_state : RAGState = {
    'question':'RAG란 무엇인가요?',
    'documents':[],
    'context':'',
    'answer':''
}

print(f'초기상태 : {initial_state}')

# 상태 업데이트
state = initial_state.copy()

# 1. 검색 노드가 문서에 추가
state['documents'] = [
    Document(page_content='RAG는 검색 증강 생성입니다.', matadata={'source': 'wiki'}),
    Document(page_content='RAG는 LLM의 한계를 극복합니다', matadata={'source': 'blog'})
]

# 조건부 엣지가 포함된 그래프
def conditional_graph():
    '''조건부 엣지를 포함된 LangGraph생성
    검색 결과에 따라 다른 경로로 분기'''
    # 상태 정의
    class ConditionalState(TypedDict):
        question: str
        documents: List[Document]
        search_type: str
        answer: str
    # 내부문서 LoadeText or ...
    INTENAL_DOCS = {
        "회사" : [Document(page_content='우리 회사의 AI전략은 RAG 시스템 구축입니다')],
        '정책' : [Document(page_content='사내 데이터 보안 정책은 외부  공유 금지입니다')]
    }
    # 노드 함수들을구현
    def internal_search_node(state: ConditionalState) -> dict:
        '''내부문서 검색 노드'''
        question = state['question']
        documents = []
        for key, docs in INTENAL_DOCS.items():
            if key in question:
                documents.extend(docs)
        return {'documents': documents, 'search_type':'internal'}
    
    def web_search_node(state: ConditionalState) -> dict:
        '''웹문서 검색 노드'''
        # 시뮬레이션 : 실제는 web retrieval 사용
        mock_result = Document(
            page_content=f'{state["question"]}에 대한 웹 검색 결과입니다.',
            metadata={'source':'web'}
        )
        return {'documents': [mock_result], 'search_type':'web'}
    def generation_node(state: ConditionalState) -> dict:
        '''생성노드 : 검색된 문서를 기반으로 답변 생성'''
        llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
        contest = '\n'.join([doc.page_content for doc in state['documents']])
        prompt = ChatPromptTemplate.from_template(
            '''컨텍스트:{contest}\n\n질문:{question}\n\n답변:'''
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            'contest': contest,
            'question': state['question']
        })
        return {'answer': answer}
    # 조건 함수
    def decide_search_path(state: ConditionalState) -> Literal['internal_search', 'web_search']:
        '''검색 경로 결정 함수'''
        if '회사' in state['question'] or '정책' in state['question']:
            return 'internal_search'
        else:
            return 'web_search'
        
    
    # 그래프 구축
    graph = StateGraph(ConditionalState)
    graph.add_node('internal_search', internal_search_node)
    graph.add_node('web_search', web_search_node)
    graph.add_node('generate', generation_node)

    # START → 조건 분기 (바로 decide_search_path로!)
    graph.add_conditional_edges(
        START,
        decide_search_path,
        {
            'internal_search': 'internal_search',
            'web_search': 'web_search'
        }
    )
    
    # 각 검색 노드 → generate로
    graph.add_edge('internal_search', 'generate')
    graph.add_edge('web_search', 'generate')
    graph.add_edge('generate', END)

    app = graph.compile()

    # 테스트 1 : 내부문서에 있는 질문
    print('\n[테스트1] 내부문서가 존재하는 경우')
    result1 = app.invoke({
        'question':'우리 회사의 AI전략은?',
        'documents':[],
        'search_type':'',
        'answer':''
    })
    print(f'답변 : {result1["answer"]}')
    # 테스트 2 : 내부문서에 없는 질문
    print('\n[테스트2] 내부문서가 존재하지 않는 경우')
    result2 = app.invoke({
        'question':'LangGraph란 무엇인가요?',
        'documents':[],
        'search_type':'',
        'answer':''
    })
    print(f'답변 : {result2["answer"]}')

# 조건부 분기 테스ㅡ
conditional_graph()