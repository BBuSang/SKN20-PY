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
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# LangGraph 관련 임포트
from langgraph.graph import StateGraph, START, END

# 환경설정
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되어 있지 않습니다.")

def langgraph_rag():
    '''VectorDB  기반 LangGraph RAG'''
    # 상태 정의
    class RAGState(TypedDict):
        question: str
        documents: List[Document]
        doc_scores : List[float]
        search_type : str
        answer : str
    # 문서 로드
    path = 'C:/SKN20/LLM3/sample_docs'
    loader = DirectoryLoader(
        path = path,
        glob = '**/*.txt',
        loader_cls = TextLoader,
        loader_kwargs = {'encoding':'utf-8'}
    )
    docs = loader.load()

    # VectorDB 구축
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50, separators=['\n\n', '\n', ' ', ''])
    splits = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
    vectorstores = Chroma.from_documents(splits, embeddings, collection_name='langgraph_rag')
    print(f'VectorDB 구충 완료 청크 개수: {len(splits)}')
    # llm 초기화
    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)

    # 노드함수들
        # 리트리버 함수
    def retrieval_node(state: RAGState) -> dict:
        '''VectorDB 검색 노드'''
        question = state['question']
        docs_with_scores = vectorstores.similarity_search_with_score(question, k=3)
        
        # 문서와 점수를 분리
        documents = [doc for doc, score in docs_with_scores]
        scores = [1-score for doc, score in docs_with_scores]  # 거리 → 유사도 변환
        
        print(f'[retriever] {len(documents)}개의 문서 검색됨.')
        return {'documents': documents, 'doc_scores': scores, 'search_type':'internal'}

    def grade_documents_node(state: RAGState) -> dict:
        '''문서 평가 노드'''
        threshold = 0.3
        filtered_docs, filtered_scores = [], []
        for doc, score in zip(state['documents'], state['doc_scores']):
            if score <= threshold:
                filtered_docs.append(doc)
                filtered_scores.append(score)
        print(f'[grader]  {len(state["documents"])}개 -> {len(filtered_docs)}개 문서가 기준 통과')
        return {'documents': filtered_docs, 'doc_scores': filtered_scores}

    def web_search_node(state: RAGState) -> dict:
        '''웹문서 검색 노드'''
        web_result = Document(
            page_content=f'웹 검색 결과: {state["question"]}에 대한 정보입니다.',
            metadata={'source':'web_search'}
        )
        return {'documents':[web_result], 'doc_scores':[0.8], 'search_type':'web'}
    
    def generation_node(state: RAGState) -> dict:
        '''생성노드 : 검색된 문서를 기반으로 답변 생성'''
        context = '\n'.join([doc.page_content for doc in state['documents']])
        prompt = ChatPromptTemplate.from_messages(
            [
                ('system', '제공된 문맥을 바탕으로 한국어로 답변하세요.'),
                ('human', '문맥:\n{context}\n\n질문:{question}\n\n답변:')
            ]
        )
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            'context': context,
            'question': state['question']
        })
        return {'answer': answer}

    # 조건함수
    def decide_to_generate(state: RAGState) -> Literal['generate', 'web_search']:
        '''조건부 분기 함수'''
        if state['documents'] and len(state['documents']) > 0:
            print('[decision] 문서 존재 -> generate 노드로 이동')
            return 'generate'
        else:
            print('[decision] 문서 없음 -> web_search 노드로 이동')
            return 'web_search'
    
    # 그래프 구축(add_node, add_edge, add_conditional_edges)
    graph = StateGraph(RAGState)
    graph.add_node('retrieval', retrieval_node)
    graph.add_node('grade', grade_documents_node)
    graph.add_node('web_search', web_search_node)
    graph.add_node('generate', generation_node)
    graph.add_edge(START, 'retrieval')
    graph.add_edge('retrieval', 'grade')
    graph.add_edge('grade', END)
    graph.add_conditional_edges(
        'grade',
        decide_to_generate,
        {
            'generate':'generate',
            'web_search':'web_search'
        }
    )
    graph.add_edge('web_search', 'generate')
    graph.add_edge('generate', END)
    # 그래프 컴파일
    app = graph.compile()

    # 그래프 invoke(질문)
    test_question = [
        'LangGraph의 핵심 개념을 설명해  주세요',
        'RAG란 무엇인가요?',
        '오늘 서울 날씨는 어떤가요?' # 내부 문서에 없음
    ]
    for question in test_question:
        print(f'\n[질문] {question}')
        result = app.invoke({
            'question': question,
            'documents': [],
            'doc_scores': [],
            'search_type': '',
            'answer': ''
        })
        print(f'[답변] {result["answer"]}')
# LangGraph RAG 실행
langgraph_rag()




# # 조건부 엣지가 포함된 그래프
# def conditional_graph():
#     '''조건부 엣지를 포함된 LangGraph생성
#     검색 결과에 따라 다른 경로로 분기'''
#     # 상태 정의
#     class ConditionalState(TypedDict):
#         question: str
#         documents: List[Document]
#         search_type: str
#         answer: str
#     # 내부문서 LoadeText or ...
#     INTENAL_DOCS = {
#         "회사" : [Document(page_content='우리 회사의 AI전략은 RAG 시스템 구축입니다')],
#         '정책' : [Document(page_content='사내 데이터 보안 정책은 외부  공유 금지입니다')]
#     }
#     # 노드 함수들을구현
#     def internal_search_node(state: ConditionalState) -> dict:
#         '''내부문서 검색 노드'''
#         question = state['question']
#         documents = []
#         for key, docs in INTENAL_DOCS.items():
#             if key in question:
#                 documents.extend(docs)
#         return {'documents': documents, 'search_type':'internal'}
    
#     def web_search_node(state: ConditionalState) -> dict:
#         '''웹문서 검색 노드'''
#         # 시뮬레이션 : 실제는 web retrieval 사용
#         mock_result = Document(
#             page_content=f'{state["question"]}에 대한 웹 검색 결과입니다.',
#             metadata={'source':'web'}
#         )
#         return {'documents': [mock_result], 'search_type':'web'}
#     def generation_node(state: ConditionalState) -> dict:
#         '''생성노드 : 검색된 문서를 기반으로 답변 생성'''
#         llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0)
#         contest = '\n'.join([doc.page_content for doc in state['documents']])
#         prompt = ChatPromptTemplate.from_template(
#             '''컨텍스트:{contest}\n\n질문:{question}\n\n답변:'''
#         )
#         chain = prompt | llm | StrOutputParser()
#         answer = chain.invoke({
#             'contest': contest,
#             'question': state['question']
#         })
#         return {'answer': answer}
#     # 조건 함수
#     def decide_search_path(state: ConditionalState) -> Literal['internal_search', 'web_search']:
#         '''검색 경로 결정 함수'''
#         if '회사' in state['question'] or '정책' in state['question']:
#             return 'internal_search'
#         else:
#             return 'web_search'
        
    
#     # 그래프 구축
#     graph = StateGraph(ConditionalState)
#     graph.add_node('internal_search', internal_search_node)
#     graph.add_node('web_search', web_search_node)
#     graph.add_node('generate', generation_node)

#     # START → 조건 분기 (바로 decide_search_path로!)
#     graph.add_conditional_edges(
#         START,
#         decide_search_path,
#         {
#             'internal_search': 'internal_search',
#             'web_search': 'web_search'
#         }
#     )
    
#     # 각 검색 노드 → generate로
#     graph.add_edge('internal_search', 'generate')
#     graph.add_edge('web_search', 'generate')
#     graph.add_edge('generate', END)

#     app = graph.compile()

#     # 테스트 1 : 내부문서에 있는 질문
#     print('\n[테스트1] 내부문서가 존재하는 경우')
#     result1 = app.invoke({
#         'question':'우리 회사의 AI전략은?',
#         'documents':[],
#         'search_type':'',
#         'answer':''
#     })
#     print(f'답변 : {result1["answer"]}')
#     # 테스트 2 : 내부문서에 없는 질문
#     print('\n[테스트2] 내부문서가 존재하지 않는 경우')
#     result2 = app.invoke({
#         'question':'LangGraph란 무엇인가요?',
#         'documents':[],
#         'search_type':'',
#         'answer':''
#     })
#     print(f'답변 : {result2["answer"]}')

# # 조건부 분기 테스ㅡ
# conditional_graph()