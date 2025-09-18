# backend/chatbot.py
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from lyrics import LYRICS_DATA

# .env 파일에서 API 키 로드
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class LyricChatbot:
    def __init__(self):
        self.vector_store = self._create_vector_store()
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
        self.prompt = self._create_prompt_template()
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def _create_vector_store(self):
        # 텍스트 데이터를 LangChain의 Document 형식으로 변환
        # 각 줄을 별도의 문서로 취급하여 검색 정확도를 높임
        documents = [doc for doc in LYRICS_DATA.split('\n') if doc.strip()]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.create_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        
        # FAISS 벡터 스토어 생성
        vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
        print("Vector store created successfully.")
        return vectorstore

    def _create_prompt_template(self):
        template = """
        당신은 양홍원의 앨범 '오보에' 가사 전문가입니다. 
        주어진 가사 내용을 바탕으로 사용자의 질문에 대해 친절하고 깊이 있게 설명해주세요. 
        가사에 없는 내용은 추측하지 말고, 주어진 내용 안에서만 답변하세요.

        --- 관련 가사 내용 ---
        {context}
        --------------------

        질문: {question}
        답변:
        """
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def get_response(self, user_query):
        try:
            # 질문과 가장 유사한 가사 구절을 벡터 스토어에서 검색
            retriever = self.vector_store.as_retriever(search_kwargs={'k': 5})
            relevant_docs = retriever.invoke(user_query)
            
            # 검색된 문서들의 내용을 context로 합침
            context = "\n".join([doc.page_content for doc in relevant_docs])
            
            # 프롬프트와 LLM을 통해 답변 생성
            response = self.chain.run({"context": context, "question": user_query})
            return response
        except Exception as e:
            print(f"Error getting response: {e}")
            return "죄송합니다, 답변을 생성하는 중에 오류가 발생했습니다."

# 챗봇 인스턴스 생성 (서버 시작 시 한 번만 실행)
chatbot_instance = LyricChatbot()