import os
from dotenv import load_dotenv

# LangChain 관련 라이브러리 임포트
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

# 가사 데이터 임포트
from lyrics import LYRICS_DATA

# .env 파일에서 API 키 로드
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class LyricChatbot:
    def __init__(self):
        self.vector_store = self._create_vector_store()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.chain = self._setup_conversational_chain()

    def _create_vector_store(self):
        """가사 데이터를 파싱하여 노래 제목 메타데이터와 함께 벡터 스토어를 생성합니다."""
        
        # 가사를 노래별로 분리
        songs = LYRICS_DATA.strip().split('\n\n')
        documents = []
        
        for song_data in songs:
            lines = song_data.strip().split('\n')
            title = lines[0].strip()
            content = "\n".join(lines[1:])
            
            # 노래 제목을 메타데이터로 추가
            doc = Document(page_content=content, metadata={"song_title": title})
            documents.append(doc)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)
        
        embeddings = OpenAIEmbeddings()
        
        vectorstore = FAISS.from_documents(split_docs, embedding=embeddings)
        print("Vector store with metadata created successfully.")
        return vectorstore

    def _setup_conversational_chain(self):
        """대화형 검색 체인을 설정합니다."""
        
        llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)
        
        # 답변 생성을 위한 커스텀 프롬프트 템플릿
        custom_prompt_template = """
        당신은 양홍원의 앨범 '오보에'를 심도 있게 분석하는 음악 평론가입니다.
        주어진 '관련 가사'와 '대화 내용'을 바탕으로 사용자의 질문에 대해 깊이 있고 통찰력 있는 답변을 제공해주세요.
        답변은 반드시 주어진 가사 내용에 근거해야 하며, 가사에 없는 내용은 절대 추측하거나 지어내면 안 됩니다.
        친절하고 전문적인 말투를 사용해주세요.

        --- 관련 가사 ---
        {context}
        -----------------

        --- 대화 내용 ---
        {chat_history}
        -----------------

        질문: {question}
        전문가 답변:
        """
        
        PROMPT = PromptTemplate(
            template=custom_prompt_template, input_variables=["context", "chat_history", "question"]
        )

        # 대화형 체인 생성
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vector_store.as_retriever(search_kwargs={'k': 4}), # 관련성 높은 구절 4개를 검색
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        return chain

    def get_response(self, user_query):
        """사용자 질문에 대한 답변을 생성합니다."""
        try:
            result = self.chain.invoke({"question": user_query})
            return result['answer']
        except Exception as e:
            print(f"Error getting response: {e}")
            return "죄송합니다, 답변을 생성하는 중에 오류가 발생했습니다. API 키 설정을 확인해주세요."

# 서버 시작 시 한 번만 챗봇 인스턴스를 생성합니다.
chatbot_instance = LyricChatbot()
