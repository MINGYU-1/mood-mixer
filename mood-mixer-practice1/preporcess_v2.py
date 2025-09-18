# 🎵 K-HIPHOP Track Mixer - Preprocessing Script 🎵
# [개선 사항]
# 1. LangChain과 OpenAI API를 활용하여 각 트랙의 특성을 담은 설명을 생성합니다.
# 2. LLM 요청을 Batch로 처리하여 처리 속도를 대폭 향상시켰습니다.
# 3. 생성된 설명을 고품질 임베딩 모델을 사용하여 벡터로 변환합니다.
# 4. tqdm을 활용하여 진행 상황을 시각적으로 표시해 사용자 경험을 개선합니다.
# 5. dotenv를 사용하여 API 키를 안전하게 관리합니다.

import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import ast
import os
from tqdm import tqdm
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

print("✅ [K-HIPHOP Track Mixer] 전처리 스크립트를 시작합니다.")

# --- 1. 환경 설정 및 API 키 로드 ---
print("🔑 환경 변수를 로드하고 API 키를 확인합니다...")
load_dotenv() # .env 파일에서 환경 변수를 로드합니다.

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("🚨 OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
    print("   프로젝트 폴더에 .env 파일을 만들고 API 키를 설정해주세요. (예: OPENAI_API_KEY=sk-...)")
    exit()
print("✔️ API 키 로드 완료.")

# LangChain LLM 초기화 (GPT-3.5-turbo 모델 사용)
llm = ChatOpenAI(temperature=0.7, openai_api_key=api_key, model_name='gpt-3.5-turbo')

# --- 2. 원본 데이터 로드 ---
print("💿 원본 노래 데이터(누에킁_수정.csv)를 로드합니다...")
try:
    df = pd.read_csv('누에킁_수정.csv')
    print(f"✔️ 총 {len(df)}개의 트랙 데이터를 로드했습니다.")
except FileNotFoundError:
    print("🚨 '누에킁_수정.csv' 파일이 없습니다. 프로젝트 폴더에 파일을 준비해주세요.")
    exit()

# --- 3. LangChain 프롬프트 및 체인 생성 ---
# LLM에게 작업을 지시할 프롬프트 템플릿을 정의합니다.
# 페르소나와 가이드를 명확히 제시하여 일관된 고품질의 결과물을 얻습니다.
prompt_template = """
당신은 한국 힙합 음악에 대한 깊은 조예와 감성적인 필력을 가진 음악 평론가입니다.
아래 주어진 노래 정보를 바탕으로, 노래의 매력이 잘 드러나는 자연스러운 한글 소개 문장을 1~2 문장으로 생성해주세요.

[노래 정보]
- 제목: {track_title}
- 아티스트: {artist}
- 발매 연도: {release_year}
- 에너지 점수 (0-100): {energy_score}
- 분위기 점수 (0-100, 높을수록 진중함): {vibe_score}
- 키워드: {keywords}

[문장 생성 가이드]
- 단순히 정보를 나열하지 말고, 감성적인 표현을 사용해 하나의 완성된 문단으로 만드세요.
- 에너지 점수가 높으면 '폭발적인', '강렬한' 등으로, 낮으면 '차분한', '몽환적인' 등으로 묘사하세요.
- 분위기 점수가 높으면 '깊이 있는', '성찰적인' 등으로, 낮으면 '가벼운', '청량한' 등으로 묘사하세요.
- 주어진 키워드를 문장에 자연스럽게 녹여주세요. (예: '#새벽 감성을 자극하는')
- 평론가다운 전문성과 감수성이 느껴지는 문장을 만들어주세요.
"""
prompt = PromptTemplate.from_template(prompt_template)
llm_chain = LLMChain(prompt=prompt, llm=llm)
print("🔗 LangChain 프롬프트 및 체인을 생성했습니다.")

# --- 4. LLM을 활용한 설명 문장 생성 (Batch 처리) ---
print("🤖 LLM을 사용하여 각 트랙의 설명 문장을 생성합니다. (Batch 처리로 효율성 증대)")

# .batch() 메서드에 전달하기 위해 데이터프레임을 딕셔너리 리스트 형태로 변환합니다.
inputs = []
for _, row in df.iterrows():
    try:
        # 'keywords' 컬럼이 문자열 형태의 리스트이므로 ast.literal_eval로 파싱
        keywords_list = ast.literal_eval(row['keywords'])
        keywords_str = ', '.join([kw.replace('#', '') for kw in keywords_list])
    except (ValueError, SyntaxError):
        # 파싱에 실패할 경우 기본값 사용
        keywords_str = "정보 없음"
    
    inputs.append({
        'track_title': row['track_title'],
        'artist': row['artist'],
        'release_year': row['release_year'],
        'energy_score': row['energy_score'],
        'vibe_score': row['vibe_score'],
        'keywords': keywords_str
    })

# llm_chain.batch()를 사용하여 모든 입력에 대한 결과를 한 번에 효율적으로 요청합니다.
# tqdm으로 래핑하여 진행 상황을 시각적으로 보여줍니다.
results = llm_chain.batch(inputs, config={"max_concurrency": 10}) # max_concurrency로 동시 요청 수 조절

# 결과(딕셔너리 리스트)에서 'text' 키의 값만 추출하여 'description' 컬럼에 추가합니다.
df['description'] = [result['text'].strip() for result in tqdm(results, desc="설명 문장 정리 중")]
print("✔️ 모든 트랙에 대한 설명 문장 생성을 완료했습니다.")

# --- 5. 설명 문장 임베딩 및 최종 데이터 저장 ---
print("📦 고성능 임베딩 모델(jhgan/ko-sroberta-multitask)을 로드합니다...")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

print("🚀 생성된 설명 문장을 벡터로 변환(임베딩)합니다...")
song_embeddings = model.encode(df['description'].tolist(), show_progress_bar=True)

# 데이터프레임과 임베딩 벡터를 함께 저장할 딕셔너리를 생성합니다.
data_to_save = {
    'dataframe': df,
    'embeddings': song_embeddings
}

# 최종 결과물을 pickle 파일로 저장합니다.
output_filename = 'song_data_embedded.pkl'
with open(output_filename, 'wb') as f:
    pickle.dump(data_to_save, f)

print(f"\n🎉 [Final ver.] 전처리 완료! '{output_filename}' 파일이 성공적으로 생성되었습니다.")
print(f"   이제 'app.py'를 실행하여 추천 서버를 시작하세요.")
