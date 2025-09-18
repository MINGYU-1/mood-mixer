# preprocess.py
# 역할: CSV 데이터를 읽어와서, 각 노래의 정보를 자연어 문장으로 만들고,
# 이를 언어 모델이 이해할 수 있는 숫자 벡터(임베딩)로 변환하여 파일로 저장합니다.
# 이 스크립트는 데이터가 변경될 때마다 딱 한 번만 실행하면 됩니다.

import pandas as pd
from sentence_transformers import SentenceTransformer
import pickle
import ast

print("✅ [Final ver.] 전처리 스크립트를 시작합니다.")

# --- 1. 데이터 로드 ---
try:
    df = pd.read_csv('누에킁_수정.csv')
except FileNotFoundError:
    print("🚨 '누에킁_수정.csv' 파일이 없습니다. 프로젝트 폴더에 파일을 준비해주세요.")
    exit()

# --- 2. 설명 문장 생성 ---
# 각 노래의 특성(에너지, 분위기, 키워드, 발매연도)을 하나의 문장으로 결합합니다.
# 이 문장의 품질이 추천 시스템의 정확도를 결정합니다.
def create_description(row):
    energy_desc = f"에너지가 {'폭발적인' if row['energy_score'] > 66 else '보통인' if row['energy_score'] > 33 else '차분한'}"
    vibe_desc = f"분위기가 {'진중한' if row['vibe_score'] > 66 else '중간 정도인' if row['vibe_score'] > 33 else '가벼운'}"
    
    try:
        keywords_list = ast.literal_eval(row['keywords'])
        keywords_str = ', '.join([kw.replace('#', '') for kw in keywords_list])
        keywords_desc = f"주로 {keywords_str} 같은 느낌을 다룹니다."
    except:
        keywords_desc = ""

    # 발매 연도를 설명에 추가하여 문맥 정보 강화
    return f"{row['artist']}의 '{row['track_title']}'는 {row['release_year']}년에 발매된 노래로, {energy_desc} 편이며 {vibe_desc} 곡입니다. {keywords_desc}"

df['description'] = df.apply(create_description, axis=1)
print("📄 각 노래에 대한 설명 문장을 생성했습니다.")

# --- 3. 임베딩 모델 로드 ---
# 한국어 문장을 이해하고 벡터로 변환하는 사전 학습된 언어 모델을 로드합니다.
print("📦 임베딩 모델(jhgan/ko-sroberta-multitask)을 로드합니다... (최초 실행 시 시간이 걸릴 수 있습니다)")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

# --- 4. 벡터 변환 (임베딩) ---
# 데이터프레임의 모든 설명 문장을 숫자 벡터의 리스트로 변환합니다.
print("🚀 설명 문장을 벡터로 변환하는 중입니다...")
song_embeddings = model.encode(df['description'].tolist())

# --- 5. 결과 저장 ---
# 나중에 서버에서 빠르게 불러올 수 있도록, 원본 데이터와 변환된 벡터를 .pkl 파일로 저장합니다.
data_to_save = {
    'dataframe': df,
    'embeddings': song_embeddings
}

with open('song_embeddings.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("\n✅ [Final ver.] 전처리 완료! 'song_embeddings.pkl' 파일이 새로 생성되었습니다.")
print("   이제 'python app.py'를 실행하여 최종 서버를 시작하세요.")