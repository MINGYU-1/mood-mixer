# app.py
# 역할: 웹 서버를 실행하고, 사용자의 요청을 받아 추천 결과를 반환합니다.
# '/recommend' 주소로 POST 요청이 오면, 사용자가 선택한 무드(에너지, 분위기, 키워드)를
# 자연어 쿼리로 변환하고, 미리 계산된 노래 벡터들과 비교하여 가장 유사한 노래를 찾습니다.

import os
from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 현재 파일의 절대 경로를 기준으로 프로젝트의 기본 경로(BASE_DIR)를 설정합니다.
# 이렇게 하면 어떤 위치에서 서버를 실행해도 파일 경로가 꼬이지 않습니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')

# --- 1. 모델 및 데이터 로드 ---
# 서버가 시작될 때 미리 전처리된 데이터와 언어 모델을 메모리에 올려둡니다.
# 이렇게 해야 매 요청마다 모델을 로드하는 비효율을 막을 수 있습니다.
print("📦 모델과 노래 데이터를 로드합니다...")
model = SentenceTransformer('jhgan/ko-sroberta-multitask')

try:
    with open('song_embeddings.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    df = saved_data['dataframe']
    song_embeddings = saved_data['embeddings']
    print("✅ 로드 완료. 서버가 준비되었습니다.")
except FileNotFoundError:
    print("🚨 'song_embeddings.pkl' 파일이 없습니다. 먼저 'python preprocess.py'를 실행하세요.")
    exit()

# --- 2. 라우팅 (경로 설정) ---

# 기본 경로('/')로 접속하면 index.html 파일을 보여줍니다.
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

# 추천 기능을 수행하는 API 엔드포인트입니다.
@app.route('/recommend', methods=['POST'])
def recommend():
    # 1. 프론트엔드(script.js)에서 보낸 JSON 데이터를 받습니다.
    data = request.json
    energy = int(data['energy'])
    vibe = int(data['vibe'])
    keywords = data['keywords']

    # 2. 받은 데이터를 자연어 쿼리(문장)로 만듭니다.
    # 이 쿼리는 preprocess.py에서 노래 정보를 문장으로 만든 형식과 동일해야 합니다.
    energy_desc = f"에너지가 {'폭발적인' if energy > 66 else '보통인' if energy > 33 else '차분한'}"
    vibe_desc = f"분위기가 {'진중한' if vibe > 66 else '중간 정도인' if vibe > 33 else '가벼운'}"
    
    if keywords:
        keywords_str = ', '.join([kw.replace('#', '') for kw in keywords])
        query_text = f"나는 {energy_desc} 노래를 원해. 분위기는 {vibe_desc} 편이 좋겠어. 특히 {keywords_str} 느낌이 나는 노래면 좋겠어."
    else:
        query_text = f"나는 {energy_desc} 노래를 원해. 분위기는 {vibe_desc} 편이 좋겠어."

    # 3. 생성된 쿼리 문장을 벡터로 변환하고, 모든 노래 벡터와 코사인 유사도를 계산합니다.
    query_embedding = model.encode([query_text])
    similarities = cosine_similarity(query_embedding, song_embeddings)
    
    # 4. 가장 유사도 점수가 높은 노래의 인덱스를 찾습니다.
    best_match_index = np.argmax(similarities)
    recommended_song = df.iloc[best_match_index]
    
    # 5. 찾은 노래의 정보를 JSON 형태로 프론트엔드에 반환합니다.
    return jsonify({
        'artist': recommended_song['artist'],
        'track_title': recommended_song['track_title'],
        'album': recommended_song['album'],
        'album_art_url': recommended_song['album_art_url']
    })

# --- 3. 서버 실행 ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)