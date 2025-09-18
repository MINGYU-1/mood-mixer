# 🎵 K-HIPHOP Track Mixer - Flask App 🎵
# 이 스크립트는 사용자의 무드 입력을 받아 가장 잘 맞는 힙합 트랙을 추천하는 웹 서버입니다.

import os
from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 현재 파일의 절대 경로를 기준으로 BASE_DIR 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Flask 앱 초기화. static 폴더를 BASE_DIR로 설정하여 index.html을 올바르게 서빙합니다.
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')

# --- 1. 모델과 전처리된 데이터 로드 ---
print("📦 모델과 노래 데이터를 로드합니다...")
try:
    # 임베딩 모델 로드
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    
    # 전처리된 노래 데이터와 임베딩 벡터가 담긴 pkl 파일 로드
    with open('song_data_embedded.pkl', 'rb') as f:
        saved_data = pickle.load(f)
    df = saved_data['dataframe']
    song_embeddings = saved_data['embeddings']
    print("✅ 로드 완료. 서버가 준비되었습니다.")
except FileNotFoundError:
    print("🚨 'song_data_embedded.pkl' 파일이 없습니다. 먼저 'python preprocess.py'를 실행하여 파일을 생성해주세요.")
    exit()

# --- 2. 라우팅 설정 ---

# 루트 URL('/') 요청 시 index.html 파일을 반환합니다.
@app.route('/')
def index():
    """메인 페이지를 렌더링합니다."""
    return send_from_directory(BASE_DIR, 'index.html')

# '/recommend' URL에 대한 POST 요청을 처리합니다.
@app.route('/recommend', methods=['POST'])
def recommend():
    """사용자 입력을 바탕으로 노래를 추천하고 JSON 형태로 반환합니다."""
    try:
        # 1. 프론트엔드에서 사용자 입력값(JSON) 받기
        data = request.json
        energy = int(data['energy'])
        vibe = int(data['vibe'])
        keywords = data['keywords']

        # 2. 사용자 입력을 자연어 쿼리로 변환
        # 에너지 값 (1-100)을 10개 구간으로 나누어 설명 생성
        if energy <= 10:
            energy_desc = "에너지가 매우 차분하고 잔잔한"
        elif energy <= 20:
            energy_desc = "에너지가 차분하고 몽환적인"
        elif energy <= 30:
            energy_desc = "에너지가 편안하고 나른한"
        elif energy <= 40:
            energy_desc = "에너지가 부드러운 그루브가 있는"
        elif energy <= 50:
            energy_desc = "에너지가 적당히 리드미컬한"
        elif energy <= 60:
            energy_desc = "에너지가 흥겨운 리듬의"
        elif energy <= 70:
            energy_desc = "에너지가 활기차고 신나는"
        elif energy <= 80:
            energy_desc = "에너지가 강렬하고 파워풀한"
        elif energy <= 90:
            energy_desc = "에너지가 폭발적이고 격렬한"
        else:
            energy_desc = "에너지가 압도적으로 폭발하는"

        # 분위기 값 (1-100)을 10개 구간으로 나누어 설명 생성
        if vibe <= 10:
            vibe_desc = "분위기가 매우 가볍고 경쾌한"
        elif vibe <= 20:
            vibe_desc = "분위기가 산뜻하고 청량한"
        elif vibe <= 30:
            vibe_desc = "분위기가 밝고 긍정적인"
        elif vibe <= 40:
            vibe_desc = "분위기가 감성적이고 부드러운"
        elif vibe <= 50:
            vibe_desc = "분위기가 담담하고 솔직한"
        elif vibe <= 60:
            vibe_desc = "분위기가 진솔하고 성찰적인"
        elif vibe <= 70:
            vibe_desc = "분위기가 진지하고 무게감 있는"
        elif vibe <= 80:
            vibe_desc = "분위기가 깊고 철학적인"
        elif vibe <= 90:
            vibe_desc = "분위기가 어둡고 진중한"
        else:
            vibe_desc = "분위기가 압도적으로 무겁고 진중한"
        
        # 선택된 키워드를 쿼리에 자연스럽게 통합
        if keywords:
            keywords_str = ', '.join([kw.replace('#', '') for kw in keywords])
            query_text = f"나는 {energy_desc} 노래를 원해. 분위기는 {vibe_desc} 편이 좋겠어. 특히 {keywords_str} 느낌이 나는 노래면 좋겠어."
        else:
            query_text = f"나는 {energy_desc} 노래를 원해. 분위기는 {vibe_desc} 편이 좋겠어."
        
        print(f"🔍 생성된 검색 쿼리: {query_text}")

        # 3. 쿼리를 벡터로 변환하고 가장 유사한 노래 찾기
        query_embedding = model.encode([query_text])
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(query_embedding, song_embeddings)
        
        # 가장 높은 유사도를 가진 노래의 인덱스 찾기
        best_match_index = np.argmax(similarities)
        recommended_song = df.iloc[best_match_index]
        
        # 4. 추천 결과를 JSON 형태로 프론트엔드에 반환
        print(f"✨ 추천 트랙: {recommended_song['artist']} - {recommended_song['track_title']}")
        return jsonify({
            'artist': recommended_song['artist'],
            'track_title': recommended_song['track_title'],
            'album': recommended_song['album'],
            'album_art_url': recommended_song['album_art_url']
        })
    except Exception as e:
        print(f"Error during recommendation: {e}")
        return jsonify({"error": "추천 중 오류가 발생했습니다."}), 500

# --- 3. 서버 실행 ---
if __name__ == '__main__':
    # 디버그 모드로 Flask 앱 실행 (개발 중에만 사용)
    app.run(debug=True, port=5001)

