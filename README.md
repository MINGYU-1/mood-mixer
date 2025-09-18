좋습니다! 올려주신 파일들(`app.py`, `style.css`, `song_embeddings.pkl`, 앨범 이미지들)을 기반으로 프로젝트용 README.md 초안을 작성해드릴게요.

---

# 한국힙합 누에킁 트랙 흥미 찾기 🎵

사용자의 무드(에너지/분위기)와 키워드(#드라이브, #낭만, #자신감 등)를 기반으로 가장 어울리는 **한국힙합 트랙**을 추천해주는 웹 애플리케이션입니다.
Flask + Sentence-BERT(`jhgan/ko-sroberta-multitask`)를 활용해 자연어 기반 추천 시스템을 구현했습니다.

---

## 📂 프로젝트 구조

```
.
├── app.py                 # Flask 서버 (추천 API 포함)
├── style.css              # 프론트엔드 스타일
├── index.html             # 메인 페이지 (사용자 입력 UI)
├── song_embeddings.pkl    # 전처리된 곡 데이터 + 임베딩
├── images/                # 앨범 아트 이미지 폴더
│   ├── 누명.jpg
│   ├── 에넥도트.jpg
│   └── ... 
```

---

## 🚀 실행 방법

### 1. 환경 세팅

```bash
# 가상환경 생성 및 실행 (옵션)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# 필수 라이브러리 설치
pip install -r requirements.txt
```

※ 주요 라이브러리

* Flask
* sentence-transformers
* scikit-learn
* numpy, pickle

### 2. 데이터 전처리

```bash
python preprocess.py
```

`song_embeddings.pkl` 파일이 생성됩니다. (곡 데이터 + SBERT 임베딩 포함)

### 3. 서버 실행

```bash
python app.py
```

브라우저에서 `http://127.0.0.1:5001` 접속 후 서비스 이용 가능.

---

## 🎛 기능 소개

* **무드 믹서 (Mood Mixer)**

  * 슬라이더로 `에너지`, `분위기` 선택
  * 키워드 버튼으로 #드라이브, #낭만, #자신감 등 선택

* **추천 시스템**

  * 입력값을 기반으로 자연어 쿼리 생성
  * SBERT 임베딩 + 코사인 유사도 계산
  * 가장 유사한 곡을 추천

* **결과 출력**

  * 앨범 아트
  * 곡 제목, 아티스트, 앨범 정보 표시

---

## 🎨 UI 미리보기

* 다크 모드 기반 세련된 디자인 (`style.css`)
* 앨범 이미지 Hover 시 확대 효과
* 추천 결과는 카드 형태로 표시

---

## 📌 향후 개선 아이디어

* 멀티곡 추천 (Top-N)
* Spotify / YouTube API 연동 → 바로 듣기
* 개인화 학습 (사용자 피드백 반영)
* 다양한 장르 확장 (힙합 외 K-Pop, R\&B 등)


