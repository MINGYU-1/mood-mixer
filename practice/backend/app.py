from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from chatbot import chatbot_instance

# .env 파일로부터 환경 변수를 로드합니다.
# 이 코드는 다른 모듈이 임포트되기 전에 실행되는 것이 좋습니다.
load_dotenv()

app = Flask(__name__)
# 모든 도메인에서의 요청을 허용 (개발용)
CORS(app) 

@app.route('/')
def home():
    return "LangChain Lyric Analyzer Backend is running."

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'Question is required'}), 400
    
    question = data['question']
    print(f"Received question: {question}")
    
    # 챗봇 로직을 통해 답변 생성
    response = chatbot_instance.get_response(question)
    
    return jsonify({'answer': response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
