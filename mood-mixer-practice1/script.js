// script.js
document.addEventListener('DOMContentLoaded', () => {
    const findTrackBtn = document.getElementById('find-track-btn');
    const resultSection = document.getElementById('result-section');
    
    // 슬라이더와 값 표시 영역 DOM 요소 가져오기
    const energySlider = document.getElementById('energy-slider');
    const vibeSlider = document.getElementById('vibe-slider');
    const energyValueSpan = document.getElementById('energy-value');
    const vibeValueSpan = document.getElementById('vibe-value');

    // 슬라이더 값을 실시간으로 업데이트하는 함수
    function updateSliderValue(slider, span) {
        span.textContent = slider.value;
    }

    // 페이지 로드 시 초기 값 설정
    updateSliderValue(energySlider, energyValueSpan);
    updateSliderValue(vibeSlider, vibeValueSpan);

    // 슬라이더를 움직일 때마다 값 업데이트
    energySlider.addEventListener('input', () => updateSliderValue(energySlider, energyValueSpan));
    vibeSlider.addEventListener('input', () => updateSliderValue(vibeSlider, vibeValueSpan));

    // 키워드 버튼 클릭 시 'active' 클래스 토글
    document.querySelectorAll('.keyword-btn').forEach(btn => {
        btn.addEventListener('click', () => btn.classList.toggle('active'));
    });

    findTrackBtn.addEventListener('click', () => {
        // 현재 슬라이더 및 키워드 값 가져오기
        const energyValue = energySlider.value;
        const vibeValue = vibeSlider.value;
        const selectedKeywords = Array.from(document.querySelectorAll('.keyword-btn.active'))
                                      .map(btn => btn.textContent);

        // 로딩 애니메이션 표시
        resultSection.innerHTML = `<div class="loader"></div>`;

        // 백엔드에 데이터 전송하여 추천 요청
        fetch('/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                energy: energyValue,
                vibe: vibeValue,
                keywords: selectedKeywords
            }),
        })
        .then(response => response.json())
        .then(data => {
            // 추천 결과를 받아와 카드 형태로 표시
            resultSection.innerHTML = `
                <div class="result-card">
                    <img class="album-art" src="${data.album_art_url}" alt="${data.album}">
                    <div>
                        <p style="color: #a0a0a0; font-weight: 500;">당신의 무드를 위한 추천 트랙</p>
                        <h3 class="track-title">${data.track_title}</h3>
                        <p class="artist-album">${data.artist} - ${data.album}</p>
                    </div>
                </div>
            `;
        })
        .catch(error => {
            console.error('Error:', error);
            resultSection.innerHTML = `<p class="error-text">⚠️ 추천 곡을 찾지 못했습니다. 잠시 후 다시 시도해주세요.</p>`;
        });
    });
});