# 딥러닝 강의 튜토리얼 시리즈

PyTorch를 사용한 포괄적인 딥러닝 교육 프로그램입니다. 기초부터 최신 고급 기법까지 단계별로 학습할 수 있도록 구성되었습니다.

## 📚 튜토리얼 목록

### 1. PyTorch 기초 (01_pytorch_basics.py)
- **데이터셋**: MNIST 손글씨 숫자
- **학습 목표**: PyTorch 기본 개념, 텐서 연산, 자동 미분
- **핵심 개념**: 
  - 텐서(Tensor) 기본 연산
  - 자동 미분(Autograd) 시스템
  - 기본 신경망 구조
  - 손실 함수와 옵티마이저

### 2. 신경망 기초 (02_neural_networks.py)
- **데이터셋**: Fashion-MNIST
- **학습 목표**: 다층 퍼셉트론, 활성화 함수, 정규화
- **핵심 개념**:
  - 다층 퍼셉트론(MLP) 구조
  - 활성화 함수 (ReLU, Sigmoid, Tanh)
  - 배치 정규화(Batch Normalization)
  - 드롭아웃(Dropout) 정규화
  - 하이퍼파라미터 튜닝

### 3. CNN 이미지 분류 (03_cnn_image_classification.py)
- **데이터셋**: CIFAR-10
- **학습 목표**: 합성곱 신경망, 특성 맵, 풀링
- **핵심 개념**:
  - 합성곱(Convolution) 연산
  - 풀링(Pooling) 레이어
  - 특성 맵(Feature Map) 시각화
  - 데이터 증강(Data Augmentation)
  - 전이 학습(Transfer Learning)

### 4. RNN 텍스트 분류 (04_rnn_text_classification.py)
- **데이터셋**: IMDB 영화 리뷰
- **학습 목표**: 순환 신경망, 시퀀스 처리, 임베딩
- **핵심 개념**:
  - 순환 신경망(RNN) 구조
  - 텍스트 전처리 및 토큰화
  - 단어 임베딩(Word Embedding)
  - 시퀀스 분류
  - 그래디언트 소실 문제

### 5. LSTM 시계열 예측 (05_lstm_sequence_prediction.py)
- **데이터셋**: 주식 가격 데이터
- **학습 목표**: LSTM, 시계열 분석, 예측
- **핵심 개념**:
  - LSTM(Long Short-Term Memory) 구조
  - 시계열 데이터 전처리
  - 윈도우 슬라이딩 기법
  - 시계열 예측 평가 메트릭
  - 예측 결과 시각화

### 6. YOLO 객체 탐지 (06_yolo_object_detection.py)
- **데이터셋**: COCO (교육용 샘플 데이터)
- **학습 목표**: 객체 탐지, 바운딩 박스, 실시간 처리
- **핵심 개념**:
  - YOLO(You Only Look Once) 알고리즘
  - 바운딩 박스 예측 및 처리
  - Non-Maximum Suppression (NMS)
  - IoU(Intersection over Union) 계산
  - mAP(mean Average Precision) 평가

### 7. GAN 이미지 생성 (07_gan_image_generation.py)
- **데이터셋**: CelebA (교육용 샘플 데이터)
- **학습 목표**: 생성적 적대 신경망, 이미지 합성
- **핵심 개념**:
  - GAN(Generative Adversarial Networks) 구조
  - 생성자(Generator)와 판별자(Discriminator)
  - 적대적 학습(Adversarial Training)
  - 모드 붕괴(Mode Collapse) 문제
  - 잠재 공간(Latent Space) 탐색

### 8. Transformer NLP (08_transformer_nlp.py)
- **데이터셋**: Multi30k 번역 (교육용 샘플 데이터)
- **학습 목표**: 어텐션 메커니즘, 기계 번역
- **핵심 개념**:
  - Transformer 아키텍처
  - Self-Attention과 Multi-Head Attention
  - 위치 인코딩(Positional Encoding)
  - 인코더-디코더 구조
  - BLEU 점수 평가

## 🛠️ 환경 설정

### 필요 라이브러리
```bash
pip install -r requirements.txt
```

### 주요 의존성
- Python 3.8+
- PyTorch 1.12+
- torchvision
- matplotlib
- numpy
- pandas
- scikit-learn
- tqdm
- opencv-python
- seaborn

## 🚀 실행 방법

각 튜토리얼은 독립적으로 실행 가능합니다:

```bash
# PyTorch 기초
python 01_pytorch_basics.py

# 신경망 기초
python 02_neural_networks.py

# CNN 이미지 분류
python 03_cnn_image_classification.py

# RNN 텍스트 분류
python 04_rnn_text_classification.py

# LSTM 시계열 예측
python 05_lstm_sequence_prediction.py

# YOLO 객체 탐지
python 06_yolo_object_detection.py

# GAN 이미지 생성
python 07_gan_image_generation.py

# Transformer NLP
python 08_transformer_nlp.py
```

## 📁 프로젝트 구조

```
deep_learning_tutorials/
├── 01_pytorch_basics.py          # PyTorch 기초
├── 02_neural_networks.py         # 신경망 기초
├── 03_cnn_image_classification.py # CNN 이미지 분류
├── 04_rnn_text_classification.py  # RNN 텍스트 분류
├── 05_lstm_sequence_prediction.py # LSTM 시계열 예측
├── 06_yolo_object_detection.py    # YOLO 객체 탐지
├── 07_gan_image_generation.py     # GAN 이미지 생성
├── 08_transformer_nlp.py          # Transformer NLP
├── utils/                         # 공통 유틸리티
│   ├── data_utils.py             # 데이터 처리 함수
│   ├── model_utils.py            # 모델 관련 함수
│   └── visualization.py          # 시각화 함수
├── requirements.txt               # 의존성 목록
└── README.md                     # 프로젝트 설명
```

## 🎯 학습 순서

### 기초 과정 (필수)
1. **01_pytorch_basics.py**: PyTorch 기본 개념 익히기
2. **02_neural_networks.py**: 신경망 구조와 훈련 과정 이해

### 중급 과정 (권장)
3. **03_cnn_image_classification.py**: 이미지 처리를 위한 CNN 학습
4. **04_rnn_text_classification.py**: 텍스트 처리를 위한 RNN 학습
5. **05_lstm_sequence_prediction.py**: 시계열 데이터 처리를 위한 LSTM 학습

### 고급 과정 (선택)
6. **06_yolo_object_detection.py**: 실시간 객체 탐지 시스템
7. **07_gan_image_generation.py**: 생성적 AI와 이미지 합성
8. **08_transformer_nlp.py**: 현대 NLP의 핵심 기술

## 💡 주요 특징

- **실습 중심**: 이론과 실습이 균형잡힌 구성
- **단계별 학습**: 기초부터 최신 기법까지 체계적 진행
- **상세한 주석**: 코드의 모든 부분에 한국어 설명
- **시각화**: 학습 과정과 결과를 그래프로 확인
- **실용적 예제**: 실제 문제에 적용 가능한 예제
- **최신 기법**: YOLO, GAN, Transformer 등 최신 딥러닝 기법 포함

## 🔧 문제 해결

### 일반적인 문제들

1. **CUDA 메모리 부족**
   - 배치 크기를 줄여보세요
   - `torch.cuda.empty_cache()` 사용
   - 모델 크기나 이미지 해상도 조정

2. **느린 훈련 속도**
   - GPU 사용 확인: `torch.cuda.is_available()`
   - 데이터 로더의 `num_workers` 조정
   - 혼합 정밀도(Mixed Precision) 사용 고려

3. **패키지 설치 오류**
   - 가상환경 사용 권장
   - PyTorch 공식 사이트에서 설치 명령어 확인
   - CUDA 버전과 PyTorch 버전 호환성 확인

4. **수렴하지 않는 모델**
   - 학습률 조정 (보통 더 낮게)
   - 배치 크기 조정
   - 정규화 기법 적용
   - 그래디언트 클리핑 사용

### 성능 최적화 팁

1. **메모리 최적화**
   - `torch.no_grad()` 사용 (추론 시)
   - 불필요한 그래디언트 계산 방지
   - 체크포인팅 활용

2. **훈련 속도 향상**
   - 데이터 로딩 병렬화
   - GPU 활용률 모니터링
   - 효율적인 데이터 전처리

## 📊 각 튜토리얼별 예상 실행 시간

| 튜토리얼 | CPU | GPU | 메모리 사용량 |
|---------|-----|-----|-------------|
| 01_pytorch_basics | 5-10분 | 2-5분 | ~2GB |
| 02_neural_networks | 10-15분 | 3-7분 | ~3GB |
| 03_cnn_image_classification | 20-30분 | 5-10분 | ~4GB |
| 04_rnn_text_classification | 15-25분 | 5-10분 | ~3GB |
| 05_lstm_sequence_prediction | 10-20분 | 3-8분 | ~2GB |
| 06_yolo_object_detection | 30-60분 | 10-20분 | ~6GB |
| 07_gan_image_generation | 60-120분 | 15-30분 | ~8GB |
| 08_transformer_nlp | 40-80분 | 10-25분 | ~6GB |

*실행 시간은 하드웨어 사양에 따라 달라질 수 있습니다.

## 📖 추가 학습 자료

### 공식 문서
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [PyTorch 공식 문서](https://pytorch.org/docs/)

### 온라인 강의
- [Deep Learning Specialization (Coursera)](https://www.coursera.org/specializations/deep-learning)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [CS224n: Natural Language Processing](http://cs224n.stanford.edu/)

### 도서
- Deep Learning with PyTorch
- Hands-On Machine Learning
- Deep Learning (Ian Goodfellow)

### 논문 및 최신 연구
- [Papers With Code](https://paperswithcode.com/)
- [arXiv.org](https://arxiv.org/)
- [Google AI Blog](https://ai.googleblog.com/)

## 🎓 학습 완료 후 다음 단계

1. **실제 프로젝트 적용**
   - Kaggle 경진대회 참여
   - 개인 프로젝트 개발
   - 오픈소스 기여

2. **전문 분야 심화**
   - Computer Vision: Object Detection, Segmentation
   - NLP: Language Models, Question Answering
   - Generative AI: Diffusion Models, Large Language Models

3. **MLOps 및 배포**
   - 모델 서빙 (FastAPI, Flask)
   - 컨테이너화 (Docker)
   - 클라우드 배포 (AWS, GCP, Azure)

## 🤝 기여하기

버그 리포트나 개선 제안은 언제든 환영합니다!

### 기여 방법
1. 이슈 등록
2. 풀 리퀘스트 제출
3. 문서 개선
4. 새로운 튜토리얼 제안

## 📄 라이선스

MIT License

---

**Happy Learning! 🚀**

이 튜토리얼 시리즈가 여러분의 딥러닝 여정에 도움이 되기를 바랍니다. 질문이나 피드백이 있으시면 언제든 연락해 주세요!