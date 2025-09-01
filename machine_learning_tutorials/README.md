# 머신러닝 튜토리얼 시리즈

데이터 마이닝 기법부터 머신러닝 알고리즘까지 포괄하는 교육용 튜토리얼 시리즈입니다.

## 📚 개요

이 튜토리얼 시리즈는 다음과 같은 특징을 가집니다:

- **체계적 학습**: 데이터 마이닝 개념부터 고급 머신러닝 알고리즘까지 단계적 학습
- **3단계 구성**: 각 주제마다 이론(MD) + 실습(PY) + 퀴즈(PY) 파일 제공
- **개념 중심**: 수학적 증명보다는 직관적 이해와 메커니즘 파악에 중점
- **실용적 접근**: 실제 데이터를 활용한 실습과 산업계 적용 사례 소개

## 🛠️ 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 설정

```python
# 주피터 노트북에서 사용하는 경우
import sys
sys.path.append('.')

from utils import *
```

## 📖 학습 순서

### 학기 전반부: 데이터 마이닝 기초 및 지도학습

1. **데이터 마이닝 개요** - 정의, 데이터 유형, 속성 분류
2. **데이터 품질 및 전처리** - 잡음, 이상치, 결측치 처리
3. **유사도와 거리** - 다양한 거리 척도, 유사도 측정
4. **다변량 회귀** - 연속형 변수 예측의 기초
5. **로지스틱 회귀** - 범주형 변수 예측의 기초
6. **의사결정나무** - 해석 가능한 분류 모델
7. **규칙 기반 학습** - 명시적 규칙을 통한 분류
8. **앙상블 학습** - 여러 모델의 조합
9. **인공신경망** - 복잡한 패턴 학습
10. **SVM** - 최적 분류 경계 찾기

### 학기 후반부: 비지도학습

11. **주성분 분석 (PCA)** - 차원 축소
12. **K-평균 클러스터링** - 기본 클러스터링
13. **계층적 클러스터링** - 계층 구조 클러스터링
14. **DBSCAN 클러스터링** - 밀도 기반 클러스터링
15. **그래프 기반 클러스터링** - 네트워크 구조 활용
16. **Apriori 알고리즘** - 연관 규칙 마이닝

## 🚀 사용 방법

### 기본 사용법

```python
# 1. 데이터 로딩
from utils.data_utils import load_sample_dataset
df = load_sample_dataset('iris')

# 2. 데이터 전처리
from utils.data_utils import preprocess_data, split_data
df_processed = preprocess_data(df, target_column='target')
X_train, X_test, y_train, y_test = split_data(df_processed, 'target')

# 3. 모델 훈련 및 평가
from sklearn.tree import DecisionTreeClassifier
from utils.evaluation_utils import calculate_classification_metrics

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

metrics = calculate_classification_metrics(y_test, y_pred)
print(metrics)

# 4. 시각화
from utils.visualization_utils import plot_confusion_matrix
plot_confusion_matrix(y_test, y_pred)
```

### 퀴즈 실행

```python
from utils.quiz_utils import run_interactive_quiz

# 특정 주제 퀴즈
results = run_interactive_quiz(topic="데이터 마이닝 개요", num_questions=5)

# 전체 퀴즈
results = run_interactive_quiz(num_questions=10)
```

## 📁 프로젝트 구조

```
machine_learning_tutorials/
├── utils/                          # 공통 유틸리티 모듈
│   ├── __init__.py
│   ├── data_utils.py              # 데이터 로딩, 전처리, 분할
│   ├── evaluation_utils.py        # 성능 평가 및 모델 비교
│   ├── visualization_utils.py     # 시각화 및 결과 분석
│   └── quiz_utils.py             # 퀴즈 생성 및 채점
├── datasets/                       # 샘플 데이터셋
│   └── README.md
├── requirements.txt               # 의존성 목록
└── README.md                     # 프로젝트 설명서
```

## 🔧 주요 기능

### 데이터 유틸리티 (`data_utils.py`)

- **데이터 로딩**: 다양한 샘플 데이터셋 제공
- **전처리**: 결측치 처리, 스케일링, 인코딩
- **탐색**: 데이터 분포 분석 및 시각화
- **품질 관리**: 이상치 탐지 및 처리

### 평가 유틸리티 (`evaluation_utils.py`)

- **분류 평가**: 정확도, 정밀도, 재현율, F1-score, ROC-AUC
- **회귀 평가**: MSE, RMSE, MAE, R², MAPE
- **클러스터링 평가**: 실루엣 점수, 조정된 랜드 지수
- **교차 검증**: 모델 성능의 안정성 평가
- **모델 비교**: 여러 알고리즘 성능 비교

### 시각화 유틸리티 (`visualization_utils.py`)

- **알고리즘 비교**: 성능 지표별 막대 그래프
- **파라미터 분석**: 하이퍼파라미터 민감도 분석
- **클러스터링 결과**: 2D/3D 클러스터 시각화
- **결정 경계**: 분류 모델의 결정 경계 표시
- **차원 축소**: PCA, t-SNE 시각화

### 퀴즈 유틸리티 (`quiz_utils.py`)

- **문제 유형**: 객관식, 참/거짓, 단답형, 수치형
- **자동 채점**: 즉시 피드백 및 해설 제공
- **진도 추적**: 학습 기록 및 성과 분석
- **적응적 학습**: 난이도별, 주제별 문제 선별

## 📊 학습 목표

각 튜토리얼을 완료하면 다음을 할 수 있게 됩니다:

1. **알고리즘 이해**: 동작 원리와 메커니즘 파악
2. **파라미터 조정**: 하이퍼파라미터의 의미와 조정 방법
3. **성능 평가**: 다양한 평가 지표의 해석과 활용
4. **알고리즘 비교**: 유사점, 차이점, 장단점 분석
5. **실무 적용**: 실제 문제에 적합한 알고리즘 선택

## 🎯 평가 기준

- **이해도**: 개념과 메커니즘의 정확한 이해
- **적용력**: 실제 데이터에 알고리즘 적용 능력
- **분석력**: 결과 해석 및 개선 방안 도출
- **비교력**: 알고리즘 간 차이점과 적용 상황 판단

## 🤝 기여 방법

1. 새로운 알고리즘 튜토리얼 추가
2. 기존 튜토리얼 개선 및 오류 수정
3. 퀴즈 문제 추가 및 품질 향상
4. 시각화 기능 확장
5. 문서화 개선

## 📝 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다. 자유롭게 사용하되, 상업적 이용 시에는 별도 문의 바랍니다.

## 📞 문의

- 버그 리포트: GitHub Issues
- 기능 제안: GitHub Discussions
- 일반 문의: 프로젝트 관리자에게 연락

---

**Happy Learning! 🎓**