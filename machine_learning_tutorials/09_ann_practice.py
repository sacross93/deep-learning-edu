"""
인공신경망 실습

이 실습에서는 MNIST 데이터셋을 사용하여 다층 퍼셉트론(MLP)을 구현하고,
하이퍼파라미터 튜닝을 통해 성능을 최적화하는 방법을 학습합니다.
"""

# 1. 라이브러리 임포트 및 설정
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_recall_fscore_support)
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("인공신경망(ANN) 실습 - MNIST 손글씨 분류")
print("=" * 60)

# 2. 데이터셋 로딩 및 탐색
print("\n1. MNIST 데이터셋 로딩 및 탐색")
print("-" * 40)

# MNIST 데이터 로딩 (간소화된 버전)
try:
    # 전체 MNIST 대신 작은 샘플 사용 (실습 목적)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # 데이터 크기 축소 (빠른 실습을 위해)
    n_samples = 10000
    indices = np.random.choice(len(X), n_samples, replace=False)
    X = X[indices]
    y = y[indices]
    
    print(f"데이터 형태: {X.shape}")
    print(f"레이블 형태: {y.shape}")
    print(f"픽셀 값 범위: {X.min():.1f} ~ {X.max():.1f}")
    print(f"클래스 수: {len(np.unique(y))}")
    print(f"클래스 분포: {np.bincount(y)}")
    
except Exception as e:
    print(f"MNIST 로딩 실패: {e}")
    print("대신 합성 데이터를 생성합니다...")
    
    # 합성 데이터 생성
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=5000, n_features=784, n_informative=100,
                             n_redundant=50, n_classes=10, n_clusters_per_class=1,
                             random_state=42)
    X = (X - X.min()) / (X.max() - X.min()) * 255  # 0-255 범위로 정규화
    
    print(f"합성 데이터 형태: {X.shape}")
    print(f"레이블 형태: {y.shape}")

# 데이터 시각화
def visualize_samples(X, y, n_samples=10):
    """샘플 이미지 시각화"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('MNIST 샘플 이미지', fontsize=16)
    
    for i in range(n_samples):
        row, col = i // 5, i % 5
        if X.shape[1] == 784:  # 28x28 이미지
            image = X[i].reshape(28, 28)
        else:  # 합성 데이터의 경우 임의 형태
            image = X[i].reshape(int(np.sqrt(X.shape[1])), -1)
        
        axes[row, col].imshow(image, cmap='gray')
        axes[row, col].set_title(f'Label: {y[i]}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

if X.shape[1] == 784:  # 실제 MNIST 데이터인 경우만 시각화
    visualize_samples(X, y)

# 3. 데이터 전처리
print("\n2. 데이터 전처리")
print("-" * 40)

# 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 훈련/검증 분할
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"훈련 데이터: {X_train.shape}")
print(f"검증 데이터: {X_val.shape}")
print(f"테스트 데이터: {X_test.shape}")

# 데이터 정규화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"정규화 후 훈련 데이터 범위: {X_train_scaled.min():.3f} ~ {X_train_scaled.max():.3f}")
print(f"정규화 후 평균: {X_train_scaled.mean():.3f}, 표준편차: {X_train_scaled.std():.3f}")

# 4. 기본 다층 퍼셉트론 구현
print("\n3. 기본 다층 퍼셉트론 모델 구현")
print("-" * 40)

# 기본 MLP 모델 생성
mlp_basic = MLPClassifier(
    hidden_layer_sizes=(100,),  # 은닉층 1개, 100개 뉴런
    activation='relu',          # ReLU 활성화 함수
    solver='adam',              # Adam 최적화
    learning_rate_init=0.001,   # 초기 학습률
    max_iter=200,              # 최대 반복 횟수
    random_state=42,
    early_stopping=True,        # 조기 종료
    validation_fraction=0.1,    # 검증 데이터 비율
    n_iter_no_change=10        # 성능 개선 없을 때 대기 횟수
)

# 모델 훈련
print("기본 MLP 모델 훈련 중...")
mlp_basic.fit(X_train_scaled, y_train)

# 예측 및 평가
y_pred_basic = mlp_basic.predict(X_test_scaled)
accuracy_basic = accuracy_score(y_test, y_pred_basic)

print(f"기본 MLP 정확도: {accuracy_basic:.4f}")
print(f"훈련 반복 횟수: {mlp_basic.n_iter_}")
print(f"최종 손실값: {mlp_basic.loss_:.6f}")

# 5. 하이퍼파라미터 튜닝 실험
print("\n4. 하이퍼파라미터 튜닝 실험")
print("-" * 40)

# 5.1 은닉층 크기 실험
print("\n4.1 은닉층 크기별 성능 비교")
hidden_sizes = [(50,), (100,), (200,), (100, 50), (200, 100), (300, 150, 75)]
size_results = []

for size in hidden_sizes:
    mlp = MLPClassifier(
        hidden_layer_sizes=size,
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp.fit(X_train_scaled, y_train)
    val_score = mlp.score(X_val_scaled, y_val)
    size_results.append((str(size), val_score, mlp.n_iter_))
    print(f"은닉층 {size}: 검증 정확도 = {val_score:.4f}, 반복 횟수 = {mlp.n_iter_}")

# 5.2 활성화 함수 비교
print("\n4.2 활성화 함수별 성능 비교")
activations = ['relu', 'tanh', 'logistic']
activation_results = []

for activation in activations:
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation=activation,
        solver='adam',
        learning_rate_init=0.001,
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp.fit(X_train_scaled, y_train)
    val_score = mlp.score(X_val_scaled, y_val)
    activation_results.append((activation, val_score, mlp.n_iter_))
    print(f"활성화 함수 {activation}: 검증 정확도 = {val_score:.4f}")

# 5.3 학습률 실험
print("\n4.3 학습률별 성능 비교")
learning_rates = [0.0001, 0.001, 0.01, 0.1]
lr_results = []

for lr in learning_rates:
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        learning_rate_init=lr,
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp.fit(X_train_scaled, y_train)
    val_score = mlp.score(X_val_scaled, y_val)
    lr_results.append((lr, val_score, mlp.n_iter_))
    print(f"학습률 {lr}: 검증 정확도 = {val_score:.4f}")

# 6. 최적 모델 선택 및 평가
print("\n5. 최적 모델 선택 및 종합 평가")
print("-" * 40)

# 그리드 서치를 통한 최적 하이퍼파라미터 탐색
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (200, 100)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01],
    'alpha': [0.0001, 0.001]  # L2 정규화 강도
}

print("그리드 서치를 통한 최적 하이퍼파라미터 탐색 중...")
mlp_grid = MLPClassifier(
    solver='adam',
    max_iter=100,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1
)

grid_search = GridSearchCV(
    mlp_grid, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0
)

grid_search.fit(X_train_scaled, y_train)

print(f"최적 하이퍼파라미터: {grid_search.best_params_}")
print(f"최적 교차 검증 점수: {grid_search.best_score_:.4f}")

# 최적 모델로 최종 평가
best_mlp = grid_search.best_estimator_
y_pred_best = best_mlp.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)

print(f"최적 모델 테스트 정확도: {accuracy_best:.4f}")

# 7. 상세 성능 분석
print("\n6. 상세 성능 분석")
print("-" * 40)

# 분류 보고서
print("분류 보고서:")
print(classification_report(y_test, y_pred_best))

# 혼동 행렬 시각화
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """혼동 행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    return cm

cm = plot_confusion_matrix(y_test, y_pred_best, "최적 MLP 모델 혼동 행렬")

# 클래스별 성능 분석
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_best)

print("\n클래스별 성능:")
for i in range(len(np.unique(y_test))):
    print(f"클래스 {i}: 정밀도={precision[i]:.3f}, 재현율={recall[i]:.3f}, "
          f"F1={f1[i]:.3f}, 샘플수={support[i]}")

# 8. 학습 곡선 분석
print("\n7. 학습 곡선 분석")
print("-" * 40)

def plot_learning_curves():
    """학습 곡선 시각화"""
    # 다양한 훈련 데이터 크기에 대한 성능 측정
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    val_scores = []
    
    for size in train_sizes:
        n_samples = int(len(X_train_scaled) * size)
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=100,
            random_state=42
        )
        
        # 부분 데이터로 훈련
        X_partial = X_train_scaled[:n_samples]
        y_partial = y_train[:n_samples]
        
        mlp.fit(X_partial, y_partial)
        
        train_score = mlp.score(X_partial, y_partial)
        val_score = mlp.score(X_val_scaled, y_val)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    # 학습 곡선 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label='Training Score', color='blue')
    plt.plot(train_sizes, val_scores, 'o-', label='Validation Score', color='red')
    plt.xlabel('Training Set Size Ratio')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves - MLP')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return train_sizes, train_scores, val_scores

train_sizes, train_scores, val_scores = plot_learning_curves()

# 9. 과적합 분석 및 정규화 효과
print("\n8. 과적합 분석 및 정규화 효과")
print("-" * 40)

# 정규화 강도별 성능 비교
alphas = [0.0001, 0.001, 0.01, 0.1, 1.0]
regularization_results = []

for alpha in alphas:
    mlp = MLPClassifier(
        hidden_layer_sizes=(200, 100, 50),  # 복잡한 모델
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        alpha=alpha,
        max_iter=100,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    mlp.fit(X_train_scaled, y_train)
    
    train_score = mlp.score(X_train_scaled, y_train)
    val_score = mlp.score(X_val_scaled, y_val)
    
    regularization_results.append((alpha, train_score, val_score))
    print(f"정규화 강도 {alpha}: 훈련 정확도={train_score:.4f}, 검증 정확도={val_score:.4f}")

# 정규화 효과 시각화
def plot_regularization_effect():
    """정규화 효과 시각화"""
    alphas_list = [result[0] for result in regularization_results]
    train_scores_reg = [result[1] for result in regularization_results]
    val_scores_reg = [result[2] for result in regularization_results]
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas_list, train_scores_reg, 'o-', label='Training Score', color='blue')
    plt.semilogx(alphas_list, val_scores_reg, 'o-', label='Validation Score', color='red')
    plt.xlabel('Regularization Strength (alpha)')
    plt.ylabel('Accuracy')
    plt.title('Regularization Effect on MLP Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

plot_regularization_effect()

# 10. 모델 해석 및 특성 중요도
print("\n9. 모델 해석 및 가중치 분석")
print("-" * 40)

def analyze_weights(model, layer_idx=0):
    """가중치 분석 및 시각화"""
    weights = model.coefs_[layer_idx]
    
    print(f"층 {layer_idx+1} 가중치 통계:")
    print(f"  형태: {weights.shape}")
    print(f"  평균: {weights.mean():.6f}")
    print(f"  표준편차: {weights.std():.6f}")
    print(f"  최솟값: {weights.min():.6f}")
    print(f"  최댓값: {weights.max():.6f}")
    
    # 가중치 분포 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(weights.flatten(), bins=50, alpha=0.7, color='skyblue')
    plt.title(f'Layer {layer_idx+1} Weight Distribution')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    if weights.shape[0] == 784:  # 입력층에서 첫 번째 은닉층으로의 가중치
        # 첫 번째 뉴런의 가중치를 이미지로 시각화
        first_neuron_weights = weights[:, 0].reshape(28, 28)
        plt.imshow(first_neuron_weights, cmap='RdBu', aspect='auto')
        plt.title('First Hidden Neuron Weights (as Image)')
        plt.colorbar()
    else:
        plt.imshow(weights, cmap='RdBu', aspect='auto')
        plt.title(f'Layer {layer_idx+1} Weights Heatmap')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()

# 최적 모델의 가중치 분석
if hasattr(best_mlp, 'coefs_') and X.shape[1] == 784:
    analyze_weights(best_mlp, 0)

# 11. 성능 최적화 팁 및 결론
print("\n10. 성능 최적화 팁 및 결론")
print("-" * 40)

print("인공신경망 성능 최적화 팁:")
print("1. 데이터 전처리:")
print("   - 입력 데이터 정규화/표준화 필수")
print("   - 적절한 훈련/검증/테스트 분할")
print("   - 클래스 불균형 문제 해결")

print("\n2. 네트워크 구조 설계:")
print("   - 문제 복잡도에 맞는 은닉층 수와 크기 선택")
print("   - 과적합 방지를 위한 적절한 복잡도 유지")
print("   - 활성화 함수 선택 (ReLU 계열 추천)")

print("\n3. 훈련 최적화:")
print("   - Adam 옵티마이저 사용 권장")
print("   - 적절한 학습률 설정 (0.001 ~ 0.01)")
print("   - 조기 종료로 과적합 방지")
print("   - 배치 크기 조정 (32, 64, 128 등)")

print("\n4. 정규화 기법:")
print("   - L2 정규화 (alpha 매개변수)")
print("   - 드롭아웃 (sklearn에서는 자동 적용 안됨)")
print("   - 배치 정규화 (고급 프레임워크에서 사용)")

print("\n5. 하이퍼파라미터 튜닝:")
print("   - 그리드 서치 또는 랜덤 서치 활용")
print("   - 교차 검증으로 일반화 성능 평가")
print("   - 베이지안 최적화 고려")

# 최종 결과 요약
print(f"\n최종 결과 요약:")
print(f"- 기본 MLP 정확도: {accuracy_basic:.4f}")
print(f"- 최적화된 MLP 정확도: {accuracy_best:.4f}")
print(f"- 성능 향상: {((accuracy_best - accuracy_basic) / accuracy_basic * 100):.2f}%")

print("\n실습 완료!")
print("다음 단계: 딥러닝 프레임워크(TensorFlow, PyTorch)를 사용한 고급 신경망 구현")