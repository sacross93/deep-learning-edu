"""
SVM (Support Vector Machine) 실습

이 실습에서는 Breast Cancer 데이터셋을 사용하여 SVM의 다양한 측면을 학습합니다:
1. 선형 SVM과 비선형 SVM 비교
2. 다양한 커널 함수 성능 비교
3. 하이퍼파라미터 튜닝
4. 결정 경계 시각화
5. 성능 평가 및 해석
"""

# 1. 라이브러리 임포트 및 설정
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, validation_curve
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("SVM (Support Vector Machine) 실습")
print("=" * 60)

# 2. 데이터셋 로딩 및 탐색
print("\n2. 데이터셋 로딩 및 탐색")
print("-" * 40)

# Breast Cancer 데이터셋 로드
data = load_breast_cancer()
X, y = data.data, data.target

print(f"데이터셋 크기: {X.shape}")
print(f"특성 수: {X.shape[1]}")
print(f"클래스 수: {len(np.unique(y))}")
print(f"클래스 분포:")
print(f"  - 악성 (0): {np.sum(y == 0)}개")
print(f"  - 양성 (1): {np.sum(y == 1)}개")

# 특성 정보 출력
print(f"\n특성 이름 (처음 10개):")
for i, feature in enumerate(data.feature_names[:10]):
    print(f"  {i+1}. {feature}")

# 기본 통계 정보
df = pd.DataFrame(X, columns=data.feature_names)
print(f"\n기본 통계 정보:")
print(df.describe().iloc[:, :5])  # 처음 5개 특성만 표시

# 3. 데이터 전처리
print("\n3. 데이터 전처리")
print("-" * 40)

# 훈련/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"훈련 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")

# 특성 스케일링 (SVM에서 매우 중요!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("특성 스케일링 완료")
print(f"스케일링 전 평균: {X_train.mean():.2f}, 표준편차: {X_train.std():.2f}")
print(f"스케일링 후 평균: {X_train_scaled.mean():.2f}, 표준편차: {X_train_scaled.std():.2f}")

# 4. 기본 SVM 모델 구현
print("\n4. 기본 SVM 모델 구현")
print("-" * 40)

# 선형 SVM
linear_svm = SVC(kernel='linear', random_state=42)
linear_svm.fit(X_train_scaled, y_train)

# 예측 및 성능 평가
y_pred_linear = linear_svm.predict(X_test_scaled)
linear_accuracy = linear_svm.score(X_test_scaled, y_test)

print(f"선형 SVM 정확도: {linear_accuracy:.4f}")
print(f"서포트 벡터 수: {linear_svm.n_support_}")
print(f"전체 서포트 벡터 비율: {sum(linear_svm.n_support_) / len(X_train_scaled):.3f}")

# 5. 다양한 커널 함수 비교
print("\n5. 다양한 커널 함수 비교")
print("-" * 40)

# 다양한 커널로 SVM 모델 훈련
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_results = {}

for kernel in kernels:
    print(f"\n{kernel.upper()} 커널:")
    
    # 모델 훈련
    if kernel == 'poly':
        svm = SVC(kernel=kernel, degree=3, random_state=42)
    else:
        svm = SVC(kernel=kernel, random_state=42)
    
    svm.fit(X_train_scaled, y_train)
    
    # 성능 평가
    train_score = svm.score(X_train_scaled, y_train)
    test_score = svm.score(X_test_scaled, y_test)
    
    # 교차 검증
    cv_scores = cross_val_score(svm, X_train_scaled, y_train, cv=5)
    
    kernel_results[kernel] = {
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_support': sum(svm.n_support_)
    }
    
    print(f"  훈련 정확도: {train_score:.4f}")
    print(f"  테스트 정확도: {test_score:.4f}")
    print(f"  CV 평균: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"  서포트 벡터 수: {sum(svm.n_support_)}")

# 커널 비교 시각화
plt.figure(figsize=(12, 8))

# 서브플롯 1: 정확도 비교
plt.subplot(2, 2, 1)
kernels_list = list(kernel_results.keys())
test_accuracies = [kernel_results[k]['test_accuracy'] for k in kernels_list]
cv_means = [kernel_results[k]['cv_mean'] for k in kernels_list]

x = np.arange(len(kernels_list))
width = 0.35

plt.bar(x - width/2, test_accuracies, width, label='Test Accuracy', alpha=0.8)
plt.bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.8)
plt.xlabel('Kernel')
plt.ylabel('Accuracy')
plt.title('Kernel Performance Comparison')
plt.xticks(x, kernels_list)
plt.legend()
plt.grid(True, alpha=0.3)

# 서브플롯 2: 서포트 벡터 수 비교
plt.subplot(2, 2, 2)
n_supports = [kernel_results[k]['n_support'] for k in kernels_list]
plt.bar(kernels_list, n_supports, alpha=0.8, color='orange')
plt.xlabel('Kernel')
plt.ylabel('Number of Support Vectors')
plt.title('Support Vectors by Kernel')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. 하이퍼파라미터 튜닝
print("\n6. 하이퍼파라미터 튜닝")
print("-" * 40)

# RBF 커널에 대한 그리드 서치
print("RBF 커널 하이퍼파라미터 튜닝:")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

grid_search = GridSearchCV(
    SVC(kernel='rbf', random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 CV 점수: {grid_search.best_score_:.4f}")

# 최적 모델로 테스트
best_svm = grid_search.best_estimator_
test_accuracy = best_svm.score(X_test_scaled, y_test)
print(f"테스트 정확도: {test_accuracy:.4f}")

# 7. 파라미터 민감도 분석
print("\n7. 파라미터 민감도 분석")
print("-" * 40)

# C 파라미터 검증 곡선
C_range = np.logspace(-2, 2, 10)
train_scores, val_scores = validation_curve(
    SVC(kernel='rbf', gamma='scale', random_state=42),
    X_train_scaled, y_train,
    param_name='C', param_range=C_range,
    cv=5, scoring='accuracy'
)

# gamma 파라미터 검증 곡선
gamma_range = np.logspace(-4, 0, 10)
train_scores_gamma, val_scores_gamma = validation_curve(
    SVC(kernel='rbf', C=1, random_state=42),
    X_train_scaled, y_train,
    param_name='gamma', param_range=gamma_range,
    cv=5, scoring='accuracy'
)

# 검증 곡선 시각화
plt.figure(figsize=(15, 5))

# C 파라미터 효과
plt.subplot(1, 3, 1)
train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

plt.semilogx(C_range, train_mean, 'o-', label='Training score', alpha=0.8)
plt.fill_between(C_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.semilogx(C_range, val_mean, 'o-', label='Validation score', alpha=0.8)
plt.fill_between(C_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('C')
plt.ylabel('Accuracy')
plt.title('Validation Curve (C parameter)')
plt.legend()
plt.grid(True, alpha=0.3)

# gamma 파라미터 효과
plt.subplot(1, 3, 2)
train_mean_gamma = train_scores_gamma.mean(axis=1)
train_std_gamma = train_scores_gamma.std(axis=1)
val_mean_gamma = val_scores_gamma.mean(axis=1)
val_std_gamma = val_scores_gamma.std(axis=1)

plt.semilogx(gamma_range, train_mean_gamma, 'o-', label='Training score', alpha=0.8)
plt.fill_between(gamma_range, train_mean_gamma - train_std_gamma, 
                train_mean_gamma + train_std_gamma, alpha=0.1)
plt.semilogx(gamma_range, val_mean_gamma, 'o-', label='Validation score', alpha=0.8)
plt.fill_between(gamma_range, val_mean_gamma - val_std_gamma, 
                val_mean_gamma + val_std_gamma, alpha=0.1)
plt.xlabel('gamma')
plt.ylabel('Accuracy')
plt.title('Validation Curve (gamma parameter)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 8. 결정 경계 시각화 (2D PCA)
print("\n8. 결정 경계 시각화")
print("-" * 40)

# PCA로 2차원으로 축소
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA 설명 분산 비율: {pca.explained_variance_ratio_}")
print(f"총 설명 분산: {pca.explained_variance_ratio_.sum():.3f}")

# 2D 데이터로 SVM 훈련
svm_2d = SVC(kernel='rbf', C=10, gamma=0.1, random_state=42)
svm_2d.fit(X_train_pca, y_train)

# 결정 경계 시각화 함수
def plot_decision_boundary(X, y, model, title):
    plt.figure(figsize=(10, 8))
    
    # 격자 생성
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # 예측
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 결정 경계 그리기
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    
    # 데이터 포인트 그리기
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    
    # 서포트 벡터 표시
    if hasattr(model, 'support_vectors_'):
        plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
                   s=100, facecolors='none', edgecolors='red', linewidth=2,
                   label=f'Support Vectors ({len(model.support_vectors_)})')
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)
    plt.colorbar(scatter)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 결정 경계 시각화
plot_decision_boundary(X_train_pca, y_train, svm_2d, 
                      'SVM Decision Boundary (2D PCA projection)')

# 9. 상세 성능 평가
print("\n9. 상세 성능 평가")
print("-" * 40)

# 최적 모델로 예측
y_pred = best_svm.predict(X_test_scaled)
y_pred_proba = best_svm.decision_function(X_test_scaled)

# 분류 리포트
print("분류 리포트:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# 혼동 행렬
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=['Malignant', 'Benign'],
           yticklabels=['Malignant', 'Benign'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ROC 곡선
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# 10. 특성 중요도 분석 (선형 SVM)
print("\n10. 특성 중요도 분석")
print("-" * 40)

# 선형 SVM의 계수 분석
linear_svm_best = SVC(kernel='linear', C=grid_search.best_params_.get('C', 1), random_state=42)
linear_svm_best.fit(X_train_scaled, y_train)

# 특성 중요도 (가중치의 절댓값)
feature_importance = np.abs(linear_svm_best.coef_[0])
feature_names = data.feature_names

# 상위 10개 중요 특성
top_indices = np.argsort(feature_importance)[-10:]
top_features = [feature_names[i] for i in top_indices]
top_importance = feature_importance[top_indices]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features)), top_importance, alpha=0.8)
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('Feature Importance (|coefficient|)')
plt.title('Top 10 Most Important Features (Linear SVM)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("상위 10개 중요 특성:")
for i, (feature, importance) in enumerate(zip(top_features, top_importance)):
    print(f"{i+1:2d}. {feature:<30} {importance:.4f}")

# 11. 모델 비교 요약
print("\n11. 모델 비교 요약")
print("-" * 40)

# 결과 요약 테이블
results_df = pd.DataFrame(kernel_results).T
results_df = results_df.round(4)
print("커널별 성능 비교:")
print(results_df)

# 최종 권장사항
print(f"\n최종 권장사항:")
print(f"- 최적 커널: RBF")
print(f"- 최적 파라미터: C={grid_search.best_params_['C']}, gamma={grid_search.best_params_['gamma']}")
print(f"- 최종 테스트 정확도: {test_accuracy:.4f}")
print(f"- ROC AUC: {roc_auc:.4f}")

# 12. 실제 적용 시 고려사항
print("\n12. 실제 적용 시 고려사항")
print("-" * 40)

print("SVM 사용 시 주의사항:")
print("1. 특성 스케일링이 필수적입니다")
print("2. 하이퍼파라미터 튜닝이 성능에 큰 영향을 미칩니다")
print("3. 대용량 데이터에서는 훈련 시간이 오래 걸릴 수 있습니다")
print("4. 확률 출력이 필요한 경우 decision_function 값을 변환해야 합니다")
print("5. 해석성이 중요한 경우 선형 커널을 고려하세요")

print("\n커널 선택 가이드:")
print("- Linear: 고차원 데이터, 빠른 훈련이 필요한 경우")
print("- RBF: 대부분의 경우에 좋은 성능, 기본 선택")
print("- Polynomial: 중간 정도의 비선형성, 특정 도메인 지식이 있는 경우")
print("- Sigmoid: 신경망과 유사한 효과가 필요한 경우")

print("\n" + "=" * 60)
print("SVM 실습 완료!")
print("=" * 60)