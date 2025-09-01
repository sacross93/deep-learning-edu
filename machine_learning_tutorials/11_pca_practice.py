"""
주성분 분석(PCA) 실습

이 실습에서는 고차원 이미지 데이터에 PCA를 적용하여 차원 축소를 수행하고,
주성분 개수 선택 및 복원 품질을 평가하는 방법을 학습합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('default')

print("=" * 60)
print("주성분 분석(PCA) 실습")
print("=" * 60)

# 1. 데이터셋 로딩 및 탐색
print("\n1. 데이터셋 로딩 및 탐색")
print("-" * 30)

# Olivetti 얼굴 데이터셋 로딩 (40명, 각 10장씩 총 400장의 64x64 얼굴 이미지)
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X, y = faces.data, faces.target

print(f"데이터 형태: {X.shape}")
print(f"타겟 형태: {y.shape}")
print(f"이미지 크기: {int(np.sqrt(X.shape[1]))}x{int(np.sqrt(X.shape[1]))}")
print(f"클래스 수: {len(np.unique(y))}")
print(f"픽셀 값 범위: [{X.min():.3f}, {X.max():.3f}]")

# 원본 이미지 시각화
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
fig.suptitle('Original Face Images', fontsize=14)

for i in range(10):
    row, col = i // 5, i % 5
    axes[row, col].imshow(X[i].reshape(64, 64), cmap='gray')
    axes[row, col].set_title(f'Person {y[i]}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# 2. 데이터 전처리
print("\n2. 데이터 전처리")
print("-" * 30)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"훈련 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")

# 데이터 중심화 (PCA는 내부적으로 수행하지만 명시적으로 보여줌)
X_train_centered = X_train - np.mean(X_train, axis=0)
X_test_centered = X_test - np.mean(X_train, axis=0)  # 훈련 데이터 평균 사용

print(f"중심화 후 훈련 데이터 평균: {np.mean(X_train_centered):.6f}")

# 3. PCA 적용 및 주성분 분석
print("\n3. PCA 적용 및 주성분 분석")
print("-" * 30)

# 전체 주성분으로 PCA 수행
pca_full = PCA()
X_train_pca_full = pca_full.fit_transform(X_train)

# 분산 설명 비율 분석
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print(f"총 주성분 개수: {len(explained_variance_ratio)}")
print(f"첫 10개 주성분의 분산 설명 비율:")
for i in range(10):
    print(f"  PC{i+1}: {explained_variance_ratio[i]:.4f} ({explained_variance_ratio[i]*100:.2f}%)")

# 누적 분산 비율로 필요한 주성분 개수 결정
variance_thresholds = [0.8, 0.9, 0.95, 0.99]
for threshold in variance_thresholds:
    n_components = np.argmax(cumulative_variance_ratio >= threshold) + 1
    print(f"{threshold*100}% 분산 설명을 위한 주성분 개수: {n_components}")

# 4. 주성분 개수별 성능 비교
print("\n4. 주성분 개수별 성능 비교")
print("-" * 30)

# 다양한 주성분 개수로 실험
n_components_list = [10, 20, 50, 100, 150, 200]
results = []

for n_comp in n_components_list:
    # PCA 적용
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # 분류 성능 평가 (KNN 사용)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 재구성 오차 계산
    X_train_reconstructed = pca.inverse_transform(X_train_pca)
    reconstruction_error = np.mean((X_train - X_train_reconstructed) ** 2)
    
    # 분산 설명 비율
    variance_explained = np.sum(pca.explained_variance_ratio_)
    
    results.append({
        'n_components': n_comp,
        'accuracy': accuracy,
        'reconstruction_error': reconstruction_error,
        'variance_explained': variance_explained
    })
    
    print(f"주성분 {n_comp:3d}개: 정확도={accuracy:.4f}, "
          f"재구성오차={reconstruction_error:.6f}, "
          f"분산설명={variance_explained:.4f}")

# 5. 시각화
print("\n5. 결과 시각화")
print("-" * 30)

# 분산 설명 비율 시각화
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 스크리 플롯
axes[0, 0].plot(range(1, 51), explained_variance_ratio[:50], 'bo-', markersize=4)
axes[0, 0].set_xlabel('Principal Component')
axes[0, 0].set_ylabel('Explained Variance Ratio')
axes[0, 0].set_title('Scree Plot (First 50 Components)')
axes[0, 0].grid(True, alpha=0.3)

# 누적 분산 비율
axes[0, 1].plot(range(1, 201), cumulative_variance_ratio[:200], 'ro-', markersize=3)
axes[0, 1].axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
axes[0, 1].axhline(y=0.9, color='orange', linestyle='--', label='90% threshold')
axes[0, 1].set_xlabel('Number of Components')
axes[0, 1].set_ylabel('Cumulative Explained Variance')
axes[0, 1].set_title('Cumulative Explained Variance')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 주성분 개수별 성능 비교
n_comps = [r['n_components'] for r in results]
accuracies = [r['accuracy'] for r in results]
recon_errors = [r['reconstruction_error'] for r in results]

axes[1, 0].plot(n_comps, accuracies, 'go-', markersize=6, linewidth=2)
axes[1, 0].set_xlabel('Number of Components')
axes[1, 0].set_ylabel('Classification Accuracy')
axes[1, 0].set_title('Classification Performance vs Components')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(n_comps, recon_errors, 'mo-', markersize=6, linewidth=2)
axes[1, 1].set_xlabel('Number of Components')
axes[1, 1].set_ylabel('Reconstruction Error')
axes[1, 1].set_title('Reconstruction Error vs Components')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. 주성분 시각화 (Eigenfaces)
print("\n6. 주성분 시각화 (Eigenfaces)")
print("-" * 30)

# 상위 20개 주성분 시각화
pca_20 = PCA(n_components=20)
pca_20.fit(X_train)

fig, axes = plt.subplots(4, 5, figsize=(12, 10))
fig.suptitle('Top 20 Principal Components (Eigenfaces)', fontsize=14)

for i in range(20):
    row, col = i // 5, i % 5
    eigenface = pca_20.components_[i].reshape(64, 64)
    axes[row, col].imshow(eigenface, cmap='gray')
    axes[row, col].set_title(f'PC{i+1}\n({pca_20.explained_variance_ratio_[i]:.3f})')
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# 7. 이미지 재구성 품질 비교
print("\n7. 이미지 재구성 품질 비교")
print("-" * 30)

# 다양한 주성분 개수로 재구성
test_image_idx = 0
original_image = X_test[test_image_idx]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle(f'Image Reconstruction with Different Numbers of Components\n(Person {y_test[test_image_idx]})', fontsize=14)

reconstruction_components = [5, 10, 20, 50, 100, 150, 200, 'Original']

for i, n_comp in enumerate(reconstruction_components):
    row, col = i // 4, i % 4
    
    if n_comp == 'Original':
        image_to_show = original_image
        title = 'Original'
    else:
        pca_recon = PCA(n_components=n_comp)
        X_train_pca_recon = pca_recon.fit_transform(X_train)
        X_test_pca_recon = pca_recon.transform(X_test[test_image_idx:test_image_idx+1])
        reconstructed = pca_recon.inverse_transform(X_test_pca_recon)
        image_to_show = reconstructed[0]
        
        # 재구성 오차 계산
        mse = np.mean((original_image - image_to_show) ** 2)
        title = f'{n_comp} components\nMSE: {mse:.4f}'
    
    axes[row, col].imshow(image_to_show.reshape(64, 64), cmap='gray')
    axes[row, col].set_title(title)
    axes[row, col].axis('off')

plt.tight_layout()
plt.show()

# 8. 2D 시각화를 위한 PCA
print("\n8. 2D 시각화를 위한 PCA")
print("-" * 30)

# 2D PCA로 데이터 시각화
pca_2d = PCA(n_components=2)
X_train_2d = pca_2d.fit_transform(X_train)
X_test_2d = pca_2d.transform(X_test)

print(f"2D PCA 분산 설명 비율: {pca_2d.explained_variance_ratio_}")
print(f"총 분산 설명 비율: {np.sum(pca_2d.explained_variance_ratio_):.4f}")

# 2D 산점도
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap='tab20', alpha=0.7)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.3f})')
plt.title('Training Data in 2D PCA Space')
plt.colorbar(scatter, label='Person ID')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap='tab20', alpha=0.7)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.3f})')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.3f})')
plt.title('Test Data in 2D PCA Space')
plt.colorbar(scatter, label='Person ID')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. 최적 주성분 개수 선택
print("\n9. 최적 주성분 개수 선택")
print("-" * 30)

# 엘보우 방법을 위한 더 세밀한 분석
n_components_detailed = range(1, 101, 5)
reconstruction_errors_detailed = []

for n_comp in n_components_detailed:
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train)
    X_train_reconstructed = pca.inverse_transform(X_train_pca)
    error = np.mean((X_train - X_train_reconstructed) ** 2)
    reconstruction_errors_detailed.append(error)

# 엘보우 지점 찾기 (2차 미분 이용)
errors_array = np.array(reconstruction_errors_detailed)
second_derivative = np.diff(errors_array, 2)
elbow_idx = np.argmax(second_derivative) + 2  # 2차 미분이므로 +2
optimal_components = list(n_components_detailed)[elbow_idx]

print(f"엘보우 방법으로 찾은 최적 주성분 개수: {optimal_components}")

# 엘보우 플롯
plt.figure(figsize=(10, 6))
plt.plot(n_components_detailed, reconstruction_errors_detailed, 'bo-', markersize=4)
plt.axvline(x=optimal_components, color='r', linestyle='--', 
           label=f'Elbow point: {optimal_components} components')
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error')
plt.title('Elbow Method for Optimal Number of Components')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 10. 최종 모델 평가
print("\n10. 최종 모델 평가")
print("-" * 30)

# 최적 주성분 개수로 최종 모델 구축
final_pca = PCA(n_components=optimal_components)
X_train_final = final_pca.fit_transform(X_train)
X_test_final = final_pca.transform(X_test)

# 최종 분류 성능
final_knn = KNeighborsClassifier(n_neighbors=3)
final_knn.fit(X_train_final, y_train)
y_pred_final = final_knn.predict(X_test_final)

print(f"최적 주성분 개수: {optimal_components}")
print(f"차원 축소 비율: {X.shape[1]} → {optimal_components} ({optimal_components/X.shape[1]*100:.1f}%)")
print(f"분산 설명 비율: {np.sum(final_pca.explained_variance_ratio_):.4f}")
print(f"최종 분류 정확도: {accuracy_score(y_test, y_pred_final):.4f}")

# 상세 분류 리포트
print("\n분류 성능 상세 리포트:")
print(classification_report(y_test, y_pred_final))

# 11. 실습 요약
print("\n" + "="*60)
print("실습 요약")
print("="*60)
print("1. Olivetti 얼굴 데이터셋(64×64=4096차원)에 PCA 적용")
print("2. 분산 설명 비율과 스크리 플롯으로 주성분 개수 분석")
print("3. 다양한 주성분 개수별 분류 성능과 재구성 오차 비교")
print("4. Eigenfaces 시각화로 주성분의 의미 파악")
print("5. 엘보우 방법으로 최적 주성분 개수 선택")
print(f"6. 최종 결과: {X.shape[1]}차원 → {optimal_components}차원 축소")
print(f"   - 분산 설명 비율: {np.sum(final_pca.explained_variance_ratio_):.1%}")
print(f"   - 분류 정확도: {accuracy_score(y_test, y_pred_final):.1%}")
print("\nPCA는 고차원 데이터의 주요 패턴을 효과적으로 추출하여")
print("차원 축소와 노이즈 제거에 유용한 기법임을 확인했습니다.")