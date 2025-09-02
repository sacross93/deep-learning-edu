"""
계층적 클러스터링 실습

이 실습에서는 생물 종 분류 데이터를 사용하여 계층적 클러스터링을 수행합니다.
다양한 연결 기준을 비교하고 덴드로그램을 해석하는 방법을 학습합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("="*60)
print("계층적 클러스터링 실습")
print("="*60)

# 1. 데이터셋 로딩 및 탐색
print("\n1. 데이터셋 로딩 및 탐색")
print("-" * 40)

# Iris 데이터셋을 생물 종 분류 데이터로 사용
iris = load_iris()
X = iris.data
y_true = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"데이터 형태: {X.shape}")
print(f"특성 이름: {feature_names}")
print(f"종 이름: {target_names}")

# 데이터 기본 통계
df = pd.DataFrame(X, columns=feature_names)
df['species'] = [target_names[i] for i in y_true]
print(f"\n데이터 기본 통계:")
print(df.describe())

# 2. 데이터 전처리
print("\n2. 데이터 전처리")
print("-" * 40)

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("표준화 완료")
print(f"표준화 전 평균: {X.mean(axis=0)}")
print(f"표준화 후 평균: {X_scaled.mean(axis=0)}")
print(f"표준화 후 표준편차: {X_scaled.std(axis=0)}")

# 3. 거리 행렬 계산
print("\n3. 거리 행렬 계산")
print("-" * 40)

# 유클리드 거리 행렬 계산
distance_matrix = pdist(X_scaled, metric='euclidean')
print(f"거리 행렬 크기: {len(distance_matrix)} (n*(n-1)/2 = {len(X_scaled)*(len(X_scaled)-1)//2})")

# 거리 행렬을 정사각 행렬로 변환
distance_square = squareform(distance_matrix)
print(f"정사각 거리 행렬 형태: {distance_square.shape}")

# 4. 다양한 연결 기준으로 계층적 클러스터링 수행
print("\n4. 다양한 연결 기준 비교")
print("-" * 40)

linkage_methods = ['single', 'complete', 'average', 'ward']
linkage_results = {}

for method in linkage_methods:
    print(f"\n{method.upper()} 연결 기준:")
    
    # 연결 행렬 계산
    if method == 'ward':
        # Ward는 유클리드 거리만 사용 가능
        Z = linkage(X_scaled, method=method)
    else:
        Z = linkage(distance_matrix, method=method)
    
    linkage_results[method] = Z
    
    # 3개 클러스터로 분할
    clusters = fcluster(Z, 3, criterion='maxclust')
    
    # 성능 평가
    ari = adjusted_rand_score(y_true, clusters)
    silhouette = silhouette_score(X_scaled, clusters)
    
    print(f"  조정된 랜드 지수: {ari:.3f}")
    print(f"  실루엣 계수: {silhouette:.3f}")
    print(f"  클러스터 분포: {np.bincount(clusters)}")

# 5. 덴드로그램 시각화
print("\n5. 덴드로그램 시각화")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, method in enumerate(linkage_methods):
    ax = axes[i]
    
    # 덴드로그램 그리기
    dendrogram(linkage_results[method], 
               ax=ax,
               truncate_mode='level',
               p=10,
               leaf_rotation=90,
               leaf_font_size=8)
    
    ax.set_title(f'{method.upper()} Linkage Dendrogram', fontsize=12, fontweight='bold')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Distance')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('machine_learning_tutorials/hierarchical_clustering_dendrograms.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# 6. 최적 클러스터 개수 결정
print("\n6. 최적 클러스터 개수 결정")
print("-" * 40)

# Ward 연결을 사용하여 클러스터 개수별 성능 평가
ward_linkage = linkage_results['ward']
n_clusters_range = range(2, 8)
silhouette_scores = []
ari_scores = []

for n_clusters in n_clusters_range:
    clusters = fcluster(ward_linkage, n_clusters, criterion='maxclust')
    
    silhouette = silhouette_score(X_scaled, clusters)
    ari = adjusted_rand_score(y_true, clusters)
    
    silhouette_scores.append(silhouette)
    ari_scores.append(ari)
    
    print(f"클러스터 {n_clusters}개: 실루엣={silhouette:.3f}, ARI={ari:.3f}")

# 성능 지표 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(n_clusters_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters')
ax1.set_ylabel('Silhouette Score')
ax1.set_title('Silhouette Score vs Number of Clusters')
ax1.grid(True, alpha=0.3)

ax2.plot(n_clusters_range, ari_scores, 'ro-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters')
ax2.set_ylabel('Adjusted Rand Index')
ax2.set_title('ARI vs Number of Clusters')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('machine_learning_tutorials/hierarchical_clustering_metrics.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# 7. 최적 결과로 상세 분석
print("\n7. 최적 결과 상세 분석 (Ward, 3 클러스터)")
print("-" * 40)

# Ward 연결로 3개 클러스터 생성
final_clusters = fcluster(ward_linkage, 3, criterion='maxclust')

# 클러스터별 통계
cluster_stats = pd.DataFrame(X_scaled, columns=feature_names)
cluster_stats['cluster'] = final_clusters
cluster_stats['true_species'] = [target_names[i] for i in y_true]

print("클러스터별 평균 특성값:")
cluster_means = cluster_stats.groupby('cluster')[feature_names].mean()
print(cluster_means)

print("\n클러스터와 실제 종의 교차표:")
confusion_matrix = pd.crosstab(cluster_stats['cluster'], 
                              cluster_stats['true_species'])
print(confusion_matrix)

# 8. 클러스터 결과 시각화
print("\n8. 클러스터 결과 시각화")
print("-" * 40)

# 주성분 분석으로 2차원 시각화
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 실제 종별 시각화
scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', s=50, alpha=0.7)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax1.set_title('True Species Labels')
ax1.grid(True, alpha=0.3)

# 범례 추가
for i, species in enumerate(target_names):
    ax1.scatter([], [], c=plt.cm.viridis(i/2), label=species, s=50)
ax1.legend()

# 클러스터링 결과 시각화
scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=final_clusters, cmap='Set1', s=50, alpha=0.7)
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax2.set_title('Hierarchical Clustering Results')
ax2.grid(True, alpha=0.3)

# 범례 추가
for i in range(1, 4):
    ax2.scatter([], [], c=plt.cm.Set1((i-1)/3), label=f'Cluster {i}', s=50)
ax2.legend()

plt.tight_layout()
plt.savefig('machine_learning_tutorials/hierarchical_clustering_results.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# 9. 덴드로그램 상세 해석
print("\n9. 덴드로그램 상세 해석")
print("-" * 40)

# Ward 연결의 덴드로그램을 더 자세히 분석
plt.figure(figsize=(12, 8))

# 색상으로 클러스터 구분
dendrogram(ward_linkage,
           color_threshold=0.7*max(ward_linkage[:,2]),
           above_threshold_color='gray',
           leaf_rotation=90,
           leaf_font_size=10)

plt.title('Ward Linkage Dendrogram with Color Threshold', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.axhline(y=0.7*max(ward_linkage[:,2]), color='red', linestyle='--', 
           label=f'Cut Height = {0.7*max(ward_linkage[:,2]):.2f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('machine_learning_tutorials/hierarchical_clustering_detailed_dendrogram.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# 병합 과정 분석
print("주요 병합 단계 (마지막 10단계):")
n_samples = len(X_scaled)
for i in range(max(0, len(ward_linkage)-10), len(ward_linkage)):
    step = i + 1
    cluster1 = int(ward_linkage[i, 0])
    cluster2 = int(ward_linkage[i, 1])
    distance = ward_linkage[i, 2]
    size = int(ward_linkage[i, 3])
    
    print(f"단계 {step:2d}: 클러스터 {cluster1:3d}와 {cluster2:3d} 병합 "
          f"(거리: {distance:.3f}, 크기: {size:2d})")

# 10. 다른 알고리즘과의 비교
print("\n10. K-평균과의 비교")
print("-" * 40)

from sklearn.cluster import KMeans

# K-평균 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_clusters = kmeans.fit_predict(X_scaled)

# 성능 비교
hierarchical_ari = adjusted_rand_score(y_true, final_clusters)
hierarchical_silhouette = silhouette_score(X_scaled, final_clusters)

kmeans_ari = adjusted_rand_score(y_true, kmeans_clusters)
kmeans_silhouette = silhouette_score(X_scaled, kmeans_clusters)

print("성능 비교:")
print(f"계층적 클러스터링 - ARI: {hierarchical_ari:.3f}, 실루엣: {hierarchical_silhouette:.3f}")
print(f"K-평균 클러스터링  - ARI: {kmeans_ari:.3f}, 실루엣: {kmeans_silhouette:.3f}")

# 결과 비교 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 계층적 클러스터링 결과
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=final_clusters, cmap='Set1', s=50, alpha=0.7)
ax1.set_title(f'Hierarchical Clustering\n(ARI: {hierarchical_ari:.3f}, Silhouette: {hierarchical_silhouette:.3f})')
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax1.grid(True, alpha=0.3)

# K-평균 결과
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_clusters, cmap='Set1', s=50, alpha=0.7)
ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='Centroids')
ax2.set_title(f'K-Means Clustering\n(ARI: {kmeans_ari:.3f}, Silhouette: {kmeans_silhouette:.3f})')
ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('machine_learning_tutorials/hierarchical_vs_kmeans.png', 
            dpi=300, bbox_inches='tight')
plt.show()

# 11. 실제 생물학적 해석
print("\n11. 생물학적 해석")
print("-" * 40)

# 클러스터별 특성 분석
print("클러스터별 생물학적 특성:")
for cluster_id in range(1, 4):
    cluster_mask = final_clusters == cluster_id
    cluster_data = df[cluster_mask]
    
    print(f"\n클러스터 {cluster_id}:")
    print(f"  포함된 종: {cluster_data['species'].value_counts().to_dict()}")
    print(f"  평균 꽃받침 길이: {cluster_data['sepal length (cm)'].mean():.2f} cm")
    print(f"  평균 꽃받침 너비: {cluster_data['sepal width (cm)'].mean():.2f} cm")
    print(f"  평균 꽃잎 길이: {cluster_data['petal length (cm)'].mean():.2f} cm")
    print(f"  평균 꽃잎 너비: {cluster_data['petal width (cm)'].mean():.2f} cm")

# 12. 요약 및 결론
print("\n12. 요약 및 결론")
print("-" * 40)

print("계층적 클러스터링 실습 결과:")
print(f"• 최적 연결 기준: Ward (ARI: {hierarchical_ari:.3f})")
print(f"• 최적 클러스터 개수: 3개")
print(f"• 실제 종 분류와의 일치도: {hierarchical_ari:.1%}")
print(f"• 클러스터 품질 (실루엣 계수): {hierarchical_silhouette:.3f}")

print("\n주요 학습 내용:")
print("• 다양한 연결 기준의 특성과 차이점")
print("• 덴드로그램을 통한 계층 구조 해석")
print("• 최적 클러스터 개수 결정 방법")
print("• K-평균과의 성능 및 특성 비교")
print("• 생물학적 데이터에서의 실제 적용 사례")

print("\n실습 완료!")