"""
DBSCAN 클러스터링 실습

이 실습에서는 DBSCAN 알고리즘을 사용하여 다양한 형태의 클러스터를 탐지하고,
노이즈 데이터를 처리하는 방법을 학습합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, adjusted_rand_score
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def generate_sample_datasets():
    """다양한 형태의 샘플 데이터셋 생성"""
    
    datasets = {}
    
    # 1. 구형 클러스터 (Gaussian blobs)
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, n_features=2, 
                                  random_state=42, cluster_std=0.8)
    datasets['blobs'] = (X_blobs, y_blobs, 'Gaussian Blobs')
    
    # 2. 달 모양 클러스터
    X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)
    datasets['moons'] = (X_moons, y_moons, 'Moons')
    
    # 3. 동심원 클러스터
    X_circles, y_circles = make_circles(n_samples=300, noise=0.05, 
                                        factor=0.6, random_state=42)
    datasets['circles'] = (X_circles, y_circles, 'Circles')
    
    # 4. 노이즈가 포함된 데이터
    np.random.seed(42)
    # 기본 클러스터
    cluster1 = np.random.normal([2, 2], 0.5, (100, 2))
    cluster2 = np.random.normal([6, 6], 0.5, (100, 2))
    cluster3 = np.random.normal([2, 6], 0.5, (80, 2))
    
    # 노이즈 포인트 추가
    noise = np.random.uniform([0, 0], [8, 8], (50, 2))
    
    X_noisy = np.vstack([cluster1, cluster2, cluster3, noise])
    y_noisy = np.hstack([np.zeros(100), np.ones(100), 
                         np.full(80, 2), np.full(50, -1)])  # -1은 노이즈
    
    datasets['noisy'] = (X_noisy, y_noisy, 'Noisy Data')
    
    return datasets

def plot_datasets(datasets):
    """데이터셋 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
    
    for i, (key, (X, y, title)) in enumerate(datasets.items()):
        ax = axes[i]
        
        # 각 클러스터별로 다른 색상으로 표시
        unique_labels = np.unique(y)
        for j, label in enumerate(unique_labels):
            if label == -1:  # 노이즈는 검은색으로
                color = 'black'
                marker = 'x'
                alpha = 0.6
            else:
                color = colors[j % len(colors)]
                marker = 'o'
                alpha = 0.8
            
            mask = y == label
            ax.scatter(X[mask, 0], X[mask, 1], c=color, marker=marker, 
                      alpha=alpha, s=50, label=f'Class {label}' if label != -1 else 'Noise')
        
        ax.set_title(f'{title} (Original)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def find_optimal_eps(X, min_samples=5, plot=True):
    """k-distance 그래프를 이용한 최적 epsilon 값 찾기"""
    
    # k번째 가장 가까운 이웃까지의 거리 계산
    k = min_samples - 1
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    
    # k번째 이웃까지의 거리 (마지막 열)
    k_distances = distances[:, k-1]
    k_distances = np.sort(k_distances, axis=0)[::-1]  # 내림차순 정렬
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(k_distances)), k_distances, 'b-', linewidth=2)
        plt.xlabel('Points sorted by distance')
        plt.ylabel(f'{k}-NN distance')
        plt.title(f'{k}-Distance Graph for Optimal Eps Selection')
        plt.grid(True, alpha=0.3)
        
        # 무릎 지점 추정 (간단한 방법)
        # 거리 변화율이 가장 큰 지점 찾기
        if len(k_distances) > 10:
            diff = np.diff(k_distances)
            knee_point = np.argmax(np.abs(diff)) + 1
            optimal_eps = k_distances[knee_point]
            
            plt.axhline(y=optimal_eps, color='red', linestyle='--', 
                       label=f'Suggested eps = {optimal_eps:.3f}')
            plt.legend()
        
        plt.show()
        
        return optimal_eps if len(k_distances) > 10 else np.median(k_distances)
    
    return np.median(k_distances)

def apply_dbscan_with_params(X, eps_values, min_samples_values):
    """다양한 파라미터로 DBSCAN 적용 및 결과 비교"""
    
    results = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            # DBSCAN 적용
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X)
            
            # 결과 분석
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # 실루엣 점수 계산 (노이즈 제외)
            if n_clusters > 1:
                mask = labels != -1
                if np.sum(mask) > 1:
                    silhouette = silhouette_score(X[mask], labels[mask])
                else:
                    silhouette = -1
            else:
                silhouette = -1
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette_score': silhouette,
                'labels': labels.copy()
            })
    
    return results

def visualize_dbscan_results(X, results, title="DBSCAN Results"):
    """DBSCAN 결과 시각화"""
    
    n_results = len(results)
    cols = min(4, n_results)
    rows = (n_results + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n_results == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    for i, result in enumerate(results):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        labels = result['labels']
        eps = result['eps']
        min_samples = result['min_samples']
        n_clusters = result['n_clusters']
        n_noise = result['n_noise']
        silhouette = result['silhouette_score']
        
        # 클러스터별로 다른 색상으로 표시
        unique_labels = set(labels)
        for j, label in enumerate(unique_labels):
            if label == -1:  # 노이즈는 검은색으로
                color = 'black'
                marker = 'x'
                alpha = 0.6
                size = 30
            else:
                color = colors[label % len(colors)]
                marker = 'o'
                alpha = 0.8
                size = 50
            
            class_member_mask = (labels == label)
            xy = X[class_member_mask]
            ax.scatter(xy[:, 0], xy[:, 1], c=color, marker=marker, 
                      alpha=alpha, s=size)
        
        ax.set_title(f'eps={eps}, min_samples={min_samples}\\n'
                    f'Clusters: {n_clusters}, Noise: {n_noise}\\n'
                    f'Silhouette: {silhouette:.3f}', fontsize=10)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
    
    # 빈 subplot 숨기기
    for i in range(n_results, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def compare_clustering_algorithms(X, true_labels=None):
    """DBSCAN과 다른 클러스터링 알고리즘 비교"""
    
    from sklearn.cluster import KMeans, AgglomerativeClustering
    
    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 최적 eps 찾기
    optimal_eps = find_optimal_eps(X_scaled, plot=False)
    
    # 알고리즘별 클러스터링
    algorithms = {
        'DBSCAN': DBSCAN(eps=optimal_eps, min_samples=5),
        'K-Means (k=3)': KMeans(n_clusters=3, random_state=42, n_init=10),
        'K-Means (k=4)': KMeans(n_clusters=4, random_state=42, n_init=10),
        'Hierarchical': AgglomerativeClustering(n_clusters=3)
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        labels = algorithm.fit_predict(X_scaled)
        
        # 성능 지표 계산
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1) if -1 in labels else 0
        
        # 실루엣 점수
        if n_clusters > 1:
            mask = labels != -1 if -1 in labels else np.ones(len(labels), dtype=bool)
            if np.sum(mask) > 1:
                silhouette = silhouette_score(X_scaled[mask], labels[mask])
            else:
                silhouette = -1
        else:
            silhouette = -1
        
        # ARI (실제 레이블이 있는 경우)
        ari = adjusted_rand_score(true_labels, labels) if true_labels is not None else None
        
        results[name] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette,
            'ari': ari
        }
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
    
    for i, (name, result) in enumerate(results.items()):
        ax = axes[i]
        labels = result['labels']
        
        unique_labels = set(labels)
        for j, label in enumerate(unique_labels):
            if label == -1:  # 노이즈
                color = 'black'
                marker = 'x'
                alpha = 0.6
            else:
                color = colors[label % len(colors)]
                marker = 'o'
                alpha = 0.8
            
            class_member_mask = (labels == label)
            xy = X[class_member_mask]
            ax.scatter(xy[:, 0], xy[:, 1], c=color, marker=marker, alpha=alpha, s=50)
        
        title = f'{name}\\nClusters: {result["n_clusters"]}'
        if result['n_noise'] > 0:
            title += f', Noise: {result["n_noise"]}'
        title += f'\\nSilhouette: {result["silhouette_score"]:.3f}'
        if result['ari'] is not None:
            title += f', ARI: {result["ari"]:.3f}'
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Clustering Algorithms Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return results

def analyze_parameter_sensitivity():
    """파라미터 민감도 분석"""
    
    # 테스트 데이터 생성
    X, _ = make_blobs(n_samples=300, centers=3, n_features=2, 
                      random_state=42, cluster_std=1.0)
    
    # 노이즈 추가
    noise = np.random.uniform(X.min()-1, X.max()+1, (50, 2))
    X = np.vstack([X, noise])
    
    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 파라미터 범위 설정
    eps_values = np.linspace(0.1, 1.0, 10)
    min_samples_values = [3, 5, 7, 10]
    
    # 결과 저장
    sensitivity_results = []
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            sensitivity_results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise
            })
    
    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(sensitivity_results)
    
    # 히트맵으로 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 클러스터 개수 히트맵
    pivot1 = df.pivot(index='min_samples', columns='eps', values='n_clusters')
    sns.heatmap(pivot1, annot=True, fmt='d', cmap='viridis', ax=ax1)
    ax1.set_title('Number of Clusters')
    ax1.set_xlabel('eps')
    ax1.set_ylabel('min_samples')
    
    # 노이즈 개수 히트맵
    pivot2 = df.pivot(index='min_samples', columns='eps', values='n_noise')
    sns.heatmap(pivot2, annot=True, fmt='d', cmap='plasma', ax=ax2)
    ax2.set_title('Number of Noise Points')
    ax2.set_xlabel('eps')
    ax2.set_ylabel('min_samples')
    
    plt.suptitle('DBSCAN Parameter Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return df

def real_world_example():
    """실제 데이터를 이용한 DBSCAN 적용 예시"""
    
    print("실제 데이터 예시: 고객 세분화")
    print("="*50)
    
    # 가상의 고객 데이터 생성
    np.random.seed(42)
    
    # 고객 그룹별 특성
    # 그룹 1: 고소득, 고지출
    group1 = np.random.multivariate_normal([80, 75], [[100, 50], [50, 80]], 100)
    
    # 그룹 2: 중간소득, 중간지출
    group2 = np.random.multivariate_normal([50, 50], [[80, 30], [30, 60]], 150)
    
    # 그룹 3: 저소득, 저지출
    group3 = np.random.multivariate_normal([25, 25], [[60, 20], [20, 40]], 120)
    
    # 이상치 고객들 (특이한 소비 패턴)
    outliers = np.array([[90, 20], [20, 80], [95, 95], [10, 10], [85, 15]])
    
    # 전체 데이터 결합
    X_customers = np.vstack([group1, group2, group3, outliers])
    
    # 특성 이름
    feature_names = ['Annual Income (k$)', 'Spending Score (1-100)']
    
    # 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_customers)
    
    # 최적 eps 찾기
    print("최적 eps 값 찾기...")
    optimal_eps = find_optimal_eps(X_scaled, min_samples=5)
    
    # DBSCAN 적용
    dbscan = DBSCAN(eps=optimal_eps, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)
    
    # 결과 분석
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"\\n클러스터링 결과:")
    print(f"- 발견된 클러스터 수: {n_clusters}")
    print(f"- 이상치 고객 수: {n_noise}")
    print(f"- 사용된 eps: {optimal_eps:.3f}")
    
    # 시각화
    plt.figure(figsize=(12, 5))
    
    # 원본 데이터
    plt.subplot(1, 2, 1)
    plt.scatter(X_customers[:, 0], X_customers[:, 1], c='blue', alpha=0.6, s=50)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title('Original Customer Data')
    plt.grid(True, alpha=0.3)
    
    # DBSCAN 결과
    plt.subplot(1, 2, 2)
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    unique_labels = set(labels)
    
    for i, label in enumerate(unique_labels):
        if label == -1:  # 이상치
            color = 'black'
            marker = 'x'
            alpha = 0.8
            size = 100
            label_name = 'Outliers'
        else:
            color = colors[label % len(colors)]
            marker = 'o'
            alpha = 0.7
            size = 50
            label_name = f'Cluster {label}'
        
        class_member_mask = (labels == label)
        xy = X_customers[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=color, marker=marker, 
                   alpha=alpha, s=size, label=label_name)
    
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(f'DBSCAN Clustering Results\\n({n_clusters} clusters, {n_noise} outliers)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 클러스터별 특성 분석
    if n_clusters > 0:
        print("\\n클러스터별 특성 분석:")
        print("-" * 40)
        
        df_customers = pd.DataFrame(X_customers, columns=feature_names)
        df_customers['Cluster'] = labels
        
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                cluster_name = "이상치"
            else:
                cluster_name = f"클러스터 {cluster_id}"
            
            cluster_data = df_customers[df_customers['Cluster'] == cluster_id]
            
            print(f"\\n{cluster_name} (고객 수: {len(cluster_data)}):")
            print(f"  평균 연소득: ${cluster_data[feature_names[0]].mean():.1f}k")
            print(f"  평균 지출점수: {cluster_data[feature_names[1]].mean():.1f}")
            
            if cluster_id != -1:
                if cluster_data[feature_names[0]].mean() > 60 and cluster_data[feature_names[1]].mean() > 60:
                    print("  → 고가치 고객 (High Value)")
                elif cluster_data[feature_names[0]].mean() > 40 and cluster_data[feature_names[1]].mean() > 40:
                    print("  → 중간 고객 (Medium Value)")
                else:
                    print("  → 저가치 고객 (Low Value)")

def main():
    """메인 실습 함수"""
    
    print("DBSCAN 클러스터링 실습")
    print("="*50)
    
    # 1. 샘플 데이터셋 생성 및 시각화
    print("\\n1. 다양한 형태의 데이터셋 생성")
    datasets = generate_sample_datasets()
    plot_datasets(datasets)
    
    # 2. 각 데이터셋에 대해 DBSCAN 적용
    print("\\n2. 데이터셋별 DBSCAN 적용")
    
    for name, (X, y_true, title) in datasets.items():
        print(f"\\n처리 중: {title}")
        
        # 데이터 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 최적 eps 찾기
        optimal_eps = find_optimal_eps(X_scaled, plot=False)
        
        # 다양한 파라미터로 DBSCAN 적용
        eps_values = [optimal_eps * 0.5, optimal_eps, optimal_eps * 1.5]
        min_samples_values = [3, 5]
        
        results = apply_dbscan_with_params(X_scaled, eps_values, min_samples_values)
        
        # 결과 시각화
        visualize_dbscan_results(X, results, f"DBSCAN Results - {title}")
        
        # 다른 알고리즘과 비교
        print(f"다른 알고리즘과 비교 - {title}")
        comparison_results = compare_clustering_algorithms(X, y_true)
    
    # 3. 파라미터 민감도 분석
    print("\\n3. 파라미터 민감도 분석")
    sensitivity_df = analyze_parameter_sensitivity()
    
    # 4. 실제 데이터 예시
    print("\\n4. 실제 데이터 적용 예시")
    real_world_example()
    
    # 5. 주요 학습 포인트 정리
    print("\\n" + "="*50)
    print("주요 학습 포인트")
    print("="*50)
    print("1. DBSCAN은 클러스터 개수를 미리 지정할 필요가 없습니다.")
    print("2. 임의의 모양을 가진 클러스터를 잘 탐지할 수 있습니다.")
    print("3. 노이즈와 이상치를 자동으로 탐지합니다.")
    print("4. eps와 min_samples 파라미터 선택이 매우 중요합니다.")
    print("5. k-distance 그래프를 이용해 적절한 eps를 찾을 수 있습니다.")
    print("6. 데이터 표준화가 결과에 큰 영향을 미칩니다.")
    print("7. 밀도가 매우 다른 클러스터가 동시에 있으면 어려울 수 있습니다.")
    
    print("\\n실습을 완료했습니다!")

if __name__ == "__main__":
    main()