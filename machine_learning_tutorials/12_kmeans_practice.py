#!/usr/bin/env python3
"""
K-평균 클러스터링 실습

이 실습에서는 고객 세분화 데이터를 활용하여 K-평균 클러스터링을 수행합니다.
클러스터 개수 선택, 클러스터링 수행, 결과 시각화 및 해석을 다룹니다.

주요 학습 내용:
1. 고객 데이터 탐색 및 전처리
2. 최적 클러스터 개수 결정 (엘보우 방법, 실루엣 분석)
3. K-평균 클러스터링 수행
4. 클러스터 특성 분석 및 시각화
5. 비즈니스 인사이트 도출
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def generate_customer_data(n_samples=1000, random_state=42):
    """
    고객 세분화를 위한 합성 데이터 생성
    
    Parameters:
    -----------
    n_samples : int
        생성할 샘플 수
    random_state : int
        랜덤 시드
        
    Returns:
    --------
    pd.DataFrame
        고객 데이터
    """
    np.random.seed(random_state)
    
    # 4개의 고객 세그먼트 정의
    segments = {
        'Premium': {'size': 0.15, 'annual_spending': (8000, 12000), 'frequency': (50, 80), 'recency': (1, 30)},
        'Regular': {'size': 0.35, 'annual_spending': (3000, 6000), 'frequency': (20, 40), 'recency': (15, 60)},
        'Occasional': {'size': 0.35, 'annual_spending': (500, 2000), 'frequency': (5, 15), 'recency': (30, 120)},
        'Inactive': {'size': 0.15, 'annual_spending': (0, 500), 'frequency': (1, 5), 'recency': (90, 365)}
    }
    
    data = []
    customer_id = 1
    
    for segment_name, params in segments.items():
        n_segment = int(n_samples * params['size'])
        
        for _ in range(n_segment):
            # 연간 지출액 (Annual Spending)
            spending = np.random.uniform(params['annual_spending'][0], params['annual_spending'][1])
            
            # 구매 빈도 (Frequency - 연간 구매 횟수)
            frequency = np.random.randint(params['frequency'][0], params['frequency'][1])
            
            # 최근 구매일 (Recency - 마지막 구매 후 경과 일수)
            recency = np.random.randint(params['recency'][0], params['recency'][1])
            
            # 평균 구매액 (Average Order Value)
            avg_order_value = spending / max(frequency, 1) if frequency > 0 else 0
            
            # 고객 나이 (세그먼트별 특성 반영)
            if segment_name == 'Premium':
                age = np.random.normal(45, 10)
            elif segment_name == 'Regular':
                age = np.random.normal(35, 8)
            elif segment_name == 'Occasional':
                age = np.random.normal(28, 12)
            else:  # Inactive
                age = np.random.normal(50, 15)
            
            age = max(18, min(80, age))  # 18-80세 범위로 제한
            
            data.append({
                'customer_id': customer_id,
                'annual_spending': spending,
                'purchase_frequency': frequency,
                'recency_days': recency,
                'avg_order_value': avg_order_value,
                'age': age,
                'true_segment': segment_name
            })
            
            customer_id += 1
    
    return pd.DataFrame(data)

def explore_customer_data(df):
    """
    고객 데이터 탐색적 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        고객 데이터
    """
    print("=== 고객 데이터 탐색적 분석 ===\n")
    
    # 기본 정보
    print("1. 데이터 기본 정보:")
    print(f"   - 고객 수: {len(df):,}명")
    print(f"   - 특성 수: {len(df.columns)-2}개")  # customer_id, true_segment 제외
    print(f"   - 결측값: {df.isnull().sum().sum()}개\n")
    
    # 기술 통계
    print("2. 기술 통계:")
    numeric_cols = ['annual_spending', 'purchase_frequency', 'recency_days', 'avg_order_value', 'age']
    print(df[numeric_cols].describe().round(2))
    print()
    
    # 실제 세그먼트 분포
    print("3. 실제 고객 세그먼트 분포:")
    segment_counts = df['true_segment'].value_counts()
    for segment, count in segment_counts.items():
        print(f"   - {segment}: {count}명 ({count/len(df)*100:.1f}%)")
    print()
    
    # 상관관계 분석
    print("4. 특성 간 상관관계:")
    correlation_matrix = df[numeric_cols].corr()
    print(correlation_matrix.round(3))
    
    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Customer Data Exploratory Analysis', fontsize=16, fontweight='bold')
    
    # 1. 연간 지출액 분포
    axes[0, 0].hist(df['annual_spending'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Annual Spending Distribution')
    axes[0, 0].set_xlabel('Annual Spending ($)')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. 구매 빈도 분포
    axes[0, 1].hist(df['purchase_frequency'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('Purchase Frequency Distribution')
    axes[0, 1].set_xlabel('Purchase Frequency (per year)')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. 최근성 분포
    axes[0, 2].hist(df['recency_days'], bins=50, alpha=0.7, color='salmon', edgecolor='black')
    axes[0, 2].set_title('Recency Distribution')
    axes[0, 2].set_xlabel('Days Since Last Purchase')
    axes[0, 2].set_ylabel('Frequency')
    
    # 4. 연간 지출액 vs 구매 빈도
    scatter = axes[1, 0].scatter(df['purchase_frequency'], df['annual_spending'], 
                                c=df['recency_days'], cmap='viridis', alpha=0.6)
    axes[1, 0].set_title('Annual Spending vs Purchase Frequency')
    axes[1, 0].set_xlabel('Purchase Frequency')
    axes[1, 0].set_ylabel('Annual Spending ($)')
    plt.colorbar(scatter, ax=axes[1, 0], label='Recency (days)')
    
    # 5. 상관관계 히트맵
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, ax=axes[1, 1])
    axes[1, 1].set_title('Feature Correlation Matrix')
    
    # 6. 세그먼트별 특성 비교
    segment_means = df.groupby('true_segment')[numeric_cols].mean()
    segment_means.plot(kind='bar', ax=axes[1, 2])
    axes[1, 2].set_title('Average Features by True Segment')
    axes[1, 2].set_xlabel('Segment')
    axes[1, 2].set_ylabel('Normalized Value')
    axes[1, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def find_optimal_k(X, max_k=10):
    """
    엘보우 방법과 실루엣 분석을 통한 최적 K 찾기
    
    Parameters:
    -----------
    X : array-like
        클러스터링할 데이터
    max_k : int
        테스트할 최대 K 값
        
    Returns:
    --------
    dict
        최적 K 값들과 평가 지표
    """
    print("=== 최적 클러스터 개수 탐색 ===\n")
    
    k_range = range(2, max_k + 1)
    wcss_values = []
    silhouette_scores = []
    
    print("K 값별 평가 지표 계산 중...")
    for k in k_range:
        # K-평균 클러스터링 수행
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # WCSS 계산
        wcss = kmeans.inertia_
        wcss_values.append(wcss)
        
        # 실루엣 점수 계산
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        print(f"K={k}: WCSS={wcss:.0f}, Silhouette Score={silhouette_avg:.3f}")
    
    # 엘보우 포인트 찾기 (간단한 방법)
    wcss_diff = np.diff(wcss_values)
    wcss_diff2 = np.diff(wcss_diff)
    elbow_k = k_range[np.argmax(wcss_diff2) + 2]  # 2차 차분이 최대인 지점
    
    # 최고 실루엣 점수의 K
    best_silhouette_k = k_range[np.argmax(silhouette_scores)]
    
    print(f"\n엘보우 방법 추천 K: {elbow_k}")
    print(f"실루엣 분석 추천 K: {best_silhouette_k}")
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 엘보우 방법 시각화
    ax1.plot(k_range, wcss_values, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=elbow_k, color='red', linestyle='--', alpha=0.7, label=f'Elbow K={elbow_k}')
    ax1.set_title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 실루엣 분석 시각화
    ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.axvline(x=best_silhouette_k, color='blue', linestyle='--', alpha=0.7, 
                label=f'Best Silhouette K={best_silhouette_k}')
    ax2.set_title('Silhouette Analysis for Optimal K', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Average Silhouette Score')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return {
        'k_range': list(k_range),
        'wcss_values': wcss_values,
        'silhouette_scores': silhouette_scores,
        'elbow_k': elbow_k,
        'best_silhouette_k': best_silhouette_k
    }

def perform_kmeans_clustering(X, k, feature_names):
    """
    K-평균 클러스터링 수행
    
    Parameters:
    -----------
    X : array-like
        클러스터링할 데이터
    k : int
        클러스터 개수
    feature_names : list
        특성 이름들
        
    Returns:
    --------
    tuple
        (kmeans 모델, 클러스터 레이블, 클러스터 중심점)
    """
    print(f"=== K-평균 클러스터링 수행 (K={k}) ===\n")
    
    # K-평균 클러스터링
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    
    # 클러스터 중심점
    centroids = kmeans.cluster_centers_
    
    # 클러스터링 결과 요약
    print("클러스터링 결과:")
    print(f"- 클러스터 개수: {k}")
    print(f"- 총 반복 횟수: {kmeans.n_iter_}")
    print(f"- WCSS: {kmeans.inertia_:.2f}")
    
    # 클러스터별 데이터 포인트 수
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print(f"- 클러스터별 크기: {dict(zip(unique, counts))}")
    
    # 실루엣 점수
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"- 평균 실루엣 점수: {silhouette_avg:.3f}")
    
    # 클러스터 중심점 출력
    print("\n클러스터 중심점:")
    centroids_df = pd.DataFrame(centroids, columns=feature_names)
    centroids_df.index = [f'Cluster {i}' for i in range(k)]
    print(centroids_df.round(3))
    
    return kmeans, cluster_labels, centroids

def analyze_clusters(df, cluster_labels, feature_names):
    """
    클러스터 특성 분석
    
    Parameters:
    -----------
    df : pd.DataFrame
        원본 데이터
    cluster_labels : array-like
        클러스터 레이블
    feature_names : list
        분석할 특성 이름들
    """
    print("\n=== 클러스터 특성 분석 ===\n")
    
    # 클러스터 레이블 추가
    df_analysis = df.copy()
    df_analysis['cluster'] = cluster_labels
    
    # 클러스터별 기술 통계
    print("1. 클러스터별 평균 특성:")
    cluster_stats = df_analysis.groupby('cluster')[feature_names].agg(['mean', 'std'])
    print(cluster_stats.round(2))
    print()
    
    # 클러스터별 크기
    print("2. 클러스터별 고객 수:")
    cluster_sizes = df_analysis['cluster'].value_counts().sort_index()
    for cluster, size in cluster_sizes.items():
        percentage = size / len(df_analysis) * 100
        print(f"   Cluster {cluster}: {size}명 ({percentage:.1f}%)")
    print()
    
    # 실제 세그먼트와의 비교 (만약 있다면)
    if 'true_segment' in df_analysis.columns:
        print("3. 클러스터와 실제 세그먼트 비교:")
        confusion_matrix = pd.crosstab(df_analysis['cluster'], df_analysis['true_segment'])
        print(confusion_matrix)
        print()
    
    # 클러스터 특성 해석
    print("4. 클러스터 특성 해석:")
    cluster_means = df_analysis.groupby('cluster')[feature_names].mean()
    
    for cluster_id in cluster_means.index:
        print(f"\n   Cluster {cluster_id}:")
        cluster_data = cluster_means.loc[cluster_id]
        
        # 각 특성별 특징 설명
        spending = cluster_data['annual_spending']
        frequency = cluster_data['purchase_frequency']
        recency = cluster_data['recency_days']
        avg_order = cluster_data['avg_order_value']
        age = cluster_data['age']
        
        # 지출 수준 분류
        if spending > 6000:
            spending_level = "높은"
        elif spending > 2000:
            spending_level = "보통"
        else:
            spending_level = "낮은"
        
        # 구매 빈도 분류
        if frequency > 30:
            frequency_level = "높은"
        elif frequency > 10:
            frequency_level = "보통"
        else:
            frequency_level = "낮은"
        
        # 최근성 분류
        if recency < 30:
            recency_level = "최근"
        elif recency < 90:
            recency_level = "보통"
        else:
            recency_level = "오래된"
        
        print(f"     - 연간 지출: ${spending:.0f} ({spending_level})")
        print(f"     - 구매 빈도: {frequency:.1f}회/년 ({frequency_level})")
        print(f"     - 최근 구매: {recency:.0f}일 전 ({recency_level})")
        print(f"     - 평균 주문액: ${avg_order:.0f}")
        print(f"     - 평균 나이: {age:.0f}세")
        
        # 비즈니스 세그먼트 추천
        if spending_level == "높은" and frequency_level == "높은":
            segment_name = "프리미엄 고객"
        elif spending_level == "보통" and frequency_level == "보통":
            segment_name = "일반 고객"
        elif recency_level == "오래된":
            segment_name = "비활성 고객"
        else:
            segment_name = "가끔 구매 고객"
        
        print(f"     → 추천 세그먼트: {segment_name}")

def visualize_clusters(X, cluster_labels, centroids, feature_names, k):
    """
    클러스터링 결과 시각화
    
    Parameters:
    -----------
    X : array-like
        클러스터링된 데이터
    cluster_labels : array-like
        클러스터 레이블
    centroids : array-like
        클러스터 중심점
    feature_names : list
        특성 이름들
    k : int
        클러스터 개수
    """
    print("\n=== 클러스터링 결과 시각화 ===\n")
    
    # PCA를 통한 2D 시각화
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    centroids_pca = pca.transform(centroids)
    
    # 설명된 분산 비율
    explained_variance = pca.explained_variance_ratio_
    print(f"PCA 설명된 분산 비율: PC1={explained_variance[0]:.3f}, PC2={explained_variance[1]:.3f}")
    print(f"총 설명된 분산: {sum(explained_variance):.3f}")
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'K-means Clustering Results (K={k})', fontsize=16, fontweight='bold')
    
    # 1. PCA 2D 클러스터 시각화
    colors = plt.cm.Set3(np.linspace(0, 1, k))
    for i in range(k):
        mask = cluster_labels == i
        axes[0, 0].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=[colors[i]], label=f'Cluster {i}', alpha=0.7, s=50)
    
    # 중심점 표시
    axes[0, 0].scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                      c='red', marker='x', s=200, linewidths=3, label='Centroids')
    axes[0, 0].set_title('Clusters in PCA Space')
    axes[0, 0].set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)')
    axes[0, 0].set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 실루엣 분석
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    y_lower = 10
    for i in range(k):
        cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = colors[i]
        axes[0, 1].fill_betweenx(np.arange(y_lower, y_upper),
                                0, cluster_silhouette_values,
                                facecolor=color, edgecolor=color, alpha=0.7)
        
        axes[0, 1].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    axes[0, 1].axvline(x=silhouette_avg, color="red", linestyle="--", 
                      label=f'Average Score: {silhouette_avg:.3f}')
    axes[0, 1].set_title('Silhouette Analysis')
    axes[0, 1].set_xlabel('Silhouette Coefficient Values')
    axes[0, 1].set_ylabel('Cluster Label')
    axes[0, 1].legend()
    
    # 3. 클러스터별 특성 비교 (레이더 차트)
    from math import pi
    
    # 정규화된 중심점 계산
    scaler = StandardScaler()
    centroids_normalized = scaler.fit_transform(centroids)
    
    angles = [n / len(feature_names) * 2 * pi for n in range(len(feature_names))]
    angles += angles[:1]  # 원형 완성
    
    ax_radar = plt.subplot(2, 2, 3, projection='polar')
    
    for i in range(k):
        values = centroids_normalized[i].tolist()
        values += values[:1]  # 원형 완성
        
        ax_radar.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {i}', color=colors[i])
        ax_radar.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(feature_names)
    ax_radar.set_title('Cluster Characteristics (Normalized)', y=1.08)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 4. 클러스터 크기 분포
    unique, counts = np.unique(cluster_labels, return_counts=True)
    axes[1, 1].pie(counts, labels=[f'Cluster {i}' for i in unique], autopct='%1.1f%%',
                   colors=colors[:len(unique)], startangle=90)
    axes[1, 1].set_title('Cluster Size Distribution')
    
    plt.tight_layout()
    plt.show()

def business_insights(df, cluster_labels, feature_names):
    """
    비즈니스 인사이트 도출
    
    Parameters:
    -----------
    df : pd.DataFrame
        원본 데이터
    cluster_labels : array-like
        클러스터 레이블
    feature_names : list
        특성 이름들
    """
    print("\n=== 비즈니스 인사이트 및 활용 방안 ===\n")
    
    df_business = df.copy()
    df_business['cluster'] = cluster_labels
    
    # 클러스터별 수익성 분석
    print("1. 클러스터별 수익성 분석:")
    cluster_revenue = df_business.groupby('cluster')['annual_spending'].agg(['sum', 'mean', 'count'])
    cluster_revenue.columns = ['Total_Revenue', 'Avg_Revenue', 'Customer_Count']
    cluster_revenue['Revenue_Share'] = cluster_revenue['Total_Revenue'] / cluster_revenue['Total_Revenue'].sum() * 100
    
    print(cluster_revenue.round(2))
    print()
    
    # 클러스터별 마케팅 전략 제안
    print("2. 클러스터별 마케팅 전략 제안:")
    
    cluster_means = df_business.groupby('cluster')[feature_names].mean()
    
    for cluster_id in cluster_means.index:
        cluster_data = cluster_means.loc[cluster_id]
        cluster_size = len(df_business[df_business['cluster'] == cluster_id])
        revenue_share = cluster_revenue.loc[cluster_id, 'Revenue_Share']
        
        print(f"\n   Cluster {cluster_id} ({cluster_size}명, 매출 기여도 {revenue_share:.1f}%):")
        
        spending = cluster_data['annual_spending']
        frequency = cluster_data['purchase_frequency']
        recency = cluster_data['recency_days']
        
        # 전략 제안
        if spending > 6000 and frequency > 30:
            print("     전략: VIP 고객 관리")
            print("     - 개인화된 프리미엄 서비스 제공")
            print("     - 로열티 프로그램 강화")
            print("     - 신제품 우선 소개")
            
        elif spending > 2000 and frequency > 10:
            print("     전략: 관계 강화")
            print("     - 정기적인 프로모션 제공")
            print("     - 교차 판매 기회 확대")
            print("     - 고객 만족도 조사")
            
        elif recency > 90:
            print("     전략: 재활성화")
            print("     - 윈백 캠페인 실시")
            print("     - 특별 할인 혜택 제공")
            print("     - 이탈 원인 분석")
            
        else:
            print("     전략: 구매 촉진")
            print("     - 타겟 광고 집행")
            print("     - 구매 인센티브 제공")
            print("     - 브랜드 인지도 향상")
    
    # 우선순위 고객 그룹 식별
    print("\n3. 우선순위 고객 그룹:")
    
    # 각 클러스터의 가치 점수 계산 (매출 기여도 + 고객 수)
    value_scores = []
    for cluster_id in cluster_revenue.index:
        revenue_score = cluster_revenue.loc[cluster_id, 'Revenue_Share'] / 100
        size_score = cluster_revenue.loc[cluster_id, 'Customer_Count'] / len(df_business)
        total_score = revenue_score * 0.7 + size_score * 0.3  # 매출 가중치 70%
        value_scores.append((cluster_id, total_score))
    
    # 점수 순으로 정렬
    value_scores.sort(key=lambda x: x[1], reverse=True)
    
    for rank, (cluster_id, score) in enumerate(value_scores, 1):
        revenue_share = cluster_revenue.loc[cluster_id, 'Revenue_Share']
        customer_count = cluster_revenue.loc[cluster_id, 'Customer_Count']
        print(f"   {rank}순위: Cluster {cluster_id} (가치점수: {score:.3f})")
        print(f"           매출 기여도: {revenue_share:.1f}%, 고객 수: {customer_count}명")
    
    print("\n4. 실행 가능한 액션 아이템:")
    print("   □ 고가치 클러스터 대상 개인화 마케팅 캠페인 설계")
    print("   □ 저활성 클러스터 대상 재활성화 프로그램 개발")
    print("   □ 클러스터별 고객 여정 맵핑 및 터치포인트 최적화")
    print("   □ 정기적인 클러스터 모니터링 및 세그먼트 업데이트")
    print("   □ A/B 테스트를 통한 클러스터별 마케팅 효과 측정")

def main():
    """
    K-평균 클러스터링 실습 메인 함수
    """
    print("=" * 60)
    print("K-평균 클러스터링을 활용한 고객 세분화 실습")
    print("=" * 60)
    
    # 1. 데이터 생성 및 탐색
    print("\n1단계: 고객 데이터 생성 및 탐색")
    df = generate_customer_data(n_samples=1000, random_state=42)
    explore_customer_data(df)
    
    # 2. 데이터 전처리
    print("\n2단계: 데이터 전처리")
    feature_names = ['annual_spending', 'purchase_frequency', 'recency_days', 'avg_order_value', 'age']
    X = df[feature_names].values
    
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("특성 표준화 완료:")
    print(f"- 원본 데이터 형태: {X.shape}")
    print(f"- 표준화된 데이터 평균: {X_scaled.mean(axis=0).round(3)}")
    print(f"- 표준화된 데이터 표준편차: {X_scaled.std(axis=0).round(3)}")
    
    # 3. 최적 클러스터 개수 찾기
    print("\n3단계: 최적 클러스터 개수 탐색")
    optimal_k_results = find_optimal_k(X_scaled, max_k=8)
    
    # 4. K-평균 클러스터링 수행
    print("\n4단계: K-평균 클러스터링 수행")
    # 실루엣 점수가 가장 높은 K 사용
    optimal_k = optimal_k_results['best_silhouette_k']
    kmeans, cluster_labels, centroids = perform_kmeans_clustering(X_scaled, optimal_k, feature_names)
    
    # 5. 클러스터 분석
    print("\n5단계: 클러스터 특성 분석")
    analyze_clusters(df, cluster_labels, feature_names)
    
    # 6. 결과 시각화
    print("\n6단계: 클러스터링 결과 시각화")
    visualize_clusters(X_scaled, cluster_labels, centroids, feature_names, optimal_k)
    
    # 7. 비즈니스 인사이트
    print("\n7단계: 비즈니스 인사이트 도출")
    business_insights(df, cluster_labels, feature_names)
    
    print("\n" + "=" * 60)
    print("K-평균 클러스터링 실습 완료!")
    print("=" * 60)
    
    return df, cluster_labels, kmeans

if __name__ == "__main__":
    # 실습 실행
    df, cluster_labels, kmeans_model = main()
    
    # 추가 분석을 위한 데이터 저장
    print("\n추가 분석을 위해 결과를 변수에 저장했습니다:")
    print("- df: 원본 고객 데이터")
    print("- cluster_labels: 클러스터 레이블")
    print("- kmeans_model: 훈련된 K-평균 모델")