"""
유사도와 거리 실습

이 실습에서는 다양한 거리 척도와 유사도 측정 방법을 구현하고 비교합니다.
다차원 데이터, 텍스트 데이터, 정보 이론 기반 측정을 다룹니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial.distance import pdist, squareform, euclidean, manhattan, minkowski
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("유사도와 거리 척도 실습")
print("=" * 60)

# 1. 라이브러리 임포트 및 설정
print("\n1. 실습 환경 설정")
print("-" * 30)

def print_section(title):
    """섹션 제목 출력"""
    print(f"\n{title}")
    print("-" * len(title.encode('utf-8')))

# 2. 다차원 데이터 생성 및 탐색
print_section("2. 다차원 데이터 생성")

# 2차원 클러스터 데이터 생성
np.random.seed(42)
X_2d, y_2d = make_blobs(n_samples=100, centers=3, n_features=2, 
                        random_state=42, cluster_std=1.5)

# 고차원 데이터 생성
X_high, y_high = make_blobs(n_samples=100, centers=3, n_features=10, 
                           random_state=42, cluster_std=1.5)

# Iris 데이터셋 로드
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

print(f"2D 데이터 형태: {X_2d.shape}")
print(f"고차원 데이터 형태: {X_high.shape}")
print(f"Iris 데이터 형태: {X_iris.shape}")

# 3. 거리 척도 구현
print_section("3. 거리 척도 구현 및 비교")

def euclidean_distance(x, y):
    """유클리드 거리 계산"""
    return np.sqrt(np.sum((x - y) ** 2))

def manhattan_distance(x, y):
    """맨해튼 거리 계산"""
    return np.sum(np.abs(x - y))

def minkowski_distance(x, y, p):
    """민코프스키 거리 계산"""
    return np.sum(np.abs(x - y) ** p) ** (1/p)

def mahalanobis_distance(x, y, cov_inv):
    """마할라노비스 거리 계산"""
    diff = x - y
    return np.sqrt(diff.T @ cov_inv @ diff)

# 샘플 포인트 선택
point1 = X_2d[0]
point2 = X_2d[1]

print(f"포인트 1: {point1}")
print(f"포인트 2: {point2}")
print()

# 다양한 거리 계산
eucl_dist = euclidean_distance(point1, point2)
manh_dist = manhattan_distance(point1, point2)
mink_dist_3 = minkowski_distance(point1, point2, 3)

print(f"유클리드 거리: {eucl_dist:.4f}")
print(f"맨해튼 거리: {manh_dist:.4f}")
print(f"민코프스키 거리 (p=3): {mink_dist_3:.4f}")

# 마할라노비스 거리 (공분산 행렬 필요)
cov_matrix = np.cov(X_2d.T)
cov_inv = np.linalg.inv(cov_matrix)
mahal_dist = mahalanobis_distance(point1, point2, cov_inv)
print(f"마할라노비스 거리: {mahal_dist:.4f}")

# 4. 거리 척도별 특성 비교
print_section("4. 거리 척도별 특성 비교")

# 다양한 p 값에 대한 민코프스키 거리
p_values = [1, 2, 3, 5, 10]
distances = []

for p in p_values:
    if p == np.inf:
        # 체비셰프 거리 (p=무한대)
        dist = np.max(np.abs(point1 - point2))
    else:
        dist = minkowski_distance(point1, point2, p)
    distances.append(dist)
    print(f"민코프스키 거리 (p={p}): {dist:.4f}")

# 체비셰프 거리 (p=무한대)
chebyshev_dist = np.max(np.abs(point1 - point2))
print(f"체비셰프 거리 (p=∞): {chebyshev_dist:.4f}")

# 5. 유사도 척도 구현
print_section("5. 유사도 척도 구현")

def cosine_similarity(x, y):
    """코사인 유사도 계산"""
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return dot_product / (norm_x * norm_y)

def jaccard_similarity(x, y):
    """자카드 유사도 계산 (이진 벡터)"""
    intersection = np.sum(np.logical_and(x, y))
    union = np.sum(np.logical_or(x, y))
    return intersection / union if union > 0 else 0

def smc_similarity(x, y):
    """단순 매칭 계수 계산 (이진 벡터)"""
    matches = np.sum(x == y)
    return matches / len(x)

# 코사인 유사도 계산
cos_sim = cosine_similarity(point1, point2)
print(f"코사인 유사도: {cos_sim:.4f}")

# 상관계수 계산
corr_coef, _ = pearsonr(point1, point2)
print(f"피어슨 상관계수: {corr_coef:.4f}")

# 이진 데이터 생성 및 유사도 계산
binary1 = np.random.binomial(1, 0.3, 10)
binary2 = np.random.binomial(1, 0.3, 10)

jaccard_sim = jaccard_similarity(binary1, binary2)
smc_sim = smc_similarity(binary1, binary2)

print(f"\n이진 벡터 1: {binary1}")
print(f"이진 벡터 2: {binary2}")
print(f"자카드 유사도: {jaccard_sim:.4f}")
print(f"SMC 유사도: {smc_sim:.4f}")

# 6. 텍스트 데이터에서 코사인 유사도
print_section("6. 텍스트 데이터 코사인 유사도")

# 샘플 문서
documents = [
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks with multiple layers",
    "data mining extracts patterns from large datasets",
    "artificial intelligence includes machine learning and deep learning",
    "neural networks are inspired by biological neurons"
]

# TF-IDF 벡터화
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

print(f"TF-IDF 행렬 형태: {tfidf_matrix.shape}")
print(f"어휘 크기: {len(vectorizer.vocabulary_)}")

# 문서 간 코사인 유사도 계산
def calculate_cosine_similarity_matrix(matrix):
    """TF-IDF 행렬에서 코사인 유사도 행렬 계산"""
    # 정규화된 벡터들의 내적이 코사인 유사도
    normalized_matrix = matrix / np.linalg.norm(matrix.toarray(), axis=1, keepdims=True)
    similarity_matrix = normalized_matrix @ normalized_matrix.T
    return similarity_matrix.toarray()

cosine_sim_matrix = calculate_cosine_similarity_matrix(tfidf_matrix)

print("\n문서 간 코사인 유사도 행렬:")
for i in range(len(documents)):
    for j in range(len(documents)):
        print(f"{cosine_sim_matrix[i,j]:.3f}", end="  ")
    print()

# 가장 유사한 문서 쌍 찾기
max_sim = 0
max_pair = (0, 0)
for i in range(len(documents)):
    for j in range(i+1, len(documents)):
        if cosine_sim_matrix[i,j] > max_sim:
            max_sim = cosine_sim_matrix[i,j]
            max_pair = (i, j)

print(f"\n가장 유사한 문서 쌍:")
print(f"문서 {max_pair[0]}: {documents[max_pair[0]]}")
print(f"문서 {max_pair[1]}: {documents[max_pair[1]]}")
print(f"유사도: {max_sim:.4f}")

# 7. 정보 이론 기반 측정
print_section("7. 정보 이론 기반 측정")

def calculate_entropy(data):
    """엔트로피 계산"""
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    return entropy

def calculate_mutual_information(x, y):
    """상호정보량 계산"""
    # 결합 분포 계산
    xy_pairs = list(zip(x, y))
    unique_pairs, pair_counts = np.unique(xy_pairs, axis=0, return_counts=True)
    
    # 개별 분포 계산
    unique_x, x_counts = np.unique(x, return_counts=True)
    unique_y, y_counts = np.unique(y, return_counts=True)
    
    n = len(x)
    mi = 0
    
    for i, (xi, yi) in enumerate(unique_pairs):
        p_xy = pair_counts[i] / n
        p_x = x_counts[np.where(unique_x == xi)[0][0]] / n
        p_y = y_counts[np.where(unique_y == yi)[0][0]] / n
        
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log2(p_xy / (p_x * p_y))
    
    return mi

# Iris 데이터로 정보 이론 측정
# 연속형 데이터를 범주형으로 변환 (이산화)
def discretize_data(data, bins=3):
    """연속형 데이터를 범주형으로 변환"""
    discretized = []
    for col in range(data.shape[1]):
        _, bin_edges = np.histogram(data[:, col], bins=bins)
        digitized = np.digitize(data[:, col], bin_edges[1:-1])
        discretized.append(digitized)
    return np.array(discretized).T

# Iris 데이터 이산화
X_iris_discrete = discretize_data(X_iris, bins=3)

# 각 특성의 엔트로피 계산
feature_names = iris.feature_names
print("특성별 엔트로피:")
for i, name in enumerate(feature_names):
    entropy = calculate_entropy(X_iris_discrete[:, i])
    print(f"{name}: {entropy:.4f}")

# 타겟 변수의 엔트로피
target_entropy = calculate_entropy(y_iris)
print(f"\n타겟 변수 엔트로피: {target_entropy:.4f}")

# 각 특성과 타겟 간 상호정보량
print("\n특성-타겟 간 상호정보량:")
for i, name in enumerate(feature_names):
    mi = calculate_mutual_information(X_iris_discrete[:, i], y_iris)
    print(f"{name}: {mi:.4f}")

# 8. 차원별 거리 척도 비교
print_section("8. 차원별 거리 척도 성능 비교")

def compare_distances_by_dimension():
    """차원에 따른 거리 척도 성능 비교"""
    dimensions = [2, 5, 10, 20, 50]
    n_samples = 100
    
    results = []
    
    for dim in dimensions:
        # 고차원 데이터 생성
        X, _ = make_blobs(n_samples=n_samples, centers=3, n_features=dim, 
                         random_state=42, cluster_std=1.0)
        
        # 표준화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 첫 번째와 두 번째 포인트 간 거리 계산
        p1, p2 = X_scaled[0], X_scaled[1]
        
        eucl = euclidean_distance(p1, p2)
        manh = manhattan_distance(p1, p2)
        cos_sim = cosine_similarity(p1, p2)
        
        results.append({
            'dimension': dim,
            'euclidean': eucl,
            'manhattan': manh,
            'cosine_similarity': cos_sim
        })
    
    return pd.DataFrame(results)

distance_comparison = compare_distances_by_dimension()
print(distance_comparison)

# 9. 시각화
print_section("9. 결과 시각화")

# 9.1 2D 데이터 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 원본 데이터
axes[0,0].scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap='viridis', alpha=0.7)
axes[0,0].scatter(point1[0], point1[1], c='red', s=100, marker='x', label='Point 1')
axes[0,0].scatter(point2[0], point2[1], c='blue', s=100, marker='x', label='Point 2')
axes[0,0].set_title('2D Data with Sample Points')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 거리 척도 비교
p_vals = [1, 2, 3, 5, 10]
distances_plot = [manhattan_distance(point1, point2), 
                 euclidean_distance(point1, point2),
                 minkowski_distance(point1, point2, 3),
                 minkowski_distance(point1, point2, 5),
                 minkowski_distance(point1, point2, 10)]

axes[0,1].bar(range(len(p_vals)), distances_plot)
axes[0,1].set_xticks(range(len(p_vals)))
axes[0,1].set_xticklabels(['Manhattan\n(p=1)', 'Euclidean\n(p=2)', 'Minkowski\n(p=3)', 
                          'Minkowski\n(p=5)', 'Minkowski\n(p=10)'])
axes[0,1].set_title('Distance Measures Comparison')
axes[0,1].set_ylabel('Distance')

# 문서 유사도 히트맵
im = axes[1,0].imshow(cosine_sim_matrix, cmap='Blues', vmin=0, vmax=1)
axes[1,0].set_title('Document Cosine Similarity Matrix')
axes[1,0].set_xlabel('Document Index')
axes[1,0].set_ylabel('Document Index')
plt.colorbar(im, ax=axes[1,0])

# 차원별 거리 변화
axes[1,1].plot(distance_comparison['dimension'], distance_comparison['euclidean'], 
               'o-', label='Euclidean', linewidth=2)
axes[1,1].plot(distance_comparison['dimension'], distance_comparison['manhattan'], 
               's-', label='Manhattan', linewidth=2)
axes[1,1].plot(distance_comparison['dimension'], distance_comparison['cosine_similarity'], 
               '^-', label='Cosine Similarity', linewidth=2)
axes[1,1].set_xlabel('Dimension')
axes[1,1].set_ylabel('Distance/Similarity')
axes[1,1].set_title('Distance Measures vs Dimension')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 10. 실제 적용 예시
print_section("10. 실제 적용 예시")

def find_similar_samples(X, query_idx, metric='euclidean', k=3):
    """주어진 샘플과 가장 유사한 k개 샘플 찾기"""
    query_sample = X[query_idx]
    distances = []
    
    for i, sample in enumerate(X):
        if i == query_idx:
            continue
            
        if metric == 'euclidean':
            dist = euclidean_distance(query_sample, sample)
        elif metric == 'manhattan':
            dist = manhattan_distance(query_sample, sample)
        elif metric == 'cosine':
            # 코사인 거리 = 1 - 코사인 유사도
            dist = 1 - cosine_similarity(query_sample, sample)
        
        distances.append((i, dist))
    
    # 거리 기준 정렬
    distances.sort(key=lambda x: x[1])
    return distances[:k]

# Iris 데이터에서 유사한 샘플 찾기
query_idx = 0
print(f"쿼리 샘플 (인덱스 {query_idx}): {X_iris[query_idx]}")
print(f"실제 클래스: {iris.target_names[y_iris[query_idx]]}")

for metric in ['euclidean', 'manhattan', 'cosine']:
    print(f"\n{metric.capitalize()} 거리 기준 유사한 샘플들:")
    similar_samples = find_similar_samples(X_iris, query_idx, metric, k=3)
    
    for i, (idx, dist) in enumerate(similar_samples):
        predicted_class = iris.target_names[y_iris[idx]]
        print(f"  {i+1}. 인덱스 {idx}: 거리 {dist:.4f}, 클래스 {predicted_class}")

# 11. 성능 분석 및 권장사항
print_section("11. 성능 분석 및 권장사항")

def analyze_distance_performance():
    """거리 척도별 성능 분석"""
    print("거리 척도별 특성 및 권장 사용 사례:")
    print()
    
    recommendations = {
        "유클리드 거리": {
            "장점": ["직관적", "기하학적 의미 명확", "널리 사용됨"],
            "단점": ["고차원에서 성능 저하", "스케일에 민감", "이상치에 민감"],
            "권장 사용": ["저차원 연속형 데이터", "물리적 거리 측정", "K-평균 클러스터링"]
        },
        "맨해튼 거리": {
            "장점": ["이상치에 강건", "고차원에서 안정적", "계산 효율적"],
            "단점": ["기하학적 직관성 부족", "회전에 민감"],
            "권장 사용": ["고차원 데이터", "이상치가 많은 데이터", "격자 구조 데이터"]
        },
        "코사인 유사도": {
            "장점": ["크기 무관", "고차원 희소 데이터 효과적", "정규화 효과"],
            "단점": ["크기 정보 손실", "음수 해석 주의"],
            "권장 사용": ["텍스트 데이터", "추천 시스템", "고차원 희소 벡터"]
        },
        "마할라노비스 거리": {
            "장점": ["상관관계 고려", "스케일 무관", "통계적 의미"],
            "단점": ["계산 복잡", "충분한 데이터 필요"],
            "권장 사용": ["다변량 이상치 탐지", "상관관계 중요한 데이터"]
        }
    }
    
    for metric, info in recommendations.items():
        print(f"【{metric}】")
        print(f"  장점: {', '.join(info['장점'])}")
        print(f"  단점: {', '.join(info['단점'])}")
        print(f"  권장: {', '.join(info['권장 사용'])}")
        print()

analyze_distance_performance()

# 12. 요약 및 결론
print_section("12. 실습 요약")

print("이번 실습에서 학습한 내용:")
print("1. 다양한 거리 척도 (유클리드, 맨해튼, 민코프스키, 마할라노비스)")
print("2. 유사도 측정 (코사인, 자카드, SMC, 상관계수)")
print("3. 텍스트 데이터에서의 코사인 유사도 적용")
print("4. 정보 이론 기반 측정 (엔트로피, 상호정보량)")
print("5. 차원에 따른 거리 척도 성능 변화")
print("6. 실제 데이터에서의 유사 샘플 검색")
print()
print("핵심 인사이트:")
print("- 데이터 특성에 따른 적절한 거리 척도 선택이 중요")
print("- 고차원에서는 유클리드 거리보다 맨해튼이나 코사인 유사도가 효과적")
print("- 텍스트나 희소 데이터에는 코사인 유사도가 적합")
print("- 상관관계가 중요한 경우 마할라노비스 거리 고려")
print("- 정보 이론 기반 측정은 특징 선택과 분류에 유용")

print("\n" + "=" * 60)
print("실습 완료!")
print("=" * 60)