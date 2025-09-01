#!/usr/bin/env python3
"""
데이터 마이닝 개요 실습

이 실습에서는 다음 내용을 다룹니다:
1. 다양한 속성 유형 (명목형, 순서형, 구간형, 비율형) 예시
2. 레코드, 문서, 거래, 그래프 데이터 샘플 생성 및 탐색
3. 데이터 마이닝의 예측 vs 설명 과제 실습
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("데이터 마이닝 개요 실습")
print("=" * 60)

# 1. 속성 유형별 데이터 예시 생성
print("\n1. 속성 유형별 데이터 예시")
print("-" * 40)

def create_attribute_examples():
    """다양한 속성 유형의 예시 데이터 생성"""
    
    np.random.seed(42)
    n_samples = 100
    
    # 명목형 속성 (Nominal)
    gender = np.random.choice(['Male', 'Female'], n_samples)
    blood_type = np.random.choice(['A', 'B', 'AB', 'O'], n_samples)
    color_preference = np.random.choice(['Red', 'Blue', 'Green', 'Yellow'], n_samples)
    
    # 순서형 속성 (Ordinal)
    education = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples)
    satisfaction = np.random.choice(['Very Unsatisfied', 'Unsatisfied', 'Neutral', 'Satisfied', 'Very Satisfied'], n_samples)
    grade = np.random.choice(['A', 'B', 'C', 'D', 'F'], n_samples)
    
    # 구간형 속성 (Interval)
    temperature_celsius = np.random.normal(25, 10, n_samples)  # 섭씨 온도
    year_born = np.random.randint(1970, 2005, n_samples)  # 출생년도
    iq_score = np.random.normal(100, 15, n_samples)  # IQ 점수
    
    # 비율형 속성 (Ratio)
    age = np.random.randint(18, 80, n_samples)  # 나이
    height = np.random.normal(170, 10, n_samples)  # 키 (cm)
    income = np.random.exponential(50000, n_samples)  # 소득
    
    # 데이터프레임 생성
    df = pd.DataFrame({
        # 명목형
        'Gender': gender,
        'Blood_Type': blood_type,
        'Color_Preference': color_preference,
        
        # 순서형
        'Education': education,
        'Satisfaction': satisfaction,
        'Grade': grade,
        
        # 구간형
        'Temperature_C': temperature_celsius,
        'Birth_Year': year_born,
        'IQ_Score': iq_score,
        
        # 비율형
        'Age': age,
        'Height_cm': height,
        'Income': income
    })
    
    return df

# 속성 예시 데이터 생성
attribute_data = create_attribute_examples()

print("속성 유형별 데이터 샘플:")
print(attribute_data.head(10))

print("\n속성별 데이터 타입 및 통계:")
print(attribute_data.dtypes)
print("\n기술통계:")
print(attribute_data.describe())

# 2. 속성 유형별 분석 및 시각화
print("\n2. 속성 유형별 분석")
print("-" * 40)

def analyze_attribute_types(df):
    """속성 유형별 분석 및 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Attribute Types Analysis', fontsize=16)
    
    # 명목형 속성 분석 - 성별 분포
    gender_counts = df['Gender'].value_counts()
    axes[0, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Nominal: Gender Distribution')
    
    # 순서형 속성 분석 - 교육 수준
    education_order = ['High School', 'Bachelor', 'Master', 'PhD']
    education_counts = df['Education'].value_counts().reindex(education_order)
    axes[0, 1].bar(range(len(education_counts)), education_counts.values)
    axes[0, 1].set_xticks(range(len(education_counts)))
    axes[0, 1].set_xticklabels(education_counts.index, rotation=45)
    axes[0, 1].set_title('Ordinal: Education Level')
    
    # 구간형 속성 분석 - 온도 분포
    axes[0, 2].hist(df['Temperature_C'], bins=20, alpha=0.7)
    axes[0, 2].set_title('Interval: Temperature Distribution')
    axes[0, 2].set_xlabel('Temperature (Celsius)')
    
    # 비율형 속성 분석 - 나이와 소득 관계
    axes[1, 0].scatter(df['Age'], df['Income'], alpha=0.6)
    axes[1, 0].set_title('Ratio: Age vs Income')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Income')
    
    # 이산형 vs 연속형 비교
    axes[1, 1].hist(df['Age'], bins=20, alpha=0.7, label='Discrete (Age)')
    axes[1, 1].set_title('Discrete vs Continuous')
    axes[1, 1].set_xlabel('Age (years)')
    axes[1, 1].legend()
    
    # 속성 간 상관관계 (수치형만)
    numeric_cols = ['Temperature_C', 'Birth_Year', 'IQ_Score', 'Age', 'Height_cm', 'Income']
    corr_matrix = df[numeric_cols].corr()
    im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
    axes[1, 2].set_xticks(range(len(numeric_cols)))
    axes[1, 2].set_yticks(range(len(numeric_cols)))
    axes[1, 2].set_xticklabels([col.replace('_', '\n') for col in numeric_cols], rotation=45)
    axes[1, 2].set_yticklabels([col.replace('_', '\n') for col in numeric_cols])
    axes[1, 2].set_title('Correlation Matrix')
    
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

correlation_matrix = analyze_attribute_types(attribute_data)

# 3. 데이터 집합 유형별 예시
print("\n3. 데이터 집합 유형별 예시")
print("-" * 40)

def create_record_data():
    """레코드 데이터 예시 - 고객 정보"""
    np.random.seed(42)
    n_customers = 50
    
    customers = pd.DataFrame({
        'CustomerID': range(1, n_customers + 1),
        'Age': np.random.randint(18, 80, n_customers),
        'Gender': np.random.choice(['M', 'F'], n_customers),
        'Income': np.random.normal(50000, 20000, n_customers),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_customers),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], n_customers)
    })
    
    return customers

def create_document_data():
    """문서 데이터 예시 - 텍스트 문서"""
    documents = [
        "Machine learning is a subset of artificial intelligence",
        "Data mining discovers patterns in large datasets",
        "Deep learning uses neural networks with multiple layers",
        "Classification predicts categorical target variables",
        "Clustering groups similar data points together",
        "Regression predicts continuous numerical values",
        "Association rules find relationships between items",
        "Dimensionality reduction simplifies high-dimensional data"
    ]
    
    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    
    # 문서-단어 행렬로 변환
    doc_word_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=[f'Doc_{i+1}' for i in range(len(documents))]
    )
    
    return documents, doc_word_df

def create_transaction_data():
    """거래 데이터 예시 - 장바구니 데이터"""
    items = ['Bread', 'Milk', 'Eggs', 'Butter', 'Cheese', 'Yogurt', 'Apples', 'Bananas']
    
    transactions = []
    np.random.seed(42)
    
    for i in range(20):
        # 각 거래에서 1-5개 아이템 선택
        n_items = np.random.randint(1, 6)
        transaction = np.random.choice(items, n_items, replace=False).tolist()
        transactions.append({
            'TransactionID': f'T{i+1:03d}',
            'Items': transaction,
            'ItemCount': len(transaction)
        })
    
    return transactions

def create_graph_data():
    """그래프 데이터 예시 - 소셜 네트워크"""
    # 간단한 소셜 네트워크 그래프 생성
    G = nx.Graph()
    
    # 노드 추가 (사용자)
    users = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry']
    G.add_nodes_from(users)
    
    # 엣지 추가 (친구 관계)
    friendships = [
        ('Alice', 'Bob'), ('Alice', 'Charlie'), ('Bob', 'David'),
        ('Charlie', 'Eve'), ('David', 'Frank'), ('Eve', 'Grace'),
        ('Frank', 'Henry'), ('Grace', 'Alice'), ('Henry', 'Bob')
    ]
    G.add_edges_from(friendships)
    
    return G

# 레코드 데이터
print("3.1 레코드 데이터 (고객 정보)")
record_data = create_record_data()
print(record_data.head())
print(f"데이터 형태: {record_data.shape}")

# 문서 데이터
print("\n3.2 문서 데이터 (TF-IDF 벡터)")
documents, doc_word_matrix = create_document_data()
print("원본 문서:")
for i, doc in enumerate(documents[:3]):
    print(f"  문서 {i+1}: {doc}")

print("\nTF-IDF 행렬 (상위 5개 특성):")
print(doc_word_matrix.iloc[:3, :5])

# 거래 데이터
print("\n3.3 거래 데이터 (장바구니)")
transaction_data = create_transaction_data()
print("거래 예시:")
for trans in transaction_data[:5]:
    print(f"  {trans['TransactionID']}: {trans['Items']}")

# 그래프 데이터
print("\n3.4 그래프 데이터 (소셜 네트워크)")
graph_data = create_graph_data()
print(f"노드 수: {graph_data.number_of_nodes()}")
print(f"엣지 수: {graph_data.number_of_edges()}")
print("연결 관계:")
for edge in list(graph_data.edges())[:5]:
    print(f"  {edge[0]} - {edge[1]}")

# 4. 데이터 집합 유형별 시각화
print("\n4. 데이터 집합 유형별 시각화")
print("-" * 40)

def visualize_data_types():
    """데이터 집합 유형별 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Data Set Types Visualization', fontsize=16)
    
    # 레코드 데이터 - 나이 vs 소득 산점도
    axes[0, 0].scatter(record_data['Age'], record_data['Income'], 
                      c=['red' if g == 'M' else 'blue' for g in record_data['Gender']], alpha=0.6)
    axes[0, 0].set_title('Record Data: Age vs Income')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Income')
    
    # 문서 데이터 - TF-IDF 히트맵
    im1 = axes[0, 1].imshow(doc_word_matrix.iloc[:5, :10].values, cmap='Blues', aspect='auto')
    axes[0, 1].set_title('Document Data: TF-IDF Matrix')
    axes[0, 1].set_xlabel('Words')
    axes[0, 1].set_ylabel('Documents')
    
    # 거래 데이터 - 아이템 빈도
    all_items = []
    for trans in transaction_data:
        all_items.extend(trans['Items'])
    item_counts = Counter(all_items)
    
    items, counts = zip(*item_counts.most_common())
    axes[1, 0].bar(items, counts)
    axes[1, 0].set_title('Transaction Data: Item Frequency')
    axes[1, 0].set_xlabel('Items')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 그래프 데이터 - 네트워크 시각화
    pos = nx.spring_layout(graph_data)
    nx.draw(graph_data, pos, ax=axes[1, 1], with_labels=True, 
            node_color='lightblue', node_size=500, font_size=8)
    axes[1, 1].set_title('Graph Data: Social Network')
    
    plt.tight_layout()
    plt.show()

visualize_data_types()

# 5. 예측 vs 설명 과제 실습
print("\n5. 예측 vs 설명 과제 실습")
print("-" * 40)

def prediction_task_example():
    """예측 과제 예시 - 분류 문제"""
    print("5.1 예측 과제 (Prediction Task)")
    
    # 합성 분류 데이터 생성
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                              n_informative=2, n_clusters_per_class=1, random_state=42)
    
    # 데이터프레임으로 변환
    pred_df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
    pred_df['Target'] = y
    
    print(f"데이터 형태: {pred_df.shape}")
    print("목표: Feature1, Feature2를 사용하여 Target 클래스 예측")
    print(f"클래스 분포: {Counter(y)}")
    
    return pred_df

def description_task_example():
    """설명 과제 예시 - 클러스터링"""
    print("\n5.2 설명 과제 (Description Task)")
    
    # 클러스터링을 위한 데이터 생성
    np.random.seed(42)
    
    # 3개 클러스터 생성
    cluster1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 50)
    cluster2 = np.random.multivariate_normal([6, 6], [[0.5, 0], [0, 0.5]], 50)
    cluster3 = np.random.multivariate_normal([2, 6], [[0.5, 0], [0, 0.5]], 50)
    
    desc_data = np.vstack([cluster1, cluster2, cluster3])
    desc_df = pd.DataFrame(desc_data, columns=['X', 'Y'])
    
    print(f"데이터 형태: {desc_df.shape}")
    print("목표: 데이터의 숨겨진 그룹 구조 발견")
    print("라벨 없이 패턴 탐색")
    
    return desc_df

# 예측 과제 실습
prediction_data = prediction_task_example()

# 설명 과제 실습  
description_data = description_task_example()

# 6. 예측 vs 설명 과제 시각화
print("\n6. 예측 vs 설명 과제 시각화")
print("-" * 40)

def visualize_tasks():
    """예측 vs 설명 과제 시각화"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Prediction vs Description Tasks', fontsize=16)
    
    # 예측 과제 시각화
    colors = ['red' if label == 0 else 'blue' for label in prediction_data['Target']]
    axes[0].scatter(prediction_data['Feature1'], prediction_data['Feature2'], 
                   c=colors, alpha=0.6)
    axes[0].set_title('Prediction Task\n(Supervised Learning)')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].text(0.05, 0.95, 'Goal: Predict class labels\nusing known features', 
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 설명 과제 시각화
    axes[1].scatter(description_data['X'], description_data['Y'], 
                   c='gray', alpha=0.6)
    axes[1].set_title('Description Task\n(Unsupervised Learning)')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].text(0.05, 0.95, 'Goal: Discover hidden\npatterns and structures', 
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

visualize_tasks()

# 7. 데이터 특성 분석
print("\n7. 데이터 특성 분석")
print("-" * 40)

def analyze_data_characteristics():
    """데이터의 주요 특성 분석"""
    
    print("7.1 데이터 규모 (Scale)")
    print(f"  - 레코드 데이터: {record_data.shape[0]} 행, {record_data.shape[1]} 열")
    print(f"  - 문서 데이터: {doc_word_matrix.shape[0]} 문서, {doc_word_matrix.shape[1]} 단어")
    print(f"  - 거래 데이터: {len(transaction_data)} 거래")
    print(f"  - 그래프 데이터: {graph_data.number_of_nodes()} 노드, {graph_data.number_of_edges()} 엣지")
    
    print("\n7.2 차원성 (Dimensionality)")
    high_dim_example = np.random.randn(100, 50)  # 100 샘플, 50 차원
    print(f"  - 고차원 데이터 예시: {high_dim_example.shape}")
    print(f"  - 차원의 저주: 차원이 증가할수록 데이터 밀도 감소")
    
    print("\n7.3 희소성 (Sparsity)")
    sparse_matrix = doc_word_matrix.copy()
    sparsity = (sparse_matrix == 0).sum().sum() / (sparse_matrix.shape[0] * sparse_matrix.shape[1])
    print(f"  - 문서-단어 행렬 희소성: {sparsity:.2%}")
    print(f"  - 대부분의 값이 0인 희소 행렬")
    
    print("\n7.4 이질성 (Heterogeneity)")
    print("  - 수치형 + 범주형 속성 혼재")
    print("  - 다양한 스케일과 분포")
    print("  - 서로 다른 데이터 타입 통합 필요")

analyze_data_characteristics()

# 8. 실습 요약 및 다음 단계
print("\n8. 실습 요약")
print("-" * 40)

print("""
이번 실습에서 학습한 내용:

1. 속성 유형 분류
   - 명목형: 단순 구분 (성별, 혈액형)
   - 순서형: 순위 의미 (교육수준, 만족도)
   - 구간형: 차이 의미, 절대 0점 없음 (온도, 연도)
   - 비율형: 비율 의미, 절대 0점 있음 (나이, 소득)

2. 데이터 집합 유형
   - 레코드: 구조화된 테이블 형태
   - 문서: 텍스트를 벡터로 변환
   - 거래: 트랜잭션과 아이템 집합
   - 그래프: 노드와 엣지의 네트워크

3. 데이터 마이닝 과제
   - 예측: 미래/미지 값 예측 (지도학습)
   - 설명: 숨겨진 패턴 발견 (비지도학습)

4. 데이터 특성
   - 대규모, 고차원, 희소, 이질적, 복잡

다음 단계: 데이터 품질 및 전처리 학습
""")

print("\n실습 완료!")
print("=" * 60)