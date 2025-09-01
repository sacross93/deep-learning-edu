#!/usr/bin/env python3
"""
데이터 품질 및 전처리 실습

이 실습에서는 다음 내용을 다룹니다:
1. 잡음 제거, 이상치 탐지, 결측치 처리 실습
2. 다양한 정규화 및 변환 기법 비교
3. 전처리 파이프라인 구축 및 평가
4. 실제 데이터셋을 활용한 종합 전처리
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("데이터 품질 및 전처리 실습")
print("=" * 60)

# 1. 문제가 있는 데이터셋 생성
print("\n1. 문제가 있는 데이터셋 생성")
print("-" * 40)

def create_problematic_dataset():
    """다양한 데이터 품질 문제를 포함한 데이터셋 생성"""
    
    np.random.seed(42)
    n_samples = 1000
    
    # 기본 데이터 생성
    data = {
        'age': np.random.normal(35, 12, n_samples),
        'income': np.random.exponential(50000, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'experience': np.random.normal(10, 5, n_samples),
        'satisfaction': np.random.randint(1, 6, n_samples),
        'city': np.random.choice(['Seoul', 'Busan', 'Incheon', 'Daegu', 'Daejeon'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 1. 결측치 추가 (다양한 패턴)
    # 완전 무작위 결측
    missing_indices = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
    df.loc[missing_indices, 'age'] = np.nan
    
    # 조건부 결측 (고소득자가 소득 정보를 더 자주 누락)
    high_income_mask = df['income'] > df['income'].quantile(0.8)
    high_income_indices = df[high_income_mask].sample(frac=0.3).index
    df.loc[high_income_indices, 'income'] = np.nan
    
    # 2. 이상치 추가
    # 나이 이상치 (음수, 과도하게 큰 값)
    outlier_indices = np.random.choice(n_samples, size=20, replace=False)
    df.loc[outlier_indices[:10], 'age'] = np.random.uniform(-5, 0, 10)  # 음수 나이
    df.loc[outlier_indices[10:], 'age'] = np.random.uniform(150, 200, 10)  # 과도한 나이
    
    # 소득 이상치
    income_outliers = np.random.choice(n_samples, size=15, replace=False)
    df.loc[income_outliers, 'income'] = np.random.uniform(500000, 1000000, 15)
    
    # 3. 잡음 추가
    # 경력에 가우시안 잡음 추가
    noise = np.random.normal(0, 2, n_samples)
    df['experience'] = df['experience'] + noise
    
    # 4. 불일치성 추가
    # 교육 수준 표기 불일치
    inconsistent_edu = np.random.choice(n_samples, size=50, replace=False)
    education_variants = {
        'High School': ['high school', 'HS', 'High_School'],
        'Bachelor': ['bachelor', 'BS', 'Bachelor\'s'],
        'Master': ['master', 'MS', 'Master\'s'],
        'PhD': ['phd', 'Ph.D', 'Doctorate']
    }
    
    for idx in inconsistent_edu:
        original_edu = df.loc[idx, 'education']
        if original_edu in education_variants:
            df.loc[idx, 'education'] = np.random.choice(education_variants[original_edu])
    
    # 5. 중복 데이터 추가
    duplicate_indices = np.random.choice(n_samples, size=30, replace=False)
    duplicates = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    
    return df

# 문제가 있는 데이터셋 생성
problematic_data = create_problematic_dataset()

print("생성된 데이터셋 정보:")
print(f"데이터 형태: {problematic_data.shape}")
print("\n데이터 타입:")
print(problematic_data.dtypes)
print("\n기본 통계:")
print(problematic_data.describe())

# 2. 데이터 품질 문제 탐지 및 시각화
print("\n2. 데이터 품질 문제 탐지")
print("-" * 40)

def analyze_data_quality(df):
    """데이터 품질 문제 분석 및 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Quality Issues Analysis', fontsize=16)
    
    # 2.1 결측치 패턴 분석
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    axes[0, 0].bar(missing_data.index, missing_percent.values)
    axes[0, 0].set_title('Missing Data Percentage by Column')
    axes[0, 0].set_ylabel('Missing Percentage (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2.2 결측치 패턴 히트맵
    missing_matrix = df.isnull().astype(int)
    sns.heatmap(missing_matrix.corr(), annot=True, cmap='coolwarm', 
                center=0, ax=axes[0, 1])
    axes[0, 1].set_title('Missing Data Correlation')
    
    # 2.3 수치형 변수 분포 (이상치 확인)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for i, col in enumerate(numeric_cols[:2]):  # 처음 2개 컬럼만
        if i < 2:
            axes[0, 2].hist(df[col].dropna(), bins=30, alpha=0.7, label=col)
    axes[0, 2].set_title('Distribution of Numeric Variables')
    axes[0, 2].legend()
    
    # 2.4 박스플롯으로 이상치 시각화
    numeric_data = df[numeric_cols].dropna()
    if len(numeric_data.columns) > 0:
        axes[1, 0].boxplot([numeric_data[col].values for col in numeric_data.columns[:3]])
        axes[1, 0].set_xticklabels(numeric_data.columns[:3], rotation=45)
        axes[1, 0].set_title('Outliers Detection (Box Plot)')
    
    # 2.5 범주형 변수 불일치성 확인
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        col = categorical_cols[0]  # 첫 번째 범주형 변수
        value_counts = df[col].value_counts()
        axes[1, 1].bar(range(len(value_counts)), value_counts.values)
        axes[1, 1].set_title(f'Category Distribution: {col}')
        axes[1, 1].set_xticks(range(len(value_counts)))
        axes[1, 1].set_xticklabels(value_counts.index, rotation=45)
    
    # 2.6 중복 데이터 분석
    duplicate_counts = df.duplicated().sum()
    total_rows = len(df)
    duplicate_percent = (duplicate_counts / total_rows) * 100
    
    axes[1, 2].pie([duplicate_percent, 100-duplicate_percent], 
                   labels=['Duplicates', 'Unique'], autopct='%1.1f%%')
    axes[1, 2].set_title(f'Duplicate Data\n({duplicate_counts} duplicates)')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'missing_data': missing_data,
        'duplicates': duplicate_counts,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }

quality_info = analyze_data_quality(problematic_data)

# 3. 결측치 처리 실습
print("\n3. 결측치 처리 실습")
print("-" * 40)

def handle_missing_values(df):
    """다양한 결측치 처리 방법 비교"""
    
    print("3.1 결측치 처리 방법 비교")
    
    # 원본 데이터 복사
    df_original = df.copy()
    
    # 방법 1: 완전 사례 분석 (행 삭제)
    df_complete_case = df.dropna()
    
    # 방법 2: 단순 대체
    df_simple_impute = df.copy()
    
    # 수치형 변수: 평균값으로 대체
    numeric_imputer = SimpleImputer(strategy='mean')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_simple_impute[numeric_cols] = numeric_imputer.fit_transform(df_simple_impute[numeric_cols])
    
    # 범주형 변수: 최빈값으로 대체
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_simple_impute[categorical_cols] = categorical_imputer.fit_transform(df_simple_impute[categorical_cols])
    
    # 방법 3: KNN 대체
    df_knn_impute = df.copy()
    
    # 범주형 변수를 수치형으로 변환 (KNN을 위해)
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_knn_impute[col] = le.fit_transform(df_knn_impute[col].astype(str))
        label_encoders[col] = le
    
    # KNN 대체 적용
    knn_imputer = KNNImputer(n_neighbors=5)
    df_knn_impute_values = knn_imputer.fit_transform(df_knn_impute)
    df_knn_impute = pd.DataFrame(df_knn_impute_values, columns=df_knn_impute.columns)
    
    # 범주형 변수를 다시 원래 형태로 변환
    for col in categorical_cols:
        df_knn_impute[col] = df_knn_impute[col].round().astype(int)
        df_knn_impute[col] = label_encoders[col].inverse_transform(df_knn_impute[col])
    
    # 결과 비교
    results = {
        'Original': df_original,
        'Complete Case': df_complete_case,
        'Simple Imputation': df_simple_impute,
        'KNN Imputation': df_knn_impute
    }
    
    print("결측치 처리 결과 비교:")
    for method, data in results.items():
        missing_count = data.isnull().sum().sum()
        print(f"  {method}: {data.shape[0]} 행, 결측치 {missing_count}개")
    
    return results

missing_results = handle_missing_values(problematic_data)

# 4. 이상치 탐지 및 처리 실습
print("\n4. 이상치 탐지 및 처리 실습")
print("-" * 40)

def detect_and_handle_outliers(df):
    """다양한 이상치 탐지 및 처리 방법"""
    
    print("4.1 이상치 탐지 방법 비교")
    
    # 수치형 데이터만 사용
    numeric_data = df.select_dtypes(include=[np.number]).dropna()
    
    if len(numeric_data) == 0:
        print("수치형 데이터가 없습니다.")
        return {}
    
    # 방법 1: Z-Score 방법
    z_scores = np.abs(stats.zscore(numeric_data))
    z_outliers = (z_scores > 3).any(axis=1)
    
    # 방법 2: IQR 방법
    Q1 = numeric_data.quantile(0.25)
    Q3 = numeric_data.quantile(0.75)
    IQR = Q3 - Q1
    iqr_outliers = ((numeric_data < (Q1 - 1.5 * IQR)) | 
                    (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)
    
    # 방법 3: Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_outliers = iso_forest.fit_predict(numeric_data) == -1
    
    # 방법 4: Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    lof_outliers = lof.fit_predict(numeric_data) == -1
    
    # 결과 비교
    outlier_methods = {
        'Z-Score': z_outliers,
        'IQR': iqr_outliers,
        'Isolation Forest': iso_outliers,
        'LOF': lof_outliers
    }
    
    print("이상치 탐지 결과:")
    for method, outliers in outlier_methods.items():
        count = np.sum(outliers)
        percentage = (count / len(numeric_data)) * 100
        print(f"  {method}: {count}개 ({percentage:.1f}%)")
    
    # 이상치 처리 방법들
    df_processed = {}
    
    # 1. 제거
    df_removed = df[~iqr_outliers].copy()
    
    # 2. 윈저화 (5%, 95% 백분위수로 클리핑)
    df_winsorized = df.copy()
    for col in numeric_data.columns:
        lower_bound = df[col].quantile(0.05)
        upper_bound = df[col].quantile(0.95)
        df_winsorized[col] = df_winsorized[col].clip(lower_bound, upper_bound)
    
    # 3. 로그 변환 (양수 데이터만)
    df_log_transformed = df.copy()
    for col in numeric_data.columns:
        if (df[col] > 0).all():
            df_log_transformed[col] = np.log1p(df[col])
    
    df_processed = {
        'Original': df,
        'Outliers Removed': df_removed,
        'Winsorized': df_winsorized,
        'Log Transformed': df_log_transformed
    }
    
    return outlier_methods, df_processed

outlier_methods, outlier_processed = detect_and_handle_outliers(missing_results['Simple Imputation'])

# 5. 정규화 및 스케일링 비교
print("\n5. 정규화 및 스케일링 비교")
print("-" * 40)

def compare_scaling_methods(df):
    """다양한 스케일링 방법 비교"""
    
    print("5.1 스케일링 방법 비교")
    
    # 수치형 데이터만 선택
    numeric_data = df.select_dtypes(include=[np.number]).dropna()
    
    if len(numeric_data) == 0:
        print("수치형 데이터가 없습니다.")
        return {}
    
    # 다양한 스케일러 적용
    scalers = {
        'Original': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler()
    }
    
    scaled_data = {}
    
    for name, scaler in scalers.items():
        if scaler is None:
            scaled_data[name] = numeric_data
        else:
            scaled_values = scaler.fit_transform(numeric_data)
            scaled_data[name] = pd.DataFrame(scaled_values, 
                                           columns=numeric_data.columns,
                                           index=numeric_data.index)
    
    # 스케일링 결과 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Scaling Methods Comparison', fontsize=16)
    
    axes = axes.ravel()
    
    for i, (method, data) in enumerate(scaled_data.items()):
        if i < 4:  # 최대 4개 방법만 시각화
            # 첫 번째 수치형 컬럼 사용
            col = data.columns[0]
            axes[i].hist(data[col], bins=30, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{method}\nMean: {data[col].mean():.2f}, Std: {data[col].std():.2f}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # 통계 요약
    print("\n스케일링 결과 통계:")
    for method, data in scaled_data.items():
        col = data.columns[0]  # 첫 번째 컬럼
        print(f"{method}:")
        print(f"  평균: {data[col].mean():.3f}, 표준편차: {data[col].std():.3f}")
        print(f"  최솟값: {data[col].min():.3f}, 최댓값: {data[col].max():.3f}")
    
    return scaled_data

scaling_results = compare_scaling_methods(outlier_processed['Winsorized'])

# 6. 범주형 변수 인코딩
print("\n6. 범주형 변수 인코딩")
print("-" * 40)

def encode_categorical_variables(df):
    """범주형 변수 인코딩 방법 비교"""
    
    print("6.1 범주형 변수 인코딩 방법")
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) == 0:
        print("범주형 변수가 없습니다.")
        return {}
    
    encoded_data = {}
    
    # 원본 데이터
    encoded_data['Original'] = df.copy()
    
    # 1. 라벨 인코딩
    df_label = df.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_label[col] = le.fit_transform(df_label[col].astype(str))
        label_encoders[col] = le
    
    encoded_data['Label Encoded'] = df_label
    
    # 2. 원-핫 인코딩
    df_onehot = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    encoded_data['One-Hot Encoded'] = df_onehot
    
    # 결과 비교
    print("인코딩 결과 비교:")
    for method, data in encoded_data.items():
        print(f"  {method}: {data.shape[1]}개 컬럼")
        if method != 'Original':
            print(f"    새로 생성된 컬럼 수: {data.shape[1] - df.shape[1]}")
    
    return encoded_data, label_encoders

encoding_results, label_encoders = encode_categorical_variables(outlier_processed['Winsorized'])

# 7. 전처리 파이프라인 구축
print("\n7. 전처리 파이프라인 구축")
print("-" * 40)

def create_preprocessing_pipeline():
    """완전한 전처리 파이프라인 구축"""
    
    print("7.1 전처리 파이프라인 설계")
    
    # 샘플 데이터 생성 (분류 문제)
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                              n_redundant=2, n_clusters_per_class=1, random_state=42)
    
    # 데이터프레임으로 변환
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # 인위적으로 결측치와 이상치 추가
    np.random.seed(42)
    
    # 결측치 추가 (10%)
    missing_indices = np.random.choice(len(df), size=int(0.1 * len(df)), replace=False)
    missing_cols = np.random.choice(feature_names, size=3, replace=False)
    for col in missing_cols:
        col_missing_indices = np.random.choice(missing_indices, 
                                             size=len(missing_indices)//3, replace=False)
        df.loc[col_missing_indices, col] = np.nan
    
    # 이상치 추가
    outlier_indices = np.random.choice(len(df), size=50, replace=False)
    outlier_cols = np.random.choice(feature_names, size=2, replace=False)
    for col in outlier_cols:
        df.loc[outlier_indices[:25], col] = df[col].mean() + 5 * df[col].std()
    
    # 훈련/테스트 분할
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 전처리 파이프라인 구성
    preprocessing_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),  # 결측치 처리
        ('scaler', RobustScaler())  # 스케일링 (이상치에 강건)
    ])
    
    # 파이프라인 적용
    X_train_processed = preprocessing_pipeline.fit_transform(X_train)
    X_test_processed = preprocessing_pipeline.transform(X_test)
    
    # 결과를 데이터프레임으로 변환
    X_train_processed = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_processed = pd.DataFrame(X_test_processed, columns=feature_names)
    
    print("파이프라인 적용 결과:")
    print(f"  훈련 데이터: {X_train_processed.shape}")
    print(f"  테스트 데이터: {X_test_processed.shape}")
    print(f"  결측치 (훈련): {X_train_processed.isnull().sum().sum()}")
    print(f"  결측치 (테스트): {X_test_processed.isnull().sum().sum()}")
    
    return {
        'pipeline': preprocessing_pipeline,
        'X_train_original': X_train,
        'X_test_original': X_test,
        'X_train_processed': X_train_processed,
        'X_test_processed': X_test_processed,
        'y_train': y_train,
        'y_test': y_test
    }

pipeline_results = create_preprocessing_pipeline()

# 8. 전처리 효과 평가
print("\n8. 전처리 효과 평가")
print("-" * 40)

def evaluate_preprocessing_effects(results):
    """전처리 전후 데이터 품질 비교"""
    
    print("8.1 전처리 전후 비교")
    
    X_train_orig = results['X_train_original']
    X_train_proc = results['X_train_processed']
    
    # 품질 지표 계산
    def calculate_quality_metrics(data, name):
        metrics = {}
        
        # 완전성 (결측치 비율)
        total_values = data.size
        missing_values = data.isnull().sum().sum()
        completeness = ((total_values - missing_values) / total_values) * 100
        
        # 일관성 (표준편차의 변화)
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            avg_std = numeric_data.std().mean()
        else:
            avg_std = 0
        
        # 이상치 비율 (IQR 방법)
        outlier_count = 0
        if len(numeric_data.columns) > 0:
            for col in numeric_data.columns:
                Q1 = numeric_data[col].quantile(0.25)
                Q3 = numeric_data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((numeric_data[col] < (Q1 - 1.5 * IQR)) | 
                           (numeric_data[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_count += outliers
        
        outlier_ratio = (outlier_count / len(data)) * 100
        
        metrics = {
            'completeness': completeness,
            'avg_std': avg_std,
            'outlier_ratio': outlier_ratio
        }
        
        print(f"{name} 품질 지표:")
        print(f"  완전성: {completeness:.1f}%")
        print(f"  평균 표준편차: {avg_std:.3f}")
        print(f"  이상치 비율: {outlier_ratio:.1f}%")
        
        return metrics
    
    orig_metrics = calculate_quality_metrics(X_train_orig, "전처리 전")
    proc_metrics = calculate_quality_metrics(X_train_proc, "전처리 후")
    
    # 시각적 비교
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Preprocessing Effects Comparison', fontsize=16)
    
    # 첫 번째 특성의 분포 비교
    col_name = X_train_orig.columns[0]
    
    # 전처리 전
    axes[0, 0].hist(X_train_orig[col_name].dropna(), bins=30, alpha=0.7, 
                    label='Before', color='red')
    axes[0, 0].set_title('Before Preprocessing')
    axes[0, 0].set_xlabel(col_name)
    axes[0, 0].set_ylabel('Frequency')
    
    # 전처리 후
    axes[0, 1].hist(X_train_proc[col_name], bins=30, alpha=0.7, 
                    label='After', color='blue')
    axes[0, 1].set_title('After Preprocessing')
    axes[0, 1].set_xlabel(col_name)
    axes[0, 1].set_ylabel('Frequency')
    
    # 박스플롯 비교
    data_to_plot = [X_train_orig[col_name].dropna(), X_train_proc[col_name]]
    axes[1, 0].boxplot(data_to_plot, labels=['Before', 'After'])
    axes[1, 0].set_title('Box Plot Comparison')
    axes[1, 0].set_ylabel(col_name)
    
    # 품질 지표 비교
    metrics_names = ['Completeness (%)', 'Avg Std', 'Outlier Ratio (%)']
    before_values = [orig_metrics['completeness'], orig_metrics['avg_std'], 
                    orig_metrics['outlier_ratio']]
    after_values = [proc_metrics['completeness'], proc_metrics['avg_std'], 
                   proc_metrics['outlier_ratio']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, before_values, width, label='Before', alpha=0.7)
    axes[1, 1].bar(x + width/2, after_values, width, label='After', alpha=0.7)
    axes[1, 1].set_xlabel('Quality Metrics')
    axes[1, 1].set_ylabel('Values')
    axes[1, 1].set_title('Quality Metrics Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics_names, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return orig_metrics, proc_metrics

quality_comparison = evaluate_preprocessing_effects(pipeline_results)

# 9. 실제 데이터셋 전처리 예제
print("\n9. 실제 데이터셋 전처리 예제")
print("-" * 40)

def comprehensive_preprocessing_example():
    """종합적인 전처리 예제"""
    
    print("9.1 종합 전처리 파이프라인")
    
    # 복잡한 실제 데이터 시뮬레이션
    np.random.seed(42)
    n_samples = 500
    
    # 다양한 타입의 데이터 생성
    data = {
        # 수치형 변수들
        'age': np.random.normal(40, 15, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples),
        'debt_ratio': np.random.beta(2, 5, n_samples),
        
        # 범주형 변수들
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                    n_samples, p=[0.3, 0.4, 0.2, 0.1]),
        'employment': np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], 
                                     n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], 
                                         n_samples, p=[0.4, 0.5, 0.1]),
        
        # 이진 변수
        'owns_home': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
        'has_car': np.random.choice([0, 1], n_samples, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # 목표 변수 생성 (대출 승인 여부)
    # 소득, 신용점수, 교육수준 등에 기반한 로지스틱 함수
    education_score = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    df['education_score'] = df['education'].map(education_score)
    
    logit = (0.001 * df['income'] + 0.01 * df['credit_score'] + 
             0.5 * df['education_score'] + 2 * df['owns_home'] - 10)
    prob = 1 / (1 + np.exp(-logit))
    df['loan_approved'] = np.random.binomial(1, prob, n_samples)
    df = df.drop('education_score', axis=1)
    
    # 데이터 품질 문제 추가
    # 1. 결측치
    missing_patterns = {
        'income': 0.15,  # 고소득자가 더 자주 누락
        'credit_score': 0.08,
        'debt_ratio': 0.12
    }
    
    for col, missing_rate in missing_patterns.items():
        if col == 'income':
            # 고소득자가 더 자주 소득 정보 누락
            high_income_mask = df[col] > df[col].quantile(0.7)
            missing_candidates = df[high_income_mask].index
            missing_count = int(len(missing_candidates) * missing_rate * 2)
        else:
            missing_count = int(len(df) * missing_rate)
            missing_candidates = df.index
        
        missing_indices = np.random.choice(missing_candidates, 
                                         size=min(missing_count, len(missing_candidates)), 
                                         replace=False)
        df.loc[missing_indices, col] = np.nan
    
    # 2. 이상치
    # 나이 이상치
    age_outliers = np.random.choice(len(df), size=10, replace=False)
    df.loc[age_outliers, 'age'] = np.random.choice([-5, 150], size=10)
    
    # 신용점수 이상치
    credit_outliers = np.random.choice(len(df), size=8, replace=False)
    df.loc[credit_outliers, 'credit_score'] = np.random.uniform(200, 300, size=8)
    
    # 3. 불일치성
    education_variants = {
        'High School': ['high school', 'HS'],
        'Bachelor': ['bachelor', 'BS', 'Bachelor\'s Degree'],
        'Master': ['master', 'MS', 'Master\'s Degree']
    }
    
    inconsistent_indices = np.random.choice(len(df), size=30, replace=False)
    for idx in inconsistent_indices:
        original_edu = df.loc[idx, 'education']
        if original_edu in education_variants:
            df.loc[idx, 'education'] = np.random.choice(education_variants[original_edu])
    
    print("생성된 실제 데이터셋:")
    print(f"  형태: {df.shape}")
    print(f"  결측치: {df.isnull().sum().sum()}개")
    print(f"  목표 변수 분포: {df['loan_approved'].value_counts().to_dict()}")
    
    # 종합 전처리 파이프라인 적용
    def comprehensive_preprocessing(df, target_col):
        """종합적인 전처리 함수"""
        
        # 1. 데이터 분할 (전처리 전에 수행)
        X = df.drop(target_col, axis=1)
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                          stratify=y, random_state=42)
        
        # 2. 범주형 변수 정규화 (불일치성 해결)
        def normalize_categories(series, mapping_dict):
            normalized = series.copy()
            for standard, variants in mapping_dict.items():
                for variant in variants:
                    normalized = normalized.replace(variant, standard)
            return normalized
        
        education_mapping = {
            'High School': ['high school', 'HS'],
            'Bachelor': ['bachelor', 'BS', 'Bachelor\'s Degree'],
            'Master': ['master', 'MS', 'Master\'s Degree']
        }
        
        X_train['education'] = normalize_categories(X_train['education'], education_mapping)
        X_test['education'] = normalize_categories(X_test['education'], education_mapping)
        
        # 3. 이상치 처리 (도메인 지식 기반)
        def handle_domain_outliers(data):
            processed = data.copy()
            
            # 나이: 0-120 범위로 제한
            processed['age'] = processed['age'].clip(0, 120)
            
            # 신용점수: 300-850 범위로 제한
            processed['credit_score'] = processed['credit_score'].clip(300, 850)
            
            # 부채비율: 0-1 범위로 제한
            processed['debt_ratio'] = processed['debt_ratio'].clip(0, 1)
            
            return processed
        
        X_train = handle_domain_outliers(X_train)
        X_test = handle_domain_outliers(X_test)
        
        # 4. 결측치 처리
        # 수치형: KNN 대체
        # 범주형: 최빈값 대체
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        categorical_cols = X_train.select_dtypes(include=['object']).columns
        
        # 범주형 변수 라벨 인코딩 (KNN을 위해)
        label_encoders = {}
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        for col in categorical_cols:
            le = LabelEncoder()
            # 훈련 데이터로만 학습
            X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))
            label_encoders[col] = le
            
            # 테스트 데이터에 적용 (새로운 범주는 -1로 처리)
            test_categories = X_test_encoded[col].astype(str)
            encoded_test = []
            for cat in test_categories:
                if cat in le.classes_:
                    encoded_test.append(le.transform([cat])[0])
                else:
                    encoded_test.append(-1)  # 새로운 범주
            X_test_encoded[col] = encoded_test
        
        # KNN 대체 적용
        knn_imputer = KNNImputer(n_neighbors=5)
        X_train_imputed = knn_imputer.fit_transform(X_train_encoded)
        X_test_imputed = knn_imputer.transform(X_test_encoded)
        
        # 데이터프레임으로 변환
        X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns, 
                                     index=X_train.index)
        X_test_imputed = pd.DataFrame(X_test_imputed, columns=X_test.columns, 
                                    index=X_test.index)
        
        # 범주형 변수를 다시 원래 형태로 변환
        for col in categorical_cols:
            # 정수로 반올림
            X_train_imputed[col] = X_train_imputed[col].round().astype(int)
            X_test_imputed[col] = X_test_imputed[col].round().astype(int)
            
            # 라벨 디코딩
            train_decoded = []
            for val in X_train_imputed[col]:
                if 0 <= val < len(label_encoders[col].classes_):
                    train_decoded.append(label_encoders[col].classes_[val])
                else:
                    train_decoded.append(label_encoders[col].classes_[0])  # 기본값
            X_train_imputed[col] = train_decoded
            
            test_decoded = []
            for val in X_test_imputed[col]:
                if 0 <= val < len(label_encoders[col].classes_):
                    test_decoded.append(label_encoders[col].classes_[val])
                else:
                    test_decoded.append(label_encoders[col].classes_[0])  # 기본값
            X_test_imputed[col] = test_decoded
        
        # 5. 특성 엔지니어링
        # 소득 대비 부채 비율
        X_train_imputed['income_debt_ratio'] = (X_train_imputed['debt_ratio'] * 
                                              X_train_imputed['income'])
        X_test_imputed['income_debt_ratio'] = (X_test_imputed['debt_ratio'] * 
                                             X_test_imputed['income'])
        
        # 나이 그룹
        def age_group(age):
            if age < 30:
                return 'Young'
            elif age < 50:
                return 'Middle'
            else:
                return 'Senior'
        
        X_train_imputed['age_group'] = X_train_imputed['age'].apply(age_group)
        X_test_imputed['age_group'] = X_test_imputed['age'].apply(age_group)
        
        # 6. 최종 인코딩 및 스케일링
        # 범주형 변수 원-핫 인코딩
        categorical_cols_final = X_train_imputed.select_dtypes(include=['object']).columns
        X_train_final = pd.get_dummies(X_train_imputed, columns=categorical_cols_final)
        X_test_final = pd.get_dummies(X_test_imputed, columns=categorical_cols_final)
        
        # 테스트 데이터에 없는 컬럼 추가 (0으로 채움)
        missing_cols = set(X_train_final.columns) - set(X_test_final.columns)
        for col in missing_cols:
            X_test_final[col] = 0
        
        # 컬럼 순서 맞추기
        X_test_final = X_test_final[X_train_final.columns]
        
        # 수치형 변수 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_final)
        X_test_scaled = scaler.transform(X_test_final)
        
        # 최종 데이터프레임
        X_train_final = pd.DataFrame(X_train_scaled, columns=X_train_final.columns, 
                                   index=X_train_final.index)
        X_test_final = pd.DataFrame(X_test_scaled, columns=X_test_final.columns, 
                                  index=X_test_final.index)
        
        return {
            'X_train': X_train_final,
            'X_test': X_test_final,
            'y_train': y_train,
            'y_test': y_test,
            'scaler': scaler,
            'label_encoders': label_encoders
        }
    
    # 전처리 실행
    processed_results = comprehensive_preprocessing(df, 'loan_approved')
    
    print("\n종합 전처리 완료:")
    print(f"  훈련 데이터: {processed_results['X_train'].shape}")
    print(f"  테스트 데이터: {processed_results['X_test'].shape}")
    print(f"  특성 수: {processed_results['X_train'].shape[1]}")
    print(f"  결측치: {processed_results['X_train'].isnull().sum().sum()}")
    
    return df, processed_results

original_data, final_results = comprehensive_preprocessing_example()

# 10. 실습 요약 및 베스트 프랙티스
print("\n10. 실습 요약 및 베스트 프랙티스")
print("-" * 40)

print("""
이번 실습에서 학습한 내용:

1. 데이터 품질 문제 탐지
   - 결측치 패턴 분석 및 시각화
   - 이상치 탐지 (Z-Score, IQR, Isolation Forest, LOF)
   - 불일치성 및 중복 데이터 식별

2. 결측치 처리 방법
   - 완전 사례 분석 (행 삭제)
   - 단순 대체 (평균, 중앙값, 최빈값)
   - 고급 대체 (KNN, 회귀 기반)

3. 이상치 처리 방법
   - 제거, 윈저화, 변환
   - 도메인 지식 기반 처리

4. 정규화 및 스케일링
   - StandardScaler: 평균 0, 표준편차 1
   - MinMaxScaler: 0-1 범위로 변환
   - RobustScaler: 이상치에 강건

5. 범주형 변수 인코딩
   - 라벨 인코딩: 순서형 변수에 적합
   - 원-핫 인코딩: 명목형 변수에 적합

6. 전처리 파이프라인
   - 일관된 처리 보장
   - 데이터 누출 방지
   - 재현 가능한 결과

베스트 프랙티스:
- 항상 데이터 분할 후 전처리 수행
- 도메인 지식을 활용한 의미 있는 전처리
- 전처리 각 단계의 효과 검증
- 파이프라인을 통한 일관된 처리
- 전처리 결과 문서화 및 해석

다음 단계: 유사도와 거리 측정 학습
""")

print("\n실습 완료!")
print("=" * 60)