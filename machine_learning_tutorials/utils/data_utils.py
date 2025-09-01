"""
데이터 유틸리티 모듈

데이터 로딩, 전처리, 분할 및 샘플 데이터셋 생성 기능을 제공합니다.
요구사항 6.1, 7.2를 충족합니다.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.datasets import make_classification, make_regression, make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')


def load_sample_dataset(dataset_name: str, **kwargs) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """
    샘플 데이터셋을 로딩합니다.
    
    Args:
        dataset_name: 데이터셋 이름 ('iris', 'titanic', 'boston', 'wine', etc.)
        **kwargs: 데이터셋별 추가 파라미터
    
    Returns:
        데이터프레임 또는 (특성, 타겟) 튜플
    """
    try:
        if dataset_name.lower() == 'iris':
            from sklearn.datasets import load_iris
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            df['species'] = [data.target_names[i] for i in data.target]
            return df
            
        elif dataset_name.lower() == 'boston':
            # Boston Housing 데이터셋 (다변량 회귀용)
            X, y = make_regression(n_samples=506, n_features=13, noise=0.1, random_state=42)
            feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 
                           'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
            df = pd.DataFrame(X, columns=feature_names)
            df['MEDV'] = y
            return df
            
        elif dataset_name.lower() == 'titanic':
            # Titanic 데이터셋 (로지스틱 회귀용)
            np.random.seed(42)
            n_samples = 891
            
            # 기본 특성 생성
            pclass = np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55])
            sex = np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35])
            age = np.random.normal(29.7, 14.5, n_samples)
            age = np.clip(age, 0.42, 80)
            sibsp = np.random.poisson(0.52, n_samples)
            parch = np.random.poisson(0.38, n_samples)
            fare = np.random.lognormal(3.2, 1.3, n_samples)
            
            # 생존 확률 계산 (현실적인 패턴)
            survival_prob = 0.5
            if sex.any() == 'female':
                survival_prob += 0.3
            survival_prob -= (pclass - 1) * 0.15
            survival_prob += (age < 16) * 0.2
            
            survived = np.random.binomial(1, np.clip(survival_prob, 0, 1), n_samples)
            
            df = pd.DataFrame({
                'PassengerId': range(1, n_samples + 1),
                'Survived': survived,
                'Pclass': pclass,
                'Sex': sex,
                'Age': age,
                'SibSp': sibsp,
                'Parch': parch,
                'Fare': fare
            })
            return df
            
        elif dataset_name.lower() == 'mushroom':
            # Mushroom 데이터셋 (규칙 기반 학습용)
            np.random.seed(42)
            n_samples = 8124
            
            # 범주형 특성들
            cap_shape = np.random.choice(['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'], n_samples)
            cap_surface = np.random.choice(['fibrous', 'grooves', 'scaly', 'smooth'], n_samples)
            cap_color = np.random.choice(['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'], n_samples)
            odor = np.random.choice(['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'], n_samples)
            
            # 독성 여부 (규칙 기반)
            poisonous = np.zeros(n_samples)
            poisonous[(odor == 'foul') | (odor == 'fishy') | (odor == 'pungent')] = 1
            poisonous[(cap_color == 'green') | (cap_color == 'purple')] = 1
            
            df = pd.DataFrame({
                'class': ['p' if p else 'e' for p in poisonous],
                'cap-shape': cap_shape,
                'cap-surface': cap_surface,
                'cap-color': cap_color,
                'odor': odor
            })
            return df
            
        else:
            raise ValueError(f"지원하지 않는 데이터셋: {dataset_name}")
            
    except Exception as e:
        print(f"데이터셋 로딩 중 오류 발생: {e}")
        return generate_synthetic_data('classification')


def generate_synthetic_data(data_type: str, n_samples: int = 1000, **kwargs) -> pd.DataFrame:
    """
    합성 데이터를 생성합니다.
    
    Args:
        data_type: 데이터 타입 ('classification', 'regression', 'clustering')
        n_samples: 샘플 수
        **kwargs: 추가 파라미터
    
    Returns:
        생성된 데이터프레임
    """
    np.random.seed(kwargs.get('random_state', 42))
    
    if data_type == 'classification':
        n_features = kwargs.get('n_features', 4)
        n_classes = kwargs.get('n_classes', 3)
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_classes=n_classes,
            n_redundant=0,
            n_informative=n_features,
            random_state=kwargs.get('random_state', 42)
        )
        
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        return df
        
    elif data_type == 'regression':
        n_features = kwargs.get('n_features', 5)
        
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=kwargs.get('noise', 0.1),
            random_state=kwargs.get('random_state', 42)
        )
        
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        return df
        
    elif data_type == 'clustering':
        n_centers = kwargs.get('n_centers', 3)
        n_features = kwargs.get('n_features', 2)
        
        X, y = make_blobs(
            n_samples=n_samples,
            centers=n_centers,
            n_features=n_features,
            random_state=kwargs.get('random_state', 42)
        )
        
        feature_names = [f'feature_{i+1}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['cluster'] = y
        return df
        
    else:
        raise ValueError(f"지원하지 않는 데이터 타입: {data_type}")


def preprocess_data(df: pd.DataFrame, target_column: str = None, 
                   scaling_method: str = 'standard', 
                   handle_missing: str = 'drop') -> pd.DataFrame:
    """
    데이터 전처리를 수행합니다.
    
    Args:
        df: 입력 데이터프레임
        target_column: 타겟 컬럼명
        scaling_method: 스케일링 방법 ('standard', 'minmax', 'none')
        handle_missing: 결측치 처리 방법 ('drop', 'fill_mean', 'fill_median')
    
    Returns:
        전처리된 데이터프레임
    """
    df_processed = df.copy()
    
    # 결측치 처리
    if handle_missing == 'drop':
        df_processed = df_processed.dropna()
    elif handle_missing == 'fill_mean':
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(
            df_processed[numeric_columns].mean()
        )
    elif handle_missing == 'fill_median':
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_columns] = df_processed[numeric_columns].fillna(
            df_processed[numeric_columns].median()
        )
    
    # 범주형 변수 인코딩
    categorical_columns = df_processed.select_dtypes(include=['object']).columns
    if target_column and target_column in categorical_columns:
        categorical_columns = categorical_columns.drop(target_column)
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # 수치형 변수 스케일링
    if scaling_method != 'none':
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        if target_column and target_column in numeric_columns:
            numeric_columns = numeric_columns.drop(target_column)
        
        if len(numeric_columns) > 0:
            if scaling_method == 'standard':
                scaler = StandardScaler()
            elif scaling_method == 'minmax':
                scaler = MinMaxScaler()
            
            df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
    
    return df_processed


def split_data(df: pd.DataFrame, target_column: str, 
               test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    데이터를 훈련/테스트 세트로 분할합니다.
    
    Args:
        df: 입력 데이터프레임
        target_column: 타겟 컬럼명
        test_size: 테스트 세트 비율
        random_state: 랜덤 시드
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) < 20 else None)


def explore_data(df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
    """
    데이터 탐색 정보를 제공합니다.
    
    Args:
        df: 입력 데이터프레임
        target_column: 타겟 컬럼명
    
    Returns:
        탐색 결과 딕셔너리
    """
    exploration_results = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        'categorical_summary': {}
    }
    
    # 범주형 변수 요약
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        exploration_results['categorical_summary'][col] = df[col].value_counts().to_dict()
    
    # 타겟 변수 분포
    if target_column and target_column in df.columns:
        exploration_results['target_distribution'] = df[target_column].value_counts().to_dict()
    
    return exploration_results


def visualize_data_distribution(df: pd.DataFrame, target_column: str = None, figsize: Tuple[int, int] = (15, 10)):
    """
    데이터 분포를 시각화합니다.
    
    Args:
        df: 입력 데이터프레임
        target_column: 타겟 컬럼명
        figsize: 그래프 크기
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if target_column and target_column in numeric_columns:
        numeric_columns = numeric_columns.drop(target_column)
    
    n_numeric = len(numeric_columns)
    
    if n_numeric > 0:
        # 수치형 변수 히스토그램
        n_cols = min(4, n_numeric)
        n_rows = (n_numeric + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_columns):
            if i < len(axes):
                axes[i].hist(df[col], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{col} 분포')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('빈도')
        
        # 빈 서브플롯 제거
        for i in range(n_numeric, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
    
    # 타겟 변수 분포
    if target_column and target_column in df.columns:
        plt.figure(figsize=(8, 6))
        if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() < 20:
            df[target_column].value_counts().plot(kind='bar')
            plt.title(f'{target_column} 분포')
            plt.xlabel(target_column)
            plt.ylabel('빈도')
        else:
            plt.hist(df[target_column], bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'{target_column} 분포')
            plt.xlabel(target_column)
            plt.ylabel('빈도')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def create_correlation_matrix(df: pd.DataFrame, target_column: str = None, figsize: Tuple[int, int] = (10, 8)):
    """
    상관관계 매트릭스를 생성하고 시각화합니다.
    
    Args:
        df: 입력 데이터프레임
        target_column: 타겟 컬럼명
        figsize: 그래프 크기
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        correlation_matrix = numeric_df.corr()
        
        plt.figure(figsize=figsize)
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5)
        plt.title('특성 간 상관관계 매트릭스')
        plt.tight_layout()
        plt.show()
        
        # 타겟 변수와의 상관관계
        if target_column and target_column in correlation_matrix.columns:
            target_corr = correlation_matrix[target_column].drop(target_column).sort_values(key=abs, ascending=False)
            
            plt.figure(figsize=(10, 6))
            target_corr.plot(kind='barh')
            plt.title(f'{target_column}과의 상관관계')
            plt.xlabel('상관계수')
            plt.tight_layout()
            plt.show()
            
            return correlation_matrix
    else:
        print("상관관계 분석을 위한 충분한 수치형 변수가 없습니다.")
        return None


# 데이터 품질 검사 함수들
def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> Dict[str, np.ndarray]:
    """
    이상치를 탐지합니다.
    
    Args:
        df: 입력 데이터프레임
        method: 탐지 방법 ('iqr', 'zscore')
        threshold: 임계값
    
    Returns:
        컬럼별 이상치 인덱스 딕셔너리
    """
    outliers = {}
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.values
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = df[z_scores > threshold].index.values
    
    return outliers


def handle_outliers(df: pd.DataFrame, method: str = 'remove', outlier_indices: Dict[str, np.ndarray] = None) -> pd.DataFrame:
    """
    이상치를 처리합니다.
    
    Args:
        df: 입력 데이터프레임
        method: 처리 방법 ('remove', 'cap', 'transform')
        outlier_indices: 이상치 인덱스 딕셔너리
    
    Returns:
        처리된 데이터프레임
    """
    df_processed = df.copy()
    
    if outlier_indices is None:
        outlier_indices = detect_outliers(df)
    
    if method == 'remove':
        # 모든 컬럼의 이상치 인덱스 합집합
        all_outlier_indices = set()
        for indices in outlier_indices.values():
            all_outlier_indices.update(indices)
        df_processed = df_processed.drop(index=list(all_outlier_indices))
        
    elif method == 'cap':
        # 이상치를 상한/하한값으로 제한
        for col, indices in outlier_indices.items():
            if len(indices) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_processed