"""
다변량 회귀 실습

이 실습에서는 Boston Housing 데이터셋을 사용하여 다변량 선형 회귀를 구현하고
회귀 모델의 훈련, 예측, 성능 평가, 파라미터 해석을 실습합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data():
    """
    1. Boston Housing 데이터셋 로딩 및 탐색
    """
    print("=" * 60)
    print("1. Boston Housing 데이터셋 로딩 및 탐색")
    print("=" * 60)
    
    # 데이터 로딩
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = pd.Series(boston.target, name='MEDV')
    
    print("데이터셋 기본 정보:")
    print(f"- 샘플 수: {X.shape[0]}")
    print(f"- 특성 수: {X.shape[1]}")
    print(f"- 타겟 변수: 주택 가격 (단위: $1000)")
    
    print("\n특성 설명:")
    feature_descriptions = {
        'CRIM': '범죄율',
        'ZN': '25,000 sq.ft. 이상 주거지역 비율',
        'INDUS': '비소매 상업지역 비율',
        'CHAS': '찰스강 인접 여부 (1: 인접, 0: 비인접)',
        'NOX': '일산화질소 농도',
        'RM': '평균 방 개수',
        'AGE': '1940년 이전 건축 주택 비율',
        'DIS': '고용센터까지 거리',
        'RAD': '고속도로 접근성',
        'TAX': '재산세율',
        'PTRATIO': '학생-교사 비율',
        'B': '흑인 거주 비율',
        'LSTAT': '저소득층 비율'
    }
    
    for feature, desc in feature_descriptions.items():
        print(f"- {feature}: {desc}")
    
    print(f"\n데이터 미리보기:")
    print(X.head())
    
    print(f"\n타겟 변수 통계:")
    print(y.describe())
    
    return X, y

def visualize_data_distribution(X, y):
    """
    2. 데이터 분포 및 상관관계 시각화
    """
    print("\n" + "=" * 60)
    print("2. 데이터 분포 및 상관관계 시각화")
    print("=" * 60)
    
    # 타겟 변수 분포
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(y, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Target Variable Distribution (MEDV)')
    plt.xlabel('House Price ($1000)')
    plt.ylabel('Frequency')
    
    # 주요 특성들과 타겟 변수의 관계
    important_features = ['RM', 'LSTAT', 'PTRATIO', 'NOX']
    
    for i, feature in enumerate(important_features, 2):
        plt.subplot(2, 3, i)
        plt.scatter(X[feature], y, alpha=0.6, color='coral')
        plt.xlabel(feature)
        plt.ylabel('MEDV')
        plt.title(f'{feature} vs MEDV')
        
        # 상관계수 표시
        corr = np.corrcoef(X[feature], y)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 상관관계 히트맵
    plt.figure(figsize=(12, 10))
    correlation_matrix = pd.concat([X, y], axis=1).corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # 타겟 변수와의 상관관계 순위
    target_corr = correlation_matrix['MEDV'].abs().sort_values(ascending=False)
    print("타겟 변수와의 상관관계 (절댓값 기준):")
    for feature, corr in target_corr.items():
        if feature != 'MEDV':
            print(f"- {feature}: {corr:.3f}")

def preprocess_data(X, y):
    """
    3. 데이터 전처리
    """
    print("\n" + "=" * 60)
    print("3. 데이터 전처리")
    print("=" * 60)
    
    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"훈련 데이터: {X_train.shape[0]} 샘플")
    print(f"테스트 데이터: {X_test.shape[0]} 샘플")
    
    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\n특성 스케일링 완료 (표준화)")
    print("스케일링 전후 비교:")
    print(f"- 원본 데이터 범위: [{X_train.min().min():.2f}, {X_train.max().max():.2f}]")
    print(f"- 스케일링 후 범위: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def train_basic_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    4. 기본 선형 회귀 모델 훈련
    """
    print("\n" + "=" * 60)
    print("4. 기본 선형 회귀 모델 훈련")
    print("=" * 60)
    
    # 모델 생성 및 훈련
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 예측
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # 성능 평가
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    print("기본 선형 회귀 성능:")
    print(f"- 훈련 MSE: {train_mse:.3f}")
    print(f"- 테스트 MSE: {test_mse:.3f}")
    print(f"- 훈련 R²: {train_r2:.3f}")
    print(f"- 테스트 R²: {test_r2:.3f}")
    
    # 과적합 여부 확인
    if train_r2 - test_r2 > 0.1:
        print("⚠️  과적합 가능성이 있습니다.")
    else:
        print("✅ 적절한 일반화 성능을 보입니다.")
    
    return model, y_test_pred

def analyze_coefficients(model, feature_names, scaler):
    """
    5. 회귀 계수 분석 및 해석
    """
    print("\n" + "=" * 60)
    print("5. 회귀 계수 분석 및 해석")
    print("=" * 60)
    
    # 회귀 계수 추출
    coefficients = model.coef_
    intercept = model.intercept_
    
    print(f"절편 (Intercept): {intercept:.3f}")
    print("\n회귀 계수 (Coefficients):")
    
    # 계수를 데이터프레임으로 정리
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(coef_df.to_string(index=False))
    
    # 계수 시각화
    plt.figure(figsize=(12, 8))
    
    # 계수 크기별 정렬
    sorted_idx = np.argsort(np.abs(coefficients))
    
    plt.subplot(1, 2, 1)
    colors = ['red' if c < 0 else 'blue' for c in coefficients[sorted_idx]]
    plt.barh(range(len(coefficients)), coefficients[sorted_idx], color=colors, alpha=0.7)
    plt.yticks(range(len(coefficients)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Coefficient Value')
    plt.title('Regression Coefficients')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # 계수 절댓값
    plt.subplot(1, 2, 2)
    plt.barh(range(len(coefficients)), np.abs(coefficients[sorted_idx]), 
             color='green', alpha=0.7)
    plt.yticks(range(len(coefficients)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('|Coefficient Value|')
    plt.title('Absolute Coefficient Values')
    
    plt.tight_layout()
    plt.show()
    
    # 계수 해석
    print("\n계수 해석:")
    print("양수 계수: 해당 특성이 증가하면 주택 가격이 증가")
    print("음수 계수: 해당 특성이 증가하면 주택 가격이 감소")
    print("절댓값이 클수록 해당 특성의 영향력이 큼")
    
    # 가장 영향력 있는 특성들
    top_features = coef_df.head(5)
    print(f"\n가장 영향력 있는 특성 Top 5:")
    for _, row in top_features.iterrows():
        direction = "증가" if row['Coefficient'] > 0 else "감소"
        print(f"- {row['Feature']}: {row['Coefficient']:.3f} ({direction} 효과)")

def compare_regularization_methods(X_train_scaled, X_test_scaled, y_train, y_test, feature_names):
    """
    6. 정규화 기법 비교 (Ridge, Lasso, Elastic Net)
    """
    print("\n" + "=" * 60)
    print("6. 정규화 기법 비교")
    print("=" * 60)
    
    # 다양한 정규화 모델
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    results = {}
    
    for name, model in models.items():
        # 모델 훈련
        model.fit(X_train_scaled, y_train)
        
        # 예측 및 평가
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 교차검증
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                   cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"{name}:")
        print(f"  - Test MSE: {mse:.3f}")
        print(f"  - Test R²: {r2:.3f}")
        print(f"  - CV R² (mean±std): {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
    
    # 계수 비교 시각화
    plt.figure(figsize=(15, 10))
    
    for i, (name, result) in enumerate(results.items(), 1):
        plt.subplot(2, 2, i)
        model = result['model']
        
        if hasattr(model, 'coef_'):
            coefficients = model.coef_
            plt.barh(range(len(coefficients)), coefficients, alpha=0.7)
            plt.yticks(range(len(coefficients)), feature_names)
            plt.xlabel('Coefficient Value')
            plt.title(f'{name} Coefficients')
            plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    # 성능 비교 표
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Test_MSE': [results[name]['mse'] for name in results.keys()],
        'Test_R2': [results[name]['r2'] for name in results.keys()],
        'CV_R2_Mean': [results[name]['cv_mean'] for name in results.keys()],
        'CV_R2_Std': [results[name]['cv_std'] for name in results.keys()]
    })
    
    print("\n모델 성능 비교:")
    print(comparison_df.to_string(index=False))
    
    return results

def polynomial_features_experiment(X_train, X_test, y_train, y_test):
    """
    7. 다항식 특성 실험
    """
    print("\n" + "=" * 60)
    print("7. 다항식 특성 실험")
    print("=" * 60)
    
    degrees = [1, 2, 3]
    poly_results = {}
    
    for degree in degrees:
        print(f"\n다항식 차수: {degree}")
        
        # 파이프라인 생성 (다항식 특성 + 스케일링 + 회귀)
        pipeline = Pipeline([
            ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        # 훈련
        pipeline.fit(X_train, y_train)
        
        # 예측 및 평가
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        # 특성 개수
        n_features = pipeline.named_steps['poly'].transform(X_train).shape[1]
        
        poly_results[degree] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'n_features': n_features,
            'overfitting': train_r2 - test_r2
        }
        
        print(f"  - 특성 개수: {n_features}")
        print(f"  - 훈련 R²: {train_r2:.3f}")
        print(f"  - 테스트 R²: {test_r2:.3f}")
        print(f"  - 과적합 정도: {train_r2 - test_r2:.3f}")
        
        if train_r2 - test_r2 > 0.2:
            print("  ⚠️  심각한 과적합!")
        elif train_r2 - test_r2 > 0.1:
            print("  ⚠️  과적합 주의")
        else:
            print("  ✅ 적절한 복잡도")
    
    # 결과 시각화
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    degrees_list = list(poly_results.keys())
    train_r2_list = [poly_results[d]['train_r2'] for d in degrees_list]
    test_r2_list = [poly_results[d]['test_r2'] for d in degrees_list]
    
    plt.plot(degrees_list, train_r2_list, 'o-', label='Training R²', color='blue')
    plt.plot(degrees_list, test_r2_list, 'o-', label='Test R²', color='red')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R² Score')
    plt.title('Model Performance vs Polynomial Degree')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    n_features_list = [poly_results[d]['n_features'] for d in degrees_list]
    plt.bar(degrees_list, n_features_list, alpha=0.7, color='green')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Number of Features')
    plt.title('Feature Count vs Polynomial Degree')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return poly_results

def residual_analysis(y_test, y_test_pred):
    """
    8. 잔차 분석
    """
    print("\n" + "=" * 60)
    print("8. 잔차 분석")
    print("=" * 60)
    
    residuals = y_test - y_test_pred
    
    plt.figure(figsize=(15, 10))
    
    # 잔차 vs 예측값
    plt.subplot(2, 3, 1)
    plt.scatter(y_test_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    
    # 잔차 히스토그램
    plt.subplot(2, 3, 2)
    plt.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Distribution')
    
    # Q-Q 플롯 (정규성 검정)
    from scipy import stats
    plt.subplot(2, 3, 3)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot (Normality Test)')
    
    # 실제값 vs 예측값
    plt.subplot(2, 3, 4)
    plt.scatter(y_test, y_test_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
             'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    
    # 잔차의 절댓값
    plt.subplot(2, 3, 5)
    plt.scatter(y_test_pred, np.abs(residuals), alpha=0.6)
    plt.xlabel('Predicted Values')
    plt.ylabel('|Residuals|')
    plt.title('Absolute Residuals vs Predicted')
    
    # 잔차 통계
    plt.subplot(2, 3, 6)
    residual_stats = {
        'Mean': np.mean(residuals),
        'Std': np.std(residuals),
        'Min': np.min(residuals),
        'Max': np.max(residuals),
        'Skewness': stats.skew(residuals),
        'Kurtosis': stats.kurtosis(residuals)
    }
    
    stats_text = '\n'.join([f'{k}: {v:.3f}' for k, v in residual_stats.items()])
    plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    plt.title('Residual Statistics')
    
    plt.tight_layout()
    plt.show()
    
    # 잔차 분석 결과
    print("잔차 분석 결과:")
    print(f"- 잔차 평균: {np.mean(residuals):.3f} (0에 가까워야 함)")
    print(f"- 잔차 표준편차: {np.std(residuals):.3f}")
    print(f"- 잔차 범위: [{np.min(residuals):.3f}, {np.max(residuals):.3f}]")
    
    # 가정 검증
    print("\n선형 회귀 가정 검증:")
    
    # 1. 선형성 (잔차 vs 예측값의 패턴)
    if np.abs(np.corrcoef(y_test_pred, residuals)[0, 1]) < 0.1:
        print("✅ 선형성: 잔차와 예측값 간 상관관계가 낮음")
    else:
        print("⚠️  선형성: 비선형 패턴이 감지됨")
    
    # 2. 정규성 (Shapiro-Wilk 검정)
    _, p_value = stats.shapiro(residuals)
    if p_value > 0.05:
        print("✅ 정규성: 잔차가 정규분포를 따름 (p > 0.05)")
    else:
        print("⚠️  정규성: 잔차가 정규분포를 따르지 않음 (p ≤ 0.05)")
    
    # 3. 등분산성 (Breusch-Pagan 검정 근사)
    residual_variance = np.var(residuals)
    if residual_variance < np.var(y_test) * 0.5:
        print("✅ 등분산성: 잔차의 분산이 적절함")
    else:
        print("⚠️  등분산성: 이분산성 가능성 있음")

def feature_importance_analysis(X, y, feature_names):
    """
    9. 특성 중요도 분석
    """
    print("\n" + "=" * 60)
    print("9. 특성 중요도 분석")
    print("=" * 60)
    
    # 단변량 상관관계
    correlations = []
    for feature in feature_names:
        corr = np.corrcoef(X[feature], y)[0, 1]
        correlations.append(abs(corr))
    
    # 순열 중요도 (간단한 버전)
    from sklearn.model_selection import cross_val_score
    
    # 기준 모델 성능
    base_model = LinearRegression()
    base_scores = cross_val_score(base_model, X, y, cv=5, scoring='r2')
    base_score = base_scores.mean()
    
    permutation_importance = []
    
    for i, feature in enumerate(feature_names):
        # 해당 특성을 셔플
        X_shuffled = X.copy()
        X_shuffled.iloc[:, i] = np.random.permutation(X_shuffled.iloc[:, i])
        
        # 성능 측정
        shuffled_scores = cross_val_score(base_model, X_shuffled, y, cv=5, scoring='r2')
        importance = base_score - shuffled_scores.mean()
        permutation_importance.append(max(0, importance))  # 음수는 0으로
    
    # 결과 정리
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Correlation': correlations,
        'Permutation_Importance': permutation_importance
    })
    
    # 정규화 (0-1 스케일)
    importance_df['Correlation_Norm'] = importance_df['Correlation'] / importance_df['Correlation'].max()
    importance_df['Permutation_Norm'] = importance_df['Permutation_Importance'] / importance_df['Permutation_Importance'].max()
    
    # 종합 점수 (평균)
    importance_df['Combined_Score'] = (importance_df['Correlation_Norm'] + importance_df['Permutation_Norm']) / 2
    
    # 정렬
    importance_df = importance_df.sort_values('Combined_Score', ascending=False)
    
    print("특성 중요도 분석 결과:")
    print(importance_df.to_string(index=False))
    
    # 시각화
    plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.barh(range(len(importance_df)), importance_df['Correlation'], alpha=0.7, color='blue')
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Absolute Correlation')
    plt.title('Feature Importance: Correlation')
    
    plt.subplot(1, 2, 2)
    plt.barh(range(len(importance_df)), importance_df['Permutation_Importance'], alpha=0.7, color='green')
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Permutation Importance')
    plt.title('Feature Importance: Permutation')
    
    plt.tight_layout()
    plt.show()
    
    return importance_df

def main():
    """
    메인 실습 함수
    """
    print("다변량 회귀 실습을 시작합니다!")
    print("Boston Housing 데이터셋을 사용하여 주택 가격을 예측해보겠습니다.")
    
    # 1. 데이터 로딩 및 탐색
    X, y = load_and_explore_data()
    
    # 2. 데이터 시각화
    visualize_data_distribution(X, y)
    
    # 3. 데이터 전처리
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = preprocess_data(X, y)
    
    # 4. 기본 선형 회귀 모델 훈련
    model, y_test_pred = train_basic_regression(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 5. 회귀 계수 분석
    analyze_coefficients(model, X.columns.tolist(), scaler)
    
    # 6. 정규화 기법 비교
    regularization_results = compare_regularization_methods(
        X_train_scaled, X_test_scaled, y_train, y_test, X.columns.tolist()
    )
    
    # 7. 다항식 특성 실험
    poly_results = polynomial_features_experiment(X_train, X_test, y_train, y_test)
    
    # 8. 잔차 분석
    residual_analysis(y_test, y_test_pred)
    
    # 9. 특성 중요도 분석
    importance_df = feature_importance_analysis(X, y, X.columns.tolist())
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("실습 요약")
    print("=" * 60)
    
    print("✅ 완료된 실습 내용:")
    print("1. Boston Housing 데이터셋 탐색")
    print("2. 데이터 전처리 및 시각화")
    print("3. 기본 선형 회귀 모델 구현")
    print("4. 회귀 계수 해석")
    print("5. 정규화 기법 비교 (Ridge, Lasso, Elastic Net)")
    print("6. 다항식 특성을 통한 복잡도 조절")
    print("7. 잔차 분석을 통한 모델 진단")
    print("8. 특성 중요도 분석")
    
    print("\n🎯 핵심 학습 포인트:")
    print("- 다변량 회귀는 여러 특성을 사용한 연속형 예측")
    print("- 회귀 계수를 통해 각 특성의 영향도 해석 가능")
    print("- 정규화 기법으로 과적합 방지")
    print("- 잔차 분석으로 모델의 가정 검증")
    print("- 특성 중요도로 모델 해석력 향상")

if __name__ == "__main__":
    main()