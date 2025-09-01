"""
다변량 회귀 퀴즈

이 퀴즈는 다변량 선형 회귀의 이론적 개념, 회귀 계수 해석, 
성능 지표 계산 및 해석에 대한 이해도를 평가합니다.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys
import os

# 상위 디렉토리의 utils 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.quiz_utils import QuizManager

def create_quiz_questions():
    """
    다변량 회귀 퀴즈 문제 생성
    """
    questions = []
    
    # 1. 개념 이해 문제 (객관식)
    questions.append({
        'id': 1,
        'type': 'multiple_choice',
        'question': '''다변량 선형 회귀의 수학적 표현 y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε에서 
각 기호의 의미로 올바른 것은?''',
        'options': [
            'A) β₀: 기울기, βᵢ: 절편, ε: 예측값',
            'B) β₀: 절편, βᵢ: 회귀계수, ε: 오차항',
            'C) β₀: 오차항, βᵢ: 특성값, ε: 절편',
            'D) β₀: 예측값, βᵢ: 실제값, ε: 차이'
        ],
        'correct': 'B',
        'explanation': '''β₀는 절편(intercept)으로 모든 특성이 0일 때의 예측값입니다. 
βᵢ는 각 특성 xᵢ의 회귀계수(가중치)이며, ε는 오차항(잔차)입니다.'''
    })
    
    questions.append({
        'id': 2,
        'type': 'multiple_choice',
        'question': '''최소제곱법(OLS)의 목적 함수는 무엇을 최소화하는 것인가?''',
        'options': [
            'A) 평균 절대 오차 (MAE)',
            'B) 잔차 제곱합 (RSS)',
            'C) 로그 우도 (Log-likelihood)',
            'D) 교차 엔트로피 (Cross-entropy)'
        ],
        'correct': 'B',
        'explanation': '''최소제곱법은 잔차 제곱합(RSS = Σ(yᵢ - ŷᵢ)²)을 최소화하여 
최적의 회귀 계수를 찾는 방법입니다.'''
    })
    
    questions.append({
        'id': 3,
        'type': 'multiple_choice',
        'question': '''다변량 회귀에서 정규방정식 β̂ = (XᵀX)⁻¹Xᵀy가 사용되지 못하는 경우는?''',
        'options': [
            'A) 데이터에 결측값이 있을 때',
            'B) XᵀX가 특이행렬(singular matrix)일 때',
            'C) 타겟 변수가 연속형이 아닐 때',
            'D) 특성의 개수가 너무 적을 때'
        ],
        'correct': 'B',
        'explanation': '''XᵀX가 특이행렬(역행렬이 존재하지 않음)일 때는 정규방정식을 사용할 수 없습니다. 
이는 주로 다중공선성이나 특성 수가 샘플 수보다 많을 때 발생합니다.'''
    })
    
    questions.append({
        'id': 4,
        'type': 'multiple_choice',
        'question': '''Ridge 회귀와 Lasso 회귀의 주요 차이점은?''',
        'options': [
            'A) Ridge는 L1 정규화, Lasso는 L2 정규화를 사용',
            'B) Ridge는 L2 정규화, Lasso는 L1 정규화를 사용',
            'C) Ridge는 분류용, Lasso는 회귀용',
            'D) Ridge는 선형, Lasso는 비선형 모델'
        ],
        'correct': 'B',
        'explanation': '''Ridge 회귀는 L2 정규화(‖β‖₂²)를 사용하여 계수 크기를 제한하고, 
Lasso 회귀는 L1 정규화(‖β‖₁)를 사용하여 특성 선택 효과를 가집니다.'''
    })
    
    # 2. 계수 해석 문제 (주관식)
    questions.append({
        'id': 5,
        'type': 'numerical',
        'question': '''주택 가격 예측 모델에서 다음과 같은 회귀 계수를 얻었습니다:
- 절편: 25.5
- 방 개수(RM): 3.8
- 범죄율(CRIM): -0.2
- 찰스강 인접(CHAS): 4.1

방 개수가 1개 증가하고 다른 조건이 동일할 때, 주택 가격은 얼마나 변화하는가? (소수점 첫째자리까지)''',
        'correct': 3.8,
        'tolerance': 0.1,
        'explanation': '''방 개수(RM)의 회귀 계수가 3.8이므로, 방 개수가 1개 증가하면 
주택 가격이 3.8 단위 증가합니다.'''
    })
    
    questions.append({
        'id': 6,
        'type': 'multiple_choice',
        'question': '''위 문제의 회귀 계수 해석에서 범죄율(CRIM)의 계수가 -0.2라는 것은?''',
        'options': [
            'A) 범죄율이 1% 증가하면 주택 가격이 0.2 단위 감소',
            'B) 범죄율이 1 단위 증가하면 주택 가격이 0.2 단위 감소',
            'C) 범죄율과 주택 가격은 양의 상관관계',
            'D) 범죄율은 주택 가격에 영향을 주지 않음'
        ],
        'correct': 'B',
        'explanation': '''음수 계수(-0.2)는 범죄율이 1 단위 증가할 때마다 주택 가격이 0.2 단위 감소함을 의미합니다. 
이는 범죄율과 주택 가격 간의 음의 관계를 나타냅니다.'''
    })
    
    # 3. 성능 지표 계산 문제
    questions.append({
        'id': 7,
        'type': 'numerical',
        'question': '''다음 예측 결과에서 MSE를 계산하세요:
실제값: [10, 15, 20, 25, 30]
예측값: [12, 13, 22, 23, 28]
MSE = ? (소수점 둘째자리까지)''',
        'correct': 6.00,
        'tolerance': 0.01,
        'explanation': '''MSE = (1/n)Σ(yᵢ - ŷᵢ)²
= (1/5)[(10-12)² + (15-13)² + (20-22)² + (25-23)² + (30-28)²]
= (1/5)[4 + 4 + 4 + 4 + 4] = 20/5 = 4.00
실제 계산: (1/5)[4 + 4 + 4 + 4 + 4] = 4.00'''
    })
    
    questions.append({
        'id': 8,
        'type': 'numerical',
        'question': '''위 문제에서 RMSE를 계산하세요. (소수점 둘째자리까지)''',
        'correct': 2.00,
        'tolerance': 0.01,
        'explanation': '''RMSE = √MSE = √4.00 = 2.00'''
    })
    
    questions.append({
        'id': 9,
        'type': 'multiple_choice',
        'question': '''R² (결정계수)가 0.85라는 것은 무엇을 의미하는가?''',
        'options': [
            'A) 모델의 정확도가 85%',
            'B) 모델이 타겟 변수 분산의 85%를 설명',
            'C) 예측 오차가 15%',
            'D) 85%의 확률로 올바른 예측'
        ],
        'correct': 'B',
        'explanation': '''R²는 모델이 설명하는 분산의 비율을 나타냅니다. 
R² = 0.85는 모델이 타겟 변수 총 분산의 85%를 설명한다는 의미입니다.'''
    })
    
    # 4. 실제 적용 시나리오 문제
    questions.append({
        'id': 10,
        'type': 'multiple_choice',
        'question': '''다음 상황에서 다변량 회귀 모델의 성능을 개선하기 위한 가장 적절한 방법은?
상황: 훈련 R² = 0.95, 테스트 R² = 0.65''',
        'options': [
            'A) 더 많은 특성 추가',
            'B) 정규화 기법 적용 (Ridge/Lasso)',
            'C) 학습률 증가',
            'D) 더 복잡한 모델 사용'
        ],
        'correct': 'B',
        'explanation': '''훈련 성능과 테스트 성능의 큰 차이(0.95 vs 0.65)는 과적합을 나타냅니다. 
정규화 기법(Ridge/Lasso)을 적용하여 모델 복잡도를 제어해야 합니다.'''
    })
    
    questions.append({
        'id': 11,
        'type': 'multiple_choice',
        'question': '''다중공선성(multicollinearity) 문제가 발생했을 때의 해결 방법이 아닌 것은?''',
        'options': [
            'A) 상관관계가 높은 특성 중 일부 제거',
            'B) 주성분 분석(PCA) 적용',
            'C) Ridge 회귀 사용',
            'D) 학습 데이터 양 증가'
        ],
        'correct': 'D',
        'explanation': '''다중공선성은 특성들 간의 강한 상관관계로 인한 문제입니다. 
학습 데이터 양을 늘리는 것은 다중공선성 자체를 해결하지 못합니다.'''
    })
    
    # 5. 고급 개념 문제
    questions.append({
        'id': 12,
        'type': 'multiple_choice',
        'question': '''잔차 분석에서 잔차 vs 예측값 그래프가 깔때기 모양을 보인다면?''',
        'options': [
            'A) 선형성 가정 위반',
            'B) 정규성 가정 위반',
            'C) 등분산성 가정 위반 (이분산성)',
            'D) 독립성 가정 위반'
        ],
        'correct': 'C',
        'explanation': '''깔때기 모양의 잔차 패턴은 예측값에 따라 잔차의 분산이 달라지는 
이분산성(heteroscedasticity)을 나타내며, 등분산성 가정 위반입니다.'''
    })
    
    questions.append({
        'id': 13,
        'type': 'multiple_choice',
        'question': '''특성 스케일링이 다변량 회귀에서 중요한 이유는?''',
        'options': [
            'A) 모델의 정확도를 높이기 위해',
            'B) 계산 속도를 향상시키기 위해',
            'C) 회귀 계수의 크기를 비교 가능하게 하기 위해',
            'D) 과적합을 방지하기 위해'
        ],
        'correct': 'C',
        'explanation': '''특성들의 단위와 스케일이 다르면 회귀 계수의 크기만으로는 
특성의 중요도를 비교할 수 없습니다. 스케일링을 통해 공정한 비교가 가능합니다.'''
    })
    
    # 6. 계산 문제
    questions.append({
        'id': 14,
        'type': 'numerical',
        'question': '''조정된 R² (Adjusted R²)를 계산하세요:
R² = 0.80, n = 100 (샘플 수), p = 5 (특성 수)
공식: R²_adj = 1 - [(1-R²)(n-1)/(n-p-1)]
결과를 소수점 셋째자리까지 입력하세요.''',
        'correct': 0.789,
        'tolerance': 0.001,
        'explanation': '''R²_adj = 1 - [(1-0.80)(100-1)/(100-5-1)]
= 1 - [0.20 × 99/94]
= 1 - [0.20 × 1.053]
= 1 - 0.211 = 0.789'''
    })
    
    questions.append({
        'id': 15,
        'type': 'multiple_choice',
        'question': '''교차검증(Cross-Validation)을 사용하는 주요 목적은?''',
        'options': [
            'A) 모델 훈련 속도 향상',
            'B) 모델 성능의 안정적 평가',
            'C) 특성 개수 감소',
            'D) 데이터 전처리 자동화'
        ],
        'correct': 'B',
        'explanation': '''교차검증은 제한된 데이터에서 모델 성능을 보다 안정적이고 
신뢰할 수 있게 평가하기 위한 방법입니다.'''
    })
    
    return questions

def run_interactive_quiz():
    """
    대화형 퀴즈 실행
    """
    print("=" * 60)
    print("다변량 회귀 퀴즈")
    print("=" * 60)
    print("다변량 선형 회귀의 개념, 계수 해석, 성능 평가에 대한 이해도를 확인해보세요!")
    print("총 15문제로 구성되어 있습니다.")
    print("=" * 60)
    
    questions = create_quiz_questions()
    quiz_manager = QuizManager()
    
    return quiz_manager.run_quiz(questions, "다변량 회귀")

def create_practice_dataset():
    """
    퀴즈용 연습 데이터셋 생성
    """
    print("\n" + "=" * 50)
    print("보너스: 실제 데이터로 계산 연습")
    print("=" * 50)
    
    # Boston Housing 데이터 로드
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = pd.Series(boston.target)
    
    # 간단한 모델 훈련
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 훈련
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 예측 및 평가
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"실제 Boston Housing 데이터 결과:")
    print(f"- 테스트 MSE: {mse:.3f}")
    print(f"- 테스트 R²: {r2:.3f}")
    print(f"- 테스트 RMSE: {np.sqrt(mse):.3f}")
    
    # 주요 특성의 계수
    feature_names = X.columns
    coefficients = model.coef_
    
    print(f"\n주요 특성의 회귀 계수:")
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print(coef_df.head(5).to_string(index=False))
    
    return model, X_test, y_test, y_pred

def show_quiz_summary():
    """
    퀴즈 요약 및 학습 포인트
    """
    print("\n" + "=" * 60)
    print("퀴즈 완료! 핵심 학습 포인트 요약")
    print("=" * 60)
    
    learning_points = [
        "1. 다변량 회귀 기본 개념",
        "   - 수학적 표현: y = β₀ + β₁x₁ + ... + βₚxₚ + ε",
        "   - 최소제곱법으로 최적 계수 찾기",
        "",
        "2. 회귀 계수 해석",
        "   - βᵢ: 다른 조건이 동일할 때 xᵢ의 단위 변화에 따른 y의 변화",
        "   - 양수: 양의 관계, 음수: 음의 관계",
        "",
        "3. 성능 평가 지표",
        "   - MSE: 평균 제곱 오차 (작을수록 좋음)",
        "   - RMSE: 평균 제곱근 오차 (해석 용이)",
        "   - R²: 설명 분산 비율 (0~1, 클수록 좋음)",
        "",
        "4. 정규화 기법",
        "   - Ridge: L2 정규화, 계수 크기 제한",
        "   - Lasso: L1 정규화, 특성 선택 효과",
        "",
        "5. 모델 진단",
        "   - 잔차 분석으로 가정 검증",
        "   - 과적합 탐지 및 해결",
        "   - 다중공선성 문제 해결"
    ]
    
    for point in learning_points:
        print(point)
    
    print("\n" + "=" * 60)
    print("수고하셨습니다! 다변량 회귀에 대한 이해가 깊어졌기를 바랍니다.")
    print("=" * 60)

def main():
    """
    메인 퀴즈 실행 함수
    """
    try:
        # 대화형 퀴즈 실행
        quiz_results = run_interactive_quiz()
        
        # 실제 데이터 연습
        create_practice_dataset()
        
        # 학습 포인트 요약
        show_quiz_summary()
        
        return quiz_results
        
    except KeyboardInterrupt:
        print("\n\n퀴즈가 중단되었습니다.")
        return None
    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")
        return None

if __name__ == "__main__":
    main()