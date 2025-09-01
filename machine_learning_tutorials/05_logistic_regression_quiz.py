"""
로지스틱 회귀 퀴즈
확률 해석 및 결정 경계 이해, 분류 성능 지표 계산 문제

이 퀴즈는 다음 내용을 다룹니다:
1. 로지스틱 회귀 기본 개념
2. 시그모이드 함수와 확률 해석
3. 회귀 계수 해석
4. 분류 성능 지표 계산
5. 실제 적용 시나리오 분석
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import sys
import os

# 상위 디렉토리의 utils 모듈 import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.quiz_utils import QuizManager

def sigmoid(z):
    """시그모이드 함수"""
    return 1 / (1 + np.exp(-z))

def main():
    quiz = QuizManager("로지스틱 회귀")
    
    print("=" * 60)
    print("로지스틱 회귀 퀴즈")
    print("=" * 60)
    print("로지스틱 회귀의 개념과 적용에 대한 이해도를 확인합니다.")
    print("각 문제를 신중히 읽고 답변해주세요.\n")
    
    # 문제 1: 기본 개념
    quiz.add_question(
        "1. 로지스틱 회귀에 대한 설명으로 올바른 것은?",
        "multiple_choice",
        [
            "연속적인 수치를 예측하는 회귀 알고리즘이다",
            "범주형 변수를 예측하는 분류 알고리즘이다", 
            "비지도 학습 알고리즘이다",
            "클러스터링에 사용되는 알고리즘이다"
        ],
        1,
        "로지스틱 회귀는 시그모이드 함수를 사용하여 0과 1 사이의 확률을 출력하는 분류 알고리즘입니다."
    )
    
    # 문제 2: 시그모이드 함수
    quiz.add_question(
        "2. 시그모이드 함수 σ(z) = 1/(1+e^(-z))에서 z=0일 때의 출력값은?",
        "multiple_choice",
        ["0", "0.5", "1", "무한대"],
        1,
        "z=0일 때: σ(0) = 1/(1+e^0) = 1/(1+1) = 0.5입니다."
    )
    
    # 문제 3: 확률 해석
    quiz.add_question(
        "3. 로지스틱 회귀에서 출력값 0.7의 의미는?",
        "multiple_choice",
        [
            "클래스 0에 속할 확률이 70%",
            "클래스 1에 속할 확률이 70%",
            "예측 오차가 70%",
            "모델 정확도가 70%"
        ],
        1,
        "로지스틱 회귀의 출력값은 양성 클래스(클래스 1)에 속할 확률을 나타냅니다."
    )
    
    # 문제 4: 계산 문제
    print("\n" + "="*50)
    print("계산 문제")
    print("="*50)
    
    # 시그모이드 함수 계산
    z_value = 2.0
    expected_sigmoid = sigmoid(z_value)
    
    quiz.add_question(
        f"4. z = {z_value}일 때 시그모이드 함수 σ(z)의 값을 소수점 셋째 자리까지 계산하세요.",
        "numeric",
        expected_sigmoid,
        None,
        f"σ({z_value}) = 1/(1+e^(-{z_value})) = {expected_sigmoid:.3f}"
    )
    
    # 문제 5: 회귀 계수 해석
    quiz.add_question(
        "5. 로지스틱 회귀에서 특성의 회귀 계수가 양수일 때의 의미는?",
        "multiple_choice",
        [
            "해당 특성이 증가하면 양성 클래스 확률이 감소한다",
            "해당 특성이 증가하면 양성 클래스 확률이 증가한다",
            "해당 특성은 예측에 영향을 주지 않는다",
            "해당 특성은 잡음이다"
        ],
        1,
        "양수 계수는 해당 특성이 증가할 때 로그 오즈가 증가하므로 양성 클래스 확률이 증가함을 의미합니다."
    )
    
    # 문제 6: 오즈비 계산
    coefficient = 0.693  # ln(2)
    odds_ratio = np.exp(coefficient)
    
    quiz.add_question(
        f"6. 회귀 계수가 {coefficient:.3f}일 때 오즈비(Odds Ratio)는?",
        "multiple_choice",
        ["1.0", "2.0", "0.5", "1.5"],
        1,
        f"오즈비 = e^(계수) = e^{coefficient:.3f} = {odds_ratio:.1f}"
    )
    
    # 문제 7: 성능 지표 계산
    print("\n다음 혼동 행렬을 보고 문제 7-9를 풀어주세요:")
    print("혼동 행렬:")
    print("        예측")
    print("실제    0   1")
    print("  0    85  15")
    print("  1    10  90")
    
    # 실제 값들
    TP, TN, FP, FN = 90, 85, 15, 10
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    quiz.add_question(
        "7. 위 혼동 행렬에서 정확도(Accuracy)는? (소수점 셋째 자리까지)",
        "numeric",
        accuracy,
        None,
        f"정확도 = (TP+TN)/(TP+TN+FP+FN) = ({TP}+{TN})/({TP}+{TN}+{FP}+{FN}) = {accuracy:.3f}"
    )
    
    quiz.add_question(
        "8. 위 혼동 행렬에서 정밀도(Precision)는? (소수점 셋째 자리까지)",
        "numeric",
        precision,
        None,
        f"정밀도 = TP/(TP+FP) = {TP}/({TP}+{FP}) = {precision:.3f}"
    )
    
    quiz.add_question(
        "9. 위 혼동 행렬에서 재현율(Recall)은? (소수점 셋째 자리까지)",
        "numeric",
        recall,
        None,
        f"재현율 = TP/(TP+FN) = {TP}/({TP}+{FN}) = {recall:.3f}"
    )
    
    # 문제 10: F1-Score
    quiz.add_question(
        "10. 위 혼동 행렬에서 F1-Score는? (소수점 셋째 자리까지)",
        "numeric",
        f1,
        None,
        f"F1-Score = 2×(정밀도×재현율)/(정밀도+재현율) = 2×({precision:.3f}×{recall:.3f})/({precision:.3f}+{recall:.3f}) = {f1:.3f}"
    )
    
    # 문제 11: 결정 경계
    quiz.add_question(
        "11. 로지스틱 회귀에서 기본 결정 임계값은?",
        "multiple_choice",
        ["0.3", "0.5", "0.7", "1.0"],
        1,
        "일반적으로 확률이 0.5 이상이면 양성 클래스, 미만이면 음성 클래스로 분류합니다."
    )
    
    # 문제 12: 정규화
    quiz.add_question(
        "12. 로지스틱 회귀에서 정규화 파라미터 C가 작을 때의 효과는?",
        "multiple_choice",
        [
            "과적합이 증가한다",
            "정규화가 강해져 과적합이 감소한다",
            "모델이 더 복잡해진다",
            "정확도가 항상 증가한다"
        ],
        1,
        "C는 정규화 강도의 역수이므로, C가 작을수록 정규화가 강해져 과적합을 방지합니다."
    )
    
    # 문제 13: 시나리오 분석
    print("\n" + "="*50)
    print("시나리오 분석")
    print("="*50)
    print("다음 상황을 읽고 문제를 풀어주세요:")
    print("의료진이 환자의 질병 진단을 위해 로지스틱 회귀 모델을 사용합니다.")
    print("이 경우 거짓 음성(실제 질병이 있는데 없다고 진단)을 최소화하는 것이 중요합니다.")
    
    quiz.add_question(
        "13. 이 상황에서 가장 중요하게 고려해야 할 성능 지표는?",
        "multiple_choice",
        [
            "정확도 (Accuracy)",
            "정밀도 (Precision)", 
            "재현율 (Recall)",
            "F1-Score"
        ],
        2,
        "거짓 음성을 최소화하려면 실제 양성을 놓치지 않는 것이 중요하므로 재현율(Recall)이 가장 중요합니다."
    )
    
    quiz.add_question(
        "14. 위 상황에서 결정 임계값을 어떻게 조정해야 할까요?",
        "multiple_choice",
        [
            "0.5보다 높게 설정",
            "0.5보다 낮게 설정",
            "정확히 0.5로 설정",
            "임계값은 조정할 필요 없음"
        ],
        1,
        "거짓 음성을 줄이려면 더 많은 케이스를 양성으로 분류해야 하므로 임계값을 낮춰야 합니다."
    )
    
    # 문제 15: 선형 회귀와의 차이
    quiz.add_question(
        "15. 로지스틱 회귀와 선형 회귀의 주요 차이점이 아닌 것은?",
        "multiple_choice",
        [
            "출력값의 범위",
            "사용하는 활성화 함수",
            "손실 함수의 종류",
            "특성의 개수"
        ],
        3,
        "특성의 개수는 두 알고리즘 모두 동일하게 다룰 수 있습니다. 주요 차이점은 출력 범위, 활성화 함수, 손실 함수입니다."
    )
    
    # 문제 16: 실제 적용
    quiz.add_question(
        "16. 로지스틱 회귀가 적합하지 않은 문제는?",
        "multiple_choice",
        [
            "이메일 스팸 분류",
            "고객 구매 의향 예측",
            "주식 가격 예측",
            "의료 진단"
        ],
        2,
        "주식 가격 예측은 연속적인 수치를 예측하는 회귀 문제이므로 로지스틱 회귀보다는 선형 회귀가 적합합니다."
    )
    
    # 문제 17: 다중 클래스
    quiz.add_question(
        "17. 로지스틱 회귀를 다중 클래스 분류에 확장하는 방법은?",
        "multiple_choice",
        [
            "One-vs-Rest (OvR)",
            "One-vs-One (OvO)",
            "Multinomial 로지스틱 회귀",
            "위의 모든 방법"
        ],
        3,
        "로지스틱 회귀는 OvR, OvO, Multinomial 등 다양한 방법으로 다중 클래스 분류에 확장할 수 있습니다."
    )
    
    # 문제 18: 가정 위반
    quiz.add_question(
        "18. 로지스틱 회귀의 가정을 위반하는 상황은?",
        "multiple_choice",
        [
            "특성 간에 완벽한 다중공선성이 있는 경우",
            "데이터에 이상치가 많은 경우",
            "특성과 로그 오즈 간에 비선형 관계가 있는 경우",
            "위의 모든 경우"
        ],
        3,
        "로지스틱 회귀는 다중공선성, 이상치, 비선형 관계 모두에 취약하므로 이러한 상황들을 주의해야 합니다."
    )
    
    # 문제 19: 해석
    print("\n다음 로지스틱 회귀 결과를 보고 문제를 풀어주세요:")
    print("특성: 나이(Age), 소득(Income)")
    print("회귀 계수: Age = -0.05, Income = 0.0001")
    print("절편: 2.0")
    
    quiz.add_question(
        "19. 위 결과에서 나이가 1세 증가할 때 로그 오즈의 변화는?",
        "multiple_choice",
        [
            "0.05 증가",
            "0.05 감소", 
            "0.0001 증가",
            "변화 없음"
        ],
        1,
        "나이의 회귀 계수가 -0.05이므로, 나이가 1세 증가하면 로그 오즈는 0.05 감소합니다."
    )
    
    # 문제 20: 종합 이해
    quiz.add_question(
        "20. 로지스틱 회귀의 장점이 아닌 것은?",
        "multiple_choice",
        [
            "확률적 해석이 가능하다",
            "계산이 효율적이다",
            "비선형 패턴을 잘 학습한다",
            "결과 해석이 용이하다"
        ],
        2,
        "로지스틱 회귀는 선형 모델이므로 비선형 패턴 학습에는 제한이 있습니다. 이는 단점 중 하나입니다."
    )
    
    # 퀴즈 실행
    score = quiz.run_quiz()
    
    # 결과 분석 및 피드백
    print("\n" + "="*60)
    print("퀴즈 결과 분석")
    print("="*60)
    
    percentage = (score / quiz.total_questions) * 100
    
    if percentage >= 90:
        print("🎉 훌륭합니다! 로지스틱 회귀를 매우 잘 이해하고 있습니다.")
        print("고급 주제나 실제 프로젝트에 적용해보세요.")
    elif percentage >= 80:
        print("👍 좋습니다! 로지스틱 회귀의 핵심 개념을 잘 파악하고 있습니다.")
        print("실습을 통해 더 깊이 있는 이해를 쌓아보세요.")
    elif percentage >= 70:
        print("📚 기본기는 갖추었습니다. 몇 가지 개념을 더 학습하면 좋겠습니다.")
        print("특히 성능 지표와 파라미터 해석 부분을 복습해보세요.")
    elif percentage >= 60:
        print("⚠️ 기본 개념은 이해했지만 더 많은 학습이 필요합니다.")
        print("이론 문서를 다시 읽고 실습 코드를 실행해보세요.")
    else:
        print("📖 로지스틱 회귀에 대한 추가 학습이 필요합니다.")
        print("이론부터 차근차근 다시 학습하시기를 권장합니다.")
    
    print(f"\n최종 점수: {score}/{quiz.total_questions} ({percentage:.1f}%)")
    
    print("\n학습 권장사항:")
    if percentage < 80:
        print("• 05_logistic_regression_theory.md 문서를 다시 읽어보세요")
        print("• 05_logistic_regression_practice.py 실습을 직접 실행해보세요")
        print("• 시그모이드 함수와 확률 해석을 중점적으로 학습하세요")
    
    if percentage < 70:
        print("• 선형 회귀와의 차이점을 명확히 이해하세요")
        print("• 분류 성능 지표들의 계산 방법을 연습하세요")
        print("• 실제 데이터셋으로 직접 모델을 구축해보세요")
    
    print("\n다음 학습 주제:")
    print("• 의사결정나무 (Decision Tree)")
    print("• 규칙 기반 학습 (Rule-based Learning)")
    print("• 앙상블 학습 (Ensemble Learning)")

if __name__ == "__main__":
    main()