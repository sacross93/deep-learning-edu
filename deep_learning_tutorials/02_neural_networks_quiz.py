#!/usr/bin/env python3
"""
신경망 퀴즈

이 퀴즈는 다음 내용을 평가합니다:
1. 순전파/역전파 과정 이해도 평가
2. 활성화 함수 특성 및 선택 기준
3. 정규화 기법 효과 분석
4. 하이퍼파라미터 튜닝 전략
5. 신경망 기본 개념 및 수학적 원리

요구사항 8.1, 8.2, 8.3을 충족합니다.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from quiz_utils import DeepLearningQuizManager, QuizQuestion
import math


class NeuralNetworksQuiz:
    def __init__(self):
        self.quiz_manager = DeepLearningQuizManager("신경망 퀴즈")
        self.setup_questions()
    
    def setup_questions(self):
        """퀴즈 문제 설정"""
        
        # 1. 순전파 과정 - Easy
        self.quiz_manager.add_question_simple(
            question_id="nn_001",
            question_type="multiple_choice",
            question="신경망의 순전파(Forward Propagation) 과정에서 일어나는 일은?",
            options=[
                "가중치를 업데이트함",
                "입력에서 출력까지 데이터가 흘러가며 예측값을 계산함",
                "손실 함수의 그래디언트를 계산함",
                "학습률을 조정함"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 입력에서 출력까지 데이터가 흘러가며 예측값을 계산함

해설:
순전파는 입력 데이터가 신경망의 각 층을 거쳐 최종 출력(예측값)을 생성하는 과정입니다.
각 층에서 가중치와 편향을 적용하고 활성화 함수를 통과시켜 다음 층으로 전달합니다.
            """,
            difficulty="easy",
            topic="신경망",
            related_theory_section="신경망 이론 - 순전파 과정"
        )
        
        # 2. 역전파 수학 - Hard
        self.quiz_manager.add_question_simple(
            question_id="nn_002",
            question_type="calculation",
            question="손실 L = (y - ŷ)²에서 예측값 ŷ에 대한 편미분 ∂L/∂ŷ는? (y=3, ŷ=1일 때)",
            correct_answer=-4,
            explanation="""
정답: -4

해설:
L = (y - ŷ)²
∂L/∂ŷ = 2(y - ŷ) × (-1) = -2(y - ŷ)

y=3, ŷ=1을 대입하면:
∂L/∂ŷ = -2(3 - 1) = -2 × 2 = -4
            """,
            difficulty="hard",
            topic="신경망",
            related_theory_section="신경망 이론 - 역전파 알고리즘",
            formula="∂L/∂ŷ = -2(y - ŷ)",
            tolerance=0.1
        )
        
        # 3. 활성화 함수 비교 - Medium
        self.quiz_manager.add_question_simple(
            question_id="nn_003",
            question_type="multiple_choice",
            question="Sigmoid 함수의 주요 단점은?",
            options=[
                "계산이 복잡함",
                "출력 범위가 제한적임",
                "그래디언트 소실 문제가 발생할 수 있음",
                "음수 입력을 처리할 수 없음"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 그래디언트 소실 문제가 발생할 수 있음

해설:
Sigmoid 함수는 입력값이 매우 크거나 작을 때 그래디언트가 0에 가까워져
깊은 신경망에서 그래디언트 소실 문제를 일으킬 수 있습니다.
이는 ReLU 등의 활성화 함수가 선호되는 이유 중 하나입니다.
            """,
            difficulty="medium",
            topic="신경망",
            related_theory_section="신경망 이론 - 활성화 함수"
        )
        
        # 4. ReLU vs Sigmoid - Medium
        self.quiz_manager.add_question_simple(
            question_id="nn_004",
            question_type="true_false",
            question="ReLU 함수는 Sigmoid 함수보다 계산 속도가 빠르다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
ReLU는 max(0, x)로 단순한 비교 연산만 필요하지만,
Sigmoid는 지수 함수 계산이 필요하여 계산 비용이 더 높습니다.
이는 ReLU가 널리 사용되는 이유 중 하나입니다.
            """,
            difficulty="medium",
            topic="신경망",
            related_theory_section="신경망 이론 - 활성화 함수"
        )
        
        # 5. 드롭아웃 정규화 - Medium
        self.quiz_manager.add_question_simple(
            question_id="nn_005",
            question_type="multiple_choice",
            question="드롭아웃(Dropout) 정규화의 주요 목적은?",
            options=[
                "학습 속도 향상",
                "과적합(Overfitting) 방지",
                "메모리 사용량 감소",
                "그래디언트 폭발 방지"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 과적합(Overfitting) 방지

해설:
드롭아웃은 훈련 중 일부 뉴런을 무작위로 비활성화하여
모델이 특정 뉴런에 과도하게 의존하는 것을 방지합니다.
이를 통해 일반화 성능을 향상시키고 과적합을 줄입니다.
            """,
            difficulty="medium",
            topic="신경망",
            related_theory_section="신경망 이론 - 정규화 기법"
        )
        
        # 6. 배치 정규화 - Hard
        self.quiz_manager.add_question_simple(
            question_id="nn_006",
            question_type="short_answer",
            question="배치 정규화(Batch Normalization)가 해결하는 주요 문제는?",
            correct_answer="내부 공변량 변화",
            explanation="""
정답: 내부 공변량 변화 (Internal Covariate Shift)

해설:
배치 정규화는 각 층의 입력 분포가 훈련 중에 계속 변하는 
내부 공변량 변화 문제를 해결합니다. 이를 통해 더 안정적이고 
빠른 학습이 가능해집니다.
            """,
            difficulty="hard",
            topic="신경망",
            related_theory_section="신경망 이론 - 정규화 기법"
        )
        
        # 7. 손실 함수 선택 - Medium
        self.quiz_manager.add_question_simple(
            question_id="nn_007",
            question_type="multiple_choice",
            question="회귀 문제에 가장 적합한 손실 함수는?",
            options=[
                "CrossEntropyLoss",
                "BCELoss",
                "MSELoss",
                "NLLLoss"
            ],
            correct_answer=3,
            explanation="""
정답: 3) MSELoss

해설:
MSE(Mean Squared Error)는 예측값과 실제값의 차이를 제곱하여 평균낸 것으로,
연속적인 값을 예측하는 회귀 문제에 가장 적합한 손실 함수입니다.
            """,
            difficulty="medium",
            topic="신경망",
            related_theory_section="신경망 이론 - 손실 함수"
        )
        
        # 8. 학습률 영향 - Medium
        self.quiz_manager.add_question_simple(
            question_id="nn_008",
            question_type="multiple_choice",
            question="학습률이 너무 클 때 발생할 수 있는 문제는?",
            options=[
                "학습이 너무 느려짐",
                "과소적합 발생",
                "최적점을 지나쳐 발산할 수 있음",
                "메모리 부족"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 최적점을 지나쳐 발산할 수 있음

해설:
학습률이 너무 크면 경사하강법에서 최적점 주변에서 진동하거나
아예 최적점을 지나쳐 손실이 발산할 수 있습니다.
적절한 학습률 선택이 중요합니다.
            """,
            difficulty="medium",
            topic="신경망",
            related_theory_section="신경망 이론 - 하이퍼파라미터"
        )
        
        # 9. 가중치 초기화 - Hard
        self.quiz_manager.add_question_simple(
            question_id="nn_009",
            question_type="multiple_choice",
            question="Xavier/Glorot 초기화의 목적은?",
            options=[
                "가중치를 0으로 초기화",
                "가중치를 1로 초기화",
                "각 층의 출력 분산을 일정하게 유지",
                "가중치를 무작위로 초기화"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 각 층의 출력 분산을 일정하게 유지

해설:
Xavier 초기화는 각 층의 입력과 출력 뉴런 수를 고려하여
가중치를 초기화함으로써 신호의 분산이 층을 거치면서
너무 커지거나 작아지지 않도록 합니다.
            """,
            difficulty="hard",
            topic="신경망",
            related_theory_section="신경망 이론 - 가중치 초기화"
        )
        
        # 10. 에포크와 배치 - Easy
        self.quiz_manager.add_question_simple(
            question_id="nn_010",
            question_type="numerical",
            question="1000개의 데이터를 배치 크기 50으로 1 에포크 학습할 때, 총 몇 번의 가중치 업데이트가 일어나는가?",
            correct_answer=20,
            explanation="""
정답: 20

해설:
1 에포크 = 전체 데이터를 한 번 학습
배치 수 = 전체 데이터 수 / 배치 크기 = 1000 / 50 = 20
각 배치마다 가중치 업데이트가 1번씩 일어나므로 총 20번입니다.
            """,
            difficulty="easy",
            topic="신경망",
            related_theory_section="신경망 이론 - 학습 과정",
            tolerance=0
        )
        
        # 11. 과적합 징후 - Medium
        self.quiz_manager.add_question_simple(
            question_id="nn_011",
            question_type="multiple_choice",
            question="과적합이 발생했을 때 나타나는 현상은?",
            options=[
                "훈련 손실과 검증 손실이 모두 감소",
                "훈련 손실은 감소하지만 검증 손실은 증가",
                "훈련 손실과 검증 손실이 모두 증가",
                "훈련 손실은 증가하지만 검증 손실은 감소"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 훈련 손실은 감소하지만 검증 손실은 증가

해설:
과적합은 모델이 훈련 데이터에만 특화되어 새로운 데이터에 대한
일반화 성능이 떨어지는 현상입니다. 이때 훈련 손실은 계속 감소하지만
검증 손실은 증가하기 시작합니다.
            """,
            difficulty="medium",
            topic="신경망",
            related_theory_section="신경망 이론 - 과적합과 일반화"
        )
        
        # 12. 모멘텀 옵티마이저 - Hard
        self.quiz_manager.add_question_simple(
            question_id="nn_012",
            question_type="true_false",
            question="모멘텀(Momentum)을 사용하면 지역 최솟값에서 벗어나는 데 도움이 된다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
모멘텀은 이전 그래디언트의 방향을 기억하여 관성을 부여합니다.
이를 통해 지역 최솟값의 작은 언덕을 넘어서 더 좋은 최솟값을
찾을 수 있도록 도와줍니다.
            """,
            difficulty="hard",
            topic="신경망",
            related_theory_section="신경망 이론 - 옵티마이저"
        )
        
        # 13. 은닉층 크기 - Medium
        self.quiz_manager.add_question_simple(
            question_id="nn_013",
            question_type="multiple_choice",
            question="은닉층의 뉴런 수를 결정할 때 고려사항이 아닌 것은?",
            options=[
                "입력 데이터의 복잡도",
                "과적합 위험성",
                "계산 비용",
                "데이터의 저장 형식"
            ],
            correct_answer=4,
            explanation="""
정답: 4) 데이터의 저장 형식

해설:
은닉층 크기는 데이터의 복잡도, 과적합 위험성, 계산 비용 등을 고려해야 하지만,
데이터의 저장 형식(CSV, JSON 등)은 신경망 구조 설계와 직접적인 관련이 없습니다.
            """,
            difficulty="medium",
            topic="신경망",
            related_theory_section="신경망 이론 - 아키텍처 설계"
        )
        
        # 14. 소프트맥스 함수 - Medium
        self.quiz_manager.add_question_simple(
            question_id="nn_014",
            question_type="calculation",
            question="입력 [1, 2, 3]에 대한 소프트맥스 출력에서 가장 큰 값은? (소수점 둘째 자리까지)",
            correct_answer=0.67,
            explanation="""
정답: 0.67

해설:
소프트맥스: softmax(x_i) = exp(x_i) / Σexp(x_j)

exp(1) = 2.718, exp(2) = 7.389, exp(3) = 20.086
합계 = 2.718 + 7.389 + 20.086 = 30.193

softmax(3) = 20.086 / 30.193 = 0.665 ≈ 0.67
            """,
            difficulty="medium",
            topic="신경망",
            related_theory_section="신경망 이론 - 활성화 함수",
            formula="softmax(x_i) = exp(x_i) / Σexp(x_j)",
            tolerance=0.02
        )
        
        # 15. 조기 종료 - Easy
        self.quiz_manager.add_question_simple(
            question_id="nn_015",
            question_type="short_answer",
            question="검증 손실이 더 이상 개선되지 않을 때 학습을 중단하는 기법은?",
            correct_answer="조기 종료",
            explanation="""
정답: 조기 종료 (Early Stopping)

해설:
조기 종료는 검증 손실이 일정 에포크 동안 개선되지 않으면
학습을 중단하여 과적합을 방지하는 정규화 기법입니다.
            """,
            difficulty="easy",
            topic="신경망",
            related_theory_section="신경망 이론 - 정규화 기법"
        )
    
    def run_quiz(self, num_questions: int = 15, difficulty: str = None):
        """퀴즈 실행"""
        print("🧠 신경망 퀴즈에 오신 것을 환영합니다!")
        print("이 퀴즈는 신경망의 기본 원리와 학습 과정을 평가합니다.")
        print("-" * 60)
        
        results = self.quiz_manager.run_full_quiz(
            topic="신경망",
            difficulty=difficulty,
            num_questions=num_questions
        )
        
        return results
    
    def run_retry_quiz(self):
        """틀린 문제 재시도"""
        return self.quiz_manager.retry_wrong_questions()


def main():
    """메인 실행 함수"""
    quiz = NeuralNetworksQuiz()
    
    print("신경망 퀴즈 시스템")
    print("=" * 50)
    print("1. 전체 퀴즈 (15문제)")
    print("2. 쉬운 문제만 (Easy)")
    print("3. 보통 문제만 (Medium)")
    print("4. 어려운 문제만 (Hard)")
    print("5. 맞춤형 퀴즈")
    
    while True:
        choice = input("\n선택하세요 (1-5, q: 종료): ").strip()
        
        if choice.lower() == 'q':
            print("퀴즈를 종료합니다. 수고하셨습니다!")
            break
        elif choice == '1':
            results = quiz.run_quiz()
        elif choice == '2':
            results = quiz.run_quiz(difficulty="easy")
        elif choice == '3':
            results = quiz.run_quiz(difficulty="medium")
        elif choice == '4':
            results = quiz.run_quiz(difficulty="hard")
        elif choice == '5':
            try:
                num_q = int(input("문제 수를 입력하세요 (1-15): "))
                if 1 <= num_q <= 15:
                    results = quiz.run_quiz(num_questions=num_q)
                else:
                    print("1-15 사이의 숫자를 입력하세요.")
                    continue
            except ValueError:
                print("올바른 숫자를 입력하세요.")
                continue
        else:
            print("올바른 선택지를 입력하세요.")
            continue
        
        # 재시도 옵션
        if results and any(not r.is_correct for r in results):
            retry = input("\n틀린 문제를 다시 풀어보시겠습니까? (y/n): ").strip().lower()
            if retry == 'y':
                quiz.run_retry_quiz()


if __name__ == "__main__":
    main()