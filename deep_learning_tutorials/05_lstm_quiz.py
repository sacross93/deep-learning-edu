#!/usr/bin/env python3
"""
LSTM 퀴즈

이 퀴즈는 다음 내용을 평가합니다:
1. LSTM 게이트 메커니즘 상세 분석
2. 셀 상태와 은닉 상태 차이점
3. 시계열 데이터 전처리 및 평가 메트릭
4. GRU와 LSTM 비교 분석
5. 장기 의존성 학습 원리

요구사항 8.1, 8.2, 8.3을 충족합니다.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from quiz_utils import DeepLearningQuizManager, QuizQuestion
import math


class LSTMQuiz:
    def __init__(self):
        self.quiz_manager = DeepLearningQuizManager("LSTM 퀴즈")
        self.setup_questions()
    
    def setup_questions(self):
        """퀴즈 문제 설정"""
        
        # 1. LSTM 기본 개념 - Easy
        self.quiz_manager.add_question_simple(
            question_id="lstm_001",
            question_type="multiple_choice",
            question="LSTM이 기존 RNN보다 개선한 주요 문제는?",
            options=[
                "계산 속도",
                "메모리 사용량",
                "그래디언트 소실 문제",
                "파라미터 수"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 그래디언트 소실 문제

해설:
LSTM은 게이트 메커니즘과 셀 상태를 통해 장기 의존성을 학습할 수 있도록 설계되어
기존 RNN의 그래디언트 소실 문제를 크게 개선했습니다.
            """,
            difficulty="easy",
            topic="LSTM",
            related_theory_section="LSTM 이론 - LSTM 구조"
        )
        
        # 2. 셀 상태 vs 은닉 상태 - Medium
        self.quiz_manager.add_question_simple(
            question_id="lstm_002",
            question_type="multiple_choice",
            question="LSTM에서 셀 상태(Cell State)와 은닉 상태(Hidden State)의 차이점은?",
            options=[
                "셀 상태는 장기 기억, 은닉 상태는 단기 기억을 담당",
                "셀 상태는 입력용, 은닉 상태는 출력용",
                "둘은 완전히 동일함",
                "셀 상태는 학습되지 않음"
            ],
            correct_answer=1,
            explanation="""
정답: 1) 셀 상태는 장기 기억, 은닉 상태는 단기 기억을 담당

해설:
셀 상태(C_t)는 장기간에 걸친 정보를 보존하는 "컨베이어 벨트" 역할을 하고,
은닉 상태(h_t)는 현재 시점의 출력과 단기 기억을 담당합니다.
            """,
            difficulty="medium",
            topic="LSTM",
            related_theory_section="LSTM 이론 - 셀 상태와 은닉 상태"
        )
        
        # 3. 망각 게이트 - Medium
        self.quiz_manager.add_question_simple(
            question_id="lstm_003",
            question_type="multiple_choice",
            question="LSTM의 망각 게이트(Forget Gate)의 역할은?",
            options=[
                "새로운 정보를 추가함",
                "출력을 결정함",
                "이전 셀 상태에서 버릴 정보를 결정함",
                "입력을 필터링함"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 이전 셀 상태에서 버릴 정보를 결정함

해설:
망각 게이트는 f_t = σ(W_f · [h_{t-1}, x_t] + b_f)로 계산되며,
이전 셀 상태에서 어떤 정보를 잊을지(0에 가까운 값) 또는 
유지할지(1에 가까운 값)를 결정합니다.
            """,
            difficulty="medium",
            topic="LSTM",
            related_theory_section="LSTM 이론 - 망각 게이트"
        )
        
        # 4. 입력 게이트 - Hard
        self.quiz_manager.add_question_simple(
            question_id="lstm_004",
            question_type="short_answer",
            question="LSTM에서 새로운 정보를 셀 상태에 저장할지 결정하는 게이트는?",
            correct_answer="입력 게이트",
            explanation="""
정답: 입력 게이트 (Input Gate)

해설:
입력 게이트는 i_t = σ(W_i · [h_{t-1}, x_t] + b_i)로 계산되며,
새로운 후보 값 중에서 어떤 부분을 셀 상태에 저장할지 결정합니다.
            """,
            difficulty="hard",
            topic="LSTM",
            related_theory_section="LSTM 이론 - 입력 게이트"
        )
        
        # 5. 출력 게이트 - Medium
        self.quiz_manager.add_question_simple(
            question_id="lstm_005",
            question_type="true_false",
            question="LSTM의 출력 게이트는 셀 상태의 어떤 부분을 은닉 상태로 출력할지 결정한다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
출력 게이트는 o_t = σ(W_o · [h_{t-1}, x_t] + b_o)로 계산되며,
셀 상태를 tanh로 활성화한 값에 출력 게이트를 곱하여 
은닉 상태 h_t = o_t * tanh(C_t)를 생성합니다.
            """,
            difficulty="medium",
            topic="LSTM",
            related_theory_section="LSTM 이론 - 출력 게이트"
        )
        
        # 6. LSTM 게이트 수 - Easy
        self.quiz_manager.add_question_simple(
            question_id="lstm_006",
            question_type="numerical",
            question="표준 LSTM에는 총 몇 개의 게이트가 있는가?",
            correct_answer=3,
            explanation="""
정답: 3

해설:
LSTM에는 다음 3개의 게이트가 있습니다:
1. 망각 게이트 (Forget Gate)
2. 입력 게이트 (Input Gate)  
3. 출력 게이트 (Output Gate)
            """,
            difficulty="easy",
            topic="LSTM",
            related_theory_section="LSTM 이론 - 게이트 메커니즘",
            tolerance=0
        )
        
        # 7. 시계열 데이터 특성 - Medium
        self.quiz_manager.add_question_simple(
            question_id="lstm_007",
            question_type="multiple_choice",
            question="시계열 데이터의 주요 특성이 아닌 것은?",
            options=[
                "시간적 순서가 중요함",
                "과거 값이 미래 값에 영향을 줌",
                "모든 시점의 값이 독립적임",
                "계절성이나 트렌드를 가질 수 있음"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 모든 시점의 값이 독립적임

해설:
시계열 데이터는 시간적 의존성이 핵심 특성입니다.
과거 값들이 현재와 미래 값에 영향을 주므로 독립적이지 않습니다.
            """,
            difficulty="medium",
            topic="LSTM",
            related_theory_section="LSTM 이론 - 시계열 데이터 특성"
        )
        
        # 8. 시계열 전처리 - Medium
        self.quiz_manager.add_question_simple(
            question_id="lstm_008",
            question_type="multiple_choice",
            question="시계열 데이터를 LSTM에 입력하기 전 일반적인 전처리 과정은?",
            options=[
                "원-핫 인코딩",
                "정규화 및 윈도우 생성",
                "PCA 적용",
                "이미지 리사이징"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 정규화 및 윈도우 생성

해설:
시계열 데이터는 보통 정규화(스케일링)를 수행하고,
슬라이딩 윈도우 방식으로 입력-출력 쌍을 생성합니다.
예: [1,2,3,4,5] → ([1,2,3], 4), ([2,3,4], 5)
            """,
            difficulty="medium",
            topic="LSTM",
            related_theory_section="LSTM 이론 - 시계열 데이터 전처리"
        )
        
        # 9. 시계열 평가 메트릭 - Hard
        self.quiz_manager.add_question_simple(
            question_id="lstm_009",
            question_type="multiple_choice",
            question="시계열 예측에서 MAPE의 의미는?",
            options=[
                "Mean Absolute Percentage Error",
                "Maximum Average Prediction Error",
                "Mean Accuracy Percentage Evaluation",
                "Minimum Absolute Prediction Error"
            ],
            correct_answer=1,
            explanation="""
정답: 1) Mean Absolute Percentage Error

해설:
MAPE = (1/n) * Σ|실제값 - 예측값| / |실제값| * 100%
실제값에 대한 예측 오차의 백분율 평균으로, 
스케일에 무관하게 오차를 평가할 수 있습니다.
            """,
            difficulty="hard",
            topic="LSTM",
            related_theory_section="LSTM 이론 - 시계열 예측 평가 메트릭"
        )
        
        # 10. GRU vs LSTM - Hard
        self.quiz_manager.add_question_simple(
            question_id="lstm_010",
            question_type="multiple_choice",
            question="GRU가 LSTM과 다른 점은?",
            options=[
                "게이트가 더 많음",
                "셀 상태와 은닉 상태를 분리함",
                "게이트 수가 적고 셀 상태가 없음",
                "계산이 더 복잡함"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 게이트 수가 적고 셀 상태가 없음

해설:
GRU는 LSTM을 단순화한 구조로, 2개의 게이트(리셋, 업데이트)만 사용하고
셀 상태 없이 은닉 상태만 사용합니다. 파라미터가 적어 계산이 빠릅니다.
            """,
            difficulty="hard",
            topic="LSTM",
            related_theory_section="LSTM 이론 - GRU와 LSTM 비교"
        )
        
        # 11. 장기 의존성 - Medium
        self.quiz_manager.add_question_simple(
            question_id="lstm_011",
            question_type="true_false",
            question="LSTM은 수백 시점 이전의 정보도 기억할 수 있다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
LSTM의 셀 상태는 게이트를 통해 선택적으로 정보를 보존하므로
이론적으로는 매우 긴 시퀀스의 의존성도 학습할 수 있습니다.
실제로는 데이터와 문제에 따라 성능이 달라집니다.
            """,
            difficulty="medium",
            topic="LSTM",
            related_theory_section="LSTM 이론 - 장기 의존성"
        )
        
        # 12. 양방향 LSTM - Medium
        self.quiz_manager.add_question_simple(
            question_id="lstm_012",
            question_type="multiple_choice",
            question="양방향 LSTM(Bidirectional LSTM)의 장점은?",
            options=[
                "계산 속도가 빠름",
                "메모리 사용량이 적음",
                "과거와 미래 정보를 모두 활용",
                "파라미터 수가 적음"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 과거와 미래 정보를 모두 활용

해설:
양방향 LSTM은 순방향과 역방향 LSTM을 결합하여
현재 시점에서 과거와 미래의 맥락을 모두 고려할 수 있습니다.
문장 분류 등에서 성능 향상을 가져옵니다.
            """,
            difficulty="medium",
            topic="LSTM",
            related_theory_section="LSTM 이론 - 양방향 LSTM"
        )
        
        # 13. LSTM 파라미터 계산 - Hard
        self.quiz_manager.add_question_simple(
            question_id="lstm_013",
            question_type="calculation",
            question="입력 차원 10, 은닉 차원 20인 LSTM 레이어의 파라미터 수는? (편향 포함)",
            correct_answer=2480,
            explanation="""
정답: 2,480

해설:
LSTM은 4개의 선형 변환이 있습니다 (망각, 입력, 후보, 출력 게이트):
각 게이트: (입력_차원 + 은닉_차원) × 은닉_차원 + 은닉_차원
= (10 + 20) × 20 + 20 = 620
총 파라미터: 620 × 4 = 2,480
            """,
            difficulty="hard",
            topic="LSTM",
            related_theory_section="LSTM 이론 - 파라미터 계산",
            formula="파라미터 = 4 × [(입력_차원 + 은닉_차원) × 은닉_차원 + 은닉_차원]",
            tolerance=0
        )
        
        # 14. 시계열 분할 - Medium
        self.quiz_manager.add_question_simple(
            question_id="lstm_014",
            question_type="multiple_choice",
            question="시계열 데이터를 훈련/검증/테스트로 분할할 때 주의사항은?",
            options=[
                "무작위로 섞어서 분할",
                "시간 순서를 유지하여 분할",
                "클래스 비율을 맞춰서 분할",
                "크기를 동일하게 분할"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 시간 순서를 유지하여 분할

해설:
시계열 데이터는 시간적 순서가 중요하므로 무작위 분할하면 안 됩니다.
과거 데이터로 훈련하고 미래 데이터로 검증/테스트해야 합니다.
            """,
            difficulty="medium",
            topic="LSTM",
            related_theory_section="LSTM 이론 - 시계열 데이터 분할"
        )
        
        # 15. LSTM 응용 분야 - Easy
        self.quiz_manager.add_question_simple(
            question_id="lstm_015",
            question_type="multiple_choice",
            question="LSTM이 적용되지 않는 분야는?",
            options=[
                "주식 가격 예측",
                "기계 번역",
                "이미지 분류",
                "음성 인식"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 이미지 분류

해설:
LSTM은 시퀀스 데이터 처리에 특화되어 있어 주식 예측, 번역, 음성 인식 등에 사용되지만,
이미지 분류는 주로 CNN이 사용됩니다. (단, 이미지 캡셔닝 등에서는 CNN+LSTM 조합 사용)
            """,
            difficulty="easy",
            topic="LSTM",
            related_theory_section="LSTM 이론 - 응용 분야"
        )
    
    def run_quiz(self, num_questions: int = 15, difficulty: str = None):
        """퀴즈 실행"""
        print("🧠 LSTM 퀴즈에 오신 것을 환영합니다!")
        print("이 퀴즈는 LSTM의 게이트 메커니즘과 시계열 데이터 처리를 평가합니다.")
        print("-" * 70)
        
        results = self.quiz_manager.run_full_quiz(
            topic="LSTM",
            difficulty=difficulty,
            num_questions=num_questions
        )
        
        return results
    
    def run_retry_quiz(self):
        """틀린 문제 재시도"""
        return self.quiz_manager.retry_wrong_questions()


def main():
    """메인 실행 함수"""
    quiz = LSTMQuiz()
    
    print("LSTM 퀴즈 시스템")
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