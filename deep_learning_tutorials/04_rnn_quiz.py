#!/usr/bin/env python3
"""
RNN 퀴즈

이 퀴즈는 다음 내용을 평가합니다:
1. 시퀀스 데이터 처리 원리
2. 그래디언트 소실/폭발 문제 이해
3. 단어 임베딩 및 시퀀스 패딩 개념
4. RNN vs CNN vs Transformer 비교
5. 자연어 처리 기초 개념

요구사항 8.1, 8.2, 8.3을 충족합니다.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from quiz_utils import DeepLearningQuizManager, QuizQuestion
import math


class RNNQuiz:
    def __init__(self):
        self.quiz_manager = DeepLearningQuizManager("RNN 퀴즈")
        self.setup_questions()
    
    def setup_questions(self):
        """퀴즈 문제 설정"""
        
        # 1. RNN 기본 개념 - Easy
        self.quiz_manager.add_question_simple(
            question_id="rnn_001",
            question_type="multiple_choice",
            question="RNN(Recurrent Neural Network)의 주요 특징은?",
            options=[
                "이미지 처리에 특화됨",
                "시퀀스 데이터를 순차적으로 처리함",
                "병렬 처리가 가능함",
                "고정 크기 입력만 처리 가능함"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 시퀀스 데이터를 순차적으로 처리함

해설:
RNN은 이전 시점의 은닉 상태를 현재 시점의 입력과 함께 사용하여
시퀀스 데이터의 시간적 의존성을 모델링할 수 있습니다.
            """,
            difficulty="easy",
            topic="RNN",
            related_theory_section="RNN 이론 - RNN 구조"
        )
        
        # 2. 은닉 상태 계산 - Medium
        self.quiz_manager.add_question_simple(
            question_id="rnn_002",
            question_type="multiple_choice",
            question="RNN에서 은닉 상태 h_t의 계산 공식은?",
            options=[
                "h_t = tanh(W_h * h_{t-1})",
                "h_t = tanh(W_x * x_t + W_h * h_{t-1} + b)",
                "h_t = sigmoid(x_t + h_{t-1})",
                "h_t = x_t * h_{t-1}"
            ],
            correct_answer=2,
            explanation="""
정답: 2) h_t = tanh(W_x * x_t + W_h * h_{t-1} + b)

해설:
RNN의 은닉 상태는 현재 입력 x_t와 이전 은닉 상태 h_{t-1}의 
선형 결합에 활성화 함수(보통 tanh)를 적용하여 계산됩니다.
            """,
            difficulty="medium",
            topic="RNN",
            related_theory_section="RNN 이론 - 순환 신경망 구조"
        )
        
        # 3. 그래디언트 소실 문제 - Medium
        self.quiz_manager.add_question_simple(
            question_id="rnn_003",
            question_type="multiple_choice",
            question="RNN에서 그래디언트 소실 문제가 발생하는 주요 원인은?",
            options=[
                "학습률이 너무 큼",
                "배치 크기가 작음",
                "긴 시퀀스에서 그래디언트가 역전파되면서 지수적으로 감소",
                "활성화 함수를 사용하지 않음"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 긴 시퀀스에서 그래디언트가 역전파되면서 지수적으로 감소

해설:
RNN에서 그래디언트는 시간을 거슬러 올라가며 전파되는데,
가중치가 1보다 작으면 그래디언트가 지수적으로 감소하여
장기 의존성 학습이 어려워집니다.
            """,
            difficulty="medium",
            topic="RNN",
            related_theory_section="RNN 이론 - 그래디언트 소실/폭발 문제"
        )
        
        # 4. 시퀀스 길이 - Easy
        self.quiz_manager.add_question_simple(
            question_id="rnn_004",
            question_type="true_false",
            question="RNN은 가변 길이 시퀀스를 처리할 수 있다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
RNN은 순환 구조로 인해 임의의 길이를 가진 시퀀스를 처리할 수 있습니다.
다만 배치 처리를 위해서는 패딩을 사용하여 길이를 맞춰줍니다.
            """,
            difficulty="easy",
            topic="RNN",
            related_theory_section="RNN 이론 - 시퀀스 데이터 처리"
        )
        
        # 5. 단어 임베딩 - Medium
        self.quiz_manager.add_question_simple(
            question_id="rnn_005",
            question_type="multiple_choice",
            question="단어 임베딩(Word Embedding)의 주요 목적은?",
            options=[
                "단어를 정수로 변환",
                "단어를 고차원 희소 벡터로 변환",
                "단어를 저차원 밀집 벡터로 변환하여 의미 표현",
                "단어의 길이를 통일"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 단어를 저차원 밀집 벡터로 변환하여 의미 표현

해설:
단어 임베딩은 단어를 의미적 유사성을 반영하는 저차원 실수 벡터로 변환합니다.
유사한 의미의 단어들은 벡터 공간에서 가까운 위치에 배치됩니다.
            """,
            difficulty="medium",
            topic="RNN",
            related_theory_section="RNN 이론 - 단어 임베딩"
        )
        
        # 6. 패딩과 마스킹 - Medium
        self.quiz_manager.add_question_simple(
            question_id="rnn_006",
            question_type="multiple_choice",
            question="시퀀스 패딩(Padding)을 사용하는 이유는?",
            options=[
                "계산 속도 향상",
                "배치 처리를 위해 시퀀스 길이를 통일",
                "메모리 사용량 감소",
                "정확도 향상"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 배치 처리를 위해 시퀀스 길이를 통일

해설:
서로 다른 길이의 시퀀스들을 하나의 배치로 처리하기 위해
짧은 시퀀스에 특별한 토큰(보통 0)을 추가하여 길이를 맞춥니다.
            """,
            difficulty="medium",
            topic="RNN",
            related_theory_section="RNN 이론 - 시퀀스 전처리"
        )
        
        # 7. Many-to-One vs One-to-Many - Hard
        self.quiz_manager.add_question_simple(
            question_id="rnn_007",
            question_type="multiple_choice",
            question="감정 분석 작업에 적합한 RNN 구조는?",
            options=[
                "One-to-One",
                "One-to-Many",
                "Many-to-One",
                "Many-to-Many"
            ],
            correct_answer=3,
            explanation="""
정답: 3) Many-to-One

해설:
감정 분석은 여러 단어로 구성된 문장(Many)을 입력받아
하나의 감정 클래스(One)를 출력하는 작업이므로 Many-to-One 구조가 적합합니다.
            """,
            difficulty="hard",
            topic="RNN",
            related_theory_section="RNN 이론 - RNN 구조 변형"
        )
        
        # 8. 양방향 RNN - Medium
        self.quiz_manager.add_question_simple(
            question_id="rnn_008",
            question_type="short_answer",
            question="과거와 미래 정보를 모두 활용하는 RNN 구조는?",
            correct_answer="양방향 RNN",
            explanation="""
정답: 양방향 RNN (Bidirectional RNN)

해설:
양방향 RNN은 순방향과 역방향 RNN을 결합하여
현재 시점에서 과거와 미래의 정보를 모두 활용할 수 있습니다.
            """,
            difficulty="medium",
            topic="RNN",
            related_theory_section="RNN 이론 - 양방향 RNN"
        )
        
        # 9. 시퀀스-투-시퀀스 - Hard
        self.quiz_manager.add_question_simple(
            question_id="rnn_009",
            question_type="multiple_choice",
            question="Seq2Seq 모델에서 인코더의 역할은?",
            options=[
                "출력 시퀀스 생성",
                "입력 시퀀스를 고정 크기 벡터로 인코딩",
                "어텐션 가중치 계산",
                "손실 함수 계산"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 입력 시퀀스를 고정 크기 벡터로 인코딩

해설:
Seq2Seq 모델의 인코더는 가변 길이 입력 시퀀스를 읽어서
모든 정보를 담은 고정 크기 컨텍스트 벡터로 압축합니다.
디코더는 이 벡터를 사용하여 출력 시퀀스를 생성합니다.
            """,
            difficulty="hard",
            topic="RNN",
            related_theory_section="RNN 이론 - 시퀀스-투-시퀀스 모델"
        )
        
        # 10. 토큰화 - Easy
        self.quiz_manager.add_question_simple(
            question_id="rnn_010",
            question_type="multiple_choice",
            question="자연어 처리에서 토큰화(Tokenization)의 목적은?",
            options=[
                "문장을 단어나 서브워드 단위로 분할",
                "단어를 숫자로 변환",
                "문법 오류 수정",
                "번역 수행"
            ],
            correct_answer=1,
            explanation="""
정답: 1) 문장을 단어나 서브워드 단위로 분할

해설:
토큰화는 연속된 텍스트를 의미 있는 단위(토큰)로 분할하는 과정입니다.
단어, 서브워드, 문자 등 다양한 단위로 분할할 수 있습니다.
            """,
            difficulty="easy",
            topic="RNN",
            related_theory_section="RNN 이론 - 자연어 처리 기초"
        )
        
        # 11. RNN vs CNN - Medium
        self.quiz_manager.add_question_simple(
            question_id="rnn_011",
            question_type="multiple_choice",
            question="RNN이 CNN보다 유리한 작업은?",
            options=[
                "이미지 분류",
                "객체 탐지",
                "시계열 예측",
                "이미지 분할"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 시계열 예측

해설:
RNN은 시간적 순서가 중요한 시계열 데이터나 자연어 처리에 적합하며,
CNN은 공간적 패턴이 중요한 이미지 처리에 더 적합합니다.
            """,
            difficulty="medium",
            topic="RNN",
            related_theory_section="RNN 이론 - RNN vs CNN"
        )
        
        # 12. 그래디언트 클리핑 - Hard
        self.quiz_manager.add_question_simple(
            question_id="rnn_012",
            question_type="true_false",
            question="그래디언트 클리핑은 RNN의 그래디언트 폭발 문제를 해결하는 데 도움이 된다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
그래디언트 클리핑은 그래디언트의 크기가 임계값을 초과할 때
그래디언트를 스케일링하여 그래디언트 폭발을 방지합니다.
RNN 훈련에서 자주 사용되는 기법입니다.
            """,
            difficulty="hard",
            topic="RNN",
            related_theory_section="RNN 이론 - 그래디언트 문제 해결"
        )
        
        # 13. 언어 모델 - Medium
        self.quiz_manager.add_question_simple(
            question_id="rnn_013",
            question_type="multiple_choice",
            question="RNN 기반 언어 모델의 목표는?",
            options=[
                "단어를 분류하는 것",
                "다음 단어의 확률 분포를 예측하는 것",
                "문장의 길이를 예측하는 것",
                "문법 오류를 찾는 것"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 다음 단어의 확률 분포를 예측하는 것

해설:
언어 모델은 이전 단어들이 주어졌을 때 다음에 올 단어의 확률을 예측합니다.
P(w_t | w_1, w_2, ..., w_{t-1}) 형태로 모델링됩니다.
            """,
            difficulty="medium",
            topic="RNN",
            related_theory_section="RNN 이론 - 언어 모델"
        )
        
        # 14. 시간 전개 - Hard
        self.quiz_manager.add_question_simple(
            question_id="rnn_014",
            question_type="calculation",
            question="길이 5인 시퀀스를 처리하는 RNN에서 시간 전개 시 총 몇 개의 시간 단계가 있는가?",
            correct_answer=5,
            explanation="""
정답: 5

해설:
시퀀스 길이가 5이면 t=1, 2, 3, 4, 5로 총 5개의 시간 단계가 있습니다.
각 시간 단계에서 하나의 입력을 처리합니다.
            """,
            difficulty="hard",
            topic="RNN",
            related_theory_section="RNN 이론 - 시간 전개",
            tolerance=0
        )
        
        # 15. RNN vs Transformer - Hard
        self.quiz_manager.add_question_simple(
            question_id="rnn_015",
            question_type="multiple_choice",
            question="RNN 대비 Transformer의 주요 장점은?",
            options=[
                "메모리 사용량이 적음",
                "병렬 처리가 가능함",
                "구조가 더 간단함",
                "파라미터 수가 적음"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 병렬 처리가 가능함

해설:
RNN은 순차적으로 처리해야 하지만, Transformer는 Self-Attention을 통해
모든 위치를 동시에 처리할 수 있어 훈련 속도가 훨씬 빠릅니다.
            """,
            difficulty="hard",
            topic="RNN",
            related_theory_section="RNN 이론 - RNN vs Transformer"
        )
    
    def run_quiz(self, num_questions: int = 15, difficulty: str = None):
        """퀴즈 실행"""
        print("🔄 RNN 퀴즈에 오신 것을 환영합니다!")
        print("이 퀴즈는 순환 신경망과 시퀀스 데이터 처리를 평가합니다.")
        print("-" * 60)
        
        results = self.quiz_manager.run_full_quiz(
            topic="RNN",
            difficulty=difficulty,
            num_questions=num_questions
        )
        
        return results
    
    def run_retry_quiz(self):
        """틀린 문제 재시도"""
        return self.quiz_manager.retry_wrong_questions()


def main():
    """메인 실행 함수"""
    quiz = RNNQuiz()
    
    print("RNN 퀴즈 시스템")
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