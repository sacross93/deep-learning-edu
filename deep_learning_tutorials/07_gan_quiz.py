#!/usr/bin/env python3
"""
GAN 퀴즈

이 퀴즈는 다음 내용을 평가합니다:
1. 적대적 학습 원리 및 게임 이론
2. 생성자와 판별자 역할 분석
3. 모드 붕괴 문제 및 해결 방법
4. GAN 변형 모델들 특성 비교
5. 생성 품질 평가 방법

요구사항 8.1, 8.2, 8.3을 충족합니다.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from quiz_utils import DeepLearningQuizManager, QuizQuestion
import math


class GANQuiz:
    def __init__(self):
        self.quiz_manager = DeepLearningQuizManager("GAN 퀴즈")
        self.setup_questions()
    
    def setup_questions(self):
        """퀴즈 문제 설정"""
        
        # 1. GAN 기본 개념 - Easy
        self.quiz_manager.add_question_simple(
            question_id="gan_001",
            question_type="multiple_choice",
            question="GAN(Generative Adversarial Network)의 핵심 아이디어는?",
            options=[
                "더 깊은 신경망 사용",
                "생성자와 판별자가 서로 경쟁하며 학습",
                "더 많은 데이터 사용",
                "복잡한 손실 함수 사용"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 생성자와 판별자가 서로 경쟁하며 학습

해설:
GAN은 데이터를 생성하는 생성자(Generator)와 진짜/가짜를 구분하는 
판별자(Discriminator)가 적대적으로 경쟁하며 서로를 발전시키는 구조입니다.
            """,
            difficulty="easy",
            topic="GAN",
            related_theory_section="GAN 이론 - GAN 개념"
        )
        
        # 2. 생성자 역할 - Easy
        self.quiz_manager.add_question_simple(
            question_id="gan_002",
            question_type="multiple_choice",
            question="GAN에서 생성자(Generator)의 목표는?",
            options=[
                "진짜 데이터를 분류하는 것",
                "가짜 데이터를 생성하여 판별자를 속이는 것",
                "손실 함수를 최소화하는 것",
                "데이터를 압축하는 것"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 가짜 데이터를 생성하여 판별자를 속이는 것

해설:
생성자는 무작위 노이즈를 입력받아 실제 데이터와 구분하기 어려운 
가짜 데이터를 생성하여 판별자를 속이는 것이 목표입니다.
            """,
            difficulty="easy",
            topic="GAN",
            related_theory_section="GAN 이론 - 생성자 구조"
        )
        
        # 3. 판별자 역할 - Easy
        self.quiz_manager.add_question_simple(
            question_id="gan_003",
            question_type="multiple_choice",
            question="GAN에서 판별자(Discriminator)의 목표는?",
            options=[
                "새로운 데이터를 생성하는 것",
                "진짜 데이터와 가짜 데이터를 정확히 구분하는 것",
                "데이터를 압축하는 것",
                "노이즈를 제거하는 것"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 진짜 데이터와 가짜 데이터를 정확히 구분하는 것

해설:
판별자는 실제 데이터와 생성자가 만든 가짜 데이터를 입력받아
이를 정확히 구분하는 이진 분류기 역할을 합니다.
            """,
            difficulty="easy",
            topic="GAN",
            related_theory_section="GAN 이론 - 판별자 구조"
        )
        
        # 4. 적대적 학습 - Medium
        self.quiz_manager.add_question_simple(
            question_id="gan_004",
            question_type="true_false",
            question="GAN에서 생성자와 판별자는 동시에 학습된다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
GAN은 생성자와 판별자를 번갈아가며 또는 동시에 학습시킵니다.
두 네트워크가 서로 경쟁하면서 함께 발전하는 것이 핵심입니다.
            """,
            difficulty="medium",
            topic="GAN",
            related_theory_section="GAN 이론 - 적대적 학습"
        )
        
        # 5. 게임 이론 - Hard
        self.quiz_manager.add_question_simple(
            question_id="gan_005",
            question_type="multiple_choice",
            question="GAN의 이론적 배경이 되는 게임 이론 개념은?",
            options=[
                "제로섬 게임",
                "내쉬 균형",
                "미니맥스 게임",
                "위의 모든 것"
            ],
            correct_answer=4,
            explanation="""
정답: 4) 위의 모든 것

해설:
GAN은 제로섬 게임(한쪽의 이득이 다른 쪽의 손실)이며,
미니맥스 게임(최소화-최대화)으로 모델링되고,
이론적으로 내쉬 균형에 수렴합니다.
            """,
            difficulty="hard",
            topic="GAN",
            related_theory_section="GAN 이론 - 게임 이론적 배경"
        )
        
        # 6. GAN 손실 함수 - Hard
        self.quiz_manager.add_question_simple(
            question_id="gan_006",
            question_type="multiple_choice",
            question="원본 GAN의 손실 함수에서 생성자가 최소화하려는 것은?",
            options=[
                "log(D(G(z)))",
                "-log(D(G(z)))",
                "log(1 - D(G(z)))",
                "-log(1 - D(G(z)))"
            ],
            correct_answer=2,
            explanation="""
정답: 2) -log(D(G(z)))

해설:
생성자는 판별자를 속이려 하므로 D(G(z))를 최대화하려 합니다.
실제로는 -log(D(G(z)))를 최소화하는 것이 log(1-D(G(z)))를 최소화하는 것보다
그래디언트가 더 강해서 학습이 잘 됩니다.
            """,
            difficulty="hard",
            topic="GAN",
            related_theory_section="GAN 이론 - 손실 함수"
        )
        
        # 7. 모드 붕괴 - Medium
        self.quiz_manager.add_question_simple(
            question_id="gan_007",
            question_type="short_answer",
            question="GAN에서 생성자가 다양성을 잃고 비슷한 샘플만 생성하는 문제를 무엇이라 하는가?",
            correct_answer="모드 붕괴",
            explanation="""
정답: 모드 붕괴 (Mode Collapse)

해설:
모드 붕괴는 생성자가 판별자를 속이기 쉬운 특정 패턴만 학습하여
다양한 샘플을 생성하지 못하고 비슷한 결과만 반복 생성하는 문제입니다.
            """,
            difficulty="medium",
            topic="GAN",
            related_theory_section="GAN 이론 - 모드 붕괴 문제"
        )
        
        # 8. 훈련 불안정성 - Medium
        self.quiz_manager.add_question_simple(
            question_id="gan_008",
            question_type="multiple_choice",
            question="GAN 훈련에서 발생할 수 있는 문제가 아닌 것은?",
            options=[
                "모드 붕괴",
                "그래디언트 소실",
                "훈련 불안정성",
                "과적합"
            ],
            correct_answer=4,
            explanation="""
정답: 4) 과적합

해설:
GAN은 생성 모델이므로 전통적인 의미의 과적합보다는
모드 붕괴, 그래디언트 소실, 훈련 불안정성 등이 주요 문제입니다.
            """,
            difficulty="medium",
            topic="GAN",
            related_theory_section="GAN 이론 - 훈련 문제"
        )
        
        # 9. DCGAN - Medium
        self.quiz_manager.add_question_simple(
            question_id="gan_009",
            question_type="multiple_choice",
            question="DCGAN(Deep Convolutional GAN)의 주요 개선사항은?",
            options=[
                "완전연결층 대신 합성곱층 사용",
                "배치 정규화 도입",
                "LeakyReLU 활성화 함수 사용",
                "위의 모든 것"
            ],
            correct_answer=4,
            explanation="""
정답: 4) 위의 모든 것

해설:
DCGAN은 안정적인 훈련을 위해 여러 기법을 도입했습니다:
- 합성곱층으로 공간 정보 보존
- 배치 정규화로 훈련 안정화
- LeakyReLU로 그래디언트 흐름 개선
            """,
            difficulty="medium",
            topic="GAN",
            related_theory_section="GAN 이론 - DCGAN"
        )
        
        # 10. WGAN - Hard
        self.quiz_manager.add_question_simple(
            question_id="gan_010",
            question_type="multiple_choice",
            question="WGAN(Wasserstein GAN)이 해결하려는 주요 문제는?",
            options=[
                "계산 속도",
                "메모리 사용량",
                "훈련 불안정성 및 모드 붕괴",
                "생성 품질"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 훈련 불안정성 및 모드 붕괴

해설:
WGAN은 Wasserstein 거리를 사용하여 더 안정적인 훈련을 제공하고
모드 붕괴 문제를 완화합니다. 또한 의미 있는 손실 메트릭을 제공합니다.
            """,
            difficulty="hard",
            topic="GAN",
            related_theory_section="GAN 이론 - WGAN"
        )
        
        # 11. 조건부 GAN - Medium
        self.quiz_manager.add_question_simple(
            question_id="gan_011",
            question_type="multiple_choice",
            question="조건부 GAN(Conditional GAN)에서 추가로 입력되는 정보는?",
            options=[
                "더 많은 노이즈",
                "클래스 라벨이나 조건 정보",
                "더 큰 이미지",
                "더 복잡한 구조"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 클래스 라벨이나 조건 정보

해설:
조건부 GAN은 생성자와 판별자 모두에 추가 조건 정보(클래스 라벨 등)를 
입력하여 원하는 특성을 가진 데이터를 생성할 수 있게 합니다.
            """,
            difficulty="medium",
            topic="GAN",
            related_theory_section="GAN 이론 - 조건부 GAN"
        )
        
        # 12. StyleGAN - Hard
        self.quiz_manager.add_question_simple(
            question_id="gan_012",
            question_type="multiple_choice",
            question="StyleGAN의 주요 혁신은?",
            options=[
                "더 깊은 네트워크",
                "스타일과 콘텐츠의 분리 제어",
                "더 빠른 훈련",
                "더 적은 파라미터"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 스타일과 콘텐츠의 분리 제어

해설:
StyleGAN은 스타일 벡터를 통해 생성되는 이미지의 다양한 속성
(거친 특징부터 세밀한 특징까지)을 계층적으로 제어할 수 있게 했습니다.
            """,
            difficulty="hard",
            topic="GAN",
            related_theory_section="GAN 이론 - StyleGAN"
        )
        
        # 13. 생성 품질 평가 - Hard
        self.quiz_manager.add_question_simple(
            question_id="gan_013",
            question_type="multiple_choice",
            question="FID(Fréchet Inception Distance)가 측정하는 것은?",
            options=[
                "생성 속도",
                "메모리 사용량",
                "생성된 이미지와 실제 이미지의 분포 차이",
                "네트워크 복잡도"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 생성된 이미지와 실제 이미지의 분포 차이

해설:
FID는 Inception 네트워크의 특성 공간에서 실제 이미지와 생성된 이미지의
분포 간 Fréchet 거리를 계산하여 생성 품질을 평가합니다.
낮을수록 좋습니다.
            """,
            difficulty="hard",
            topic="GAN",
            related_theory_section="GAN 이론 - 생성 품질 평가"
        )
        
        # 14. IS 점수 - Medium
        self.quiz_manager.add_question_simple(
            question_id="gan_014",
            question_type="true_false",
            question="IS(Inception Score)는 높을수록 생성 품질이 좋다는 의미이다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
IS는 생성된 이미지의 다양성과 품질을 동시에 측정합니다.
높은 IS는 이미지가 명확하게 분류되면서도(품질) 다양한 클래스를 
포함한다는(다양성) 의미입니다.
            """,
            difficulty="medium",
            topic="GAN",
            related_theory_section="GAN 이론 - IS 점수"
        )
        
        # 15. GAN 응용 분야 - Easy
        self.quiz_manager.add_question_simple(
            question_id="gan_015",
            question_type="multiple_choice",
            question="GAN이 적용되지 않는 분야는?",
            options=[
                "이미지 생성",
                "데이터 증강",
                "이미지 변환",
                "시계열 분류"
            ],
            correct_answer=4,
            explanation="""
정답: 4) 시계열 분류

해설:
GAN은 주로 생성 작업에 사용됩니다:
- 이미지/음성/텍스트 생성
- 데이터 증강
- 스타일 변환
시계열 분류는 판별 작업으로 RNN/LSTM이 더 적합합니다.
            """,
            difficulty="easy",
            topic="GAN",
            related_theory_section="GAN 이론 - 응용 분야"
        )
    
    def run_quiz(self, num_questions: int = 15, difficulty: str = None):
        """퀴즈 실행"""
        print("🎨 GAN 퀴즈에 오신 것을 환영합니다!")
        print("이 퀴즈는 생성적 적대 신경망의 원리와 응용을 평가합니다.")
        print("-" * 60)
        
        results = self.quiz_manager.run_full_quiz(
            topic="GAN",
            difficulty=difficulty,
            num_questions=num_questions
        )
        
        return results
    
    def run_retry_quiz(self):
        """틀린 문제 재시도"""
        return self.quiz_manager.retry_wrong_questions()


def main():
    """메인 실행 함수"""
    quiz = GANQuiz()
    
    print("GAN 퀴즈 시스템")
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