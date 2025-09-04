#!/usr/bin/env python3
"""
CNN 퀴즈

이 퀴즈는 다음 내용을 평가합니다:
1. 합성곱 연산 및 특성 맵 크기 계산
2. 풀링 레이어 효과 및 파라미터 수 계산
3. CNN 아키텍처 설계 원칙
4. 전이 학습 개념 및 적용 방법
5. CNN의 장단점 및 응용 분야

요구사항 8.1, 8.2, 8.3을 충족합니다.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from quiz_utils import DeepLearningQuizManager, QuizQuestion
import math


class CNNQuiz:
    def __init__(self):
        self.quiz_manager = DeepLearningQuizManager("CNN 퀴즈")
        self.setup_questions()
    
    def setup_questions(self):
        """퀴즈 문제 설정"""
        
        # 1. 합성곱 연산 기초 - Easy
        self.quiz_manager.add_question_simple(
            question_id="cnn_001",
            question_type="multiple_choice",
            question="합성곱(Convolution) 연산의 주요 목적은?",
            options=[
                "이미지 크기를 줄임",
                "지역적 특성을 추출함",
                "색상을 변경함",
                "노이즈를 제거함"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 지역적 특성을 추출함

해설:
합성곱 연산은 필터(커널)를 사용하여 이미지의 지역적 패턴과 특성을 추출합니다.
에지, 코너, 텍스처 등의 특성을 감지하여 이미지를 이해하는 데 도움을 줍니다.
            """,
            difficulty="easy",
            topic="CNN",
            related_theory_section="CNN 이론 - 합성곱 연산"
        )
        
        # 2. 특성 맵 크기 계산 - Medium
        self.quiz_manager.add_question_simple(
            question_id="cnn_002",
            question_type="calculation",
            question="32×32 이미지에 5×5 필터, stride=1, padding=0을 적용할 때 출력 크기는?",
            correct_answer=28,
            explanation="""
정답: 28

해설:
출력 크기 공식: (입력 크기 - 필터 크기 + 2×패딩) / 스트라이드 + 1
= (32 - 5 + 2×0) / 1 + 1
= 27 / 1 + 1 = 28

따라서 출력은 28×28 크기가 됩니다.
            """,
            difficulty="medium",
            topic="CNN",
            related_theory_section="CNN 이론 - 합성곱 연산",
            formula="출력 크기 = (입력 크기 - 필터 크기 + 2×패딩) / 스트라이드 + 1",
            tolerance=0
        )
        
        # 3. 패딩의 역할 - Easy
        self.quiz_manager.add_question_simple(
            question_id="cnn_003",
            question_type="multiple_choice",
            question="패딩(Padding)을 사용하는 주요 이유는?",
            options=[
                "계산 속도 향상",
                "출력 크기 유지 및 경계 정보 보존",
                "메모리 사용량 감소",
                "색상 정보 보존"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 출력 크기 유지 및 경계 정보 보존

해설:
패딩은 입력 주변에 0을 추가하여 합성곱 후에도 원본 크기를 유지하고,
이미지 경계의 정보가 손실되는 것을 방지합니다.
            """,
            difficulty="easy",
            topic="CNN",
            related_theory_section="CNN 이론 - 패딩"
        )
        
        # 4. 풀링 레이어 효과 - Medium
        self.quiz_manager.add_question_simple(
            question_id="cnn_004",
            question_type="multiple_choice",
            question="Max Pooling의 주요 효과가 아닌 것은?",
            options=[
                "특성 맵 크기 감소",
                "위치 불변성 증가",
                "파라미터 수 증가",
                "과적합 방지"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 파라미터 수 증가

해설:
Max Pooling은 학습 가능한 파라미터가 없으므로 파라미터 수를 증가시키지 않습니다.
오히려 특성 맵 크기를 줄여 전체 파라미터 수를 감소시키는 효과가 있습니다.
            """,
            difficulty="medium",
            topic="CNN",
            related_theory_section="CNN 이론 - 풀링 레이어"
        )
        
        # 5. CNN 파라미터 계산 - Hard
        self.quiz_manager.add_question_simple(
            question_id="cnn_005",
            question_type="calculation",
            question="3×3 필터 64개를 가진 합성곱 레이어(입력 채널 32)의 파라미터 수는? (편향 포함)",
            correct_answer=18496,
            explanation="""
정답: 18,496

해설:
파라미터 수 = (필터 크기 × 입력 채널 × 출력 채널) + 편향
= (3 × 3 × 32 × 64) + 64
= 18,432 + 64 = 18,496
            """,
            difficulty="hard",
            topic="CNN",
            related_theory_section="CNN 이론 - 파라미터 계산",
            formula="파라미터 수 = (필터 크기 × 입력 채널 × 출력 채널) + 편향",
            tolerance=0
        )
        
        # 6. 스트라이드 효과 - Medium
        self.quiz_manager.add_question_simple(
            question_id="cnn_006",
            question_type="true_false",
            question="스트라이드(Stride)가 클수록 출력 특성 맵의 크기는 작아진다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
스트라이드는 필터가 이동하는 간격을 의미합니다.
스트라이드가 클수록 필터가 큰 간격으로 이동하므로 출력 크기가 작아집니다.
            """,
            difficulty="medium",
            topic="CNN",
            related_theory_section="CNN 이론 - 스트라이드"
        )
        
        # 7. 전이 학습 - Medium
        self.quiz_manager.add_question_simple(
            question_id="cnn_007",
            question_type="multiple_choice",
            question="전이 학습(Transfer Learning)에서 일반적으로 고정(freeze)하는 부분은?",
            options=[
                "마지막 분류 레이어만",
                "첫 번째 레이어만",
                "초기 특성 추출 레이어들",
                "모든 레이어"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 초기 특성 추출 레이어들

해설:
전이 학습에서는 사전 훈련된 모델의 초기 레이어들(에지, 텍스처 등 일반적 특성 추출)을 
고정하고, 후반부 레이어들만 새로운 데이터에 맞게 미세 조정합니다.
            """,
            difficulty="medium",
            topic="CNN",
            related_theory_section="CNN 이론 - 전이 학습"
        )
        
        # 8. 데이터 증강 - Easy
        self.quiz_manager.add_question_simple(
            question_id="cnn_008",
            question_type="multiple_choice",
            question="데이터 증강(Data Augmentation)의 주요 목적은?",
            options=[
                "훈련 속도 향상",
                "과적합 방지 및 일반화 성능 향상",
                "메모리 사용량 감소",
                "이미지 품질 향상"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 과적합 방지 및 일반화 성능 향상

해설:
데이터 증강은 회전, 이동, 크기 조정 등을 통해 훈련 데이터를 인위적으로 늘려
모델이 다양한 변형에 강건하게 만들어 과적합을 방지합니다.
            """,
            difficulty="easy",
            topic="CNN",
            related_theory_section="CNN 이론 - 데이터 증강"
        )
        
        # 9. CNN 아키텍처 - Medium
        self.quiz_manager.add_question_simple(
            question_id="cnn_009",
            question_type="multiple_choice",
            question="ResNet의 핵심 아이디어는?",
            options=[
                "더 큰 필터 사용",
                "잔차 연결(Residual Connection)",
                "더 많은 풀링 레이어",
                "배치 크기 증가"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 잔차 연결(Residual Connection)

해설:
ResNet은 잔차 연결을 통해 그래디언트 소실 문제를 해결하여
매우 깊은 네트워크 훈련을 가능하게 했습니다.
F(x) + x 형태로 입력을 출력에 직접 더합니다.
            """,
            difficulty="medium",
            topic="CNN",
            related_theory_section="CNN 이론 - CNN 아키텍처"
        )
        
        # 10. 1×1 합성곱 - Hard
        self.quiz_manager.add_question_simple(
            question_id="cnn_010",
            question_type="multiple_choice",
            question="1×1 합성곱의 주요 용도는?",
            options=[
                "이미지 크기 변경",
                "채널 수 조정 및 차원 축소",
                "노이즈 제거",
                "색상 변환"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 채널 수 조정 및 차원 축소

해설:
1×1 합성곱은 공간적 크기는 유지하면서 채널 수만 변경할 수 있어
차원 축소나 채널 간 정보 결합에 사용됩니다. (Network in Network에서 제안)
            """,
            difficulty="hard",
            topic="CNN",
            related_theory_section="CNN 이론 - 1×1 합성곱"
        )
        
        # 11. 배치 정규화 위치 - Medium
        self.quiz_manager.add_question_simple(
            question_id="cnn_011",
            question_type="multiple_choice",
            question="CNN에서 배치 정규화를 적용하는 일반적인 위치는?",
            options=[
                "입력 이미지에 직접",
                "합성곱 후, 활성화 함수 전",
                "풀링 레이어 후",
                "손실 함수 계산 전"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 합성곱 후, 활성화 함수 전

해설:
일반적으로 Conv → BatchNorm → Activation 순서로 적용합니다.
이는 각 레이어의 입력 분포를 안정화하여 학습을 개선합니다.
            """,
            difficulty="medium",
            topic="CNN",
            related_theory_section="CNN 이론 - 배치 정규화"
        )
        
        # 12. 수용 영역 - Hard
        self.quiz_manager.add_question_simple(
            question_id="cnn_012",
            question_type="short_answer",
            question="CNN에서 깊은 레이어의 뉴런이 입력 이미지에서 영향받는 영역을 무엇이라 하는가?",
            correct_answer="수용 영역",
            explanation="""
정답: 수용 영역 (Receptive Field)

해설:
수용 영역은 특정 뉴런의 출력에 영향을 미치는 입력 영역의 크기입니다.
CNN에서 레이어가 깊어질수록 수용 영역이 커져 더 넓은 범위의 정보를 고려합니다.
            """,
            difficulty="hard",
            topic="CNN",
            related_theory_section="CNN 이론 - 수용 영역"
        )
        
        # 13. 깊이별 분리 합성곱 - Hard
        self.quiz_manager.add_question_simple(
            question_id="cnn_013",
            question_type="multiple_choice",
            question="Depthwise Separable Convolution의 장점은?",
            options=[
                "정확도 향상",
                "파라미터 수 및 연산량 감소",
                "메모리 사용량 증가",
                "훈련 시간 증가"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 파라미터 수 및 연산량 감소

해설:
깊이별 분리 합성곱은 공간적 합성곱과 점별 합성곱을 분리하여
표준 합성곱 대비 파라미터 수와 연산량을 크게 줄입니다.
MobileNet에서 사용되는 핵심 기법입니다.
            """,
            difficulty="hard",
            topic="CNN",
            related_theory_section="CNN 이론 - 효율적 아키텍처"
        )
        
        # 14. CNN vs FC - Easy
        self.quiz_manager.add_question_simple(
            question_id="cnn_014",
            question_type="true_false",
            question="CNN은 완전연결층보다 이미지의 공간적 구조를 더 잘 보존한다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
CNN은 지역적 연결과 가중치 공유를 통해 이미지의 공간적 구조와 
위치 정보를 보존하는 반면, 완전연결층은 모든 픽셀을 평면화하여 
공간 정보를 잃습니다.
            """,
            difficulty="easy",
            topic="CNN",
            related_theory_section="CNN 이론 - CNN vs 완전연결층"
        )
        
        # 15. 객체 탐지 vs 분류 - Medium
        self.quiz_manager.add_question_simple(
            question_id="cnn_015",
            question_type="multiple_choice",
            question="이미지 분류와 객체 탐지의 주요 차이점은?",
            options=[
                "사용하는 CNN 구조가 완전히 다름",
                "객체 탐지는 위치 정보도 예측해야 함",
                "이미지 분류가 더 복잡함",
                "데이터셋 크기가 다름"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 객체 탐지는 위치 정보도 예측해야 함

해설:
이미지 분류는 "무엇인가?"만 답하지만, 객체 탐지는 "무엇이 어디에 있는가?"를 
답해야 하므로 클래스 분류와 바운딩 박스 회귀를 동시에 수행합니다.
            """,
            difficulty="medium",
            topic="CNN",
            related_theory_section="CNN 이론 - 응용 분야"
        )
    
    def run_quiz(self, num_questions: int = 15, difficulty: str = None):
        """퀴즈 실행"""
        print("🖼️ CNN 퀴즈에 오신 것을 환영합니다!")
        print("이 퀴즈는 합성곱 신경망의 구조와 원리를 평가합니다.")
        print("-" * 60)
        
        results = self.quiz_manager.run_full_quiz(
            topic="CNN",
            difficulty=difficulty,
            num_questions=num_questions
        )
        
        return results
    
    def run_retry_quiz(self):
        """틀린 문제 재시도"""
        return self.quiz_manager.retry_wrong_questions()


def main():
    """메인 실행 함수"""
    quiz = CNNQuiz()
    
    print("CNN 퀴즈 시스템")
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