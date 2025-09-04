#!/usr/bin/env python3
"""
YOLO 퀴즈

이 퀴즈는 다음 내용을 평가합니다:
1. 객체 탐지 vs 분류 차이점
2. 바운딩 박스 좌표 계산 및 IoU 개념
3. NMS 알고리즘 원리 및 필요성
4. mAP 평가 메트릭 이해
5. YOLO 아키텍처 및 발전 과정

요구사항 8.1, 8.2, 8.3을 충족합니다.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from quiz_utils import DeepLearningQuizManager, QuizQuestion
import math


class YOLOQuiz:
    def __init__(self):
        self.quiz_manager = DeepLearningQuizManager("YOLO 퀴즈")
        self.setup_questions()
    
    def setup_questions(self):
        """퀴즈 문제 설정"""
        
        # 1. 객체 탐지 기본 개념 - Easy
        self.quiz_manager.add_question_simple(
            question_id="yolo_001",
            question_type="multiple_choice",
            question="객체 탐지(Object Detection)와 이미지 분류(Image Classification)의 주요 차이점은?",
            options=[
                "사용하는 신경망 구조가 완전히 다름",
                "객체 탐지는 '무엇'과 '어디'를 모두 예측함",
                "이미지 분류가 더 복잡함",
                "데이터셋 크기가 다름"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 객체 탐지는 '무엇'과 '어디'를 모두 예측함

해설:
이미지 분류는 이미지에 무엇이 있는지만 분류하지만,
객체 탐지는 무엇이 있는지(분류)와 어디에 있는지(위치)를 모두 예측합니다.
            """,
            difficulty="easy",
            topic="YOLO",
            related_theory_section="YOLO 이론 - 객체 탐지 개념"
        )
        
        # 2. YOLO 의미 - Easy
        self.quiz_manager.add_question_simple(
            question_id="yolo_002",
            question_type="short_answer",
            question="YOLO의 풀네임은?",
            correct_answer="You Only Look Once",
            explanation="""
정답: You Only Look Once

해설:
YOLO는 "You Only Look Once"의 줄임말로, 이미지를 한 번만 보고
모든 객체를 동시에 탐지한다는 의미입니다.
            """,
            difficulty="easy",
            topic="YOLO",
            related_theory_section="YOLO 이론 - YOLO 알고리즘 원리"
        )
        
        # 3. 바운딩 박스 - Medium
        self.quiz_manager.add_question_simple(
            question_id="yolo_003",
            question_type="multiple_choice",
            question="바운딩 박스(Bounding Box)를 표현하는 일반적인 방법은?",
            options=[
                "(x, y, width, height)",
                "(x1, y1, x2, y2)",
                "(center_x, center_y, width, height)",
                "위의 모든 것"
            ],
            correct_answer=4,
            explanation="""
정답: 4) 위의 모든 것

해설:
바운딩 박스는 여러 방식으로 표현 가능합니다:
- (x, y, w, h): 좌상단 좌표 + 크기
- (x1, y1, x2, y2): 좌상단 + 우하단 좌표
- (cx, cy, w, h): 중심점 + 크기 (YOLO에서 주로 사용)
            """,
            difficulty="medium",
            topic="YOLO",
            related_theory_section="YOLO 이론 - 바운딩 박스"
        )
        
        # 4. IoU 계산 - Hard
        self.quiz_manager.add_question_simple(
            question_id="yolo_004",
            question_type="calculation",
            question="두 박스 A(0,0,4,4)와 B(2,2,6,6)의 IoU는? (소수점 둘째 자리까지)",
            correct_answer=0.14,
            explanation="""
정답: 0.14

해설:
교집합 영역: (2,2)에서 (4,4)까지 = 2×2 = 4
합집합 영역: A면적 + B면적 - 교집합 = 16 + 16 - 4 = 28
IoU = 교집합/합집합 = 4/28 = 0.143 ≈ 0.14
            """,
            difficulty="hard",
            topic="YOLO",
            related_theory_section="YOLO 이론 - IoU 계산",
            formula="IoU = 교집합 면적 / 합집합 면적",
            tolerance=0.02
        )
        
        # 5. 그리드 시스템 - Medium
        self.quiz_manager.add_question_simple(
            question_id="yolo_005",
            question_type="multiple_choice",
            question="YOLO v1에서 이미지를 7×7 그리드로 나누는 이유는?",
            options=[
                "계산 속도 향상",
                "각 그리드 셀이 해당 영역의 객체를 담당하도록 함",
                "메모리 사용량 감소",
                "정확도 향상"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 각 그리드 셀이 해당 영역의 객체를 담당하도록 함

해설:
YOLO는 이미지를 그리드로 나누고, 각 그리드 셀이 그 영역에 있는 
객체의 중심점을 담당하도록 하여 객체 탐지를 단순화합니다.
            """,
            difficulty="medium",
            topic="YOLO",
            related_theory_section="YOLO 이론 - 그리드 기반 탐지"
        )
        
        # 6. 신뢰도 점수 - Medium
        self.quiz_manager.add_question_simple(
            question_id="yolo_006",
            question_type="multiple_choice",
            question="YOLO에서 신뢰도 점수(Confidence Score)는 무엇을 나타내는가?",
            options=[
                "클래스 확률만",
                "바운딩 박스 정확도만",
                "객체 존재 확률 × IoU",
                "손실 함수 값"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 객체 존재 확률 × IoU

해설:
신뢰도 점수 = Pr(Object) × IoU
객체가 존재할 확률과 예측 박스의 정확도(IoU)를 곱한 값으로,
박스의 전반적인 품질을 나타냅니다.
            """,
            difficulty="medium",
            topic="YOLO",
            related_theory_section="YOLO 이론 - 신뢰도 점수"
        )
        
        # 7. NMS 필요성 - Medium
        self.quiz_manager.add_question_simple(
            question_id="yolo_007",
            question_type="multiple_choice",
            question="Non-Maximum Suppression(NMS)이 필요한 이유는?",
            options=[
                "계산 속도 향상",
                "같은 객체에 대한 중복 탐지 제거",
                "메모리 사용량 감소",
                "정확도 향상"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 같은 객체에 대한 중복 탐지 제거

해설:
객체 탐지 모델은 하나의 객체에 대해 여러 개의 바운딩 박스를 생성할 수 있습니다.
NMS는 IoU가 높은 중복 박스들 중 신뢰도가 가장 높은 것만 남기고 나머지를 제거합니다.
            """,
            difficulty="medium",
            topic="YOLO",
            related_theory_section="YOLO 이론 - NMS 알고리즘"
        )
        
        # 8. NMS 과정 - Hard
        self.quiz_manager.add_question_simple(
            question_id="yolo_008",
            question_type="true_false",
            question="NMS에서는 신뢰도가 가장 높은 박스를 선택하고, 이와 IoU가 임계값보다 높은 박스들을 제거한다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
NMS 알고리즘:
1. 신뢰도 순으로 박스 정렬
2. 가장 높은 신뢰도 박스 선택
3. 이 박스와 IoU > 임계값인 박스들 제거
4. 남은 박스들에 대해 반복
            """,
            difficulty="hard",
            topic="YOLO",
            related_theory_section="YOLO 이론 - NMS 알고리즘"
        )
        
        # 9. mAP 개념 - Hard
        self.quiz_manager.add_question_simple(
            question_id="yolo_009",
            question_type="multiple_choice",
            question="mAP(mean Average Precision)에서 'mean'이 의미하는 것은?",
            options=[
                "여러 이미지의 평균",
                "여러 클래스의 평균",
                "여러 IoU 임계값의 평균",
                "여러 모델의 평균"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 여러 클래스의 평균

해설:
mAP는 각 클래스별로 계산된 Average Precision(AP)의 평균입니다.
예: 3개 클래스의 AP가 0.8, 0.7, 0.9라면 mAP = (0.8+0.7+0.9)/3 = 0.8
            """,
            difficulty="hard",
            topic="YOLO",
            related_theory_section="YOLO 이론 - mAP 평가 메트릭"
        )
        
        # 10. Precision vs Recall - Medium
        self.quiz_manager.add_question_simple(
            question_id="yolo_010",
            question_type="multiple_choice",
            question="객체 탐지에서 Precision의 정의는?",
            options=[
                "탐지된 객체 중 올바른 것의 비율",
                "전체 객체 중 탐지된 것의 비율",
                "올바르게 분류된 픽셀의 비율",
                "IoU의 평균값"
            ],
            correct_answer=1,
            explanation="""
정답: 1) 탐지된 객체 중 올바른 것의 비율

해설:
Precision = TP / (TP + FP)
모델이 탐지한 것 중에서 실제로 맞는 것의 비율입니다.
Recall = TP / (TP + FN)은 실제 객체 중 탐지된 것의 비율입니다.
            """,
            difficulty="medium",
            topic="YOLO",
            related_theory_section="YOLO 이론 - Precision과 Recall"
        )
        
        # 11. YOLO 손실 함수 - Hard
        self.quiz_manager.add_question_simple(
            question_id="yolo_011",
            question_type="multiple_choice",
            question="YOLO의 손실 함수에 포함되지 않는 것은?",
            options=[
                "바운딩 박스 좌표 손실",
                "신뢰도 손실",
                "클래스 분류 손실",
                "IoU 손실"
            ],
            correct_answer=4,
            explanation="""
정답: 4) IoU 손실

해설:
YOLO v1의 손실 함수는 다음을 포함합니다:
1. 바운딩 박스 좌표 손실 (x, y, w, h)
2. 신뢰도 손실 (객체 있음/없음)
3. 클래스 분류 손실
IoU는 평가 메트릭이지 직접적인 손실 항목은 아닙니다.
            """,
            difficulty="hard",
            topic="YOLO",
            related_theory_section="YOLO 이론 - 손실 함수"
        )
        
        # 12. 앵커 박스 - Medium
        self.quiz_manager.add_question_simple(
            question_id="yolo_012",
            question_type="multiple_choice",
            question="YOLO v2부터 도입된 앵커 박스(Anchor Box)의 목적은?",
            options=[
                "계산 속도 향상",
                "다양한 크기와 비율의 객체를 더 잘 탐지",
                "메모리 사용량 감소",
                "클래스 수 증가"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 다양한 크기와 비율의 객체를 더 잘 탐지

해설:
앵커 박스는 미리 정의된 다양한 크기와 비율의 박스 템플릿으로,
다양한 형태의 객체를 더 효과적으로 탐지할 수 있게 해줍니다.
            """,
            difficulty="medium",
            topic="YOLO",
            related_theory_section="YOLO 이론 - 앵커 박스"
        )
        
        # 13. YOLO 발전 과정 - Medium
        self.quiz_manager.add_question_simple(
            question_id="yolo_013",
            question_type="multiple_choice",
            question="YOLO v3의 주요 개선사항은?",
            options=[
                "더 빠른 속도",
                "멀티스케일 예측 및 FPN 사용",
                "더 적은 파라미터",
                "더 간단한 구조"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 멀티스케일 예측 및 FPN 사용

해설:
YOLO v3는 3개의 서로 다른 스케일에서 예측을 수행하고
Feature Pyramid Network(FPN) 구조를 사용하여
다양한 크기의 객체를 더 잘 탐지할 수 있게 되었습니다.
            """,
            difficulty="medium",
            topic="YOLO",
            related_theory_section="YOLO 이론 - YOLO 버전별 발전"
        )
        
        # 14. 실시간 탐지 - Easy
        self.quiz_manager.add_question_simple(
            question_id="yolo_014",
            question_type="true_false",
            question="YOLO는 실시간 객체 탐지가 가능하도록 설계되었다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
YOLO는 "You Only Look Once"라는 이름처럼 이미지를 한 번만 보고
모든 객체를 동시에 탐지하여 빠른 속도를 달성합니다.
실시간 응용(비디오, 자율주행 등)을 목표로 설계되었습니다.
            """,
            difficulty="easy",
            topic="YOLO",
            related_theory_section="YOLO 이론 - 실시간 탐지"
        )
        
        # 15. COCO 데이터셋 - Easy
        self.quiz_manager.add_question_simple(
            question_id="yolo_015",
            question_type="numerical",
            question="COCO 데이터셋에는 총 몇 개의 객체 클래스가 있는가?",
            correct_answer=80,
            explanation="""
정답: 80

해설:
COCO(Common Objects in Context) 데이터셋은 80개의 일반적인 객체 클래스를 포함합니다.
사람, 자동차, 동물, 가구 등 일상생활에서 볼 수 있는 다양한 객체들이 포함되어 있습니다.
            """,
            difficulty="easy",
            topic="YOLO",
            related_theory_section="YOLO 이론 - COCO 데이터셋",
            tolerance=0
        )
    
    def run_quiz(self, num_questions: int = 15, difficulty: str = None):
        """퀴즈 실행"""
        print("🎯 YOLO 퀴즈에 오신 것을 환영합니다!")
        print("이 퀴즈는 객체 탐지와 YOLO 알고리즘을 평가합니다.")
        print("-" * 60)
        
        results = self.quiz_manager.run_full_quiz(
            topic="YOLO",
            difficulty=difficulty,
            num_questions=num_questions
        )
        
        return results
    
    def run_retry_quiz(self):
        """틀린 문제 재시도"""
        return self.quiz_manager.retry_wrong_questions()


def main():
    """메인 실행 함수"""
    quiz = YOLOQuiz()
    
    print("YOLO 퀴즈 시스템")
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