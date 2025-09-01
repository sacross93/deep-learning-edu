#!/usr/bin/env python3
"""
데이터 마이닝 개요 퀴즈

이 퀴즈는 다음 내용을 평가합니다:
1. 데이터 마이닝 정의 및 개념 이해
2. 속성 유형 분류 능력
3. 데이터 집합 유형 구분
4. 예측 vs 설명 과제 이해
5. 데이터 특성 파악
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from quiz_utils import QuizManager
import random

class DataMiningOverviewQuiz:
    def __init__(self):
        self.quiz_manager = QuizManager("데이터 마이닝 개요")
        self.setup_questions()
    
    def setup_questions(self):
        """퀴즈 문제 설정"""
        
        # 1. 데이터 마이닝 정의 관련 문제
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="데이터 마이닝의 정의로 가장 적절한 것은?",
            options=[
                "A) 단순히 데이터베이스에서 정보를 검색하는 과정",
                "B) 전통적 데이터 분석과 대규모 데이터 처리 알고리즘을 결합하여 숨겨진 패턴을 발견하는 과정",
                "C) 데이터를 시각화하여 보여주는 과정",
                "D) 데이터를 정리하고 저장하는 과정"
            ],
            correct_answer="B",
            explanation="""
정답: B) 전통적 데이터 분석과 대규모 데이터 처리 알고리즘을 결합하여 숨겨진 패턴을 발견하는 과정

해설:
데이터 마이닝은 단순한 데이터 검색이나 시각화가 아닙니다. 
대용량 데이터에서 암묵적이고 숨겨진 유용한 정보를 비자명하게 추출하는 과정으로,
전통적 분석 기법과 현대적 알고리즘을 결합한 것이 핵심입니다.
            """,
            difficulty="easy"
        )
        
        # 2. 전통적 분석과의 차이점
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="전통적 데이터 분석과 데이터 마이닝의 주요 차이점이 아닌 것은?",
            options=[
                "A) 데이터 규모: 소규모 vs 대규모",
                "B) 분석 방법: 가설 기반 vs 패턴 발견",
                "C) 처리 방식: 수동적 vs 자동화",
                "D) 사용 도구: 엑셀 vs 파이썬만 사용"
            ],
            correct_answer="D",
            explanation="""
정답: D) 사용 도구: 엑셀 vs 파이썬만 사용

해설:
도구의 차이는 본질적인 차이가 아닙니다. 데이터 마이닝도 다양한 도구를 사용할 수 있습니다.
주요 차이점은:
- 데이터 규모 (소규모 vs 대규모)
- 분석 접근법 (가설 검증 vs 패턴 발견)
- 처리 능력 (제한적 vs 자동화된 대용량 처리)
            """,
            difficulty="medium"
        )
        
        # 3. 속성 유형 분류 - 명목형
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="다음 중 명목형(Nominal) 속성에 해당하는 것은?",
            options=[
                "A) 학점 (A, B, C, D, F)",
                "B) 온도 (섭씨)",
                "C) 혈액형 (A, B, AB, O)",
                "D) 나이"
            ],
            correct_answer="C",
            explanation="""
정답: C) 혈액형 (A, B, AB, O)

해설:
명목형 속성은 단순한 구분을 위한 라벨로, 순서나 크기의 의미가 없습니다.
- A) 학점: 순서형 (A > B > C > D > F)
- B) 온도: 구간형 (차이는 의미 있지만 절대 0점 없음)
- C) 혈액형: 명목형 (단순 구분, 순서 없음) ✓
- D) 나이: 비율형 (절대 0점 있음)
            """,
            difficulty="easy"
        )
        
        # 4. 속성 유형 분류 - 비율형
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="비율형(Ratio) 속성의 특징으로 옳지 않은 것은?",
            options=[
                "A) 절대적 0점이 존재한다",
                "B) 비율 계산이 의미가 있다",
                "C) 모든 수학적 연산이 가능하다",
                "D) 순서는 있지만 차이는 의미가 없다"
            ],
            correct_answer="D",
            explanation="""
정답: D) 순서는 있지만 차이는 의미가 없다

해설:
비율형 속성은 가장 높은 수준의 측정으로 모든 특성을 가집니다:
- 구분 (명목형 특성)
- 순서 (순서형 특성)  
- 차이 (구간형 특성)
- 비율 (비율형 특성)

따라서 순서뿐만 아니라 차이도 의미가 있습니다.
예: 몸무게 60kg과 30kg의 차이는 30kg이고, 60kg은 30kg의 2배입니다.
            """,
            difficulty="medium"
        )
        
        # 5. 데이터 집합 유형
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="장바구니 분석에서 사용되는 데이터 집합 유형은?",
            options=[
                "A) 레코드 데이터",
                "B) 문서 데이터", 
                "C) 거래 데이터",
                "D) 그래프 데이터"
            ],
            correct_answer="C",
            explanation="""
정답: C) 거래 데이터

해설:
장바구니 분석은 고객의 구매 패턴을 분석하는 것으로, 각 거래(트랜잭션)에서
구매된 상품들의 집합을 다룹니다. 이는 전형적인 거래 데이터입니다.

데이터 집합 유형별 특징:
- 레코드: 고정된 속성을 가진 테이블 형태
- 문서: 텍스트를 벡터로 변환
- 거래: 트랜잭션과 관련 항목들의 집합 ✓
- 그래프: 노드와 엣지의 네트워크 구조
            """,
            difficulty="easy"
        )
        
        # 6. 예측 vs 설명 과제
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="다음 중 '설명(Description)' 과제에 해당하는 것은?",
            options=[
                "A) 고객 이탈 예측",
                "B) 스팸 메일 분류",
                "C) 고객 세분화",
                "D) 주가 예측"
            ],
            correct_answer="C",
            explanation="""
정답: C) 고객 세분화

해설:
설명 과제는 데이터의 숨겨진 패턴과 구조를 발견하는 것입니다.

예측 과제 (지도학습):
- A) 고객 이탈 예측: 미래 이탈 여부 예측
- B) 스팸 메일 분류: 메일의 스팸 여부 분류  
- D) 주가 예측: 미래 주가 예측

설명 과제 (비지도학습):
- C) 고객 세분화: 고객들의 숨겨진 그룹 구조 발견 ✓
            """,
            difficulty="medium"
        )
        
        # 7. 속성 유형 종합 문제
        self.quiz_manager.add_question(
            question_type="matching",
            question="다음 속성들을 올바른 유형과 연결하세요:",
            options={
                "온도(섭씨)": ["명목형", "순서형", "구간형", "비율형"],
                "만족도(1-5점)": ["명목형", "순서형", "구간형", "비율형"],
                "성별": ["명목형", "순서형", "구간형", "비율형"],
                "소득": ["명목형", "순서형", "구간형", "비율형"]
            },
            correct_answer={
                "온도(섭씨)": "구간형",
                "만족도(1-5점)": "순서형", 
                "성별": "명목형",
                "소득": "비율형"
            },
            explanation="""
정답:
- 온도(섭씨): 구간형 (차이는 의미 있지만 절대 0점 없음)
- 만족도(1-5점): 순서형 (순위는 있지만 간격이 동일하지 않을 수 있음)
- 성별: 명목형 (단순 구분, 순서 없음)
- 소득: 비율형 (절대 0점 있고 모든 연산 가능)

각 유형의 특징을 기억하세요:
- 명목형: 구분만
- 순서형: 구분 + 순서
- 구간형: 구분 + 순서 + 차이
- 비율형: 구분 + 순서 + 차이 + 비율
            """,
            difficulty="hard"
        )
        
        # 8. 데이터 특성 이해
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="현대 데이터 마이닝이 다루는 데이터의 특성이 아닌 것은?",
            options=[
                "A) 대규모(Large-scale)",
                "B) 고차원(High-dimensional)", 
                "C) 단순구조(Simple-structured)",
                "D) 분산형(Distributed)"
            ],
            correct_answer="C",
            explanation="""
정답: C) 단순구조(Simple-structured)

해설:
현대 데이터 마이닝이 다루는 데이터는 복잡한 특성을 가집니다:

✓ 대규모: 테라바이트 이상의 방대한 데이터
✓ 고차원: 수백, 수천 개의 속성
✗ 단순구조: 실제로는 복잡(Complex)한 구조
✓ 분산형: 여러 위치에 분산 저장

추가 특성:
- 이질적(Heterogeneous): 다양한 형태와 구조
- 복잡(Complex): 비선형적이고 복잡한 관계
            """,
            difficulty="medium"
        )
        
        # 9. 실제 적용 시나리오
        self.quiz_manager.add_question(
            question_type="short_answer",
            question="온라인 쇼핑몰에서 '이 상품을 구매한 고객들이 함께 구매한 상품' 추천 시스템을 구축하려고 합니다. 이는 예측 과제인가요, 설명 과제인가요? 그 이유도 함께 설명하세요.",
            correct_answer="설명",
            explanation="""
정답: 설명 과제

해설:
이는 연관 규칙 마이닝을 통한 설명 과제입니다.

이유:
1. 목표: 상품 간의 숨겨진 연관성 패턴 발견
2. 방법: 거래 데이터에서 함께 구매되는 패턴 탐색
3. 특성: 라벨이 없는 비지도학습 방식
4. 결과: "A를 구매하면 B도 구매할 확률이 높다"는 규칙 발견

예측 과제라면 "특정 고객이 특정 상품을 구매할 확률"을 예측하는 것이지만,
여기서는 상품 간의 관계 패턴을 설명하는 것이 목적입니다.
            """,
            difficulty="hard"
        )
        
        # 10. 종합 이해도 문제
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="다음 중 데이터 마이닝과 관련 학문의 연결이 올바르지 않은 것은?",
            options=[
                "A) 머신러닝 - 패턴 학습 및 예측 알고리즘",
                "B) 통계학 - 데이터 분석 및 추론 방법론",
                "C) 데이터베이스 - 대용량 데이터 저장 및 관리",
                "D) 심리학 - 데이터 시각화 및 인터페이스 설계"
            ],
            correct_answer="D",
            explanation="""
정답: D) 심리학 - 데이터 시각화 및 인터페이스 설계

해설:
데이터 마이닝의 주요 관련 학문:

✓ A) 머신러닝/AI: 패턴 학습, 예측 알고리즘 제공
✓ B) 통계학: 데이터 분석, 추론 방법론 제공  
✓ C) 데이터베이스: 대용량 데이터 저장, 관리 기술 제공
✗ D) 심리학: 직접적 관련성 낮음

실제 관련 학문:
- 패턴 인식: 데이터에서 의미 있는 패턴 식별
- 정보 이론: 정보량, 엔트로피 등의 개념 제공
            """,
            difficulty="medium"
        )

    def run_quiz(self):
        """퀴즈 실행"""
        print("=" * 60)
        print("데이터 마이닝 개요 퀴즈")
        print("=" * 60)
        print("이 퀴즈는 데이터 마이닝의 기본 개념과 이론을 평가합니다.")
        print("총 10문제로 구성되어 있으며, 각 문제마다 상세한 해설이 제공됩니다.")
        print("-" * 60)
        
        # 퀴즈 시작
        self.quiz_manager.start_quiz()
        
        # 결과 출력
        score = self.quiz_manager.get_score()
        total = self.quiz_manager.get_total_questions()
        percentage = (score / total) * 100
        
        print(f"\n퀴즈 완료!")
        print(f"점수: {score}/{total} ({percentage:.1f}%)")
        
        # 성취도별 피드백
        if percentage >= 90:
            print("🎉 우수! 데이터 마이닝 개념을 매우 잘 이해하고 있습니다.")
        elif percentage >= 80:
            print("👍 양호! 대부분의 개념을 잘 이해하고 있습니다.")
        elif percentage >= 70:
            print("📚 보통! 기본 개념은 이해했지만 더 학습이 필요합니다.")
        else:
            print("💪 노력 필요! 이론 문서를 다시 읽어보시기 바랍니다.")
        
        # 틀린 문제 복습 제안
        wrong_questions = self.quiz_manager.get_wrong_questions()
        if wrong_questions:
            print(f"\n복습 권장 주제:")
            for topic in wrong_questions:
                print(f"  - {topic}")
        
        print("\n다음 단계: 데이터 품질 및 전처리 학습")
        print("=" * 60)

def main():
    """메인 함수"""
    quiz = DataMiningOverviewQuiz()
    quiz.run_quiz()

if __name__ == "__main__":
    main()