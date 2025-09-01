#!/usr/bin/env python3
"""
데이터 품질 및 전처리 퀴즈

이 퀴즈는 다음 내용을 평가합니다:
1. 데이터 품질 문제 유형 이해
2. 결측치 처리 방법 선택
3. 이상치 탐지 및 처리 기법
4. 정규화 및 스케일링 방법
5. 전처리 파이프라인 설계
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from quiz_utils import QuizManager
import random
import numpy as np

class DataPreprocessingQuiz:
    def __init__(self):
        self.quiz_manager = QuizManager("데이터 품질 및 전처리")
        self.setup_questions()
    
    def setup_questions(self):
        """퀴즈 문제 설정"""
        
        # 1. 결측치 유형 이해
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="다음 중 MCAR(Missing Completely At Random)에 해당하는 예시는?",
            options=[
                "A) 고소득자가 소득 정보를 더 자주 누락하는 경우",
                "B) 우울증 환자가 우울증 관련 문항을 미응답하는 경우",
                "C) 설문 조사에서 무작위로 일부 문항이 인쇄되지 않은 경우",
                "D) 나이가 많은 응답자가 기술 관련 문항을 더 자주 건너뛰는 경우"
            ],
            correct_answer="C",
            explanation="""
정답: C) 설문 조사에서 무작위로 일부 문항이 인쇄되지 않은 경우

해설:
MCAR(Missing Completely At Random)는 결측이 다른 어떤 변수와도 관련이 없이 완전히 무작위로 발생하는 경우입니다.

- A) MAR: 소득(관측된 변수)과 관련된 결측
- B) MNAR: 결측값 자체(우울증 정도)와 관련된 결측  
- C) MCAR: 인쇄 오류로 인한 완전 무작위 결측 ✓
- D) MAR: 나이(관측된 변수)와 관련된 결측
            """,
            difficulty="medium"
        )
        
        # 2. 결측치 처리 방법 선택
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="소규모 데이터셋에서 중요한 정보를 담고 있는 변수에 결측치가 많을 때 가장 적절한 처리 방법은?",
            options=[
                "A) 결측치가 있는 모든 행을 삭제",
                "B) 평균값으로 단순 대체",
                "C) KNN 기반 대체",
                "D) 해당 변수를 분석에서 제외"
            ],
            correct_answer="C",
            explanation="""
정답: C) KNN 기반 대체

해설:
소규모 데이터셋에서 중요한 변수의 결측치 처리 시 고려사항:

- A) 행 삭제: 소규모 데이터에서 더 큰 정보 손실
- B) 평균값 대체: 분산 축소, 관계 왜곡 위험
- C) KNN 대체: 다른 변수들과의 관계를 고려한 정교한 대체 ✓
- D) 변수 제외: 중요한 정보 완전 손실

KNN 대체는 유사한 패턴을 가진 관측치들의 정보를 활용하여 더 정확한 대체값을 제공합니다.
            """,
            difficulty="hard"
        )
        
        # 3. 이상치 탐지 방법
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="다음 이상치 탐지 방법 중 고차원 데이터에 가장 효과적인 것은?",
            options=[
                "A) Z-Score 방법",
                "B) IQR 방법",
                "C) Isolation Forest",
                "D) 박스플롯 시각화"
            ],
            correct_answer="C",
            explanation="""
정답: C) Isolation Forest

해설:
고차원 데이터에서의 이상치 탐지 방법 비교:

- A) Z-Score: 단변량 방법, 고차원에서 효과 제한적
- B) IQR: 단변량 방법, 차원별 개별 적용 필요
- C) Isolation Forest: 고차원 데이터에 특화된 앙상블 방법 ✓
- D) 박스플롯: 시각화 도구, 고차원 데이터 처리 어려움

Isolation Forest는 이상치가 정상 데이터보다 쉽게 분리된다는 원리를 이용하여 
고차원 공간에서도 효과적으로 이상치를 탐지할 수 있습니다.
            """,
            difficulty="medium"
        )
        
        # 4. 정규화 방법 선택
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="이상치가 많은 데이터에 가장 적합한 스케일링 방법은?",
            options=[
                "A) StandardScaler (Z-score 정규화)",
                "B) MinMaxScaler",
                "C) RobustScaler",
                "D) Normalizer"
            ],
            correct_answer="C",
            explanation="""
정답: C) RobustScaler

해설:
이상치가 있는 데이터의 스케일링 방법 비교:

- A) StandardScaler: 평균과 표준편차 사용, 이상치에 민감
- B) MinMaxScaler: 최솟값과 최댓값 사용, 이상치에 매우 민감
- C) RobustScaler: 중앙값과 IQR 사용, 이상치에 강건 ✓
- D) Normalizer: 각 샘플을 단위 벡터로 변환, 스케일링과 다른 목적

RobustScaler는 중앙값(median)과 IQR(Interquartile Range)을 사용하여
이상치의 영향을 최소화하면서 데이터를 스케일링합니다.
            """,
            difficulty="easy"
        )
        
        # 5. 범주형 변수 인코딩
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="'교육수준(초등학교 < 중학교 < 고등학교 < 대학교)'과 같은 순서가 있는 범주형 변수에 가장 적합한 인코딩 방법은?",
            options=[
                "A) One-Hot Encoding",
                "B) Label Encoding",
                "C) Target Encoding",
                "D) Binary Encoding"
            ],
            correct_answer="B",
            explanation="""
정답: B) Label Encoding

해설:
순서형(Ordinal) 범주형 변수의 인코딩 방법:

- A) One-Hot Encoding: 순서 정보 손실, 차원 증가
- B) Label Encoding: 순서 정보 보존, 적절한 수치 매핑 ✓
- C) Target Encoding: 목표 변수 정보 사용, 과적합 위험
- D) Binary Encoding: 순서 정보 손실

교육수준과 같이 자연스러운 순서가 있는 변수는 Label Encoding을 사용하여
순서 관계를 수치로 표현하는 것이 적절합니다.
(예: 초등학교=1, 중학교=2, 고등학교=3, 대학교=4)
            """,
            difficulty="easy"
        )
        
        # 6. 전처리 파이프라인 순서
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="올바른 전처리 파이프라인 순서는?",
            options=[
                "A) 스케일링 → 결측치 처리 → 이상치 처리 → 특성 선택",
                "B) 결측치 처리 → 이상치 처리 → 스케일링 → 특성 선택",
                "C) 특성 선택 → 스케일링 → 결측치 처리 → 이상치 처리",
                "D) 이상치 처리 → 스케일링 → 결측치 처리 → 특성 선택"
            ],
            correct_answer="B",
            explanation="""
정답: B) 결측치 처리 → 이상치 처리 → 스케일링 → 특성 선택

해설:
전처리 파이프라인의 논리적 순서:

1. 결측치 처리: 분석 가능한 완전한 데이터 확보
2. 이상치 처리: 극값이 후속 처리에 미치는 영향 최소화
3. 스케일링: 정제된 데이터의 스케일 통일
4. 특성 선택: 최종 정제된 데이터에서 중요 특성 선별

잘못된 순서의 문제점:
- 결측치가 있으면 스케일링 계산 불가
- 이상치가 있으면 스케일링 결과 왜곡
- 스케일링 전에 특성 선택하면 중요도 계산 부정확
            """,
            difficulty="medium"
        )
        
        # 7. 데이터 누출 방지
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="교차 검증에서 데이터 누출(Data Leakage)을 방지하는 올바른 방법은?",
            options=[
                "A) 전체 데이터를 전처리한 후 훈련/검증 분할",
                "B) 훈련/검증 분할 후 각각 독립적으로 전처리",
                "C) 훈련 데이터로 전처리 파라미터를 학습하고 검증 데이터에 적용",
                "D) 검증 데이터로 전처리 파라미터를 학습하고 훈련 데이터에 적용"
            ],
            correct_answer="C",
            explanation="""
정답: C) 훈련 데이터로 전처리 파라미터를 학습하고 검증 데이터에 적용

해설:
데이터 누출 방지를 위한 올바른 전처리 방법:

- A) 잘못된 방법: 미래 정보(검증 데이터)가 전처리에 사용됨
- B) 부분적으로 맞지만 불완전: 일관성 부족 문제
- C) 올바른 방법: 훈련 데이터만으로 학습, 동일한 변환을 검증에 적용 ✓
- D) 완전히 잘못된 방법: 검증 데이터 정보 사용

올바른 절차:
1. 데이터 분할 (훈련/검증)
2. 훈련 데이터로 스케일러 학습 (fit)
3. 훈련 데이터 변환 (transform)
4. 동일한 스케일러로 검증 데이터 변환 (transform only)
            """,
            difficulty="hard"
        )
        
        # 8. 윈저화(Winsorization) 이해
        self.quiz_manager.add_question(
            question_type="short_answer",
            question="윈저화(Winsorization)가 무엇인지 설명하고, 언제 사용하는 것이 적절한지 서술하세요.",
            correct_answer="극값 대체",
            explanation="""
정답: 윈저화는 극값을 특정 백분위수 값으로 대체하는 이상치 처리 방법

해설:
윈저화(Winsorization)의 정의와 특징:

정의:
- 데이터의 극값(이상치)을 특정 백분위수 값으로 대체하는 방법
- 예: 5% 윈저화 시 하위 5%, 상위 5%를 각각 5%, 95% 백분위수 값으로 대체

장점:
- 이상치의 영향을 줄이면서 데이터 손실 방지
- 분포의 전체적인 형태 유지
- 극값의 정보를 완전히 제거하지 않음

적절한 사용 시기:
- 이상치가 측정 오류로 의심되지만 확실하지 않을 때
- 데이터 손실을 최소화하면서 이상치 영향을 줄이고 싶을 때
- 금융 데이터처럼 극값이 의미가 있지만 분석에 과도한 영향을 줄 때
            """,
            difficulty="medium"
        )
        
        # 9. 실제 시나리오 문제
        self.quiz_manager.add_question(
            question_type="multiple_choice",
            question="온라인 쇼핑몰의 고객 데이터에서 '구매금액' 변수에 0원이 많이 포함되어 있습니다. 이를 로그 변환할 때 가장 적절한 방법은?",
            options=[
                "A) log(구매금액)을 직접 계산",
                "B) log(구매금액 + 1)을 계산",
                "C) 0원 데이터를 제거한 후 log(구매금액) 계산",
                "D) 0원을 평균값으로 대체한 후 log(구매금액) 계산"
            ],
            correct_answer="B",
            explanation="""
정답: B) log(구매금액 + 1)을 계산

해설:
0을 포함한 데이터의 로그 변환 방법:

- A) log(0) = -∞이므로 계산 불가능
- B) log1p 변환: log(x+1) 사용으로 0 처리 가능 ✓
- C) 데이터 손실 발생, 0원도 의미 있는 정보
- D) 0원의 의미 왜곡, 부자연스러운 대체

log1p 변환의 장점:
- 0값을 자연스럽게 처리 (log(0+1) = 0)
- 작은 값들의 차이를 더 잘 구분
- 구매하지 않은 고객(0원)의 정보 보존
- 통계적으로 안정적인 변환

실제로 많은 라이브러리에서 np.log1p() 함수를 제공합니다.
            """,
            difficulty="medium"
        )
        
        # 10. 종합 이해도 문제
        self.quiz_manager.add_question(
            question_type="matching",
            question="다음 전처리 기법들을 적절한 상황과 연결하세요:",
            options={
                "StandardScaler": ["정규분포 데이터", "이상치 많은 데이터", "범위 고정 필요", "신경망 입력"],
                "RobustScaler": ["정규분포 데이터", "이상치 많은 데이터", "범위 고정 필요", "신경망 입력"],
                "MinMaxScaler": ["정규분포 데이터", "이상치 많은 데이터", "범위 고정 필요", "신경망 입력"],
                "KNN Imputation": ["단순한 결측치", "복잡한 결측치 패턴", "범주형 결측치", "실시간 처리"]
            },
            correct_answer={
                "StandardScaler": "신경망 입력",
                "RobustScaler": "이상치 많은 데이터",
                "MinMaxScaler": "범위 고정 필요",
                "KNN Imputation": "복잡한 결측치 패턴"
            },
            explanation="""
정답:
- StandardScaler: 신경망 입력 (평균 0, 표준편차 1로 정규화)
- RobustScaler: 이상치 많은 데이터 (중앙값과 IQR 사용)
- MinMaxScaler: 범위 고정 필요 (0-1 범위로 스케일링)
- KNN Imputation: 복잡한 결측치 패턴 (다변량 관계 고려)

각 방법의 특징:
- StandardScaler: 신경망, SVM 등에서 수렴 속도 향상
- RobustScaler: 이상치에 강건한 중앙값 기반 스케일링
- MinMaxScaler: 고정된 범위가 필요한 알고리즘에 적합
- KNN Imputation: 변수 간 관계를 고려한 정교한 대체
            """,
            difficulty="hard"
        )

    def run_quiz(self):
        """퀴즈 실행"""
        print("=" * 60)
        print("데이터 품질 및 전처리 퀴즈")
        print("=" * 60)
        print("이 퀴즈는 데이터 전처리의 핵심 개념과 실무 적용을 평가합니다.")
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
            print("🎉 우수! 데이터 전처리 개념을 매우 잘 이해하고 있습니다.")
            print("실무에서 고품질 데이터 파이프라인을 구축할 수 있는 수준입니다.")
        elif percentage >= 80:
            print("👍 양호! 대부분의 전처리 개념을 잘 이해하고 있습니다.")
            print("몇 가지 고급 기법을 더 학습하면 완벽합니다.")
        elif percentage >= 70:
            print("📚 보통! 기본 개념은 이해했지만 실무 적용을 위해 더 학습이 필요합니다.")
            print("특히 파이프라인 설계와 방법 선택 기준을 복습하세요.")
        else:
            print("💪 노력 필요! 이론 문서와 실습을 다시 진행해보시기 바랍니다.")
            print("전처리는 머신러닝의 핵심이므로 충분한 이해가 필요합니다.")
        
        # 틀린 문제 복습 제안
        wrong_questions = self.quiz_manager.get_wrong_questions()
        if wrong_questions:
            print(f"\n복습 권장 주제:")
            topic_mapping = {
                "결측치": "결측치 유형과 처리 방법",
                "이상치": "이상치 탐지 및 처리 기법", 
                "스케일링": "정규화 및 스케일링 방법",
                "인코딩": "범주형 변수 인코딩",
                "파이프라인": "전처리 파이프라인 설계",
                "데이터 누출": "교차 검증과 데이터 누출 방지"
            }
            
            for topic in wrong_questions:
                for key, description in topic_mapping.items():
                    if key in topic:
                        print(f"  - {description}")
                        break
        
        print("\n실무 적용 팁:")
        print("1. 항상 도메인 지식을 활용한 전처리 수행")
        print("2. 전처리 각 단계의 효과를 시각화로 확인")
        print("3. 파이프라인을 통한 일관된 처리 보장")
        print("4. 전처리 결과를 문서화하여 재현 가능성 확보")
        
        print("\n다음 단계: 유사도와 거리 측정 학습")
        print("=" * 60)

def main():
    """메인 함수"""
    quiz = DataPreprocessingQuiz()
    quiz.run_quiz()

if __name__ == "__main__":
    main()