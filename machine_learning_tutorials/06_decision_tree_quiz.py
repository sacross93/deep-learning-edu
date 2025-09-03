#!/usr/bin/env python3
"""
의사결정나무 퀴즈
분할 기준 선택, 트리 해석, 과적합 탐지 및 해결 방법에 대한 문제
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.quiz_utils import QuizManager
import numpy as np
import pandas as pd

def create_decision_tree_quiz():
    """의사결정나무 퀴즈 생성"""
    
    quiz = QuizManager("의사결정나무 퀴즈")
    
    # 문제 1: 기본 개념
    quiz.add_question(
        question="의사결정나무에서 '불순도(Impurity)'란 무엇인가요?",
        options=[
            "노드에서 서로 다른 클래스가 섞여 있는 정도",
            "트리의 깊이가 깊어지는 정도", 
            "특성의 개수가 많아지는 정도",
            "데이터의 노이즈 정도"
        ],
        correct_answer=0,
        explanation="""
불순도(Impurity)는 한 노드에서 서로 다른 클래스의 샘플들이 섞여 있는 정도를 나타냅니다.
- 불순도가 0이면 모든 샘플이 같은 클래스 (완전 순수)
- 불순도가 높으면 여러 클래스가 고르게 섞여 있음
- 의사결정나무는 불순도를 최소화하는 방향으로 분할을 수행합니다.
        """,
        category="기본 개념"
    )
    
    # 문제 2: 분할 기준 계산
    quiz.add_question(
        question="""
다음 노드에서 지니 불순도를 계산하세요.
노드 정보: 클래스 A 40개, 클래스 B 60개 (총 100개)
        """,
        options=[
            "0.24",
            "0.48", 
            "0.60",
            "0.76"
        ],
        correct_answer=1,
        explanation="""
지니 불순도 계산:
Gini = 1 - Σ(p_i²)
p_A = 40/100 = 0.4
p_B = 60/100 = 0.6
Gini = 1 - (0.4² + 0.6²) = 1 - (0.16 + 0.36) = 1 - 0.52 = 0.48

지니 불순도는 0(완전 순수)에서 0.5(최대 불순도, 이진 분류 시) 사이의 값을 가집니다.
        """,
        category="분할 기준"
    )
    
    # 문제 3: 엔트로피 vs 지니
    quiz.add_question(
        question="엔트로피와 지니 불순도의 차이점으로 옳지 않은 것은?",
        options=[
            "엔트로피는 로그 연산을 사용하여 계산이 더 복잡하다",
            "지니 불순도는 계산이 빠르다",
            "엔트로피는 항상 지니 불순도보다 큰 값을 가진다",
            "둘 다 불순도 측정에 사용되며 결과가 유사하다"
        ],
        correct_answer=2,
        explanation="""
엔트로피가 항상 지니 불순도보다 큰 값을 가진다는 것은 틀렸습니다.

올바른 특성:
- 엔트로피: 0 ~ log₂(c) 범위, 로그 연산 사용
- 지니 불순도: 0 ~ (1-1/c) 범위, 제곱 연산 사용
- 두 지표 모두 같은 분할을 선택하는 경우가 많음
- 지니가 계산상 더 효율적
        """,
        category="분할 기준"
    )
    
    # 문제 4: 헌트 알고리즘
    quiz.add_question(
        question="""
헌트(Hunt) 알고리즘에서 노드 t에 도달한 훈련 레코드 집합 Dₜ가 하나의 클래스만 포함할 때 어떻게 처리하나요?
        """,
        options=[
            "더 세밀하게 분할한다",
            "리프 노드를 생성하고 해당 클래스 라벨을 할당한다",
            "부모 노드로 되돌아간다",
            "무작위로 클래스를 선택한다"
        ],
        correct_answer=1,
        explanation="""
헌트 알고리즘의 기본 구조:
- Dₜ가 하나의 클래스만 포함하면 → 리프 노드 생성 (해당 클래스 라벨 할당)
- 그렇지 않으면 → 속성 테스트로 더 작은 부분집합으로 분할 → 각 부분집합에 대해 재귀적 적용

이는 의사결정나무의 기본이 되는 알고리즘으로, 모든 현대 의사결정나무 알고리즘의 기초가 됩니다.
        """,
        category="기본 개념"
    )
    
    # 문제 5: 정보 이득
    quiz.add_question(
        question="""
정보 이득(Information Gain)이 높다는 것은 무엇을 의미하나요?
        """,
        options=[
            "분할 후 불순도가 크게 감소했다",
            "분할 후 불순도가 증가했다",
            "트리의 깊이가 깊어졌다",
            "특성의 개수가 증가했다"
        ],
        correct_answer=0,
        explanation="""
정보 이득 = 분할 전 불순도 - 분할 후 가중평균 불순도

정보 이득이 높다는 것은:
- 분할로 인해 불순도가 크게 감소했음을 의미
- 해당 분할이 클래스를 잘 구분한다는 뜻
- 의사결정나무는 정보 이득이 최대인 분할을 선택
- 정보 이득이 0이면 분할의 효과가 없음
        """,
        category="분할 기준"
    )
    
    # 문제 6: 속성 유형별 분할
    quiz.add_question(
        question="""
연속형 속성에서 이분 분할을 수행할 때 주로 사용하는 방법은?
        """,
        options=[
            "(A = v) vs (A ≠ v) 형태의 분할",
            "(A < v) vs (A ≥ v) 형태의 분할",
            "모든 값을 개별 노드로 분할",
            "무작위로 두 그룹으로 분할"
        ],
        correct_answer=1,
        explanation="""
연속형 속성의 이분 분할:
- (A < v) vs (A ≥ v) 형태를 사용
- v는 절단점(threshold)으로, 모든 가능한 v 후보를 고려해 최적 절단점 선택
- 순서 관계를 유지하면서 분할
- 다분할의 경우 구간화 사용 (예: <10K, [10K,25K), ..., >80K)

명목형 속성과 달리 연속형은 순서가 중요하므로 임계값 기반 분할을 사용합니다.
        """,
        category="분할 기준"
    )
    
    # 문제 7: 트리 해석
    quiz.add_question(
        question="""
다음 의사결정나무 규칙을 해석하세요:
IF (나이 > 30) AND (소득 > 50000) THEN 승인
ELSE IF (나이 <= 30) AND (신용점수 > 700) THEN 승인  
ELSE 거부

35세, 소득 45000, 신용점수 750인 고객의 결과는?
        """,
        options=[
            "승인",
            "거부",
            "판단 불가",
            "추가 정보 필요"
        ],
        correct_answer=1,
        explanation="""
단계별 판단:
1. 나이 > 30? → 35 > 30 = True
2. 소득 > 50000? → 45000 > 50000 = False
3. 첫 번째 조건 (나이 > 30 AND 소득 > 50000) = True AND False = False
4. 두 번째 조건으로 이동: 나이 <= 30? → 35 <= 30 = False
5. 모든 승인 조건을 만족하지 않으므로 → 거부

의사결정나무는 AND 조건을 모두 만족해야 해당 경로를 따라갑니다.
        """,
        category="트리 해석"
    )
    
    # 문제 8: XOR 문제
    quiz.add_question(
        question="""
XOR 문제에서 의사결정나무가 어려움을 겪는 주된 이유는?
        """,
        options=[
            "데이터가 부족해서",
            "개별 특성의 정보 이득이 낮아 탐욕적 분할이 최적이 아니기 때문",
            "클래스가 너무 많아서",
            "특성이 연속형이어서"
        ],
        correct_answer=1,
        explanation="""
XOR 문제의 특성:
- X=0,Y=0 → Class A, X=0,Y=1 → Class B, X=1,Y=0 → Class B, X=1,Y=1 → Class A
- 개별적으로 X나 Y만 보면 엔트로피가 높음 (≈0.99)
- 탐욕적 기준으로는 둘 다 좋지 않은 분할로 판단
- 하지만 X와 Y를 함께 고려하면 완벽하게 구분 가능

이는 상호작용이 필요한 속성 조합에서 탐욕적 분할의 한계를 보여줍니다.
        """,
        category="한계점"
    )
    
    # 문제 9: 특성 중요도
    quiz.add_question(
        question="의사결정나무에서 특성 중요도는 어떻게 계산되나요?",
        options=[
            "해당 특성이 사용된 노드에서의 정보 이득의 가중합",
            "해당 특성의 분산",
            "해당 특성과 타겟의 상관계수",
            "해당 특성이 사용된 횟수"
        ],
        correct_answer=0,
        explanation="""
특성 중요도 계산:
- 각 특성이 분할에 사용된 모든 노드에서의 정보 이득을 계산
- 노드의 샘플 비율로 가중평균하여 합산
- 전체 합이 1이 되도록 정규화

중요도가 높은 특성:
- 트리 상단(루트 근처)에서 사용
- 많은 샘플을 포함한 노드에서 사용
- 큰 정보 이득을 제공
        """,
        category="트리 해석"
    )
    
    # 문제 10: 결정 경계
    quiz.add_question(
        question="""
의사결정나무가 생성하는 결정 경계의 특징은?
        """,
        options=[
            "곡선 형태의 부드러운 경계",
            "축 정렬(axis-aligned) 직사각형 경계",
            "원형 경계",
            "무작위 형태의 경계"
        ],
        correct_answer=1,
        explanation="""
의사결정나무의 결정 경계:
- 축 정렬(axis-aligned) 직사각형 경계 생성
- 각 분할이 단일 속성을 기준으로 수행되기 때문
- 2차원에서는 수직선과 수평선으로만 구성
- 복잡한 비축정렬 경계 표현에 제한적

다른 알고리즘과 비교:
- 로지스틱 회귀: 선형 경계
- SVM: 선형 또는 비선형 경계 (커널에 따라)
- 신경망: 복잡한 비선형 경계
        """,
        category="한계점"
    )
    
    # 문제 11: 과적합 탐지
    quiz.add_question(
        question="""
다음 중 의사결정나무의 과적합을 나타내는 신호가 아닌 것은?
        """,
        options=[
            "훈련 정확도 95%, 테스트 정확도 70%",
            "트리 깊이가 20 이상",
            "리프 노드에 샘플이 1개씩만 있음",
            "모든 특성의 중요도가 비슷함"
        ],
        correct_answer=3,
        explanation="""
과적합 신호들:
1. 훈련-테스트 성능 차이가 큼 (예: 95% vs 70%)
2. 트리가 너무 깊음 (일반적으로 10 이상이면 의심)
3. 리프 노드의 샘플 수가 너무 적음
4. 트리가 너무 복잡함

특성 중요도가 비슷한 것은 과적합과 직접적 관련이 없습니다.
오히려 한 특성에만 의존하는 것이 더 문제가 될 수 있습니다.
        """,
        category="과적합"
    )
    
    # 문제 8: 가지치기 방법
    quiz.add_question(
        question="사전 가지치기(Pre-pruning)의 방법이 아닌 것은?",
        options=[
            "최대 깊이 제한 (max_depth)",
            "최소 분할 샘플 수 설정 (min_samples_split)",
            "완전한 트리 구축 후 노드 제거",
            "최소 불순도 감소량 설정 (min_impurity_decrease)"
        ],
        correct_answer=2,
        explanation="""
사전 가지치기 (Pre-pruning):
- 트리 구축 중에 성장을 조기 중단
- max_depth: 최대 깊이 제한
- min_samples_split: 분할 최소 샘플 수
- min_samples_leaf: 리프 노드 최소 샘플 수
- min_impurity_decrease: 최소 불순도 감소량

사후 가지치기 (Post-pruning):
- 완전한 트리 구축 후 불필요한 노드 제거
- 비용 복잡도 가지치기 등
        """,
        category="과적합"
    )
    
    # 문제 9: 하이퍼파라미터
    quiz.add_question(
        question="""
다음 상황에서 어떤 하이퍼파라미터를 조정해야 할까요?
"트리가 너무 복잡해서 과적합이 발생하고 있습니다."
        """,
        options=[
            "max_depth를 증가시킨다",
            "min_samples_split을 감소시킨다",
            "max_depth를 감소시키거나 min_samples_split을 증가시킨다",
            "criterion을 'gini'에서 'entropy'로 변경한다"
        ],
        correct_answer=2,
        explanation="""
과적합 해결 방법:
1. max_depth 감소: 트리 깊이 제한으로 복잡도 감소
2. min_samples_split 증가: 분할 조건을 까다롭게 만듦
3. min_samples_leaf 증가: 리프 노드 최소 크기 증가
4. max_features 감소: 분할 시 고려할 특성 수 제한

criterion 변경은 과적합과 직접적 관련이 적습니다.
        """,
        category="과적합"
    )
    
    # 문제 10: 실제 적용
    quiz.add_question(
        question="""
의료 진단 시스템에서 의사결정나무를 사용할 때 가장 중요한 고려사항은?
        """,
        options=[
            "실행 속도",
            "해석 가능성과 설명 가능성",
            "메모리 사용량",
            "특성 개수"
        ],
        correct_answer=1,
        explanation="""
의료 진단에서 중요한 요소:
1. 해석 가능성: 의사가 진단 과정을 이해할 수 있어야 함
2. 설명 가능성: 환자에게 진단 근거를 설명할 수 있어야 함
3. 투명성: 의료진이 모델의 결정 과정을 검증할 수 있어야 함

의사결정나무의 장점:
- IF-THEN 규칙으로 명확한 설명 가능
- 각 분할 조건의 의미를 쉽게 이해
- 도메인 전문가의 검증 용이
        """,
        category="실제 적용"
    )
    
    # 문제 11: 알고리즘 비교
    quiz.add_question(
        question="의사결정나무와 로지스틱 회귀의 주요 차이점은?",
        options=[
            "의사결정나무는 선형 결정 경계만 만들 수 있다",
            "로지스틱 회귀는 확률을 출력하지만 의사결정나무는 클래스만 출력한다",
            "의사결정나무는 범주형 데이터를 처리할 수 없다",
            "로지스틱 회귀는 다중 클래스 분류가 불가능하다"
        ],
        correct_answer=1,
        explanation="""
주요 차이점:
1. 출력 형태:
   - 로지스틱 회귀: 각 클래스의 확률 출력
   - 의사결정나무: 클래스 레이블 출력 (확률 계산 가능하지만 기본은 클래스)

2. 결정 경계:
   - 로지스틱 회귀: 선형 결정 경계
   - 의사결정나무: 비선형, 직사각형 형태 경계

3. 해석성:
   - 둘 다 해석 가능하지만 방식이 다름
   - 의사결정나무: 규칙 기반
   - 로지스틱 회귀: 계수 기반
        """,
        category="알고리즘 비교"
    )
    
    # 문제 12: 고급 개념
    quiz.add_question(
        question="""
다음 중 의사결정나무의 한계를 극복하기 위한 앙상블 방법이 아닌 것은?
        """,
        options=[
            "Random Forest",
            "Gradient Boosting",
            "AdaBoost",
            "K-Means Clustering"
        ],
        correct_answer=3,
        explanation="""
의사결정나무 기반 앙상블 방법:
1. Random Forest: 배깅 + 특성 무작위 선택
2. Gradient Boosting: 순차적 부스팅
3. AdaBoost: 적응적 부스팅
4. Extra Trees: 극도로 무작위화된 트리

K-Means는 클러스터링 알고리즘으로 의사결정나무와 관련이 없습니다.

앙상블의 장점:
- 개별 트리의 과적합 문제 해결
- 예측 성능 향상
- 안정성 증가
        """,
        category="고급 개념"
    )
    
    return quiz

def main():
    """퀴즈 실행"""
    print("=" * 60)
    print("의사결정나무 퀴즈")
    print("=" * 60)
    print("의사결정나무의 분할 기준, 트리 해석, 과적합 방지에 대한 이해도를 확인합니다.")
    print()
    
    # 퀴즈 생성 및 실행
    quiz = create_decision_tree_quiz()
    results = quiz.run_quiz()
    
    # 상세 결과 분석
    print("\n" + "=" * 60)
    print("상세 분석 결과")
    print("=" * 60)
    
    categories = ["기본 개념", "분할 기준", "트리 해석", "한계점", "과적합", "실제 적용", "알고리즘 비교", "고급 개념"]
    
    for category in categories:
        category_questions = [q for q in quiz.questions if q.get('category') == category]
        if category_questions:
            correct = sum(1 for q in category_questions if results['answers'].get(q['question'], -1) == q['correct_answer'])
            total = len(category_questions)
            percentage = (correct / total) * 100
            print(f"{category}: {correct}/{total} ({percentage:.1f}%)")
    
    # 학습 권장사항
    print(f"\n총점: {results['score']}/{results['total']} ({results['percentage']:.1f}%)")
    
    if results['percentage'] >= 90:
        print("🎉 훌륭합니다! 의사결정나무를 완전히 이해하고 있습니다.")
    elif results['percentage'] >= 80:
        print("👍 좋습니다! 대부분의 개념을 잘 이해하고 있습니다.")
    elif results['percentage'] >= 70:
        print("📚 기본기는 탄탄합니다. 고급 개념을 더 학습해보세요.")
    else:
        print("💪 더 많은 학습이 필요합니다. 이론과 실습을 다시 복습해보세요.")
    
    # 취약 분야 분석
    weak_categories = []
    for category in categories:
        category_questions = [q for q in quiz.questions if q.get('category') == category]
        if category_questions:
            correct = sum(1 for q in category_questions if results['answers'].get(q['question'], -1) == q['correct_answer'])
            total = len(category_questions)
            if (correct / total) < 0.7:  # 70% 미만
                weak_categories.append(category)
    
    if weak_categories:
        print(f"\n📖 추가 학습 권장 분야: {', '.join(weak_categories)}")
        print("해당 분야의 이론 문서와 실습 코드를 다시 확인해보세요.")

if __name__ == "__main__":
    main()