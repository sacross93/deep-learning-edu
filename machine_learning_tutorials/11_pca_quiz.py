"""
주성분 분석(PCA) 퀴즈

이 퀴즈는 PCA의 핵심 개념, 주성분 해석, 차원 축소 효과 등에 대한 
이해도를 평가합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.quiz_utils import QuizManager
import numpy as np

def create_pca_quiz():
    """PCA 퀴즈 문제들을 생성합니다."""
    
    questions = []
    
    # 문제 1: PCA 기본 개념
    questions.append({
        'id': 1,
        'type': 'multiple_choice',
        'question': '''주성분 분석(PCA)의 주요 목적으로 가장 적절한 것은?
        
A) 데이터의 클래스를 분류하기 위해
B) 데이터의 분산을 최대한 보존하면서 차원을 축소하기 위해  
C) 데이터의 이상치를 탐지하기 위해
D) 데이터의 결측값을 채우기 위해''',
        'options': ['A', 'B', 'C', 'D'],
        'correct': 'B',
        'explanation': '''정답: B
PCA의 주요 목적은 고차원 데이터를 저차원으로 축소하면서 원본 데이터의 분산(정보)을 최대한 보존하는 것입니다. 
이를 통해 데이터의 주요 패턴을 유지하면서 차원의 저주 문제를 해결할 수 있습니다.'''
    })
    
    # 문제 2: 고유값과 고유벡터
    questions.append({
        'id': 2,
        'type': 'multiple_choice',
        'question': '''PCA에서 고유값(eigenvalue)과 고유벡터(eigenvector)의 의미로 올바른 것은?

A) 고유값: 주성분의 방향, 고유벡터: 해당 방향의 분산 크기
B) 고유값: 해당 방향의 분산 크기, 고유벡터: 주성분의 방향
C) 고유값: 데이터의 평균, 고유벡터: 데이터의 표준편차  
D) 고유값: 클래스 개수, 고유벡터: 특성 개수''',
        'options': ['A', 'B', 'C', 'D'],
        'correct': 'B',
        'explanation': '''정답: B
PCA에서 고유벡터는 주성분의 방향(새로운 축의 방향)을 나타내고, 
고유값은 해당 방향으로의 분산 크기를 나타냅니다. 
고유값이 클수록 해당 주성분이 더 많은 정보를 포함합니다.'''
    })
    
    # 문제 3: 주성분 개수 선택
    questions.append({
        'id': 3,
        'type': 'multiple_choice',
        'question': '''다음 중 PCA에서 적절한 주성분 개수를 선택하는 방법이 아닌 것은?

A) 누적 분산 설명 비율이 95% 이상이 되는 지점
B) 스크리 플롯에서 엘보우(급격한 감소) 지점
C) 카이저 기준(고유값 > 1)
D) 교차 검증을 통한 분류 정확도 최대화''',
        'options': ['A', 'B', 'C', 'D'],
        'correct': 'D',
        'explanation': '''정답: D
A, B, C는 모두 PCA에서 주성분 개수를 선택하는 일반적인 방법들입니다.
D는 지도학습에서 사용하는 방법으로, PCA는 비지도 학습이므로 
분류 정확도를 직접적인 기준으로 사용하지 않습니다.'''
    })
    
    # 문제 4: 계산 문제
    questions.append({
        'id': 4,
        'type': 'numerical',
        'question': '''어떤 데이터셋에 PCA를 적용했을 때 처음 3개 주성분의 분산 설명 비율이 
각각 0.4, 0.3, 0.2라고 합니다. 
처음 3개 주성분으로 설명되는 총 분산 비율은 몇 %입니까? (정수로 답하세요)''',
        'correct': 90,
        'tolerance': 0,
        'explanation': '''정답: 90
누적 분산 설명 비율 = 0.4 + 0.3 + 0.2 = 0.9 = 90%
처음 3개 주성분으로 전체 데이터 분산의 90%를 설명할 수 있습니다.'''
    })
    
    # 문제 5: 재구성 오차
    questions.append({
        'id': 5,
        'type': 'multiple_choice',
        'question': '''PCA에서 재구성 오차(reconstruction error)에 대한 설명으로 올바른 것은?

A) 재구성 오차가 클수록 차원 축소 품질이 좋다
B) 재구성 오차는 원본 데이터와 복원된 데이터 간의 차이를 나타낸다
C) 주성분 개수가 많을수록 재구성 오차가 증가한다
D) 재구성 오차는 항상 0이어야 한다''',
        'options': ['A', 'B', 'C', 'D'],
        'correct': 'B',
        'explanation': '''정답: B
재구성 오차는 원본 데이터와 PCA로 축소했다가 다시 복원한 데이터 간의 차이를 나타냅니다.
재구성 오차가 작을수록 정보 손실이 적어 차원 축소 품질이 좋으며,
주성분 개수가 많을수록 재구성 오차는 감소합니다.'''
    })
    
    # 문제 6: 데이터 전처리
    questions.append({
        'id': 6,
        'type': 'multiple_choice',
        'question': '''PCA 적용 전 데이터 전처리에 대한 설명으로 올바른 것은?

A) 항상 표준화(standardization)를 해야 한다
B) 특성들의 단위가 다를 때만 표준화를 고려한다
C) 중심화(centering)는 필요하지 않다
D) 결측값이 있어도 PCA를 바로 적용할 수 있다''',
        'options': ['A', 'B', 'C', 'D'],
        'correct': 'B',
        'explanation': '''정답: B
특성들의 스케일이나 단위가 크게 다를 때 표준화를 고려해야 합니다.
중심화는 PCA에서 필수적이며(sklearn에서 자동 수행), 
결측값은 반드시 처리한 후 PCA를 적용해야 합니다.'''
    })
    
    # 문제 7: PCA vs 다른 기법
    questions.append({
        'id': 7,
        'type': 'multiple_choice',
        'question': '''PCA와 LDA(Linear Discriminant Analysis)의 차이점으로 올바른 것은?

A) PCA는 지도학습, LDA는 비지도학습이다
B) PCA는 분산 최대화, LDA는 클래스 분리 최대화를 목표로 한다
C) PCA는 분류에만 사용, LDA는 회귀에만 사용된다
D) PCA와 LDA는 완전히 동일한 알고리즘이다''',
        'options': ['A', 'B', 'C', 'D'],
        'correct': 'B',
        'explanation': '''정답: B
PCA는 비지도학습으로 데이터의 분산을 최대화하는 방향을 찾고,
LDA는 지도학습으로 클래스 간 분리를 최대화하는 방향을 찾습니다.
둘 다 차원 축소 기법이지만 목적과 방법이 다릅니다.'''
    })
    
    # 문제 8: 실제 적용 시나리오
    questions.append({
        'id': 8,
        'type': 'multiple_choice',
        'question': '''다음 상황 중 PCA 적용이 가장 적절한 경우는?

A) 10개 특성으로 이진 분류 문제를 해결할 때
B) 1000차원 이미지 데이터를 시각화하고 싶을 때
C) 범주형 변수만 있는 데이터를 분석할 때  
D) 시계열 데이터의 미래값을 예측할 때''',
        'options': ['A', 'B', 'C', 'D'],
        'correct': 'B',
        'explanation': '''정답: B
고차원 이미지 데이터의 시각화는 PCA의 대표적인 활용 사례입니다.
PCA는 연속형 변수에 적합하며, 차원이 높을 때 효과적입니다.
범주형 변수에는 다른 차원 축소 기법이 더 적합합니다.'''
    })
    
    # 문제 9: 수치 계산 문제
    questions.append({
        'id': 9,
        'type': 'numerical',
        'question': '''100차원 데이터를 PCA로 20차원으로 축소했습니다. 
차원 축소 비율은 몇 %입니까? (소수점 첫째 자리까지, 예: 12.5)''',
        'correct': 80.0,
        'tolerance': 0.1,
        'explanation': '''정답: 80.0
차원 축소 비율 = (원본 차원 - 축소 차원) / 원본 차원 × 100
= (100 - 20) / 100 × 100 = 80%'''
    })
    
    # 문제 10: 해석 문제
    questions.append({
        'id': 10,
        'type': 'multiple_choice',
        'question': '''얼굴 인식에서 PCA를 적용했을 때 얻어지는 주성분들을 "Eigenfaces"라고 합니다.
첫 번째 Eigenface가 나타내는 것은?

A) 가장 밝은 얼굴 이미지
B) 가장 어두운 얼굴 이미지
C) 얼굴 이미지들 간의 가장 큰 변동 패턴
D) 평균 얼굴 이미지''',
        'options': ['A', 'B', 'C', 'D'],
        'correct': 'C',
        'explanation': '''정답: C
첫 번째 Eigenface(첫 번째 주성분)는 얼굴 이미지들 간의 가장 큰 변동 패턴을 나타냅니다.
이는 얼굴들 사이에서 가장 많이 변하는 특징(예: 조명, 표정 등)을 포착합니다.
평균 얼굴은 주성분이 아닌 데이터의 중심점입니다.'''
    })
    
    return questions

def main():
    """퀴즈를 실행합니다."""
    print("=" * 60)
    print("주성분 분석(PCA) 퀴즈")
    print("=" * 60)
    print("PCA의 핵심 개념과 적용 방법에 대한 이해도를 확인해보세요!")
    print()
    
    # 퀴즈 매니저 생성 및 퀴즈 실행
    quiz_manager = QuizManager()
    questions = create_pca_quiz()
    
    score = quiz_manager.run_quiz(questions)
    
    # 최종 평가 및 학습 가이드
    print("\n" + "=" * 60)
    print("퀴즈 완료!")
    print("=" * 60)
    
    percentage = (score / len(questions)) * 100
    
    if percentage >= 90:
        print("🎉 훌륭합니다! PCA에 대한 이해가 매우 뛰어납니다.")
        print("고급 차원 축소 기법들(t-SNE, UMAP 등)을 학습해보세요.")
    elif percentage >= 70:
        print("👍 좋습니다! PCA의 기본 개념을 잘 이해하고 있습니다.")
        print("실제 데이터에 PCA를 적용하는 실습을 더 해보세요.")
    elif percentage >= 50:
        print("📚 기본기는 있습니다. 조금 더 학습이 필요합니다.")
        print("PCA의 수학적 원리와 주성분 해석 방법을 복습해보세요.")
    else:
        print("💪 더 열심히 공부해야 합니다!")
        print("PCA 이론 문서를 다시 읽고 실습 코드를 실행해보세요.")
    
    print(f"\n최종 점수: {score}/{len(questions)} ({percentage:.1f}%)")
    
    # 추가 학습 자료 추천
    print("\n📖 추가 학습 자료:")
    print("1. 11_pca_theory.md - PCA 이론 완전 가이드")
    print("2. 11_pca_practice.py - 얼굴 인식 데이터 실습")
    print("3. sklearn.decomposition.PCA 공식 문서")
    print("4. 'Pattern Recognition and Machine Learning' - Bishop")

if __name__ == "__main__":
    main()