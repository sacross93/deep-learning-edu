"""
DBSCAN 클러스터링 퀴즈

이 퀴즈는 DBSCAN의 핵심 개념, 파라미터 조정, 이상치 탐지,
그리고 다른 클러스터링 알고리즘과의 비교에 대한 이해도를 평가합니다.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.quiz_utils import QuizManager
import numpy as np
import matplotlib.pyplot as plt

def create_dbscan_quiz():
    """DBSCAN 클러스터링 퀴즈 생성"""
    
    quiz_manager = QuizManager("DBSCAN 클러스터링")
    
    # 문제 1: 기본 개념 이해
    quiz_manager.add_question(
        question="DBSCAN에 대한 설명으로 옳지 않은 것은?",
        options=[
            "클러스터 개수를 미리 지정할 필요가 없다",
            "임의의 모양을 가진 클러스터를 탐지할 수 있다",
            "모든 데이터 포인트가 반드시 어떤 클러스터에 속해야 한다",
            "밀도 기반 클러스터링 알고리즘이다"
        ],
        correct_answer=2,
        explanation="""
정답: 3번 (모든 데이터 포인트가 반드시 어떤 클러스터에 속해야 한다)

해설:
DBSCAN의 주요 특징 중 하나는 노이즈 포인트를 자동으로 탐지하는 것입니다.

• **클러스터에 속하지 않는 점들**: 노이즈 포인트로 분류
• **자동 이상치 탐지**: DBSCAN의 핵심 장점
• **레이블 -1**: 노이즈 포인트에 할당되는 특별한 레이블

DBSCAN의 장점:
- 클러스터 개수 사전 지정 불필요
- 임의 모양 클러스터 탐지 가능
- 노이즈 자동 탐지
- 밀도 기반 접근법으로 자연스러운 클러스터 형성
        """
    )
    
    # 문제 2: 핵심점과 경계점
    quiz_manager.add_question(
        question="DBSCAN에서 '핵심점(Core Point)'의 정의는?",
        options=[
            "클러스터의 중심에 위치한 점",
            "ε 반경 내에 MinPts 개 이상의 점이 있는 점",
            "가장 많은 이웃을 가진 점",
            "클러스터 경계에 위치한 점"
        ],
        correct_answer=1,
        explanation="""
정답: 2번 (ε 반경 내에 MinPts 개 이상의 점이 있는 점)

해설:
DBSCAN에서 점의 분류:

**1. 핵심점 (Core Point):**
- ε 반경 내에 MinPts 개 이상의 점이 있는 점
- 클러스터의 내부를 형성
- 새로운 클러스터의 시작점이 될 수 있음

**2. 경계점 (Border Point):**
- 핵심점은 아니지만 핵심점의 ε 반경 내에 있는 점
- 클러스터에 속하지만 확장할 수 없음

**3. 노이즈점 (Noise Point):**
- 핵심점도 경계점도 아닌 점
- 어떤 클러스터에도 속하지 않음

이 분류는 DBSCAN 알고리즘의 핵심 개념입니다.
        """
    )
    
    # 문제 3: 파라미터 ε의 영향
    quiz_manager.add_question(
        question="DBSCAN에서 ε(epsilon) 값을 너무 크게 설정하면 어떤 일이 발생합니까?",
        options=[
            "너무 많은 작은 클러스터가 생성된다",
            "모든 점이 노이즈로 분류된다",
            "서로 다른 클러스터들이 하나로 병합될 수 있다",
            "알고리즘이 수렴하지 않는다"
        ],
        correct_answer=2,
        explanation="""
정답: 3번 (서로 다른 클러스터들이 하나로 병합될 수 있다)

해설:
ε 값의 영향:

**ε가 너무 클 때:**
- 이웃 범위가 넓어짐
- 서로 다른 자연스러운 클러스터들이 연결됨
- 전체 데이터가 하나의 큰 클러스터로 병합될 수 있음
- 클러스터 분해능(resolution) 저하

**ε가 너무 작을 때:**
- 이웃 범위가 좁아짐
- 많은 작은 클러스터 생성
- 많은 점들이 노이즈로 분류됨
- 자연스러운 클러스터가 분할됨

**적절한 ε 선택:**
- k-distance 그래프의 "무릎" 지점
- 도메인 지식 활용
- 여러 값으로 실험 후 최적값 선택
        """
    )
    
    # 문제 4: MinPts 파라미터
    quiz_manager.add_question(
        question="DBSCAN에서 MinPts 파라미터를 선택할 때 일반적인 가이드라인은?",
        options=[
            "항상 2로 설정한다",
            "데이터 포인트 개수의 10%로 설정한다",
            "차원 수 + 1 이상으로 설정한다",
            "클러스터 개수와 같게 설정한다"
        ],
        correct_answer=2,
        explanation="""
정답: 3번 (차원 수 + 1 이상으로 설정한다)

해설:
MinPts 선택 가이드라인:

**일반적 규칙:**
- MinPts ≥ 차원 수 + 1
- 2차원 데이터: MinPts ≥ 3 (보통 4 사용)
- 고차원 데이터: MinPts = 2 × 차원 수

**MinPts의 영향:**
- **작은 값**: 더 많은 클러스터, 노이즈에 민감
- **큰 값**: 더 적은 클러스터, 작은 클러스터 무시

**실제 선택 고려사항:**
- 데이터의 노이즈 수준
- 예상되는 클러스터 크기
- 도메인 특성

**추천 시작값:**
- 2D: MinPts = 4
- 3D: MinPts = 6
- 고차원: 실험을 통한 조정 필요
        """
    )
    
    # 문제 5: k-distance 그래프
    quiz_manager.add_question(
        question="k-distance 그래프에서 최적의 ε 값을 찾는 방법은?",
        options=[
            "가장 높은 거리 값을 선택한다",
            "가장 낮은 거리 값을 선택한다",
            "거리가 급격히 증가하는 '무릎' 지점을 찾는다",
            "거리의 평균값을 사용한다"
        ],
        correct_answer=2,
        explanation="""
정답: 3번 (거리가 급격히 증가하는 '무릎' 지점을 찾는다)

해설:
k-distance 그래프 활용법:

**그래프 생성:**
1. 각 점에서 k번째 가장 가까운 이웃까지의 거리 계산
2. k = MinPts - 1로 설정
3. 거리를 내림차순으로 정렬하여 그래프 생성

**무릎 지점의 의미:**
- 거리가 급격히 증가하는 지점
- 밀집된 영역과 희소한 영역의 경계
- 자연스러운 클러스터 분리 기준

**선택 방법:**
- 시각적 판단: 그래프에서 급격한 변화 지점
- 수학적 방법: 1차/2차 미분을 이용한 변곡점 탐지
- 도메인 지식: 문제 특성을 고려한 조정

이 방법은 DBSCAN의 가장 중요한 파라미터 선택 기법입니다.
        """
    )
    
    # 문제 6: 계산 복잡도
    quiz_manager.add_question(
        question="DBSCAN의 시간 복잡도는? (효율적인 공간 인덱스 사용 시)",
        options=[
            "O(n)",
            "O(n log n)",
            "O(n²)",
            "O(n³)"
        ],
        correct_answer=1,
        explanation="""
정답: 2번 (O(n log n))

해설:
DBSCAN의 복잡도:

**효율적인 구현 (공간 인덱스 사용):**
- **시간 복잡도**: O(n log n)
- KD-tree, R-tree 등의 공간 인덱스 활용
- 이웃 탐색이 O(log n)으로 개선

**단순한 구현:**
- **시간 복잡도**: O(n²)
- 모든 점 쌍에 대해 거리 계산
- 이웃 탐색이 O(n)

**공간 복잡도:**
- 일반적으로 O(n)
- 공간 인덱스 사용 시 추가 메모리 필요

**다른 알고리즘과 비교:**
- K-평균: O(nkt) (k=클러스터 수, t=반복 횟수)
- 계층적: O(n³)
- DBSCAN은 중간 정도의 효율성을 가짐
        """
    )
    
    # 문제 7: 밀도 차이 문제
    quiz_manager.add_question(
        question="DBSCAN의 주요 한계점 중 하나는?",
        options=[
            "클러스터 개수를 미리 알아야 한다",
            "구형 클러스터만 탐지할 수 있다",
            "매우 다른 밀도를 가진 클러스터들을 동시에 처리하기 어렵다",
            "노이즈를 탐지할 수 없다"
        ],
        correct_answer=2,
        explanation="""
정답: 3번 (매우 다른 밀도를 가진 클러스터들을 동시에 처리하기 어렵다)

해설:
DBSCAN의 주요 한계점:

**1. 밀도 차이 문제:**
- 전역 파라미터 (ε, MinPts) 사용
- 한 번에 하나의 밀도 기준만 적용 가능
- 밀도가 매우 다른 클러스터 동시 처리 어려움

**2. 고차원 데이터 문제:**
- 차원의 저주로 거리 개념 의미 약화
- 모든 점들이 비슷한 거리를 가지게 됨

**3. 파라미터 민감도:**
- ε, MinPts 선택이 결과에 큰 영향
- 적절한 파라미터 찾기 어려움

**해결 방안:**
- OPTICS: 다양한 밀도 레벨 처리
- HDBSCAN: 계층적 접근으로 밀도 차이 해결
- 지역적 파라미터 조정 기법
        """
    )
    
    # 문제 8: 데이터 전처리
    quiz_manager.add_question(
        question="DBSCAN을 적용하기 전에 데이터 표준화가 중요한 이유는?",
        options=[
            "알고리즘의 수렴 속도를 높이기 위해",
            "서로 다른 스케일의 특성들이 거리 계산에 미치는 영향을 균등하게 하기 위해",
            "클러스터 개수를 자동으로 결정하기 위해",
            "노이즈 탐지 성능을 향상시키기 위해"
        ],
        correct_answer=1,
        explanation="""
정답: 2번 (서로 다른 스케일의 특성들이 거리 계산에 미치는 영향을 균등하게 하기 위해)

해설:
데이터 표준화의 중요성:

**문제 상황:**
- 특성들의 스케일이 다름 (예: 나이 0-100, 소득 0-100,000)
- 큰 스케일의 특성이 거리 계산을 지배
- ε 값 선택이 왜곡됨

**표준화 효과:**
- 모든 특성을 동일한 스케일로 변환
- 각 특성이 거리 계산에 공평하게 기여
- 더 의미 있는 클러스터링 결과

**표준화 방법:**
- Z-score 정규화: (x - μ) / σ
- Min-Max 정규화: (x - min) / (max - min)
- Robust 스케일링: 이상치에 덜 민감

DBSCAN은 거리 기반 알고리즘이므로 표준화가 특히 중요합니다.
        """
    )
    
    # 문제 9: 성능 평가
    quiz_manager.add_question(
        question="DBSCAN 결과를 평가할 때 실루엣 계수를 계산하는 올바른 방법은?",
        options=[
            "모든 데이터 포인트를 포함하여 계산한다",
            "노이즈 포인트를 제외하고 클러스터에 속한 점들만으로 계산한다",
            "핵심점들만 포함하여 계산한다",
            "경계점들만 포함하여 계산한다"
        ],
        correct_answer=1,
        explanation="""
정답: 2번 (노이즈 포인트를 제외하고 클러스터에 속한 점들만으로 계산한다)

해설:
DBSCAN 결과 평가 시 고려사항:

**실루엣 계수 계산:**
- 노이즈 포인트 (레이블 -1) 제외
- 클러스터에 속한 점들만 사용
- 클러스터가 2개 이상일 때만 의미 있음

**이유:**
- 노이즈 포인트는 클러스터에 속하지 않음
- 실루엣 계수는 클러스터 내 응집도와 클러스터 간 분리도 측정
- 노이즈 포인트 포함 시 의미 없는 결과

**다른 평가 지표:**
- Davies-Bouldin 지수: 노이즈 제외
- Calinski-Harabasz 지수: 노이즈 제외
- 조정 랜드 지수 (ARI): 노이즈 포함 가능 (실제 레이블과 비교)

**노이즈 비율 평가:**
- 전체 데이터 대비 노이즈 비율 확인
- 너무 많으면 파라미터 조정 필요
        """
    )
    
    # 문제 10: 실제 적용 시나리오
    quiz_manager.add_question(
        question="다음 중 DBSCAN이 가장 적합한 상황은?",
        options=[
            "구형 클러스터가 명확히 분리된 데이터",
            "클러스터 개수를 정확히 알고 있는 경우",
            "임의의 모양을 가진 클러스터와 이상치가 섞여 있는 공간 데이터",
            "모든 데이터 포인트가 반드시 클러스터에 속해야 하는 경우"
        ],
        correct_answer=2,
        explanation="""
정답: 3번 (임의의 모양을 가진 클러스터와 이상치가 섞여 있는 공간 데이터)

해설:
DBSCAN이 적합한 상황:

**최적 적용 사례:**
- 임의 모양의 클러스터 (달 모양, S자 모양 등)
- 이상치와 노이즈가 포함된 데이터
- 공간 데이터 (지리 정보, 이미지 분할)
- 클러스터 개수를 모르는 탐색적 분석

**구체적 예시:**
- 지리 정보: 범죄 핫스팟, 상권 분석
- 이미지 처리: 객체 분할, 의료 영상
- 고객 분석: 이상 행동 패턴 탐지
- 네트워크 분석: 커뮤니티 탐지

**부적합한 상황:**
- 구형 클러스터만 있는 경우 → K-평균이 더 효율적
- 클러스터 개수를 정확히 아는 경우 → K-평균 고려
- 모든 점이 클러스터에 속해야 하는 경우 → 다른 알고리즘 사용
- 매우 고차원 데이터 → 차원 축소 후 적용

DBSCAN의 강점을 최대한 활용할 수 있는 상황을 선택하는 것이 중요합니다.
        """
    )
    
    return quiz_manager

def main():
    """퀴즈 실행 메인 함수"""
    print("="*60)
    print("DBSCAN 클러스터링 퀴즈")
    print("="*60)
    print()
    print("이 퀴즈는 DBSCAN의 핵심 개념들을 다룹니다:")
    print("• 기본 개념과 특징")
    print("• 핵심점, 경계점, 노이즈점 분류")
    print("• 파라미터 (ε, MinPts) 선택")
    print("• k-distance 그래프 활용")
    print("• 성능 평가 방법")
    print("• 다른 알고리즘과의 비교")
    print("• 실제 적용 시나리오")
    print()
    
    # 퀴즈 생성 및 실행
    quiz_manager = create_dbscan_quiz()
    results = quiz_manager.run_quiz()
    
    # 결과 출력
    print("\n" + "="*60)
    print("퀴즈 결과 분석")
    print("="*60)
    
    score = results['score']
    total = results['total']
    percentage = results['percentage']
    
    print(f"총 점수: {score}/{total} ({percentage:.1f}%)")
    
    if percentage >= 90:
        print("🏆 훌륭합니다! DBSCAN을 완벽하게 이해하고 있습니다.")
    elif percentage >= 80:
        print("👍 잘했습니다! DBSCAN의 핵심 개념을 잘 파악하고 있습니다.")
    elif percentage >= 70:
        print("👌 괜찮습니다! 몇 가지 개념을 더 복습해보세요.")
    elif percentage >= 60:
        print("📚 더 공부가 필요합니다. 이론 문서를 다시 읽어보세요.")
    else:
        print("💪 포기하지 마세요! 기초부터 차근차근 다시 학습해보세요.")
    
    # 틀린 문제 분석
    wrong_questions = results['wrong_questions']
    if wrong_questions:
        print(f"\n틀린 문제: {len(wrong_questions)}개")
        print("다음 주제들을 더 공부해보세요:")
        
        topics = {
            1: "DBSCAN 기본 개념",
            2: "핵심점과 경계점 정의",
            3: "ε 파라미터의 영향",
            4: "MinPts 파라미터 선택",
            5: "k-distance 그래프 활용",
            6: "계산 복잡도",
            7: "밀도 차이 문제",
            8: "데이터 전처리",
            9: "성능 평가 방법",
            10: "실제 적용 시나리오"
        }
        
        for q_num in wrong_questions:
            print(f"  • {topics.get(q_num, f'문제 {q_num}')}")
    
    print("\n" + "="*60)
    print("학습 권장사항")
    print("="*60)
    
    if percentage < 70:
        print("📖 14_dbscan_theory.md 문서를 다시 읽어보세요.")
        print("💻 14_dbscan_practice.py 실습을 다시 실행해보세요.")
    
    print("🔍 추가 학습 주제:")
    print("  • OPTICS와 HDBSCAN 등 DBSCAN 변형 알고리즘")
    print("  • 고차원 데이터에서의 DBSCAN 적용 방법")
    print("  • 스트리밍 데이터를 위한 온라인 DBSCAN")
    print("  • 지리 정보 시스템에서의 DBSCAN 활용")
    
    return results

if __name__ == "__main__":
    main()