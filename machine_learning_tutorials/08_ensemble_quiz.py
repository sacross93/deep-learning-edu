"""
앙상블 학습 퀴즈

이 퀴즈는 앙상블 학습의 핵심 개념, 다양한 앙상블 기법의 특성,
모델 선택 기준, 그리고 앙상블 효과에 대한 이해도를 평가합니다.

주요 평가 영역:
1. 앙상블 학습의 기본 개념
2. 배깅, 부스팅, 스태킹의 차이점
3. 편향-분산 트레이드오프
4. 앙상블 기법 선택 기준
5. 모델 다양성과 성능 관계
6. 실제 적용 시나리오
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.quiz_utils import QuizManager
import numpy as np

def create_ensemble_quiz():
    """
    앙상블 학습 퀴즈 생성
    """
    
    quiz_questions = [
        # 1. 기본 개념 문제
        {
            'id': 1,
            'type': 'multiple_choice',
            'question': """앙상블 학습의 기본 원리에 대한 설명으로 가장 적절한 것은?""",
            'options': [
                "단일 모델의 성능을 극대화하여 최고 성능을 달성한다",
                "여러 모델의 예측을 조합하여 개별 모델보다 나은 성능을 얻는다", 
                "가장 복잡한 모델을 선택하여 모든 패턴을 학습한다",
                "데이터를 여러 부분으로 나누어 각각 다른 알고리즘을 적용한다"
            ],
            'correct': 1,
            'explanation': """앙상블 학습의 핵심은 "집단 지성"의 원리입니다. 여러 개의 개별 모델(약한 학습기)을 조합하여 
단일 모델보다 더 나은 예측 성능을 달성하는 것이 목표입니다. 개별 모델들이 서로 다른 오류를 범할 때, 
이들을 적절히 조합하면 전체적인 오류를 줄일 수 있습니다."""
        },
        
        # 2. 배깅 vs 부스팅 비교
        {
            'id': 2,
            'type': 'multiple_choice',
            'question': """배깅(Bagging)과 부스팅(Boosting)의 주요 차이점으로 옳은 것은?""",
            'options': [
                "배깅은 순차적 학습, 부스팅은 병렬 학습을 수행한다",
                "배깅은 분산 감소, 부스팅은 편향 감소에 주로 효과적이다",
                "배깅은 복잡한 모델, 부스팅은 단순한 모델을 사용한다", 
                "배깅은 분류용, 부스팅은 회귀용 기법이다"
            ],
            'correct': 1,
            'explanation': """배깅은 부트스트랩 샘플링을 통해 여러 모델을 병렬로 훈련하여 주로 분산(variance)을 감소시킵니다. 
반면 부스팅은 이전 모델의 오류를 다음 모델이 보완하도록 순차적으로 학습하여 주로 편향(bias)을 감소시킵니다. 
이는 편향-분산 트레이드오프 관점에서 각 기법의 핵심적인 차이점입니다."""
        },
        
        # 3. Random Forest 특성
        {
            'id': 3,
            'type': 'multiple_choice',
            'question': """Random Forest에서 개별 의사결정나무의 다양성을 확보하는 방법이 아닌 것은?""",
            'options': [
                "부트스트랩 샘플링으로 서로 다른 훈련 데이터 사용",
                "각 노드에서 전체 특성 중 일부만 무작위로 선택",
                "모든 나무에서 동일한 최대 깊이 제한 적용",
                "서로 다른 랜덤 시드를 사용한 분할 기준 선택"
            ],
            'correct': 2,
            'explanation': """Random Forest는 배깅과 특성 무작위 선택을 결합한 기법입니다. 부트스트랩 샘플링, 
특성 부분집합 선택, 랜덤 시드 사용 등으로 개별 나무들의 다양성을 확보합니다. 
하지만 모든 나무에 동일한 최대 깊이를 적용하는 것은 다양성을 제한하는 요소입니다."""
        },
        
        # 4. AdaBoost 동작 원리
        {
            'id': 4,
            'type': 'multiple_choice',
            'question': """AdaBoost 알고리즘에서 다음 라운드의 학습을 위해 수행하는 작업은?""",
            'options': [
                "새로운 특성을 추가하여 데이터를 확장한다",
                "잘못 분류된 샘플의 가중치를 증가시킨다",
                "가장 성능이 좋은 모델만 선택하여 사용한다",
                "모든 샘플의 가중치를 동일하게 초기화한다"
            ],
            'correct': 1,
            'explanation': """AdaBoost는 적응적 부스팅 알고리즘으로, 각 라운드에서 이전 모델이 잘못 분류한 샘플들에 
더 높은 가중치를 부여합니다. 이를 통해 다음 모델이 어려운 샘플들에 더 집중하도록 하여 
전체적인 성능을 점진적으로 개선합니다."""
        },
        
        # 5. 스태킹 구조
        {
            'id': 5,
            'type': 'multiple_choice',
            'question': """스태킹(Stacking) 앙상블에서 메타 모델(Meta-model)의 역할은?""",
            'options': [
                "기본 모델들의 하이퍼파라미터를 자동으로 튜닝한다",
                "기본 모델들의 예측 결과를 입력으로 받아 최종 예측을 수행한다",
                "기본 모델들 중에서 가장 좋은 모델을 선택한다",
                "기본 모델들의 훈련 데이터를 생성한다"
            ],
            'correct': 1,
            'explanation': """스태킹에서 메타 모델은 1단계에서 훈련된 기본 모델들의 예측 결과를 새로운 특성으로 사용하여 
최종 예측을 수행합니다. 이는 기본 모델들의 예측을 어떻게 조합할지를 학습하는 것으로, 
단순한 평균이나 투표보다 더 정교한 조합 방법을 제공합니다."""
        },
        
        # 6. 편향-분산 트레이드오프
        {
            'id': 6,
            'type': 'multiple_choice',
            'question': """높은 분산(High Variance) 문제를 가진 모델에 가장 적합한 앙상블 기법은?""",
            'options': [
                "AdaBoost (적응적 부스팅)",
                "Gradient Boosting",
                "Random Forest (배깅)",
                "Stacking"
            ],
            'correct': 2,
            'explanation': """높은 분산 문제는 모델이 훈련 데이터에 과도하게 민감하여 예측이 불안정한 상황입니다. 
Random Forest와 같은 배깅 기법은 여러 모델의 예측을 평균화하여 분산을 효과적으로 감소시킵니다. 
반면 부스팅은 주로 편향 감소에 효과적입니다."""
        },
        
        # 7. 투표 방식 비교
        {
            'id': 7,
            'type': 'multiple_choice',
            'question': """소프트 투표(Soft Voting)가 하드 투표(Hard Voting)보다 유리한 경우는?""",
            'options': [
                "모든 모델의 성능이 동일할 때",
                "모델들의 예측 확신도가 다를 때",
                "이진 분류 문제일 때만",
                "계산 속도가 중요할 때"
            ],
            'correct': 1,
            'explanation': """소프트 투표는 각 모델의 예측 확률을 고려하여 가중 평균을 계산합니다. 
따라서 모델들이 예측에 대해 서로 다른 확신도를 가질 때, 이 정보를 활용하여 
더 정확한 최종 예측을 만들 수 있습니다."""
        },
        
        # 8. 앙상블 성공 조건
        {
            'id': 8,
            'type': 'multiple_choice',
            'question': """효과적인 앙상블을 위한 필수 조건이 아닌 것은?""",
            'options': [
                "개별 모델들이 랜덤 추측보다 나은 성능을 가져야 함",
                "개별 모델들이 서로 다른 오류 패턴을 가져야 함",
                "모든 개별 모델들이 동일한 알고리즘을 사용해야 함",
                "개별 모델들의 예측이 독립적이어야 함"
            ],
            'correct': 2,
            'explanation': """효과적인 앙상블을 위해서는 개별 모델들이 정확성(랜덤보다 나은 성능), 다양성(서로 다른 오류), 
독립성(낮은 상관관계)을 가져야 합니다. 오히려 서로 다른 알고리즘을 사용하는 것이 
모델 다양성 확보에 도움이 됩니다."""
        },
        
        # 9. 실제 적용 시나리오
        {
            'id': 9,
            'type': 'multiple_choice',
            'question': """다음 상황 중 앙상블 기법 사용이 가장 부적절한 경우는?""",
            'options': [
                "높은 예측 정확도가 필요한 의료 진단 시스템",
                "실시간 응답이 중요한 온라인 광고 시스템",
                "복잡한 패턴을 가진 이미지 분류 문제",
                "안정적인 성능이 요구되는 금융 리스크 모델"
            ],
            'correct': 1,
            'explanation': """앙상블은 일반적으로 단일 모델보다 높은 계산 비용과 지연 시간을 요구합니다. 
따라서 실시간 응답이 중요한 시스템에서는 성능 향상 대비 지연 시간 증가가 
문제가 될 수 있습니다."""
        },
        
        # 10. 모델 다양성과 성능
        {
            'id': 10,
            'type': 'short_answer',
            'question': """앙상블에서 개별 모델들의 다양성(Diversity)이 전체 성능에 미치는 영향을 설명하고, 
다양성을 확보하는 구체적인 방법 3가지를 제시하세요.""",
            'sample_answer': """다양성의 영향: 개별 모델들이 서로 다른 오류를 범할 때 앙상블 효과가 극대화됩니다. 
모델들이 독립적인 실수를 하면 다수결이나 평균을 통해 오류를 상쇄할 수 있지만, 
모든 모델이 같은 실수를 하면 앙상블 효과가 제한됩니다.

다양성 확보 방법:
1. 서로 다른 알고리즘 사용 (의사결정나무, SVM, 신경망 등)
2. 서로 다른 데이터 부분집합 사용 (부트스트랩, 특성 선택)
3. 서로 다른 하이퍼파라미터 설정 (깊이, 학습률, 정규화 등)""",
            'keywords': ['다양성', '독립적', '오류', '상쇄', '알고리즘', '데이터', '하이퍼파라미터'],
            'explanation': """모델 다양성은 앙상블 성공의 핵심 요소입니다. 개별 모델들이 서로 다른 강점과 약점을 가질 때, 
이들을 조합하면 개별 모델의 약점을 보완하고 강점을 극대화할 수 있습니다. 
다양성 확보 방법으로는 알고리즘 다양화, 데이터 다양화, 파라미터 다양화 등이 있습니다."""
        },
        
        # 11. 앙상블 기법 선택
        {
            'id': 11,
            'type': 'scenario',
            'question': """다음 시나리오에서 가장 적합한 앙상블 기법을 선택하고 그 이유를 설명하세요.

시나리오: 의료 영상 진단 시스템을 개발 중입니다. 높은 정확도가 필요하지만 해석 가능성도 중요합니다. 
현재 CNN, SVM, Random Forest 모델이 각각 다른 특성에서 좋은 성능을 보이고 있습니다. 
실시간 처리는 필요하지 않지만 안정적인 성능이 중요합니다.""",
            'options': [
                "Random Forest (배깅)",
                "AdaBoost (부스팅)", 
                "Soft Voting",
                "Stacking"
            ],
            'correct': 2,
            'explanation': """이 시나리오에서는 Soft Voting이 가장 적합합니다. 

이유:
1. 서로 다른 알고리즘(CNN, SVM, RF)의 장점을 모두 활용 가능
2. 각 모델의 예측 확률을 고려하여 더 정교한 결정 가능
3. 스태킹보다 상대적으로 해석하기 쉬움
4. 안정적인 성능 제공 (여러 모델의 합의)
5. 실시간 처리가 필요하지 않으므로 계산 비용 부담 적음

의료 진단에서는 정확도와 안정성이 가장 중요하며, Soft Voting은 이를 잘 만족합니다."""
        },
        
        # 12. 과적합과 앙상블
        {
            'id': 12,
            'type': 'multiple_choice',
            'question': """앙상블 기법이 과적합(Overfitting) 방지에 효과적인 주된 이유는?""",
            'options': [
                "더 복잡한 모델을 사용하기 때문에",
                "훈련 데이터를 더 많이 사용하기 때문에",
                "여러 모델의 평균화 효과로 분산이 감소하기 때문에",
                "하이퍼파라미터 튜닝이 자동화되기 때문에"
            ],
            'correct': 2,
            'explanation': """앙상블은 여러 모델의 예측을 평균화하거나 투표를 통해 결합합니다. 
이 과정에서 개별 모델들의 과적합으로 인한 높은 분산이 감소하여 
더 안정적이고 일반화된 예측을 제공합니다."""
        },
        
        # 13. 계산 복잡도
        {
            'id': 13,
            'type': 'multiple_choice',
            'question': """앙상블 모델의 계산 복잡도를 줄이는 방법으로 적절하지 않은 것은?""",
            'options': [
                "개별 모델의 복잡도를 줄이기",
                "앙상블에 포함되는 모델 수를 늘리기",
                "병렬 처리를 활용하기",
                "조기 종료(Early Stopping) 적용하기"
            ],
            'correct': 1,
            'explanation': """앙상블에 포함되는 모델 수를 늘리면 계산 복잡도가 증가합니다. 
계산 복잡도를 줄이려면 개별 모델 단순화, 병렬 처리, 조기 종료 등의 방법을 사용해야 합니다."""
        },
        
        # 14. 앙상블 평가
        {
            'id': 14,
            'type': 'multiple_choice',
            'question': """앙상블 모델의 성능을 평가할 때 주의해야 할 점은?""",
            'options': [
                "개별 모델들의 성능만 확인하면 된다",
                "훈련 데이터에서의 성능만 중요하다",
                "교차 검증을 통해 일반화 성능을 확인해야 한다",
                "가장 복잡한 모델의 성능을 기준으로 한다"
            ],
            'correct': 2,
            'explanation': """앙상블 모델도 과적합 위험이 있으므로 교차 검증을 통해 일반화 성능을 확인하는 것이 중요합니다. 
특히 스태킹과 같은 복잡한 앙상블에서는 더욱 주의깊은 검증이 필요합니다."""
        },
        
        # 15. 종합 이해도 평가
        {
            'id': 15,
            'type': 'short_answer',
            'question': """다음 상황에서 앙상블 기법의 효과가 제한적일 수 있는 이유를 분석하고, 
개선 방안을 제시하세요.

상황: 동일한 데이터셋에서 동일한 알고리즘(의사결정나무)으로 훈련된 10개의 모델을 
단순 평균으로 앙상블했지만 성능 향상이 미미합니다.""",
            'sample_answer': """효과 제한 이유:
1. 모델 다양성 부족: 동일한 알고리즘과 데이터 사용으로 유사한 예측 패턴
2. 높은 상관관계: 모델들이 비슷한 오류를 범할 가능성
3. 단순한 조합 방법: 평균화만으로는 모델별 강점 활용 제한

개선 방안:
1. 다양성 확보: 부트스트랩 샘플링, 특성 부분집합 선택, 다른 하이퍼파라미터 사용
2. 알고리즘 다양화: SVM, 로지스틱 회귀 등 다른 알고리즘 추가
3. 가중 조합: 개별 모델 성능에 따른 가중치 부여
4. 스태킹 적용: 메타 모델로 더 정교한 조합 학습""",
            'keywords': ['다양성', '상관관계', '부트스트랩', '알고리즘', '가중치', '스태킹'],
            'explanation': """앙상블의 효과는 모델 다양성에 크게 의존합니다. 동일한 조건에서 훈련된 모델들은 
유사한 예측을 하므로 앙상블 효과가 제한됩니다. 다양성 확보와 정교한 조합 방법이 필요합니다."""
        }
    ]
    
    return quiz_questions

def main():
    """
    앙상블 학습 퀴즈 실행
    """
    print("=" * 60)
    print("앙상블 학습 퀴즈")
    print("=" * 60)
    print()
    print("이 퀴즈는 앙상블 학습의 핵심 개념과 다양한 기법들에 대한")
    print("이해도를 평가합니다.")
    print()
    print("퀴즈 구성:")
    print("- 객관식 문제: 앙상블 기법의 특성과 원리")
    print("- 주관식 문제: 모델 다양성과 성능 관계")
    print("- 시나리오 문제: 실제 적용 상황에서의 기법 선택")
    print()
    
    # 퀴즈 질문 생성
    questions = create_ensemble_quiz()
    
    # 퀴즈 매니저 생성 및 실행
    quiz_manager = QuizManager()
    
    try:
        results = quiz_manager.run_quiz(
            questions=questions,
            quiz_title="앙상블 학습 퀴즈",
            time_limit=1800,  # 30분
            passing_score=70
        )
        
        # 상세 결과 분석
        print("\n" + "=" * 60)
        print("퀴즈 완료 - 상세 분석")
        print("=" * 60)
        
        # 영역별 성과 분석
        concept_questions = [1, 2, 3, 8, 12]  # 기본 개념
        technique_questions = [4, 5, 6, 7, 13, 14]  # 기법별 특성
        application_questions = [9, 10, 11, 15]  # 실제 적용
        
        concept_score = sum(1 for q_id in concept_questions if results['answers'].get(q_id, {}).get('correct', False))
        technique_score = sum(1 for q_id in technique_questions if results['answers'].get(q_id, {}).get('correct', False))
        application_score = sum(1 for q_id in application_questions if results['answers'].get(q_id, {}).get('correct', False))
        
        print(f"\n영역별 성과:")
        print(f"기본 개념 이해: {concept_score}/{len(concept_questions)} ({concept_score/len(concept_questions)*100:.1f}%)")
        print(f"기법별 특성: {technique_score}/{len(technique_questions)} ({technique_score/len(technique_questions)*100:.1f}%)")
        print(f"실제 적용: {application_score}/{len(application_questions)} ({application_score/len(application_questions)*100:.1f}%)")
        
        # 학습 권장사항
        print(f"\n학습 권장사항:")
        
        if concept_score < len(concept_questions) * 0.7:
            print("- 앙상블 학습의 기본 원리와 편향-분산 트레이드오프 개념을 복습하세요")
            
        if technique_score < len(technique_questions) * 0.7:
            print("- 배깅, 부스팅, 스태킹 등 각 기법의 특성과 차이점을 정리하세요")
            
        if application_score < len(application_questions) * 0.7:
            print("- 실제 문제 상황에서의 앙상블 기법 선택 기준을 학습하세요")
            
        if results['score'] >= 85:
            print("- 우수한 성과입니다! 고급 앙상블 기법들을 추가로 학습해보세요")
        elif results['score'] >= 70:
            print("- 양호한 이해도입니다. 실습을 통해 경험을 쌓아보세요")
        else:
            print("- 이론 학습을 다시 한 번 점검하고 실습 예제를 따라해보세요")
            
    except KeyboardInterrupt:
        print("\n\n퀴즈가 중단되었습니다.")
        print("언제든 다시 시작할 수 있습니다!")
    except Exception as e:
        print(f"\n퀴즈 실행 중 오류가 발생했습니다: {e}")
        print("프로그램을 다시 실행해주세요.")

if __name__ == "__main__":
    main()