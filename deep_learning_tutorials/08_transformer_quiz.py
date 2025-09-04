#!/usr/bin/env python3
"""
Transformer 퀴즈

이 퀴즈는 다음 내용을 평가합니다:
1. 어텐션 메커니즘 발전사 (Bahdanau → Luong → Self-Attention)
2. Self-Attention vs Cross-Attention 차이점
3. Multi-Head Attention 계산 과정
4. 위치 인코딩 필요성 및 구현 방법
5. BERT, GPT 등 Transformer 기반 모델 특성

요구사항 8.1, 8.2, 8.3, 7.1, 7.2, 7.3, 7.4, 7.5를 충족합니다.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from quiz_utils import DeepLearningQuizManager, QuizQuestion
import math


class TransformerQuiz:
    def __init__(self):
        self.quiz_manager = DeepLearningQuizManager("Transformer 퀴즈")
        self.setup_questions()
    
    def setup_questions(self):
        """퀴즈 문제 설정"""
        
        # 1. 전통적 seq2seq 한계 - Easy
        self.quiz_manager.add_question_simple(
            question_id="transformer_001",
            question_type="multiple_choice",
            question="전통적인 seq2seq 모델의 주요 한계는?",
            options=[
                "계산 속도가 너무 빠름",
                "모든 입력 정보를 고정 크기 벡터로 압축하는 정보 병목",
                "GPU 메모리를 너무 많이 사용함",
                "오직 영어만 처리 가능함"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 모든 입력 정보를 고정 크기 벡터로 압축하는 정보 병목

해설:
전통적인 seq2seq 모델은 인코더의 모든 정보를 하나의 고정 크기 컨텍스트 벡터로 
압축해야 하므로, 긴 시퀀스에서 초기 정보가 손실되는 정보 병목 현상이 발생합니다.
이는 어텐션 메커니즘이 해결하고자 한 핵심 문제입니다.
            """,
            difficulty="easy",
            topic="Transformer",
            related_theory_section="Transformer 이론 - 어텐션 메커니즘 발전사"
        )
        
        # 2. Bahdanau Attention 이해 - Medium
        self.quiz_manager.add_question_simple(
            question_id="transformer_002",
            question_type="multiple_choice",
            question="Bahdanau Attention (2014)의 핵심 아이디어는?",
            options=[
                "RNN을 완전히 제거함",
                "디코더가 각 시점에서 인코더의 모든 은닉 상태를 참조할 수 있게 함",
                "GPU 병렬 처리를 가능하게 함",
                "위치 정보를 자동으로 학습함"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 디코더가 각 시점에서 인코더의 모든 은닉 상태를 참조할 수 있게 함

해설:
Bahdanau Attention은 디코더의 각 시점에서 인코더의 모든 은닉 상태에 
동적으로 가중치를 부여하여 컨텍스트 벡터를 생성합니다. 
이를 통해 정보 병목 문제를 해결했습니다.
            """,
            difficulty="medium",
            topic="Transformer",
            related_theory_section="Transformer 이론 - Bahdanau Attention"
        )
        
        # 3. Luong vs Bahdanau - Hard
        self.quiz_manager.add_question_simple(
            question_id="transformer_003",
            question_type="multiple_choice",
            question="Luong Attention (2015)이 Bahdanau Attention 대비 개선한 점은?",
            options=[
                "완전히 다른 수식 사용",
                "Global/Local Attention 구분 및 다양한 스코어 함수 제안",
                "RNN 구조 완전 제거",
                "Self-Attention 도입"
            ],
            correct_answer=2,
            explanation="""
정답: 2) Global/Local Attention 구분 및 다양한 스코어 함수 제안

해설:
Luong Attention은 Global Attention(모든 위치 참조)과 Local Attention(특정 윈도우만 참조)을 
구분하고, dot, general, concat 등 다양한 스코어 함수를 제안하여 계산 효율성을 개선했습니다.
            """,
            difficulty="hard",
            topic="Transformer",
            related_theory_section="Transformer 이론 - Luong Attention"
        )
        
        # 4. Self-Attention 혁신 - Medium
        self.quiz_manager.add_question_simple(
            question_id="transformer_004",
            question_type="short_answer",
            question="Self-Attention이 기존 어텐션과 다른 핵심적인 차이점은?",
            correct_answer="같은 시퀀스 내부의 관계를 학습",
            explanation="""
정답: 같은 시퀀스 내부의 관계를 학습

해설:
기존 어텐션은 인코더-디코더 간의 관계를 학습했지만, 
Self-Attention은 같은 시퀀스 내의 토큰들 간의 관계를 학습합니다.
이를 통해 문맥적 표현을 더 풍부하게 만들 수 있습니다.
            """,
            difficulty="medium",
            topic="Transformer",
            related_theory_section="Transformer 이론 - Self-Attention"
        )
        
        # 5. Cross-Attention 연관성 - Hard
        self.quiz_manager.add_question_simple(
            question_id="transformer_005",
            question_type="true_false",
            question="Transformer의 Cross-Attention은 전통적인 인코더-디코더 어텐션의 일반화된 형태이다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
Cross-Attention은 Query가 디코더에서, Key/Value가 인코더에서 오는 구조로,
전통적인 인코더-디코더 어텐션과 본질적으로 같은 역할을 합니다.
다만 Self-Attention을 거친 더 풍부한 표현을 사용한다는 점이 다릅니다.
            """,
            difficulty="hard",
            topic="Transformer",
            related_theory_section="Transformer 이론 - Cross-Attention"
        )
        
        # 6. Scaled Dot-Product Attention 수식 - Medium
        self.quiz_manager.add_question_simple(
            question_id="transformer_006",
            question_type="multiple_choice",
            question="Scaled Dot-Product Attention에서 √d_k로 나누는 이유는?",
            options=[
                "계산 속도 향상",
                "메모리 사용량 감소",
                "내적 값이 커져서 softmax가 포화되는 것을 방지",
                "음수 값을 양수로 변환"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 내적 값이 커져서 softmax가 포화되는 것을 방지

해설:
d_k가 클 때 Q와 K의 내적 값이 매우 커질 수 있고, 
이는 softmax 함수를 포화 영역으로 밀어넣어 그래디언트가 매우 작아지게 합니다.
√d_k로 스케일링하여 이를 방지합니다.
            """,
            difficulty="medium",
            topic="Transformer",
            related_theory_section="Transformer 이론 - Self-Attention 메커니즘"
        )
        
        # 7. Multi-Head Attention 계산 - Hard
        self.quiz_manager.add_question_simple(
            question_id="transformer_007",
            question_type="calculation",
            question="d_model=512, num_heads=8일 때, 각 헤드의 d_k 차원은?",
            correct_answer=64,
            explanation="""
정답: 64

해설:
Multi-Head Attention에서 각 헤드의 차원은:
d_k = d_model / num_heads = 512 / 8 = 64

이렇게 차원을 나누어 여러 헤드가 서로 다른 표현 공간에서 
어텐션을 계산할 수 있게 합니다.
            """,
            difficulty="hard",
            topic="Transformer",
            related_theory_section="Transformer 이론 - Multi-Head Attention",
            formula="d_k = d_model / num_heads",
            tolerance=0
        )
        
        # 8. 위치 인코딩 필요성 - Easy
        self.quiz_manager.add_question_simple(
            question_id="transformer_008",
            question_type="multiple_choice",
            question="Transformer에서 위치 인코딩이 필요한 이유는?",
            options=[
                "계산 속도 향상을 위해",
                "Self-Attention은 순서 정보를 고려하지 않기 때문",
                "메모리 사용량 감소를 위해",
                "GPU 병렬 처리를 위해"
            ],
            correct_answer=2,
            explanation="""
정답: 2) Self-Attention은 순서 정보를 고려하지 않기 때문

해설:
Self-Attention은 모든 위치를 동시에 참조하므로 토큰의 순서 정보가 없습니다.
따라서 위치 인코딩을 추가하여 각 토큰의 위치 정보를 제공해야 합니다.
            """,
            difficulty="easy",
            topic="Transformer",
            related_theory_section="Transformer 이론 - 위치 인코딩"
        )
        
        # 9. 사인/코사인 위치 인코딩 - Hard
        self.quiz_manager.add_question_simple(
            question_id="transformer_009",
            question_type="multiple_choice",
            question="사인/코사인 위치 인코딩의 장점은?",
            options=[
                "학습이 필요 없음",
                "훈련 시보다 긴 시퀀스에도 적용 가능",
                "상대적 위치 관계 학습 가능",
                "위의 모든 것"
            ],
            correct_answer=4,
            explanation="""
정답: 4) 위의 모든 것

해설:
사인/코사인 위치 인코딩은 고정된 수식을 사용하므로 학습이 불필요하고,
주기적 패턴으로 인해 훈련 시보다 긴 시퀀스에도 적용 가능하며,
삼각함수의 성질로 상대적 위치 관계도 학습할 수 있습니다.
            """,
            difficulty="hard",
            topic="Transformer",
            related_theory_section="Transformer 이론 - 위치 인코딩"
        )
        
        # 10. Masked Self-Attention - Medium
        self.quiz_manager.add_question_simple(
            question_id="transformer_010",
            question_type="multiple_choice",
            question="디코더의 Masked Self-Attention에서 마스킹하는 이유는?",
            options=[
                "계산 속도 향상",
                "메모리 절약",
                "미래 토큰을 참조하지 못하게 하여 자기회귀적 생성 보장",
                "노이즈 제거"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 미래 토큰을 참조하지 못하게 하여 자기회귀적 생성 보장

해설:
디코더는 순차적으로 토큰을 생성해야 하므로, 현재 위치에서 
미래 위치의 토큰을 참조하면 안 됩니다. 마스킹을 통해 이를 방지합니다.
            """,
            difficulty="medium",
            topic="Transformer",
            related_theory_section="Transformer 이론 - 디코더 구조"
        )
        
        # 11. Layer Normalization - Medium
        self.quiz_manager.add_question_simple(
            question_id="transformer_011",
            question_type="true_false",
            question="Transformer에서 Layer Normalization은 Residual Connection 이후에 적용된다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
Transformer의 각 서브레이어는 LayerNorm(x + Sublayer(x)) 구조를 사용합니다.
즉, 잔차 연결(x + Sublayer(x)) 후에 Layer Normalization을 적용합니다.
            """,
            difficulty="medium",
            topic="Transformer",
            related_theory_section="Transformer 이론 - 아키텍처 상세 분석"
        )
        
        # 12. BERT vs GPT - Medium
        self.quiz_manager.add_question_simple(
            question_id="transformer_012",
            question_type="multiple_choice",
            question="BERT와 GPT의 주요 차이점은?",
            options=[
                "BERT는 인코더만, GPT는 디코더만 사용",
                "BERT는 양방향, GPT는 단방향 어텐션",
                "BERT는 MLM, GPT는 자기회귀 학습",
                "위의 모든 것"
            ],
            correct_answer=4,
            explanation="""
정답: 4) 위의 모든 것

해설:
BERT는 인코더 구조로 양방향 어텐션과 MLM(Masked Language Model)을 사용하고,
GPT는 디코더 구조로 단방향 어텐션과 자기회귀적 언어 모델링을 사용합니다.
            """,
            difficulty="medium",
            topic="Transformer",
            related_theory_section="Transformer 이론 - Transformer 기반 모델들"
        )
        
        # 13. 어텐션 복잡도 - Hard
        self.quiz_manager.add_question_simple(
            question_id="transformer_013",
            question_type="multiple_choice",
            question="Self-Attention의 시간 복잡도는?",
            options=[
                "O(n)",
                "O(n log n)",
                "O(n²)",
                "O(n³)"
            ],
            correct_answer=3,
            explanation="""
정답: 3) O(n²)

해설:
Self-Attention은 길이 n인 시퀀스에서 모든 위치 쌍 간의 어텐션을 계산하므로
시간 복잡도가 O(n²)입니다. 이는 긴 시퀀스에서 메모리와 계산 비용이 
제곱적으로 증가하는 한계를 가집니다.
            """,
            difficulty="hard",
            topic="Transformer",
            related_theory_section="Transformer 이론 - 실용적 고려사항"
        )
        
        # 14. Transformer 장점 - Easy
        self.quiz_manager.add_question_simple(
            question_id="transformer_014",
            question_type="multiple_choice",
            question="Transformer의 주요 장점이 아닌 것은?",
            options=[
                "병렬 처리 가능",
                "장거리 의존성 학습",
                "메모리 사용량이 적음",
                "해석 가능성"
            ],
            correct_answer=3,
            explanation="""
정답: 3) 메모리 사용량이 적음

해설:
Transformer는 O(n²) 복잡도로 인해 긴 시퀀스에서 메모리 사용량이 많습니다.
병렬 처리, 장거리 의존성 학습, 어텐션 가중치를 통한 해석 가능성은 장점입니다.
            """,
            difficulty="easy",
            topic="Transformer",
            related_theory_section="Transformer 이론 - 실용적 고려사항"
        )
        
        # 15. 어텐션 발전사 종합 - Hard
        self.quiz_manager.add_question_simple(
            question_id="transformer_015",
            question_type="multiple_choice",
            question="어텐션 메커니즘의 발전 순서로 올바른 것은?",
            options=[
                "Self-Attention → Bahdanau → Luong → Cross-Attention",
                "Bahdanau → Self-Attention → Luong → Cross-Attention",
                "Bahdanau → Luong → Self-Attention → Cross-Attention",
                "Luong → Bahdanau → Self-Attention → Cross-Attention"
            ],
            correct_answer=3,
            explanation="""
정답: 3) Bahdanau → Luong → Self-Attention → Cross-Attention

해설:
어텐션 발전사:
1. Bahdanau (2014): 최초 어텐션, 정보 병목 해결
2. Luong (2015): 효율성 개선, 다양한 스코어 함수
3. Self-Attention (2017): 같은 시퀀스 내부 관계 학습
4. Cross-Attention: Transformer에서 인코더-디코더 간 어텐션
            """,
            difficulty="hard",
            topic="Transformer",
            related_theory_section="Transformer 이론 - 어텐션 메커니즘 발전사"
        )
        
        # 16. T5 모델 특징 - Medium
        self.quiz_manager.add_question_simple(
            question_id="transformer_016",
            question_type="short_answer",
            question="T5 모델의 핵심 아이디어를 한 단어로 표현하면?",
            correct_answer="Text-to-Text",
            explanation="""
정답: Text-to-Text

해설:
T5(Text-to-Text Transfer Transformer)는 모든 NLP 작업을 
텍스트 입력을 텍스트 출력으로 변환하는 통일된 프레임워크로 처리합니다.
            """,
            difficulty="medium",
            topic="Transformer",
            related_theory_section="Transformer 이론 - T5"
        )
        
        # 17. Vision Transformer - Medium
        self.quiz_manager.add_question_simple(
            question_id="transformer_017",
            question_type="multiple_choice",
            question="Vision Transformer(ViT)에서 이미지를 처리하는 방법은?",
            options=[
                "픽셀 단위로 처리",
                "이미지를 패치로 나누어 시퀀스로 처리",
                "CNN과 결합하여 처리",
                "주파수 도메인으로 변환하여 처리"
            ],
            correct_answer=2,
            explanation="""
정답: 2) 이미지를 패치로 나누어 시퀀스로 처리

해설:
ViT는 이미지를 고정 크기 패치로 나누고, 각 패치를 벡터로 변환하여
시퀀스로 만든 후 표준 Transformer 구조로 처리합니다.
            """,
            difficulty="medium",
            topic="Transformer",
            related_theory_section="Transformer 이론 - Vision Transformer"
        )
        
        # 18. 어텐션 가중치 해석 - Easy
        self.quiz_manager.add_question_simple(
            question_id="transformer_018",
            question_type="true_false",
            question="어텐션 가중치를 시각화하면 모델이 어떤 부분에 집중하는지 알 수 있다.",
            correct_answer=True,
            explanation="""
정답: 참 (True)

해설:
어텐션 가중치는 각 토큰이 다른 토큰들에 얼마나 집중하는지를 나타내므로,
이를 시각화하면 모델의 의사결정 과정을 어느 정도 해석할 수 있습니다.
            """,
            difficulty="easy",
            topic="Transformer",
            related_theory_section="Transformer 이론 - 해석 가능성"
        )
    
    def run_quiz(self, num_questions: int = 18, difficulty: str = None):
        """퀴즈 실행"""
        print("🤖 Transformer 퀴즈에 오신 것을 환영합니다!")
        print("이 퀴즈는 어텐션 메커니즘의 발전사와 Transformer 구조를 평가합니다.")
        print("-" * 70)
        
        results = self.quiz_manager.run_full_quiz(
            topic="Transformer",
            difficulty=difficulty,
            num_questions=num_questions
        )
        
        return results
    
    def run_retry_quiz(self):
        """틀린 문제 재시도"""
        return self.quiz_manager.retry_wrong_questions()
    
    def run_attention_history_quiz(self):
        """어텐션 발전사 특화 퀴즈"""
        print("📚 어텐션 메커니즘 발전사 특화 퀴즈")
        print("Bahdanau → Luong → Self-Attention → Cross-Attention")
        print("-" * 60)
        
        # 어텐션 발전사 관련 문제만 선별
        attention_history_questions = [
            "transformer_001", "transformer_002", "transformer_003", 
            "transformer_004", "transformer_005", "transformer_015"
        ]
        
        selected_questions = [self.quiz_manager.questions[qid] for qid in attention_history_questions 
                            if qid in self.quiz_manager.questions]
        
        if not selected_questions:
            print("어텐션 발전사 문제를 찾을 수 없습니다.")
            return []
        
        self.quiz_manager.current_session = selected_questions
        self.quiz_manager.session_results = []
        self.quiz_manager.start_time = datetime.now()
        
        for i, question in enumerate(selected_questions, 1):
            print(f"\n{'='*15} 문제 {i}/{len(selected_questions)} {'='*15}")
            result = self.quiz_manager.ask_question(question)
            if result:
                self.quiz_manager.session_results.append(result)
            
            if i < len(selected_questions):
                input("\n다음 문제로 넘어가려면 Enter를 누르세요...")
        
        self.quiz_manager.show_final_results()
        return self.quiz_manager.session_results


def main():
    """메인 실행 함수"""
    quiz = TransformerQuiz()
    
    print("Transformer 퀴즈 시스템")
    print("=" * 50)
    print("1. 전체 퀴즈 (18문제)")
    print("2. 어텐션 발전사 특화 퀴즈")
    print("3. 쉬운 문제만 (Easy)")
    print("4. 보통 문제만 (Medium)")
    print("5. 어려운 문제만 (Hard)")
    print("6. 맞춤형 퀴즈")
    
    while True:
        choice = input("\n선택하세요 (1-6, q: 종료): ").strip()
        
        if choice.lower() == 'q':
            print("퀴즈를 종료합니다. 수고하셨습니다!")
            break
        elif choice == '1':
            results = quiz.run_quiz()
        elif choice == '2':
            results = quiz.run_attention_history_quiz()
        elif choice == '3':
            results = quiz.run_quiz(difficulty="easy")
        elif choice == '4':
            results = quiz.run_quiz(difficulty="medium")
        elif choice == '5':
            results = quiz.run_quiz(difficulty="hard")
        elif choice == '6':
            try:
                num_q = int(input("문제 수를 입력하세요 (1-18): "))
                if 1 <= num_q <= 18:
                    results = quiz.run_quiz(num_questions=num_q)
                else:
                    print("1-18 사이의 숫자를 입력하세요.")
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