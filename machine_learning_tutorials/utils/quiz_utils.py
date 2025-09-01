"""
퀴즈 유틸리티 모듈

퀴즈 문제 생성, 답안 검증, 해설 제공 및 학습 진도 추적 기능을 제공합니다.
요구사항 6.2, 6.3을 충족합니다.
"""

import json
import os
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import pickle


@dataclass
class QuizQuestion:
    """퀴즈 문제 데이터 클래스"""
    id: str
    question: str
    question_type: str  # 'multiple_choice', 'true_false', 'short_answer', 'numerical'
    options: List[str] = None  # 객관식 선택지
    correct_answer: Union[str, int, float] = None
    explanation: str = ""
    difficulty: str = "medium"  # 'easy', 'medium', 'hard'
    topic: str = ""
    points: int = 1


@dataclass
class QuizResult:
    """퀴즈 결과 데이터 클래스"""
    question_id: str
    user_answer: Union[str, int, float]
    correct_answer: Union[str, int, float]
    is_correct: bool
    points_earned: int
    time_taken: float  # 초 단위
    timestamp: str


class QuizManager:
    """퀴즈 관리 클래스"""
    
    def __init__(self, quiz_data_path: str = "quiz_data"):
        self.quiz_data_path = quiz_data_path
        self.questions: Dict[str, QuizQuestion] = {}
        self.user_progress: Dict[str, List[QuizResult]] = {}
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """데이터 디렉토리 생성"""
        if not os.path.exists(self.quiz_data_path):
            os.makedirs(self.quiz_data_path)
    
    def add_question(self, question: QuizQuestion):
        """퀴즈 문제 추가"""
        self.questions[question.id] = question
    
    def load_questions_from_file(self, filepath: str):
        """파일에서 퀴즈 문제 로드"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)
            
            for q_data in questions_data:
                question = QuizQuestion(**q_data)
                self.add_question(question)
                
        except FileNotFoundError:
            print(f"퀴즈 파일을 찾을 수 없습니다: {filepath}")
        except Exception as e:
            print(f"퀴즈 로딩 중 오류 발생: {e}")
    
    def save_questions_to_file(self, filepath: str):
        """퀴즈 문제를 파일에 저장"""
        questions_data = [asdict(q) for q in self.questions.values()]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(questions_data, f, ensure_ascii=False, indent=2)
    
    def get_questions_by_topic(self, topic: str) -> List[QuizQuestion]:
        """주제별 문제 조회"""
        return [q for q in self.questions.values() if q.topic == topic]
    
    def get_questions_by_difficulty(self, difficulty: str) -> List[QuizQuestion]:
        """난이도별 문제 조회"""
        return [q for q in self.questions.values() if q.difficulty == difficulty]
    
    def create_quiz_session(self, topic: str = None, difficulty: str = None, 
                          num_questions: int = 10) -> List[QuizQuestion]:
        """퀴즈 세션 생성"""
        available_questions = list(self.questions.values())
        
        # 필터링
        if topic:
            available_questions = [q for q in available_questions if q.topic == topic]
        if difficulty:
            available_questions = [q for q in available_questions if q.difficulty == difficulty]
        
        # 랜덤 선택
        if len(available_questions) > num_questions:
            selected_questions = random.sample(available_questions, num_questions)
        else:
            selected_questions = available_questions
        
        return selected_questions


class InteractiveQuiz:
    """인터랙티브 퀴즈 인터페이스"""
    
    def __init__(self, quiz_manager: QuizManager):
        self.quiz_manager = quiz_manager
        self.current_session: List[QuizQuestion] = []
        self.session_results: List[QuizResult] = []
        self.start_time: datetime = None
    
    def start_quiz(self, topic: str = None, difficulty: str = None, 
                  num_questions: int = 10):
        """퀴즈 시작"""
        self.current_session = self.quiz_manager.create_quiz_session(
            topic, difficulty, num_questions
        )
        self.session_results = []
        self.start_time = datetime.now()
        
        print(f"=== 퀴즈 시작 ===")
        print(f"주제: {topic or '전체'}")
        print(f"난이도: {difficulty or '전체'}")
        print(f"문제 수: {len(self.current_session)}")
        print("=" * 50)
    
    def ask_question(self, question: QuizQuestion) -> QuizResult:
        """개별 문제 출제"""
        print(f"\n문제 ID: {question.id}")
        print(f"주제: {question.topic}")
        print(f"난이도: {question.difficulty}")
        print(f"배점: {question.points}점")
        print("-" * 30)
        print(f"문제: {question.question}")
        
        question_start_time = datetime.now()
        
        if question.question_type == 'multiple_choice':
            return self._handle_multiple_choice(question, question_start_time)
        elif question.question_type == 'true_false':
            return self._handle_true_false(question, question_start_time)
        elif question.question_type == 'short_answer':
            return self._handle_short_answer(question, question_start_time)
        elif question.question_type == 'numerical':
            return self._handle_numerical(question, question_start_time)
        else:
            print("지원하지 않는 문제 유형입니다.")
            return None
    
    def _handle_multiple_choice(self, question: QuizQuestion, start_time: datetime) -> QuizResult:
        """객관식 문제 처리"""
        print("\n선택지:")
        for i, option in enumerate(question.options, 1):
            print(f"{i}. {option}")
        
        while True:
            try:
                user_input = input("\n답을 선택하세요 (번호 입력): ").strip()
                user_answer = int(user_input)
                
                if 1 <= user_answer <= len(question.options):
                    break
                else:
                    print(f"1부터 {len(question.options)} 사이의 번호를 입력하세요.")
            except ValueError:
                print("올바른 번호를 입력하세요.")
        
        time_taken = (datetime.now() - start_time).total_seconds()
        is_correct = user_answer == question.correct_answer
        points_earned = question.points if is_correct else 0
        
        result = QuizResult(
            question_id=question.id,
            user_answer=user_answer,
            correct_answer=question.correct_answer,
            is_correct=is_correct,
            points_earned=points_earned,
            time_taken=time_taken,
            timestamp=datetime.now().isoformat()
        )
        
        self._show_result(question, result)
        return result
    
    def _handle_true_false(self, question: QuizQuestion, start_time: datetime) -> QuizResult:
        """참/거짓 문제 처리"""
        print("\n1. 참 (True)")
        print("2. 거짓 (False)")
        
        while True:
            user_input = input("\n답을 선택하세요 (1 또는 2): ").strip()
            if user_input in ['1', '2']:
                user_answer = user_input == '1'
                break
            else:
                print("1 또는 2를 입력하세요.")
        
        time_taken = (datetime.now() - start_time).total_seconds()
        is_correct = user_answer == question.correct_answer
        points_earned = question.points if is_correct else 0
        
        result = QuizResult(
            question_id=question.id,
            user_answer=user_answer,
            correct_answer=question.correct_answer,
            is_correct=is_correct,
            points_earned=points_earned,
            time_taken=time_taken,
            timestamp=datetime.now().isoformat()
        )
        
        self._show_result(question, result)
        return result
    
    def _handle_short_answer(self, question: QuizQuestion, start_time: datetime) -> QuizResult:
        """단답형 문제 처리"""
        user_answer = input("\n답을 입력하세요: ").strip()
        
        time_taken = (datetime.now() - start_time).total_seconds()
        is_correct = self._check_text_similarity(user_answer, question.correct_answer)
        points_earned = question.points if is_correct else 0
        
        result = QuizResult(
            question_id=question.id,
            user_answer=user_answer,
            correct_answer=question.correct_answer,
            is_correct=is_correct,
            points_earned=points_earned,
            time_taken=time_taken,
            timestamp=datetime.now().isoformat()
        )
        
        self._show_result(question, result)
        return result
    
    def _handle_numerical(self, question: QuizQuestion, start_time: datetime) -> QuizResult:
        """수치형 문제 처리"""
        while True:
            try:
                user_input = input("\n답을 입력하세요 (숫자): ").strip()
                user_answer = float(user_input)
                break
            except ValueError:
                print("올바른 숫자를 입력하세요.")
        
        time_taken = (datetime.now() - start_time).total_seconds()
        is_correct = abs(user_answer - question.correct_answer) < 0.01  # 오차 허용
        points_earned = question.points if is_correct else 0
        
        result = QuizResult(
            question_id=question.id,
            user_answer=user_answer,
            correct_answer=question.correct_answer,
            is_correct=is_correct,
            points_earned=points_earned,
            time_taken=time_taken,
            timestamp=datetime.now().isoformat()
        )
        
        self._show_result(question, result)
        return result
    
    def _check_text_similarity(self, user_answer: str, correct_answer: str) -> bool:
        """텍스트 유사도 검사"""
        user_clean = user_answer.lower().strip()
        correct_clean = correct_answer.lower().strip()
        
        # 정확히 일치하는 경우
        if user_clean == correct_clean:
            return True
        
        # 키워드 포함 검사
        correct_keywords = correct_clean.split()
        user_keywords = user_clean.split()
        
        # 주요 키워드가 포함되어 있는지 확인
        keyword_match_ratio = sum(1 for keyword in correct_keywords 
                                if keyword in user_keywords) / len(correct_keywords)
        
        return keyword_match_ratio >= 0.7  # 70% 이상 키워드 일치
    
    def _show_result(self, question: QuizQuestion, result: QuizResult):
        """결과 표시"""
        print("\n" + "="*30)
        if result.is_correct:
            print("✅ 정답입니다!")
        else:
            print("❌ 틀렸습니다.")
            print(f"정답: {question.correct_answer}")
        
        print(f"획득 점수: {result.points_earned}/{question.points}")
        print(f"소요 시간: {result.time_taken:.1f}초")
        
        if question.explanation:
            print(f"\n📝 해설:")
            print(question.explanation)
        
        print("="*30)
    
    def run_full_quiz(self, topic: str = None, difficulty: str = None, 
                     num_questions: int = 10):
        """전체 퀴즈 실행"""
        self.start_quiz(topic, difficulty, num_questions)
        
        if not self.current_session:
            print("출제할 문제가 없습니다.")
            return
        
        for i, question in enumerate(self.current_session, 1):
            print(f"\n{'='*20} 문제 {i}/{len(self.current_session)} {'='*20}")
            result = self.ask_question(question)
            if result:
                self.session_results.append(result)
            
            # 다음 문제로 넘어갈지 확인
            if i < len(self.current_session):
                input("\n다음 문제로 넘어가려면 Enter를 누르세요...")
        
        self.show_final_results()
    
    def show_final_results(self):
        """최종 결과 표시"""
        if not self.session_results:
            return
        
        total_questions = len(self.session_results)
        correct_answers = sum(1 for r in self.session_results if r.is_correct)
        total_points = sum(r.points_earned for r in self.session_results)
        max_points = sum(self.quiz_manager.questions[r.question_id].points 
                        for r in self.session_results)
        total_time = sum(r.time_taken for r in self.session_results)
        
        print("\n" + "="*50)
        print("🎯 퀴즈 완료! 최종 결과")
        print("="*50)
        print(f"총 문제 수: {total_questions}")
        print(f"정답 수: {correct_answers}")
        print(f"정답률: {correct_answers/total_questions*100:.1f}%")
        print(f"총 점수: {total_points}/{max_points}")
        print(f"총 소요 시간: {total_time:.1f}초")
        print(f"평균 소요 시간: {total_time/total_questions:.1f}초/문제")
        
        # 성과 평가
        accuracy = correct_answers / total_questions
        if accuracy >= 0.9:
            print("\n🏆 탁월한 성과입니다!")
        elif accuracy >= 0.8:
            print("\n🥇 우수한 성과입니다!")
        elif accuracy >= 0.7:
            print("\n🥈 양호한 성과입니다!")
        else:
            print("\n📚 더 학습이 필요합니다!")
        
        print("="*50)


def generate_data_mining_questions() -> List[QuizQuestion]:
    """데이터 마이닝 개요 퀴즈 문제 생성"""
    questions = [
        QuizQuestion(
            id="dm_001",
            question="데이터 마이닝의 정의로 가장 적절한 것은?",
            question_type="multiple_choice",
            options=[
                "단순한 데이터 검색 과정",
                "전통적 데이터 분석과 대규모 데이터 처리 알고리즘의 결합",
                "데이터베이스 관리 시스템",
                "통계적 분석 방법"
            ],
            correct_answer=2,
            explanation="데이터 마이닝은 전통적 데이터 분석 기법과 대규모 데이터를 처리할 수 있는 알고리즘을 결합하여 암묵적이고 숨겨진 유용한 정보를 추출하는 과정입니다.",
            difficulty="easy",
            topic="데이터 마이닝 개요",
            points=2
        ),
        
        QuizQuestion(
            id="dm_002",
            question="다음 중 명목형(Nominal) 속성의 예는?",
            question_type="multiple_choice",
            options=[
                "온도 (섭씨)",
                "학점 (A, B, C, D, F)",
                "성별 (남, 여)",
                "키 (cm)"
            ],
            correct_answer=3,
            explanation="명목형 속성은 단순히 구분을 위한 라벨 역할만 하며, 순서나 크기의 의미가 없습니다. 성별은 단순 구분 목적의 명목형 속성입니다.",
            difficulty="medium",
            topic="데이터 마이닝 개요",
            points=2
        ),
        
        QuizQuestion(
            id="dm_003",
            question="데이터 마이닝의 주요 과제는 예측(Prediction)과 설명(Description)이다.",
            question_type="true_false",
            correct_answer=True,
            explanation="데이터 마이닝의 주요 과제는 크게 예측 과제(미래 값 예측)와 설명 과제(데이터 패턴 발견 및 설명)로 나뉩니다.",
            difficulty="easy",
            topic="데이터 마이닝 개요",
            points=1
        ),
        
        QuizQuestion(
            id="dm_004",
            question="비율형(Ratio) 속성의 특징을 설명하세요.",
            question_type="short_answer",
            correct_answer="절대적 0점이 존재하고 단위가 있어 비율 계산이 가능한 속성",
            explanation="비율형 속성은 구간형 속성의 특성에 더해 절대적인 0점이 존재하여 비율 계산이 의미가 있는 속성입니다. 예: 키, 몸무게, 나이 등",
            difficulty="medium",
            topic="데이터 마이닝 개요",
            points=3
        ),
        
        QuizQuestion(
            id="dm_005",
            question="대규모 데이터의 특성 중 '고차원'이 의미하는 바는?",
            question_type="multiple_choice",
            options=[
                "데이터의 크기가 매우 큼",
                "속성(특성)의 개수가 매우 많음",
                "데이터가 여러 형태로 구성됨",
                "데이터가 여러 위치에 분산됨"
            ],
            correct_answer=2,
            explanation="고차원은 데이터의 속성(특성, feature) 개수가 매우 많다는 의미입니다. 이는 차원의 저주 문제를 야기할 수 있습니다.",
            difficulty="medium",
            topic="데이터 마이닝 개요",
            points=2
        )
    ]
    
    return questions


def generate_preprocessing_questions() -> List[QuizQuestion]:
    """데이터 전처리 퀴즈 문제 생성"""
    questions = [
        QuizQuestion(
            id="prep_001",
            question="다음 중 잡음(Noise) 처리 방법이 아닌 것은?",
            question_type="multiple_choice",
            options=[
                "필터링 기법 적용",
                "스무딩 기법 사용",
                "데이터 정규화",
                "이상치 제거"
            ],
            correct_answer=3,
            explanation="데이터 정규화는 스케일 조정을 위한 방법이며, 잡음 처리와는 다른 목적입니다. 잡음 처리에는 필터링, 스무딩 등이 사용됩니다.",
            difficulty="medium",
            topic="데이터 전처리",
            points=2
        ),
        
        QuizQuestion(
            id="prep_002",
            question="결측치 처리 방법으로 적절하지 않은 것은?",
            question_type="multiple_choice",
            options=[
                "해당 레코드 제거",
                "평균값으로 대체",
                "중앙값으로 대체",
                "최댓값으로 대체"
            ],
            correct_answer=4,
            explanation="최댓값으로 대체하는 것은 데이터 분포를 크게 왜곡시킬 수 있어 적절하지 않습니다. 평균값, 중앙값, 최빈값 등이 일반적으로 사용됩니다.",
            difficulty="easy",
            topic="데이터 전처리",
            points=2
        )
    ]
    
    return questions


def save_quiz_questions_to_file(questions: List[QuizQuestion], filepath: str):
    """퀴즈 문제를 JSON 파일로 저장"""
    questions_data = [asdict(q) for q in questions]
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, ensure_ascii=False, indent=2)


def load_quiz_questions_from_file(filepath: str) -> List[QuizQuestion]:
    """JSON 파일에서 퀴즈 문제 로드"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        questions = []
        for q_data in questions_data:
            questions.append(QuizQuestion(**q_data))
        
        return questions
        
    except FileNotFoundError:
        print(f"퀴즈 파일을 찾을 수 없습니다: {filepath}")
        return []
    except Exception as e:
        print(f"퀴즈 로딩 중 오류 발생: {e}")
        return []


class ProgressTracker:
    """학습 진도 추적 클래스"""
    
    def __init__(self, data_path: str = "progress_data"):
        self.data_path = data_path
        self.user_progress: Dict[str, Dict] = {}
        self._ensure_data_directory()
        self.load_progress()
    
    def _ensure_data_directory(self):
        """데이터 디렉토리 생성"""
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
    
    def save_progress(self):
        """진도 데이터 저장"""
        progress_file = os.path.join(self.data_path, "user_progress.json")
        with open(progress_file, 'w', encoding='utf-8') as f:
            json.dump(self.user_progress, f, ensure_ascii=False, indent=2)
    
    def load_progress(self):
        """진도 데이터 로드"""
        progress_file = os.path.join(self.data_path, "user_progress.json")
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                self.user_progress = json.load(f)
        except FileNotFoundError:
            self.user_progress = {}
    
    def update_progress(self, user_id: str, topic: str, quiz_results: List[QuizResult]):
        """사용자 진도 업데이트"""
        if user_id not in self.user_progress:
            self.user_progress[user_id] = {}
        
        if topic not in self.user_progress[user_id]:
            self.user_progress[user_id][topic] = {
                'attempts': 0,
                'best_score': 0,
                'total_questions': 0,
                'correct_answers': 0,
                'last_attempt': None,
                'quiz_history': []
            }
        
        # 현재 퀴즈 결과 계산
        total_questions = len(quiz_results)
        correct_answers = sum(1 for r in quiz_results if r.is_correct)
        score = correct_answers / total_questions * 100 if total_questions > 0 else 0
        
        # 진도 업데이트
        topic_progress = self.user_progress[user_id][topic]
        topic_progress['attempts'] += 1
        topic_progress['best_score'] = max(topic_progress['best_score'], score)
        topic_progress['total_questions'] += total_questions
        topic_progress['correct_answers'] += correct_answers
        topic_progress['last_attempt'] = datetime.now().isoformat()
        
        # 퀴즈 기록 추가
        quiz_record = {
            'timestamp': datetime.now().isoformat(),
            'score': score,
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'results': [asdict(r) for r in quiz_results]
        }
        topic_progress['quiz_history'].append(quiz_record)
        
        self.save_progress()
    
    def get_user_progress(self, user_id: str) -> Dict:
        """사용자 진도 조회"""
        return self.user_progress.get(user_id, {})
    
    def get_topic_progress(self, user_id: str, topic: str) -> Dict:
        """특정 주제 진도 조회"""
        user_data = self.get_user_progress(user_id)
        return user_data.get(topic, {})
    
    def generate_progress_report(self, user_id: str) -> str:
        """진도 보고서 생성"""
        user_data = self.get_user_progress(user_id)
        
        if not user_data:
            return f"사용자 {user_id}의 학습 기록이 없습니다."
        
        report = f"=== {user_id} 학습 진도 보고서 ===\n\n"
        
        total_attempts = 0
        total_score = 0
        topic_count = 0
        
        for topic, progress in user_data.items():
            attempts = progress.get('attempts', 0)
            best_score = progress.get('best_score', 0)
            last_attempt = progress.get('last_attempt', 'N/A')
            
            report += f"📚 주제: {topic}\n"
            report += f"   시도 횟수: {attempts}\n"
            report += f"   최고 점수: {best_score:.1f}%\n"
            report += f"   마지막 시도: {last_attempt}\n\n"
            
            total_attempts += attempts
            total_score += best_score
            topic_count += 1
        
        if topic_count > 0:
            avg_score = total_score / topic_count
            report += f"📊 전체 통계\n"
            report += f"   학습한 주제 수: {topic_count}\n"
            report += f"   총 시도 횟수: {total_attempts}\n"
            report += f"   평균 최고 점수: {avg_score:.1f}%\n"
        
        return report


# 퀴즈 실행 함수들
def run_interactive_quiz(topic: str = None, difficulty: str = None, num_questions: int = 10):
    """인터랙티브 퀴즈 실행"""
    # 퀴즈 매니저 초기화
    quiz_manager = QuizManager()
    
    # 기본 문제들 추가
    dm_questions = generate_data_mining_questions()
    prep_questions = generate_preprocessing_questions()
    
    for q in dm_questions + prep_questions:
        quiz_manager.add_question(q)
    
    # 퀴즈 실행
    interactive_quiz = InteractiveQuiz(quiz_manager)
    interactive_quiz.run_full_quiz(topic, difficulty, num_questions)
    
    return interactive_quiz.session_results


def create_sample_quiz_files():
    """샘플 퀴즈 파일 생성"""
    # 데이터 마이닝 개요 퀴즈
    dm_questions = generate_data_mining_questions()
    save_quiz_questions_to_file(dm_questions, "data_mining_overview_quiz.json")
    
    # 데이터 전처리 퀴즈
    prep_questions = generate_preprocessing_questions()
    save_quiz_questions_to_file(prep_questions, "data_preprocessing_quiz.json")
    
    print("샘플 퀴즈 파일이 생성되었습니다:")
    print("- data_mining_overview_quiz.json")
    print("- data_preprocessing_quiz.json")


if __name__ == "__main__":
    # 샘플 퀴즈 실행 예제
    print("머신러닝 튜토리얼 퀴즈 시스템")
    print("1. 데이터 마이닝 개요 퀴즈")
    print("2. 데이터 전처리 퀴즈")
    print("3. 전체 퀴즈")
    
    choice = input("선택하세요 (1-3): ").strip()
    
    if choice == "1":
        results = run_interactive_quiz(topic="데이터 마이닝 개요", num_questions=5)
    elif choice == "2":
        results = run_interactive_quiz(topic="데이터 전처리", num_questions=5)
    elif choice == "3":
        results = run_interactive_quiz(num_questions=10)
    else:
        print("올바른 선택지를 입력하세요.")