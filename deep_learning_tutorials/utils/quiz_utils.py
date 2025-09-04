"""
딥러닝 튜토리얼 퀴즈 유틸리티 모듈

딥러닝 특화 퀴즈 문제 생성, 답안 검증, 해설 제공 및 학습 진도 추적 기능을 제공합니다.
요구사항 8.1, 8.2, 8.3, 8.4, 8.5를 충족합니다.
"""

import json
import os
import random
import math
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
    question_type: str  # 'multiple_choice', 'true_false', 'short_answer', 'numerical', 'calculation'
    options: List[str] = None  # 객관식 선택지
    correct_answer: Union[str, int, float, bool] = None
    explanation: str = ""
    difficulty: str = "medium"  # 'easy', 'medium', 'hard'
    topic: str = ""
    points: int = 1
    related_theory_section: str = ""  # 관련 이론 섹션 참조
    formula: str = ""  # 수식이 포함된 문제의 경우
    tolerance: float = 0.01  # 수치 답안의 허용 오차


@dataclass
class QuizResult:
    """퀴즈 결과 데이터 클래스"""
    question_id: str
    user_answer: Union[str, int, float, bool]
    correct_answer: Union[str, int, float, bool]
    is_correct: bool
    points_earned: int
    time_taken: float  # 초 단위
    timestamp: str
    difficulty: str
    topic: str


class DeepLearningQuizManager:
    """딥러닝 퀴즈 관리 클래스"""
    
    def __init__(self, quiz_name: str = "딥러닝 퀴즈"):
        self.quiz_name = quiz_name
        self.questions: Dict[str, QuizQuestion] = {}
        self.current_session: List[QuizQuestion] = []
        self.session_results: List[QuizResult] = []
        self.start_time: datetime = None
    
    def add_question(self, question: QuizQuestion):
        """퀴즈 문제 추가"""
        self.questions[question.id] = question
    
    def add_question_simple(self, question_id: str, question_type: str, question: str, 
                           options: list = None, correct_answer: Union[str, int, float, bool] = None, 
                           explanation: str = "", difficulty: str = "medium", topic: str = "",
                           points: int = None, related_theory_section: str = "", 
                           formula: str = "", tolerance: float = 0.01):
        """간단한 방식으로 퀴즈 문제 추가"""
        # 난이도별 기본 점수 설정
        if points is None:
            points = {"easy": 1, "medium": 2, "hard": 3}.get(difficulty, 2)
        
        question_obj = QuizQuestion(
            id=question_id,
            question=question,
            question_type=question_type,
            options=options,
            correct_answer=correct_answer,
            explanation=explanation,
            difficulty=difficulty,
            topic=topic,
            points=points,
            related_theory_section=related_theory_section,
            formula=formula,
            tolerance=tolerance
        )
        self.add_question(question_obj)
    
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
    
    def start_quiz_session(self, topic: str = None, difficulty: str = None, 
                          num_questions: int = 10):
        """퀴즈 세션 시작"""
        self.current_session = self.create_quiz_session(topic, difficulty, num_questions)
        self.session_results = []
        self.start_time = datetime.now()
        
        print(f"=== {self.quiz_name} 시작 ===")
        print(f"주제: {topic or '전체'}")
        print(f"난이도: {difficulty or '전체'}")
        print(f"문제 수: {len(self.current_session)}")
        print("=" * 50)
    
    def ask_question(self, question: QuizQuestion) -> QuizResult:
        """개별 문제 출제"""
        print(f"\n문제 ID: {question.id}")
        print(f"주제: {question.topic}")
        print(f"난이도: {question.difficulty} ({question.points}점)")
        if question.related_theory_section:
            print(f"관련 이론: {question.related_theory_section}")
        print("-" * 40)
        print(f"문제: {question.question}")
        
        if question.formula:
            print(f"참고 수식: {question.formula}")
        
        question_start_time = datetime.now()
        
        if question.question_type == 'multiple_choice':
            return self._handle_multiple_choice(question, question_start_time)
        elif question.question_type == 'true_false':
            return self._handle_true_false(question, question_start_time)
        elif question.question_type == 'short_answer':
            return self._handle_short_answer(question, question_start_time)
        elif question.question_type == 'numerical':
            return self._handle_numerical(question, question_start_time)
        elif question.question_type == 'calculation':
            return self._handle_calculation(question, question_start_time)
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
            timestamp=datetime.now().isoformat(),
            difficulty=question.difficulty,
            topic=question.topic
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
            timestamp=datetime.now().isoformat(),
            difficulty=question.difficulty,
            topic=question.topic
        )
        
        self._show_result(question, result)
        return result
    
    def _handle_short_answer(self, question: QuizQuestion, start_time: datetime) -> QuizResult:
        """단답형 문제 처리"""
        user_answer = input("\n답을 입력하세요: ").strip()
        
        time_taken = (datetime.now() - start_time).total_seconds()
        is_correct = self._check_text_similarity(user_answer, str(question.correct_answer))
        points_earned = question.points if is_correct else 0
        
        result = QuizResult(
            question_id=question.id,
            user_answer=user_answer,
            correct_answer=question.correct_answer,
            is_correct=is_correct,
            points_earned=points_earned,
            time_taken=time_taken,
            timestamp=datetime.now().isoformat(),
            difficulty=question.difficulty,
            topic=question.topic
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
        is_correct = abs(user_answer - float(question.correct_answer)) <= question.tolerance
        points_earned = question.points if is_correct else 0
        
        result = QuizResult(
            question_id=question.id,
            user_answer=user_answer,
            correct_answer=question.correct_answer,
            is_correct=is_correct,
            points_earned=points_earned,
            time_taken=time_taken,
            timestamp=datetime.now().isoformat(),
            difficulty=question.difficulty,
            topic=question.topic
        )
        
        self._show_result(question, result)
        return result
    
    def _handle_calculation(self, question: QuizQuestion, start_time: datetime) -> QuizResult:
        """계산 문제 처리 (딥러닝 특화)"""
        print("\n💡 힌트: 계산 과정을 단계별로 생각해보세요.")
        if question.formula:
            print(f"참고 수식: {question.formula}")
        
        while True:
            try:
                user_input = input("\n최종 답을 입력하세요 (숫자): ").strip()
                user_answer = float(user_input)
                break
            except ValueError:
                print("올바른 숫자를 입력하세요.")
        
        time_taken = (datetime.now() - start_time).total_seconds()
        is_correct = abs(user_answer - float(question.correct_answer)) <= question.tolerance
        points_earned = question.points if is_correct else 0
        
        result = QuizResult(
            question_id=question.id,
            user_answer=user_answer,
            correct_answer=question.correct_answer,
            is_correct=is_correct,
            points_earned=points_earned,
            time_taken=time_taken,
            timestamp=datetime.now().isoformat(),
            difficulty=question.difficulty,
            topic=question.topic
        )
        
        self._show_result(question, result)
        return result
    
    def _check_text_similarity(self, user_answer: str, correct_answer: str) -> bool:
        """텍스트 유사도 검사 (딥러닝 용어 특화)"""
        user_clean = user_answer.lower().strip()
        correct_clean = correct_answer.lower().strip()
        
        # 정확히 일치하는 경우
        if user_clean == correct_clean:
            return True
        
        # 딥러닝 용어 동의어 처리
        synonyms = {
            'cnn': ['convolutional neural network', 'convnet'],
            'rnn': ['recurrent neural network'],
            'lstm': ['long short-term memory'],
            'gru': ['gated recurrent unit'],
            'mlp': ['multi-layer perceptron', 'multilayer perceptron'],
            'relu': ['rectified linear unit'],
            'softmax': ['soft max'],
            'backprop': ['backpropagation', 'back propagation'],
            'gradient descent': ['경사하강법'],
            'overfitting': ['과적합'],
            'underfitting': ['과소적합']
        }
        
        # 동의어 확인
        for key, values in synonyms.items():
            if (key in user_clean and correct_clean in values) or \
               (correct_clean == key and any(v in user_clean for v in values)):
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
        print("\n" + "="*40)
        if result.is_correct:
            print("✅ 정답입니다!")
        else:
            print("❌ 틀렸습니다.")
            if question.question_type in ['numerical', 'calculation']:
                print(f"정답: {question.correct_answer}")
                if question.tolerance > 0:
                    print(f"허용 오차: ±{question.tolerance}")
            else:
                print(f"정답: {question.correct_answer}")
        
        print(f"획득 점수: {result.points_earned}/{question.points}")
        print(f"소요 시간: {result.time_taken:.1f}초")
        
        if question.explanation:
            print(f"\n📝 해설:")
            print(question.explanation)
        
        if question.related_theory_section:
            print(f"\n📚 복습 추천: {question.related_theory_section}")
        
        print("="*40)
    
    def run_full_quiz(self, topic: str = None, difficulty: str = None, 
                     num_questions: int = 10):
        """전체 퀴즈 실행"""
        self.start_quiz_session(topic, difficulty, num_questions)
        
        if not self.current_session:
            print("출제할 문제가 없습니다.")
            return []
        
        for i, question in enumerate(self.current_session, 1):
            print(f"\n{'='*15} 문제 {i}/{len(self.current_session)} {'='*15}")
            result = self.ask_question(question)
            if result:
                self.session_results.append(result)
            
            # 다음 문제로 넘어갈지 확인
            if i < len(self.current_session):
                input("\n다음 문제로 넘어가려면 Enter를 누르세요...")
        
        self.show_final_results()
        return self.session_results
    
    def show_final_results(self):
        """최종 결과 표시"""
        if not self.session_results:
            return
        
        total_questions = len(self.session_results)
        correct_answers = sum(1 for r in self.session_results if r.is_correct)
        total_points = sum(r.points_earned for r in self.session_results)
        max_points = sum(self.questions[r.question_id].points for r in self.session_results)
        total_time = sum(r.time_taken for r in self.session_results)
        
        print("\n" + "="*60)
        print("🎯 퀴즈 완료! 최종 결과")
        print("="*60)
        print(f"총 문제 수: {total_questions}")
        print(f"정답 수: {correct_answers}")
        print(f"정답률: {correct_answers/total_questions*100:.1f}%")
        print(f"총 점수: {total_points}/{max_points}")
        print(f"총 소요 시간: {total_time:.1f}초")
        print(f"평균 소요 시간: {total_time/total_questions:.1f}초/문제")
        
        # 난이도별 분석
        difficulty_stats = {}
        for result in self.session_results:
            diff = result.difficulty
            if diff not in difficulty_stats:
                difficulty_stats[diff] = {'correct': 0, 'total': 0}
            difficulty_stats[diff]['total'] += 1
            if result.is_correct:
                difficulty_stats[diff]['correct'] += 1
        
        print(f"\n📊 난이도별 성과:")
        for diff, stats in difficulty_stats.items():
            accuracy = stats['correct'] / stats['total'] * 100
            print(f"   {diff.capitalize()}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
        
        # 주제별 분석
        topic_stats = {}
        for result in self.session_results:
            topic = result.topic
            if topic not in topic_stats:
                topic_stats[topic] = {'correct': 0, 'total': 0}
            topic_stats[topic]['total'] += 1
            if result.is_correct:
                topic_stats[topic]['correct'] += 1
        
        if len(topic_stats) > 1:
            print(f"\n📚 주제별 성과:")
            for topic, stats in topic_stats.items():
                accuracy = stats['correct'] / stats['total'] * 100
                print(f"   {topic}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
        
        # 성과 평가
        accuracy = correct_answers / total_questions
        if accuracy >= 0.9:
            print("\n🏆 탁월한 성과입니다! 딥러닝 마스터!")
        elif accuracy >= 0.8:
            print("\n🥇 우수한 성과입니다! 딥러닝 전문가 수준!")
        elif accuracy >= 0.7:
            print("\n🥈 양호한 성과입니다! 조금 더 학습하면 전문가!")
        elif accuracy >= 0.6:
            print("\n🥉 기본기는 갖춰졌습니다! 더 연습해보세요!")
        else:
            print("\n📚 더 학습이 필요합니다! 이론 복습을 추천합니다!")
        
        # 취약 영역 분석
        wrong_results = [r for r in self.session_results if not r.is_correct]
        if wrong_results:
            print(f"\n🔍 복습 추천 영역:")
            wrong_topics = {}
            for result in wrong_results:
                topic = result.topic
                wrong_topics[topic] = wrong_topics.get(topic, 0) + 1
            
            for topic, count in sorted(wrong_topics.items(), key=lambda x: x[1], reverse=True):
                print(f"   - {topic} ({count}문제)")
        
        print("="*60)
    
    def retry_wrong_questions(self):
        """틀린 문제 재시도"""
        if not self.session_results:
            print("이전 퀴즈 결과가 없습니다.")
            return
        
        wrong_question_ids = [r.question_id for r in self.session_results if not r.is_correct]
        
        if not wrong_question_ids:
            print("틀린 문제가 없습니다! 완벽한 점수입니다!")
            return
        
        print(f"\n🔄 틀린 문제 재시도 ({len(wrong_question_ids)}문제)")
        print("="*50)
        
        retry_results = []
        for i, question_id in enumerate(wrong_question_ids, 1):
            question = self.questions[question_id]
            print(f"\n{'='*10} 재시도 {i}/{len(wrong_question_ids)} {'='*10}")
            result = self.ask_question(question)
            if result:
                retry_results.append(result)
            
            if i < len(wrong_question_ids):
                input("\n다음 문제로 넘어가려면 Enter를 누르세요...")
        
        # 재시도 결과 분석
        if retry_results:
            correct_retry = sum(1 for r in retry_results if r.is_correct)
            print(f"\n🎯 재시도 결과: {correct_retry}/{len(retry_results)} 정답")
            print(f"개선율: {correct_retry/len(retry_results)*100:.1f}%")
        
        return retry_results


class DeepLearningProgressTracker:
    """딥러닝 학습 진도 추적 클래스"""
    
    def __init__(self, data_path: str = "deep_learning_progress"):
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
                'best_accuracy': 0,
                'total_questions': 0,
                'correct_answers': 0,
                'last_attempt': None,
                'quiz_history': [],
                'weak_areas': {},
                'strong_areas': {}
            }
        
        # 현재 퀴즈 결과 계산
        total_questions = len(quiz_results)
        correct_answers = sum(1 for r in quiz_results if r.is_correct)
        total_points = sum(r.points_earned for r in quiz_results)
        accuracy = correct_answers / total_questions * 100 if total_questions > 0 else 0
        
        # 진도 업데이트
        topic_progress = self.user_progress[user_id][topic]
        topic_progress['attempts'] += 1
        topic_progress['best_score'] = max(topic_progress['best_score'], total_points)
        topic_progress['best_accuracy'] = max(topic_progress['best_accuracy'], accuracy)
        topic_progress['total_questions'] += total_questions
        topic_progress['correct_answers'] += correct_answers
        topic_progress['last_attempt'] = datetime.now().isoformat()
        
        # 취약/강점 영역 분석
        for result in quiz_results:
            difficulty = result.difficulty
            if not result.is_correct:
                topic_progress['weak_areas'][difficulty] = \
                    topic_progress['weak_areas'].get(difficulty, 0) + 1
            else:
                topic_progress['strong_areas'][difficulty] = \
                    topic_progress['strong_areas'].get(difficulty, 0) + 1
        
        # 퀴즈 기록 추가
        quiz_record = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'total_points': total_points,
            'total_questions': total_questions,
            'correct_answers': correct_answers,
            'difficulty_breakdown': self._analyze_difficulty_breakdown(quiz_results),
            'results': [asdict(r) for r in quiz_results]
        }
        topic_progress['quiz_history'].append(quiz_record)
        
        self.save_progress()
    
    def _analyze_difficulty_breakdown(self, quiz_results: List[QuizResult]) -> Dict:
        """난이도별 성과 분석"""
        breakdown = {}
        for result in quiz_results:
            diff = result.difficulty
            if diff not in breakdown:
                breakdown[diff] = {'correct': 0, 'total': 0}
            breakdown[diff]['total'] += 1
            if result.is_correct:
                breakdown[diff]['correct'] += 1
        
        return breakdown
    
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
        
        report = f"=== {user_id} 딥러닝 학습 진도 보고서 ===\n\n"
        
        total_attempts = 0
        total_accuracy = 0
        topic_count = 0
        
        for topic, progress in user_data.items():
            attempts = progress.get('attempts', 0)
            best_accuracy = progress.get('best_accuracy', 0)
            last_attempt = progress.get('last_attempt', 'N/A')
            weak_areas = progress.get('weak_areas', {})
            strong_areas = progress.get('strong_areas', {})
            
            report += f"🧠 주제: {topic}\n"
            report += f"   시도 횟수: {attempts}\n"
            report += f"   최고 정확도: {best_accuracy:.1f}%\n"
            report += f"   마지막 시도: {last_attempt}\n"
            
            if weak_areas:
                report += f"   취약 영역: {', '.join(weak_areas.keys())}\n"
            if strong_areas:
                report += f"   강점 영역: {', '.join(strong_areas.keys())}\n"
            report += "\n"
            
            total_attempts += attempts
            total_accuracy += best_accuracy
            topic_count += 1
        
        if topic_count > 0:
            avg_accuracy = total_accuracy / topic_count
            report += f"📊 전체 통계\n"
            report += f"   학습한 주제 수: {topic_count}\n"
            report += f"   총 시도 횟수: {total_attempts}\n"
            report += f"   평균 최고 정확도: {avg_accuracy:.1f}%\n"
            
            # 전체 레벨 평가
            if avg_accuracy >= 90:
                report += f"   현재 레벨: 🏆 딥러닝 마스터\n"
            elif avg_accuracy >= 80:
                report += f"   현재 레벨: 🥇 딥러닝 전문가\n"
            elif avg_accuracy >= 70:
                report += f"   현재 레벨: 🥈 딥러닝 중급자\n"
            elif avg_accuracy >= 60:
                report += f"   현재 레벨: 🥉 딥러닝 초급자\n"
            else:
                report += f"   현재 레벨: 📚 딥러닝 입문자\n"
        
        return report


# 퀴즈 실행 함수들
def run_deep_learning_quiz(topic: str = None, difficulty: str = None, num_questions: int = 10):
    """딥러닝 퀴즈 실행"""
    quiz_manager = DeepLearningQuizManager()
    
    # 기본 문제들이 있다면 추가 (각 튜토리얼에서 구현)
    print("딥러닝 퀴즈를 시작합니다!")
    
    results = quiz_manager.run_full_quiz(topic, difficulty, num_questions)
    
    # 진도 추적
    if results:
        tracker = DeepLearningProgressTracker()
        tracker.update_progress("default_user", topic or "전체", results)
    
    return results


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


if __name__ == "__main__":
    # 딥러닝 퀴즈 시스템 테스트
    print("딥러닝 튜토리얼 퀴즈 시스템")
    print("퀴즈 유틸리티가 성공적으로 로드되었습니다!")
    
    # 샘플 퀴즈 매니저 생성
    quiz_manager = DeepLearningQuizManager("딥러닝 기초 퀴즈")
    
    # 샘플 문제 추가
    quiz_manager.add_question_simple(
        question_id="dl_sample_001",
        question_type="multiple_choice",
        question="딥러닝에서 역전파(Backpropagation)의 주요 목적은?",
        options=[
            "순전파 계산 속도 향상",
            "가중치 업데이트를 위한 그래디언트 계산",
            "메모리 사용량 최적화",
            "데이터 전처리"
        ],
        correct_answer=2,
        explanation="역전파는 손실 함수에 대한 각 가중치의 그래디언트를 계산하여 가중치를 업데이트하는 데 사용됩니다.",
        difficulty="medium",
        topic="딥러닝 기초",
        related_theory_section="신경망 이론 - 역전파 알고리즘"
    )
    
    print(f"샘플 문제 {len(quiz_manager.questions)}개가 추가되었습니다.")
    print("각 튜토리얼에서 이 유틸리티를 사용하여 퀴즈를 구현할 수 있습니다.")