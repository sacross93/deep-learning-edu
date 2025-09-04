#!/usr/bin/env python3
"""
딥러닝 튜토리얼 퀴즈 통합 테스트

모든 퀴즈의 정상 작동을 확인하고 통계를 제공합니다.
요구사항 8.4, 8.5를 충족합니다.
"""

import sys
import os
import importlib
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from quiz_utils import DeepLearningQuizManager


class QuizTester:
    def __init__(self):
        self.quiz_modules = [
            ('01_pytorch_basics_quiz', 'PyTorchBasicsQuiz', 'PyTorch 기초'),
            ('02_neural_networks_quiz', 'NeuralNetworksQuiz', '신경망'),
            ('03_cnn_quiz', 'CNNQuiz', 'CNN'),
            ('04_rnn_quiz', 'RNNQuiz', 'RNN'),
            ('05_lstm_quiz', 'LSTMQuiz', 'LSTM'),
            ('06_yolo_quiz', 'YOLOQuiz', 'YOLO'),
            ('07_gan_quiz', 'GANQuiz', 'GAN'),
            ('08_transformer_quiz', 'TransformerQuiz', 'Transformer')
        ]
        self.test_results = {}
    
    def test_quiz_loading(self):
        """퀴즈 모듈 로딩 테스트"""
        print("🔍 퀴즈 모듈 로딩 테스트 시작...")
        print("=" * 60)
        
        for module_name, class_name, display_name in self.quiz_modules:
            try:
                # 모듈 임포트
                module = importlib.import_module(module_name)
                
                # 클래스 인스턴스 생성
                quiz_class = getattr(module, class_name)
                quiz_instance = quiz_class()
                
                # 문제 수 확인
                question_count = len(quiz_instance.quiz_manager.questions)
                
                print(f"✅ {display_name}: {question_count}개 문제 로드 성공")
                
                self.test_results[display_name] = {
                    'status': 'success',
                    'question_count': question_count,
                    'error': None
                }
                
            except Exception as e:
                print(f"❌ {display_name}: 로드 실패 - {str(e)}")
                self.test_results[display_name] = {
                    'status': 'failed',
                    'question_count': 0,
                    'error': str(e)
                }
        
        print("=" * 60)
    
    def test_question_structure(self):
        """문제 구조 검증 테스트"""
        print("\n🔍 문제 구조 검증 테스트 시작...")
        print("=" * 60)
        
        for module_name, class_name, display_name in self.quiz_modules:
            if self.test_results[display_name]['status'] != 'success':
                continue
                
            try:
                module = importlib.import_module(module_name)
                quiz_class = getattr(module, class_name)
                quiz_instance = quiz_class()
                
                questions = quiz_instance.quiz_manager.questions
                issues = []
                
                for q_id, question in questions.items():
                    # 필수 필드 확인
                    if not question.question:
                        issues.append(f"문제 {q_id}: 질문 내용 없음")
                    if not question.explanation:
                        issues.append(f"문제 {q_id}: 해설 없음")
                    if question.correct_answer is None:
                        issues.append(f"문제 {q_id}: 정답 없음")
                    
                    # 객관식 문제 검증
                    if question.question_type == 'multiple_choice':
                        if not question.options or len(question.options) < 2:
                            issues.append(f"문제 {q_id}: 선택지 부족")
                        if not isinstance(question.correct_answer, int) or \
                           question.correct_answer < 1 or \
                           question.correct_answer > len(question.options or []):
                            issues.append(f"문제 {q_id}: 정답 번호 오류")
                
                if issues:
                    print(f"⚠️  {display_name}: {len(issues)}개 문제 발견")
                    for issue in issues[:3]:  # 처음 3개만 표시
                        print(f"   - {issue}")
                    if len(issues) > 3:
                        print(f"   ... 외 {len(issues)-3}개")
                else:
                    print(f"✅ {display_name}: 문제 구조 검증 통과")
                
                self.test_results[display_name]['structure_issues'] = len(issues)
                
            except Exception as e:
                print(f"❌ {display_name}: 구조 검증 실패 - {str(e)}")
                self.test_results[display_name]['structure_issues'] = -1
        
        print("=" * 60)
    
    def test_difficulty_distribution(self):
        """난이도 분포 분석"""
        print("\n📊 난이도 분포 분석...")
        print("=" * 60)
        
        total_stats = {'easy': 0, 'medium': 0, 'hard': 0}
        
        for module_name, class_name, display_name in self.quiz_modules:
            if self.test_results[display_name]['status'] != 'success':
                continue
                
            try:
                module = importlib.import_module(module_name)
                quiz_class = getattr(module, class_name)
                quiz_instance = quiz_class()
                
                questions = quiz_instance.quiz_manager.questions
                difficulty_count = {'easy': 0, 'medium': 0, 'hard': 0}
                
                for question in questions.values():
                    difficulty = question.difficulty.lower()
                    if difficulty in difficulty_count:
                        difficulty_count[difficulty] += 1
                        total_stats[difficulty] += 1
                
                total = sum(difficulty_count.values())
                if total > 0:
                    easy_pct = difficulty_count['easy'] / total * 100
                    medium_pct = difficulty_count['medium'] / total * 100
                    hard_pct = difficulty_count['hard'] / total * 100
                    
                    print(f"{display_name:12}: Easy {difficulty_count['easy']:2d}({easy_pct:4.1f}%) "
                          f"Medium {difficulty_count['medium']:2d}({medium_pct:4.1f}%) "
                          f"Hard {difficulty_count['hard']:2d}({hard_pct:4.1f}%)")
                
                self.test_results[display_name]['difficulty_dist'] = difficulty_count
                
            except Exception as e:
                print(f"❌ {display_name}: 난이도 분석 실패 - {str(e)}")
        
        # 전체 통계
        total_questions = sum(total_stats.values())
        if total_questions > 0:
            print("-" * 60)
            print(f"{'전체 통계':12}: Easy {total_stats['easy']:2d}({total_stats['easy']/total_questions*100:4.1f}%) "
                  f"Medium {total_stats['medium']:2d}({total_stats['medium']/total_questions*100:4.1f}%) "
                  f"Hard {total_stats['hard']:2d}({total_stats['hard']/total_questions*100:4.1f}%)")
        
        print("=" * 60)
    
    def test_quiz_functionality(self):
        """퀴즈 기능 테스트 (샘플 실행)"""
        print("\n🧪 퀴즈 기능 테스트...")
        print("=" * 60)
        
        for module_name, class_name, display_name in self.quiz_modules:
            if self.test_results[display_name]['status'] != 'success':
                continue
                
            try:
                module = importlib.import_module(module_name)
                quiz_class = getattr(module, class_name)
                quiz_instance = quiz_class()
                
                # 퀴즈 세션 생성 테스트
                session = quiz_instance.quiz_manager.create_quiz_session(num_questions=1)
                
                if len(session) > 0:
                    # 첫 번째 문제로 기능 테스트
                    question = session[0]
                    
                    # 문제 출력 형식 확인
                    if question.question and question.explanation:
                        print(f"✅ {display_name}: 기능 테스트 통과")
                        self.test_results[display_name]['functionality'] = 'pass'
                    else:
                        print(f"⚠️  {display_name}: 문제 내용 부족")
                        self.test_results[display_name]['functionality'] = 'warning'
                else:
                    print(f"❌ {display_name}: 문제 생성 실패")
                    self.test_results[display_name]['functionality'] = 'fail'
                
            except Exception as e:
                print(f"❌ {display_name}: 기능 테스트 실패 - {str(e)}")
                self.test_results[display_name]['functionality'] = 'error'
        
        print("=" * 60)
    
    def generate_report(self):
        """최종 보고서 생성"""
        print("\n📋 최종 테스트 보고서")
        print("=" * 60)
        
        total_quizzes = len(self.quiz_modules)
        successful_quizzes = sum(1 for result in self.test_results.values() 
                               if result['status'] == 'success')
        total_questions = sum(result['question_count'] for result in self.test_results.values())
        
        print(f"테스트 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 퀴즈 수: {total_quizzes}")
        print(f"성공한 퀴즈: {successful_quizzes}")
        print(f"총 문제 수: {total_questions}")
        print(f"성공률: {successful_quizzes/total_quizzes*100:.1f}%")
        
        print("\n📊 퀴즈별 상세 정보:")
        print("-" * 60)
        
        for display_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            structure_info = ""
            if 'structure_issues' in result:
                if result['structure_issues'] == 0:
                    structure_info = " 구조✅"
                elif result['structure_issues'] > 0:
                    structure_info = f" 구조⚠️({result['structure_issues']})"
                else:
                    structure_info = " 구조❌"
            
            func_info = ""
            if 'functionality' in result:
                func_map = {'pass': '✅', 'warning': '⚠️', 'fail': '❌', 'error': '❌'}
                func_info = f" 기능{func_map.get(result['functionality'], '❓')}"
            
            print(f"{status_icon} {display_name:12}: {result['question_count']:2d}문제{structure_info}{func_info}")
        
        print("=" * 60)
        
        # 권장사항
        print("\n💡 권장사항:")
        failed_quizzes = [name for name, result in self.test_results.items() 
                         if result['status'] != 'success']
        
        if failed_quizzes:
            print(f"- 실패한 퀴즈 수정 필요: {', '.join(failed_quizzes)}")
        
        structure_issues = sum(result.get('structure_issues', 0) 
                             for result in self.test_results.values() 
                             if result.get('structure_issues', 0) > 0)
        
        if structure_issues > 0:
            print(f"- 총 {structure_issues}개 구조 문제 수정 권장")
        
        if successful_quizzes == total_quizzes and structure_issues == 0:
            print("- 모든 퀴즈가 정상 작동합니다! 🎉")
        
        print("=" * 60)
    
    def run_all_tests(self):
        """모든 테스트 실행"""
        print("🚀 딥러닝 튜토리얼 퀴즈 통합 테스트 시작")
        print("=" * 60)
        
        self.test_quiz_loading()
        self.test_question_structure()
        self.test_difficulty_distribution()
        self.test_quiz_functionality()
        self.generate_report()


def main():
    """메인 실행 함수"""
    print("딥러닝 튜토리얼 퀴즈 통합 테스트 시스템")
    print("=" * 60)
    print("1. 전체 테스트 실행")
    print("2. 로딩 테스트만")
    print("3. 구조 검증만")
    print("4. 난이도 분석만")
    print("5. 기능 테스트만")
    
    tester = QuizTester()
    
    choice = input("\n선택하세요 (1-5): ").strip()
    
    if choice == '1':
        tester.run_all_tests()
    elif choice == '2':
        tester.test_quiz_loading()
    elif choice == '3':
        tester.test_quiz_loading()
        tester.test_question_structure()
    elif choice == '4':
        tester.test_quiz_loading()
        tester.test_difficulty_distribution()
    elif choice == '5':
        tester.test_quiz_loading()
        tester.test_quiz_functionality()
    else:
        print("올바른 선택지를 입력하세요.")
        return
    
    print("\n테스트 완료!")


if __name__ == "__main__":
    main()