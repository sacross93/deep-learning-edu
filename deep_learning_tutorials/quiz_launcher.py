#!/usr/bin/env python3
"""
딥러닝 튜토리얼 퀴즈 런처

모든 퀴즈에 대한 통합 접근점을 제공합니다.
요구사항 8.1, 8.2, 8.3, 8.4, 8.5를 충족합니다.
"""

import sys
import os
import importlib
from datetime import datetime

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from quiz_utils import DeepLearningProgressTracker


class QuizLauncher:
    def __init__(self):
        self.quizzes = [
            ('01_pytorch_basics_quiz', 'PyTorchBasicsQuiz', '1. PyTorch 기초', 'PyTorch 텐서, Autograd, 기본 신경망'),
            ('02_neural_networks_quiz', 'NeuralNetworksQuiz', '2. 신경망', '순전파, 역전파, 활성화 함수, 정규화'),
            ('03_cnn_quiz', 'CNNQuiz', '3. CNN', '합성곱, 풀링, 전이학습, 아키텍처'),
            ('04_rnn_quiz', 'RNNQuiz', '4. RNN', '시퀀스 처리, 그래디언트 문제, 자연어 처리'),
            ('05_lstm_quiz', 'LSTMQuiz', '5. LSTM', '게이트 메커니즘, 시계열 데이터, GRU 비교'),
            ('06_yolo_quiz', 'YOLOQuiz', '6. YOLO', '객체 탐지, 바운딩 박스, NMS, mAP'),
            ('07_gan_quiz', 'GANQuiz', '7. GAN', '적대적 학습, 생성자/판별자, 모드 붕괴'),
            ('08_transformer_quiz', 'TransformerQuiz', '8. Transformer', '어텐션 발전사, Self-Attention, BERT/GPT')
        ]
        self.progress_tracker = DeepLearningProgressTracker()
    
    def show_main_menu(self):
        """메인 메뉴 표시"""
        print("\n" + "="*70)
        print("🧠 딥러닝 튜토리얼 퀴즈 시스템")
        print("="*70)
        print("개별 퀴즈:")
        
        for _, _, display_name, description in self.quizzes:
            print(f"  {display_name}: {description}")
        
        print("\n특별 옵션:")
        print("  9. 전체 통합 퀴즈 (모든 주제에서 문제 선별)")
        print("  10. 맞춤형 퀴즈 (난이도/주제 선택)")
        print("  11. 학습 진도 확인")
        print("  12. 퀴즈 시스템 테스트")
        print("  0. 종료")
        print("="*70)
    
    def run_individual_quiz(self, quiz_index):
        """개별 퀴즈 실행"""
        if quiz_index < 0 or quiz_index >= len(self.quizzes):
            print("올바른 퀴즈 번호를 선택하세요.")
            return
        
        module_name, class_name, display_name, description = self.quizzes[quiz_index]
        
        try:
            print(f"\n🚀 {display_name} 시작!")
            print(f"주제: {description}")
            print("-" * 50)
            
            # 모듈 로드 및 퀴즈 실행
            module = importlib.import_module(module_name)
            quiz_class = getattr(module, class_name)
            quiz_instance = quiz_class()
            
            # 퀴즈 실행
            results = quiz_instance.run_quiz()
            
            # 진도 추적 업데이트
            if results:
                topic_name = display_name.split('. ')[1]  # "1. PyTorch 기초" -> "PyTorch 기초"
                self.progress_tracker.update_progress("default_user", topic_name, results)
                print(f"\n📊 학습 진도가 업데이트되었습니다!")
            
        except Exception as e:
            print(f"❌ 퀴즈 실행 중 오류 발생: {str(e)}")
            print("퀴즈 파일이 올바르게 설치되었는지 확인하세요.")
    
    def run_integrated_quiz(self):
        """전체 통합 퀴즈 실행"""
        print("\n🌟 전체 통합 퀴즈")
        print("모든 주제에서 문제를 선별하여 출제합니다.")
        print("-" * 50)
        
        try:
            num_questions = int(input("총 문제 수를 입력하세요 (10-50): "))
            if num_questions < 10 or num_questions > 50:
                print("10-50 사이의 숫자를 입력하세요.")
                return
        except ValueError:
            print("올바른 숫자를 입력하세요.")
            return
        
        # 모든 퀴즈에서 문제 수집
        all_questions = []
        
        for module_name, class_name, display_name, _ in self.quizzes:
            try:
                module = importlib.import_module(module_name)
                quiz_class = getattr(module, class_name)
                quiz_instance = quiz_class()
                
                questions = list(quiz_instance.quiz_manager.questions.values())
                all_questions.extend(questions)
                
            except Exception as e:
                print(f"⚠️ {display_name} 로드 실패: {str(e)}")
        
        if len(all_questions) < num_questions:
            print(f"사용 가능한 문제 수({len(all_questions)})가 요청한 문제 수보다 적습니다.")
            num_questions = len(all_questions)
        
        # 통합 퀴즈 매니저 생성
        from quiz_utils import DeepLearningQuizManager
        integrated_quiz = DeepLearningQuizManager("딥러닝 통합 퀴즈")
        
        # 문제 추가
        import random
        selected_questions = random.sample(all_questions, num_questions)
        for i, question in enumerate(selected_questions):
            question.id = f"integrated_{i+1:03d}"
            integrated_quiz.add_question(question)
        
        # 퀴즈 실행
        results = integrated_quiz.run_full_quiz(
            topic="딥러닝 통합",
            num_questions=num_questions
        )
        
        # 진도 추적
        if results:
            self.progress_tracker.update_progress("default_user", "딥러닝 통합", results)
    
    def run_custom_quiz(self):
        """맞춤형 퀴즈 실행"""
        print("\n🎯 맞춤형 퀴즈")
        print("-" * 50)
        
        # 주제 선택
        print("주제를 선택하세요:")
        for i, (_, _, display_name, _) in enumerate(self.quizzes):
            print(f"  {i+1}. {display_name}")
        print("  0. 모든 주제")
        
        try:
            topic_choice = int(input("선택: "))
            if topic_choice < 0 or topic_choice > len(self.quizzes):
                print("올바른 번호를 선택하세요.")
                return
        except ValueError:
            print("올바른 숫자를 입력하세요.")
            return
        
        # 난이도 선택
        print("\n난이도를 선택하세요:")
        print("  1. Easy (쉬움)")
        print("  2. Medium (보통)")
        print("  3. Hard (어려움)")
        print("  0. 모든 난이도")
        
        try:
            difficulty_choice = int(input("선택: "))
            difficulty_map = {0: None, 1: "easy", 2: "medium", 3: "hard"}
            if difficulty_choice not in difficulty_map:
                print("올바른 번호를 선택하세요.")
                return
            difficulty = difficulty_map[difficulty_choice]
        except ValueError:
            print("올바른 숫자를 입력하세요.")
            return
        
        # 문제 수 선택
        try:
            num_questions = int(input("문제 수를 입력하세요 (5-30): "))
            if num_questions < 5 or num_questions > 30:
                print("5-30 사이의 숫자를 입력하세요.")
                return
        except ValueError:
            print("올바른 숫자를 입력하세요.")
            return
        
        # 퀴즈 실행
        if topic_choice == 0:
            # 모든 주제에서 선별
            self.run_integrated_quiz_with_options(difficulty, num_questions)
        else:
            # 특정 주제
            module_name, class_name, display_name, _ = self.quizzes[topic_choice - 1]
            try:
                module = importlib.import_module(module_name)
                quiz_class = getattr(module, class_name)
                quiz_instance = quiz_class()
                
                results = quiz_instance.quiz_manager.run_full_quiz(
                    topic=display_name.split('. ')[1],
                    difficulty=difficulty,
                    num_questions=num_questions
                )
                
                if results:
                    topic_name = display_name.split('. ')[1]
                    self.progress_tracker.update_progress("default_user", topic_name, results)
                
            except Exception as e:
                print(f"❌ 퀴즈 실행 중 오류 발생: {str(e)}")
    
    def run_integrated_quiz_with_options(self, difficulty, num_questions):
        """옵션이 적용된 통합 퀴즈"""
        all_questions = []
        
        for module_name, class_name, display_name, _ in self.quizzes:
            try:
                module = importlib.import_module(module_name)
                quiz_class = getattr(module, class_name)
                quiz_instance = quiz_class()
                
                questions = list(quiz_instance.quiz_manager.questions.values())
                if difficulty:
                    questions = [q for q in questions if q.difficulty == difficulty]
                
                all_questions.extend(questions)
                
            except Exception as e:
                print(f"⚠️ {display_name} 로드 실패: {str(e)}")
        
        if len(all_questions) < num_questions:
            print(f"조건에 맞는 문제 수({len(all_questions)})가 부족합니다.")
            num_questions = len(all_questions)
        
        # 통합 퀴즈 실행
        from quiz_utils import DeepLearningQuizManager
        custom_quiz = DeepLearningQuizManager("맞춤형 딥러닝 퀴즈")
        
        import random
        selected_questions = random.sample(all_questions, num_questions)
        for i, question in enumerate(selected_questions):
            question.id = f"custom_{i+1:03d}"
            custom_quiz.add_question(question)
        
        results = custom_quiz.run_full_quiz(
            topic="맞춤형 퀴즈",
            difficulty=difficulty,
            num_questions=num_questions
        )
        
        if results:
            self.progress_tracker.update_progress("default_user", "맞춤형 퀴즈", results)
    
    def show_progress(self):
        """학습 진도 표시"""
        print("\n📊 학습 진도 확인")
        print("-" * 50)
        
        report = self.progress_tracker.generate_progress_report("default_user")
        print(report)
    
    def run_system_test(self):
        """시스템 테스트 실행"""
        print("\n🧪 퀴즈 시스템 테스트")
        print("-" * 50)
        
        try:
            from test_all_quizzes import QuizTester
            tester = QuizTester()
            tester.run_all_tests()
        except Exception as e:
            print(f"❌ 테스트 실행 중 오류 발생: {str(e)}")
    
    def run(self):
        """메인 실행 루프"""
        print("🎓 딥러닝 튜토리얼 퀴즈 시스템에 오신 것을 환영합니다!")
        
        while True:
            self.show_main_menu()
            
            try:
                choice = input("\n선택하세요: ").strip()
                
                if choice == '0':
                    print("퀴즈 시스템을 종료합니다. 수고하셨습니다! 🎉")
                    break
                elif choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
                    self.run_individual_quiz(int(choice) - 1)
                elif choice == '9':
                    self.run_integrated_quiz()
                elif choice == '10':
                    self.run_custom_quiz()
                elif choice == '11':
                    self.show_progress()
                elif choice == '12':
                    self.run_system_test()
                else:
                    print("올바른 선택지를 입력하세요.")
                
                input("\n계속하려면 Enter를 누르세요...")
                
            except KeyboardInterrupt:
                print("\n\n퀴즈 시스템을 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")
                input("계속하려면 Enter를 누르세요...")


def main():
    """메인 함수"""
    launcher = QuizLauncher()
    launcher.run()


if __name__ == "__main__":
    main()