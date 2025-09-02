"""
머신러닝 튜토리얼 시리즈 통합 테스트

모든 튜토리얼 파일들이 정상적으로 실행되는지 확인하는 테스트 스크립트입니다.
"""

import os
import sys
import importlib.util
import traceback
from pathlib import Path

def test_import(module_path):
    """모듈 임포트 테스트"""
    try:
        spec = importlib.util.spec_from_file_location("test_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return True, "성공"
    except Exception as e:
        return False, str(e)

def test_quiz_functionality(module_path):
    """퀴즈 모듈의 기본 기능 테스트"""
    try:
        spec = importlib.util.spec_from_file_location("quiz_module", module_path)
        module = importlib.util.module_from_spec(spec)
        
        # 표준 출력을 임시로 리다이렉트
        import io
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            spec.loader.exec_module(module)
            
            # 퀴즈 생성 함수가 있는지 확인
            if hasattr(module, 'create_' + Path(module_path).stem.replace('_quiz', '') + '_quiz'):
                quiz_func = getattr(module, 'create_' + Path(module_path).stem.replace('_quiz', '') + '_quiz')
                quiz_manager = quiz_func()
                
                # 퀴즈 매니저가 정상적으로 생성되었는지 확인
                if hasattr(quiz_manager, 'questions') and len(quiz_manager.questions) > 0:
                    return True, f"퀴즈 {len(quiz_manager.questions)}문제 생성 성공"
                else:
                    return False, "퀴즈 문제가 생성되지 않음"
            else:
                return False, "퀴즈 생성 함수를 찾을 수 없음"
                
    except Exception as e:
        return False, f"퀴즈 테스트 실패: {str(e)}"

def run_comprehensive_test():
    """포괄적인 테스트 실행"""
    
    print("머신러닝 튜토리얼 시리즈 통합 테스트")
    print("="*60)
    
    # 현재 디렉토리 확인
    current_dir = Path(__file__).parent
    print(f"테스트 디렉토리: {current_dir}")
    
    # 테스트할 파일 패턴들
    test_patterns = {
        'theory': '*_theory.md',
        'practice': '*_practice.py', 
        'quiz': '*_quiz.py'
    }
    
    results = {
        'theory': {'total': 0, 'success': 0, 'failed': []},
        'practice': {'total': 0, 'success': 0, 'failed': []},
        'quiz': {'total': 0, 'success': 0, 'failed': []}
    }
    
    # 1. 이론 문서 파일 존재 확인
    print("\\n1. 이론 문서 파일 확인")
    print("-" * 30)
    
    theory_files = list(current_dir.glob(test_patterns['theory']))
    results['theory']['total'] = len(theory_files)
    
    for theory_file in sorted(theory_files):
        if theory_file.exists() and theory_file.stat().st_size > 0:
            print(f"✓ {theory_file.name}")
            results['theory']['success'] += 1
        else:
            print(f"✗ {theory_file.name} (파일 없음 또는 빈 파일)")
            results['theory']['failed'].append(theory_file.name)
    
    # 2. 실습 코드 임포트 테스트
    print("\\n2. 실습 코드 임포트 테스트")
    print("-" * 30)
    
    practice_files = list(current_dir.glob(test_patterns['practice']))
    # 테스트 파일 자체는 제외
    practice_files = [f for f in practice_files if f.name != 'test_all_tutorials.py']
    results['practice']['total'] = len(practice_files)
    
    for practice_file in sorted(practice_files):
        success, message = test_import(practice_file)
        if success:
            print(f"✓ {practice_file.name}")
            results['practice']['success'] += 1
        else:
            print(f"✗ {practice_file.name}: {message}")
            results['practice']['failed'].append(practice_file.name)
    
    # 3. 퀴즈 모듈 테스트
    print("\\n3. 퀴즈 모듈 테스트")
    print("-" * 30)
    
    quiz_files = list(current_dir.glob(test_patterns['quiz']))
    results['quiz']['total'] = len(quiz_files)
    
    for quiz_file in sorted(quiz_files):
        success, message = test_quiz_functionality(quiz_file)
        if success:
            print(f"✓ {quiz_file.name}: {message}")
            results['quiz']['success'] += 1
        else:
            print(f"✗ {quiz_file.name}: {message}")
            results['quiz']['failed'].append(quiz_file.name)
    
    # 4. 유틸리티 모듈 테스트
    print("\\n4. 유틸리티 모듈 테스트")
    print("-" * 30)
    
    utils_dir = current_dir / 'utils'
    if utils_dir.exists():
        util_files = list(utils_dir.glob('*.py'))
        util_files = [f for f in util_files if f.name != '__init__.py']
        
        for util_file in sorted(util_files):
            success, message = test_import(util_file)
            if success:
                print(f"✓ utils/{util_file.name}")
            else:
                print(f"✗ utils/{util_file.name}: {message}")
    else:
        print("✗ utils 디렉토리가 존재하지 않습니다")
    
    # 5. 결과 요약
    print("\\n" + "="*60)
    print("테스트 결과 요약")
    print("="*60)
    
    total_files = sum(results[category]['total'] for category in results)
    total_success = sum(results[category]['success'] for category in results)
    
    print(f"전체 파일: {total_files}개")
    print(f"성공: {total_success}개")
    print(f"실패: {total_files - total_success}개")
    print(f"성공률: {total_success/total_files*100:.1f}%")
    
    print("\\n카테고리별 결과:")
    for category, result in results.items():
        success_rate = result['success'] / result['total'] * 100 if result['total'] > 0 else 0
        print(f"  {category.capitalize()}: {result['success']}/{result['total']} ({success_rate:.1f}%)")
        
        if result['failed']:
            print(f"    실패한 파일: {', '.join(result['failed'])}")
    
    # 6. 권장사항
    print("\\n" + "="*60)
    print("권장사항")
    print("="*60)
    
    if total_success == total_files:
        print("🎉 모든 테스트가 성공했습니다!")
        print("   튜토리얼 시리즈가 정상적으로 구성되었습니다.")
    else:
        print("⚠️  일부 테스트가 실패했습니다.")
        print("   실패한 파일들을 확인하고 수정해주세요.")
        
        # 실패한 파일들에 대한 구체적인 조치 방안
        all_failed = []
        for category in results:
            all_failed.extend(results[category]['failed'])
        
        if all_failed:
            print("\\n실패한 파일 조치 방안:")
            for failed_file in all_failed:
                if failed_file.endswith('.py'):
                    print(f"  - {failed_file}: 구문 오류나 의존성 문제 확인")
                elif failed_file.endswith('.md'):
                    print(f"  - {failed_file}: 파일 존재 여부 및 내용 확인")
    
    print("\\n추가 테스트 권장사항:")
    print("  1. 개별 튜토리얼을 실제로 실행해보세요")
    print("  2. 퀴즈를 풀어보며 내용을 검증하세요")
    print("  3. 실습 코드의 출력 결과를 확인하세요")
    print("  4. 필요한 패키지가 모두 설치되었는지 확인하세요")
    
    return results

def check_dependencies():
    """의존성 패키지 확인"""
    
    print("\\n의존성 패키지 확인")
    print("-" * 30)
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 
        'seaborn', 'scipy', 'networkx'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (설치 필요)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\\n누락된 패키지: {', '.join(missing_packages)}")
        print("다음 명령어로 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\\n모든 필수 패키지가 설치되어 있습니다.")
        return True

def main():
    """메인 테스트 함수"""
    
    # 의존성 확인
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\\n⚠️  필수 패키지를 먼저 설치해주세요.")
        return
    
    # 통합 테스트 실행
    results = run_comprehensive_test()
    
    # 최종 상태 반환
    total_files = sum(results[category]['total'] for category in results)
    total_success = sum(results[category]['success'] for category in results)
    
    if total_success == total_files:
        print("\\n✅ 모든 테스트 통과!")
        sys.exit(0)
    else:
        print("\\n❌ 일부 테스트 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()