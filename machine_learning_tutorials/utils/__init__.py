"""
머신러닝 튜토리얼 시리즈 공통 유틸리티 모듈

이 패키지는 머신러닝 튜토리얼 시리즈에서 사용되는 공통 기능들을 제공합니다:
- data_utils: 데이터 로딩, 전처리, 분할 기능
- evaluation_utils: 성능 평가 및 모델 비교 기능  
- visualization_utils: 시각화 및 결과 분석 기능
- quiz_utils: 퀴즈 생성 및 채점 기능
"""

from .data_utils import *
from .evaluation_utils import *
from .visualization_utils import *
from .quiz_utils import *

__version__ = "1.0.0"
__author__ = "머신러닝 튜토리얼 시리즈"