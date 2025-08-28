# 딥러닝 튜토리얼 시리즈 유틸리티 패키지
# 
# 이 패키지는 모든 튜토리얼에서 공통으로 사용되는 유틸리티 함수들을 포함합니다.
# 
# 모듈 구성:
# - data_utils: 데이터 다운로드, 로딩, 탐색 관련 함수
# - visualization: 시각화 및 플롯 생성 함수  
# - model_utils: 모델 관련 유틸리티 (저장, 로드, 평가 등)

__version__ = "1.0.0"
__author__ = "딥러닝 강의 시리즈"

# 주요 함수들을 패키지 레벨에서 임포트 가능하도록 설정
from .data_utils import (
    download_and_extract,
    explore_dataset, 
    visualize_samples
)

from .visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_feature_maps
)

from .model_utils import (
    count_parameters,
    save_checkpoint,
    load_checkpoint
)

__all__ = [
    'download_and_extract',
    'explore_dataset', 
    'visualize_samples',
    'plot_training_curves',
    'plot_confusion_matrix', 
    'plot_feature_maps',
    'count_parameters',
    'save_checkpoint',
    'load_checkpoint'
]