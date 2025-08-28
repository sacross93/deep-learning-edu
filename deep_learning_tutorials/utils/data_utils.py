"""
데이터 유틸리티 함수 모듈

이 모듈은 딥러닝 튜토리얼에서 사용되는 데이터셋 관련 공통 기능을 제공합니다.
- 데이터셋 다운로드 및 압축 해제
- 데이터셋 기본 정보 탐색
- 샘플 데이터 시각화

각 함수는 교육 목적에 맞게 상세한 설명과 함께 구현되었습니다.
"""

import os
import zipfile
import tarfile
import requests
from urllib.parse import urlparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Optional, Union, Any
from tqdm import tqdm


def download_and_extract(
    url: str, 
    download_path: str = "./data", 
    extract_path: Optional[str] = None,
    force_download: bool = False
) -> str:
    """
    URL에서 파일을 다운로드하고 압축을 해제합니다.
    
    Args:
        url: 다운로드할 파일의 URL
        download_path: 파일을 저장할 경로 (기본값: "./data")
        extract_path: 압축을 해제할 경로 (None이면 download_path와 동일)
        force_download: True이면 기존 파일이 있어도 다시 다운로드
    
    Returns:
        str: 압축 해제된 파일들이 있는 경로
    
    교육적 목적:
    - 실제 데이터 과학 프로젝트에서는 데이터를 외부에서 가져오는 경우가 많음
    - 네트워크 오류, 파일 손상 등의 예외 상황 처리 방법 학습
    - 진행률 표시를 통한 사용자 경험 개선
    """
    
    # 경로 설정 및 디렉토리 생성
    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)
    
    if extract_path is None:
        extract_path = download_path
    else:
        extract_path = Path(extract_path)
        extract_path.mkdir(parents=True, exist_ok=True)
    
    # URL에서 파일명 추출
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path)
    if not filename:
        filename = "downloaded_file"
    
    file_path = download_path / filename
    
    # 기존 파일 확인
    if file_path.exists() and not force_download:
        print(f"파일이 이미 존재합니다: {file_path}")
    else:
        print(f"다운로드 시작: {url}")
        print(f"저장 위치: {file_path}")
        
        try:
            # 스트리밍 다운로드로 메모리 효율성 확보
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # 파일 크기 확인 (진행률 표시용)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as file:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="다운로드") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"다운로드 완료: {file_path}")
            
        except requests.RequestException as e:
            print(f"다운로드 오류: {e}")
            return str(download_path)
    
    # 압축 파일 확인 및 해제
    if filename.endswith(('.zip', '.tar', '.tar.gz', '.tgz')):
        print(f"압축 해제 시작: {filename}")
        
        try:
            if filename.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            elif filename.endswith(('.tar', '.tar.gz', '.tgz')):
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_path)
            
            print(f"압축 해제 완료: {extract_path}")
            
        except Exception as e:
            print(f"압축 해제 오류: {e}")
    
    return str(extract_path)


def explore_dataset(
    dataset: Any, 
    dataset_name: str = "Dataset",
    show_samples: int = 5,
    show_distribution: bool = True
) -> None:
    """
    데이터셋의 기본 정보를 탐색하고 출력합니다.
    
    Args:
        dataset: 탐색할 데이터셋 (PyTorch Dataset 객체)
        dataset_name: 데이터셋 이름 (출력용)
        show_samples: 보여줄 샘플 수
        show_distribution: 클래스 분포를 보여줄지 여부
    
    교육적 목적:
    - 데이터 탐색은 모든 머신러닝 프로젝트의 첫 번째 단계
    - 데이터의 구조, 크기, 분포를 이해해야 적절한 모델과 전처리 방법 선택 가능
    - 클래스 불균형, 이상치 등의 문제를 미리 파악할 수 있음
    """
    
    print(f"\n{'='*50}")
    print(f"📊 {dataset_name} 데이터셋 탐색")
    print(f"{'='*50}")
    
    # 기본 정보
    print(f"📈 기본 정보:")
    print(f"  - 총 샘플 수: {len(dataset):,}개")
    
    # 첫 번째 샘플로 데이터 구조 파악
    try:
        sample = dataset[0]
        if isinstance(sample, tuple):
            data, target = sample
            print(f"  - 데이터 형태: {type(data).__name__}")
            print(f"  - 타겟 형태: {type(target).__name__}")
            
            # 텐서인 경우 shape 정보
            if hasattr(data, 'shape'):
                print(f"  - 데이터 크기: {data.shape}")
            if hasattr(target, 'shape'):
                print(f"  - 타겟 크기: {target.shape}")
                
        else:
            print(f"  - 샘플 형태: {type(sample).__name__}")
            if hasattr(sample, 'shape'):
                print(f"  - 샘플 크기: {sample.shape}")
                
    except Exception as e:
        print(f"  - 샘플 구조 분석 실패: {e}")
    
    # 클래스 분포 분석 (분류 문제인 경우)
    if show_distribution:
        try:
            print(f"\n📊 클래스 분포 분석:")
            targets = []
            
            # 전체 데이터셋을 순회하는 것은 비효율적이므로 샘플링
            sample_size = min(1000, len(dataset))
            indices = np.random.choice(len(dataset), sample_size, replace=False)
            
            for idx in indices:
                sample = dataset[idx]
                if isinstance(sample, tuple):
                    _, target = sample
                    targets.append(target)
            
            if targets:
                # numpy 배열로 변환
                if hasattr(targets[0], 'item'):  # 텐서인 경우
                    targets = [t.item() for t in targets]
                
                unique, counts = np.unique(targets, return_counts=True)
                
                print(f"  - 클래스 수: {len(unique)}개")
                for cls, count in zip(unique, counts):
                    percentage = (count / len(targets)) * 100
                    print(f"  - 클래스 {cls}: {count}개 ({percentage:.1f}%)")
                
                # 클래스 불균형 경고
                max_ratio = max(counts) / min(counts)
                if max_ratio > 3:
                    print(f"  ⚠️  클래스 불균형 감지 (최대/최소 비율: {max_ratio:.1f})")
                    print(f"     -> 가중치 손실함수나 샘플링 기법 고려 필요")
                    
        except Exception as e:
            print(f"  - 클래스 분포 분석 실패: {e}")
    
    # 샘플 데이터 미리보기
    if show_samples > 0:
        print(f"\n🔍 샘플 데이터 미리보기 ({show_samples}개):")
        for i in range(min(show_samples, len(dataset))):
            try:
                sample = dataset[i]
                if isinstance(sample, tuple):
                    data, target = sample
                    print(f"  샘플 {i+1}: 데이터 타입={type(data).__name__}, 타겟={target}")
                else:
                    print(f"  샘플 {i+1}: {type(sample).__name__}")
            except Exception as e:
                print(f"  샘플 {i+1}: 로드 실패 - {e}")
    
    print(f"{'='*50}\n")


def visualize_samples(
    dataset: Any,
    num_samples: int = 8,
    num_cols: int = 4,
    figsize: Tuple[int, int] = (12, 8),
    title: str = "데이터셋 샘플",
    class_names: Optional[list] = None
) -> None:
    """
    데이터셋의 샘플들을 시각화합니다.
    
    Args:
        dataset: 시각화할 데이터셋
        num_samples: 보여줄 샘플 수
        num_cols: 한 행에 보여줄 열 수
        figsize: 그림 크기
        title: 그림 제목
        class_names: 클래스 이름 리스트 (분류 문제인 경우)
    
    교육적 목적:
    - 시각화는 데이터를 이해하는 가장 직관적인 방법
    - 이미지 데이터의 품질, 전처리 효과, 라벨링 정확성 등을 확인 가능
    - 모델이 학습할 데이터의 특성을 미리 파악하여 적절한 아키텍처 선택에 도움
    """
    
    print(f"🖼️  {title} 시각화")
    
    # 그리드 설정
    num_rows = (num_samples + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # axes를 1차원 배열로 변환 (단일 행인 경우 처리)
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    elif num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # 샘플 시각화
    for i in range(num_samples):
        row = i // num_cols
        col = i % num_cols
        
        try:
            # 데이터셋에서 샘플 가져오기
            sample = dataset[i]
            
            if isinstance(sample, tuple):
                data, target = sample
            else:
                data = sample
                target = None
            
            # 텐서를 numpy로 변환
            if hasattr(data, 'numpy'):
                img = data.numpy()
            else:
                img = np.array(data)
            
            # 이미지 차원 조정
            if len(img.shape) == 3:
                # (C, H, W) -> (H, W, C) 변환
                if img.shape[0] in [1, 3]:  # 채널이 첫 번째 차원인 경우
                    img = np.transpose(img, (1, 2, 0))
                
                # 단일 채널인 경우 squeeze
                if img.shape[2] == 1:
                    img = img.squeeze(axis=2)
            
            # 이미지 정규화 (0-1 범위로)
            if img.max() > 1.0:
                img = img / 255.0
            
            # 음수 값이 있는 경우 정규화 (예: 표준화된 데이터)
            if img.min() < 0:
                img = (img - img.min()) / (img.max() - img.min())
            
            # 이미지 표시
            if len(img.shape) == 2:  # 흑백 이미지
                axes[row, col].imshow(img, cmap='gray')
            else:  # 컬러 이미지
                axes[row, col].imshow(img)
            
            # 제목 설정
            if target is not None:
                if class_names and hasattr(target, 'item'):
                    title_text = f"클래스: {class_names[target.item()]}"
                else:
                    title_text = f"타겟: {target}"
            else:
                title_text = f"샘플 {i+1}"
            
            axes[row, col].set_title(title_text, fontsize=10)
            axes[row, col].axis('off')
            
        except Exception as e:
            # 오류 발생 시 빈 플롯 표시
            axes[row, col].text(0.5, 0.5, f"로드 실패\n{str(e)[:30]}...", 
                               ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].axis('off')
    
    # 빈 서브플롯 숨기기
    for i in range(num_samples, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"✅ {num_samples}개 샘플 시각화 완료")


def get_data_statistics(dataloader: DataLoader) -> dict:
    """
    데이터로더에서 데이터의 통계 정보를 계산합니다.
    
    Args:
        dataloader: PyTorch DataLoader 객체
    
    Returns:
        dict: 평균, 표준편차 등의 통계 정보
    
    교육적 목적:
    - 데이터 정규화를 위해서는 전체 데이터의 평균과 표준편차를 알아야 함
    - 배치별로 통계를 계산하여 메모리 효율성 확보
    - 채널별 통계 정보로 컬러 이미지의 각 채널 특성 파악
    """
    
    print("📊 데이터 통계 정보 계산 중...")
    
    # 통계 정보 저장용 변수
    mean = 0.0
    std = 0.0
    total_samples = 0
    
    # 배치별로 통계 계산
    for batch_idx, (data, _) in enumerate(tqdm(dataloader, desc="통계 계산")):
        batch_samples = data.size(0)
        
        # 배치를 (batch_size, -1) 형태로 reshape
        data = data.view(batch_samples, -1)
        
        # 누적 평균 계산
        mean += data.mean(1).sum(0)
        std += data.std(1).sum(0)
        total_samples += batch_samples
    
    # 전체 평균과 표준편차 계산
    mean /= total_samples
    std /= total_samples
    
    statistics = {
        'mean': mean.item() if hasattr(mean, 'item') else mean,
        'std': std.item() if hasattr(std, 'item') else std,
        'total_samples': total_samples
    }
    
    print(f"✅ 통계 계산 완료:")
    print(f"  - 평균: {statistics['mean']:.4f}")
    print(f"  - 표준편차: {statistics['std']:.4f}")
    print(f"  - 총 샘플 수: {statistics['total_samples']:,}개")
    
    return statistics


def create_data_split(
    dataset: Any, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[Any, Any, Any]:
    """
    데이터셋을 훈련/검증/테스트 세트로 분할합니다.
    
    Args:
        dataset: 분할할 데이터셋
        train_ratio: 훈련 세트 비율
        val_ratio: 검증 세트 비율  
        test_ratio: 테스트 세트 비율
        random_seed: 랜덤 시드
    
    Returns:
        Tuple: (train_dataset, val_dataset, test_dataset)
    
    교육적 목적:
    - 적절한 데이터 분할은 모델의 일반화 성능 평가에 필수
    - 검증 세트는 하이퍼파라미터 튜닝에, 테스트 세트는 최종 성능 평가에 사용
    - 재현 가능한 결과를 위해 랜덤 시드 설정 중요
    """
    
    # 비율 검증
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "훈련/검증/테스트 비율의 합이 1이 되어야 합니다."
    
    # 랜덤 시드 설정
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 데이터셋 크기
    dataset_size = len(dataset)
    
    # 각 세트 크기 계산
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    print(f"📊 데이터 분할:")
    print(f"  - 전체: {dataset_size:,}개")
    print(f"  - 훈련: {train_size:,}개 ({train_ratio*100:.1f}%)")
    print(f"  - 검증: {val_size:,}개 ({val_ratio*100:.1f}%)")
    print(f"  - 테스트: {test_size:,}개 ({test_ratio*100:.1f}%)")
    
    # 데이터셋 분할
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    return train_dataset, val_dataset, test_dataset