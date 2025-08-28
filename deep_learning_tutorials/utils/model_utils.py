"""
모델 유틸리티 함수 모듈

이 모듈은 딥러닝 모델 관련 공통 기능을 제공합니다.
- 모델 파라미터 수 계산 및 분석
- 모델 체크포인트 저장/로드
- 모델 성능 평가 및 메트릭 계산
- 모델 구조 분석 및 시각화

각 함수는 교육 목적에 맞게 상세한 설명과 분석 결과를 포함합니다.
"""

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import time
from collections import OrderedDict
import json


def count_parameters(
    model: nn.Module, 
    trainable_only: bool = False,
    detailed: bool = True
) -> Dict[str, Any]:
    """
    모델의 파라미터 수를 계산하고 분석합니다.
    
    Args:
        model: 분석할 PyTorch 모델
        trainable_only: True이면 훈련 가능한 파라미터만 계산
        detailed: True이면 레이어별 상세 정보 출력
    
    Returns:
        Dict: 파라미터 수 및 분석 결과
    
    교육적 목적:
    - 모델의 복잡도를 파라미터 수로 정량화하여 이해
    - 메모리 사용량 및 연산량 추정에 도움
    - 모델 경량화나 압축 필요성 판단 기준 제공
    - 레이어별 파라미터 분포로 모델 구조 이해
    """
    
    print("🔍 모델 파라미터 분석 중...")
    
    total_params = 0
    trainable_params = 0
    layer_info = []
    
    # 레이어별 파라미터 계산
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
        
        if detailed:
            layer_info.append({
                'name': name,
                'shape': list(param.shape),
                'params': param_count,
                'trainable': param.requires_grad,
                'dtype': str(param.dtype)
            })
    
    # 결과 딕셔너리 생성
    result = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'non_trainable_params': total_params - trainable_params,
        'layer_info': layer_info if detailed else None
    }
    
    # 결과 출력
    print(f"\n📊 모델 파라미터 분석 결과:")
    print(f"  - 전체 파라미터: {total_params:,}개")
    print(f"  - 훈련 가능한 파라미터: {trainable_params:,}개")
    print(f"  - 고정된 파라미터: {total_params - trainable_params:,}개")
    
    # 메모리 사용량 추정 (float32 기준)
    memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
    print(f"  - 예상 메모리 사용량: {memory_mb:.2f} MB")
    
    # 모델 크기 분류
    if total_params < 1e6:
        size_category = "소형 모델 (< 1M)"
    elif total_params < 1e7:
        size_category = "중형 모델 (1M - 10M)"
    elif total_params < 1e8:
        size_category = "대형 모델 (10M - 100M)"
    else:
        size_category = "초대형 모델 (> 100M)"
    
    print(f"  - 모델 크기 분류: {size_category}")
    
    # 상세 정보 출력
    if detailed and layer_info:
        print(f"\n📋 레이어별 파라미터 분포:")
        print(f"{'레이어 이름':<30} {'형태':<20} {'파라미터 수':<15} {'훈련가능'}")
        print("-" * 80)
        
        for info in layer_info:
            shape_str = str(info['shape'])
            trainable_str = "✓" if info['trainable'] else "✗"
            print(f"{info['name']:<30} {shape_str:<20} {info['params']:<15,} {trainable_str}")
    
    # 성능 분석 및 조언
    print(f"\n💡 모델 분석 및 조언:")
    
    if total_params > 1e8:
        print(f"  ⚠️  매우 큰 모델입니다. 다음을 고려하세요:")
        print(f"     - GPU 메모리 부족 시 배치 크기 감소")
        print(f"     - 그래디언트 체크포인팅 사용")
        print(f"     - 모델 병렬화 고려")
    elif total_params < 1e5:
        print(f"  💡 작은 모델입니다. 성능 향상을 위해:")
        print(f"     - 모델 깊이나 너비 증가 고려")
        print(f"     - 더 복잡한 아키텍처 시도")
    
    # 훈련 가능한 파라미터 비율 분석
    trainable_ratio = trainable_params / total_params if total_params > 0 else 0
    if trainable_ratio < 0.5:
        print(f"  📌 고정된 파라미터가 많습니다 ({trainable_ratio:.1%} 훈련 가능)")
        print(f"     - 전이 학습이나 파인 튜닝 중인 것으로 보입니다")
    
    return result


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: Optional[float] = None,
    save_path: str = "checkpoint.pth",
    additional_info: Optional[Dict] = None
) -> None:
    """
    모델 체크포인트를 저장합니다.
    
    Args:
        model: 저장할 모델
        optimizer: 옵티마이저 상태
        epoch: 현재 에포크
        loss: 현재 손실값
        accuracy: 현재 정확도 (선택사항)
        save_path: 저장 경로
        additional_info: 추가 정보 딕셔너리
    
    교육적 목적:
    - 긴 훈련 과정에서 중간 결과 보존
    - 최적 모델 상태 저장으로 과적합 방지
    - 훈련 재개 및 실험 재현성 확보
    - 모델 배포를 위한 상태 저장
    """
    
    print(f"💾 체크포인트 저장 중: {save_path}")
    
    # 저장할 정보 구성
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': time.time(),
        'model_class': model.__class__.__name__
    }
    
    # 선택적 정보 추가
    if accuracy is not None:
        checkpoint['accuracy'] = accuracy
    
    if additional_info:
        checkpoint.update(additional_info)
    
    # 디렉토리 생성
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        torch.save(checkpoint, save_path)
        
        # 저장 정보 출력
        file_size = os.path.getsize(save_path) / (1024 * 1024)  # MB
        print(f"✅ 체크포인트 저장 완료:")
        print(f"  - 파일 크기: {file_size:.2f} MB")
        print(f"  - 에포크: {epoch}")
        print(f"  - 손실: {loss:.6f}")
        if accuracy is not None:
            print(f"  - 정확도: {accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ 체크포인트 저장 실패: {e}")


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu',
    strict: bool = True
) -> Dict[str, Any]:
    """
    저장된 체크포인트를 로드합니다.
    
    Args:
        model: 체크포인트를 로드할 모델
        checkpoint_path: 체크포인트 파일 경로
        optimizer: 옵티마이저 (상태 복원 시 필요)
        device: 로드할 장치
        strict: 엄격한 상태 딕셔너리 매칭 여부
    
    Returns:
        Dict: 로드된 체크포인트 정보
    
    교육적 목적:
    - 저장된 모델 상태 복원으로 훈련 재개
    - 사전 훈련된 모델 활용 (전이 학습)
    - 모델 배포 및 추론 환경 구성
    - 실험 결과 재현 및 비교
    """
    
    print(f"📂 체크포인트 로드 중: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다: {checkpoint_path}")
    
    try:
        # 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 모델 상태 로드
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            print(f"✅ 모델 상태 로드 완료")
        else:
            print(f"⚠️  모델 상태를 찾을 수 없습니다")
        
        # 옵티마이저 상태 로드
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ 옵티마이저 상태 로드 완료")
        
        # 로드 정보 출력
        print(f"📊 체크포인트 정보:")
        if 'epoch' in checkpoint:
            print(f"  - 에포크: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            print(f"  - 손실: {checkpoint['loss']:.6f}")
        if 'accuracy' in checkpoint:
            print(f"  - 정확도: {checkpoint['accuracy']:.4f}")
        if 'timestamp' in checkpoint:
            save_time = time.ctime(checkpoint['timestamp'])
            print(f"  - 저장 시간: {save_time}")
        
        return checkpoint
        
    except Exception as e:
        print(f"❌ 체크포인트 로드 실패: {e}")
        raise


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = 'cpu',
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    모델의 성능을 평가합니다.
    
    Args:
        model: 평가할 모델
        dataloader: 평가 데이터 로더
        criterion: 손실 함수
        device: 연산 장치
        num_classes: 클래스 수 (분류 문제인 경우)
    
    Returns:
        Dict: 평가 메트릭 결과
    
    교육적 목적:
    - 모델의 일반화 성능 정량적 측정
    - 다양한 메트릭으로 모델 성능 다각도 분석
    - 클래스별 성능 분석으로 모델 편향성 확인
    - 추론 속도 측정으로 실용성 평가
    """
    
    print("🎯 모델 성능 평가 중...")
    
    model.eval()
    model = model.to(device)
    
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # 클래스별 통계 (분류 문제인 경우)
    if num_classes:
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
    
    # 추론 시간 측정
    inference_times = []
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            
            # 추론 시간 측정
            start_time = time.time()
            outputs = model(data)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # 손실 계산
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            # 정확도 계산 (분류 문제인 경우)
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                predictions = torch.argmax(outputs, dim=1)
                correct_predictions += (predictions == targets).sum().item()
                
                # 클래스별 정확도 계산
                if num_classes:
                    for i in range(len(targets)):
                        label = targets[i].item()
                        class_total[label] += 1
                        if predictions[i] == targets[i]:
                            class_correct[label] += 1
            
            total_samples += len(targets)
    
    # 평균 메트릭 계산
    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
    avg_inference_time = np.mean(inference_times)
    
    # 결과 딕셔너리 구성
    results = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'total_samples': total_samples,
        'avg_inference_time': avg_inference_time,
        'throughput': total_samples / sum(inference_times)  # 초당 처리 샘플 수
    }
    
    # 결과 출력
    print(f"\n📊 평가 결과:")
    print(f"  - 평균 손실: {avg_loss:.6f}")
    print(f"  - 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  - 총 샘플 수: {total_samples:,}개")
    print(f"  - 평균 추론 시간: {avg_inference_time*1000:.2f} ms/배치")
    print(f"  - 처리량: {results['throughput']:.1f} 샘플/초")
    
    # 클래스별 성능 분석
    if num_classes and class_total.sum() > 0:
        print(f"\n📋 클래스별 정확도:")
        class_accuracies = []
        
        for i in range(num_classes):
            if class_total[i] > 0:
                class_acc = class_correct[i] / class_total[i]
                class_accuracies.append(class_acc)
                print(f"  - 클래스 {i}: {class_acc:.4f} ({class_acc*100:.2f}%) - {int(class_total[i])}개 샘플")
            else:
                class_accuracies.append(0.0)
        
        # 클래스 간 성능 편차 분석
        if len(class_accuracies) > 1:
            acc_std = np.std(class_accuracies)
            acc_min = min(class_accuracies)
            acc_max = max(class_accuracies)
            
            results['class_accuracies'] = class_accuracies
            results['accuracy_std'] = acc_std
            results['accuracy_range'] = acc_max - acc_min
            
            print(f"\n📈 클래스 성능 분석:")
            print(f"  - 성능 표준편차: {acc_std:.4f}")
            print(f"  - 성능 범위: {acc_min:.4f} ~ {acc_max:.4f}")
            
            if acc_std > 0.1:
                print(f"  ⚠️  클래스 간 성능 편차가 큽니다")
                print(f"     -> 데이터 불균형이나 모델 편향성 확인 필요")
    
    # 성능 분석 및 개선 제안
    print(f"\n💡 성능 분석 및 제안:")
    
    if accuracy < 0.6:
        print(f"  📈 성능 개선 필요:")
        print(f"     - 모델 아키텍처 복잡도 증가")
        print(f"     - 하이퍼파라미터 튜닝")
        print(f"     - 데이터 전처리 개선")
        print(f"     - 훈련 에포크 증가")
    elif accuracy > 0.95:
        print(f"  🎯 우수한 성능:")
        print(f"     - 과적합 여부 확인")
        print(f"     - 더 어려운 데이터셋으로 일반화 테스트")
    
    if avg_inference_time > 0.1:  # 100ms 이상
        print(f"  ⚡ 추론 속도 개선 고려:")
        print(f"     - 모델 경량화 (pruning, quantization)")
        print(f"     - 배치 크기 최적화")
        print(f"     - GPU 활용 최적화")
    
    return results


def get_model_summary(
    model: nn.Module,
    input_size: Tuple[int, ...],
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    모델의 구조와 정보를 요약합니다.
    
    Args:
        model: 분석할 모델
        input_size: 입력 텐서 크기 (배치 차원 제외)
        device: 연산 장치
    
    Returns:
        Dict: 모델 요약 정보
    
    교육적 목적:
    - 모델 구조의 전체적인 이해
    - 각 레이어의 출력 크기 및 파라미터 수 파악
    - 메모리 사용량 및 연산량 추정
    - 모델 설계의 적절성 검증
    """
    
    print(f"📋 모델 구조 분석 중...")
    
    model = model.to(device)
    model.eval()
    
    # 더미 입력 생성
    dummy_input = torch.randn(1, *input_size).to(device)
    
    # 레이어별 정보 수집
    layer_info = []
    hooks = []
    
    def register_hook(module, name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                output_shape = list(output.shape)
                num_params = sum(p.numel() for p in module.parameters())
            elif isinstance(output, (list, tuple)):
                output_shape = [list(o.shape) if isinstance(o, torch.Tensor) else str(type(o)) for o in output]
                num_params = sum(p.numel() for p in module.parameters())
            else:
                output_shape = str(type(output))
                num_params = sum(p.numel() for p in module.parameters())
            
            layer_info.append({
                'name': name,
                'type': module.__class__.__name__,
                'output_shape': output_shape,
                'num_params': num_params
            })
        
        return hook
    
    # 모든 모듈에 훅 등록
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 리프 모듈만
            hook = module.register_forward_hook(register_hook(module, name))
            hooks.append(hook)
    
    # 순전파 실행
    try:
        with torch.no_grad():
            _ = model(dummy_input)
    finally:
        # 훅 제거
        for hook in hooks:
            hook.remove()
    
    # 전체 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 메모리 사용량 추정
    param_memory = total_params * 4 / (1024**2)  # MB, float32 기준
    
    # 결과 구성
    summary = {
        'model_name': model.__class__.__name__,
        'input_size': input_size,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_memory_mb': param_memory,
        'layer_info': layer_info
    }
    
    # 결과 출력
    print(f"\n📊 모델 요약:")
    print(f"  - 모델명: {summary['model_name']}")
    print(f"  - 입력 크기: {input_size}")
    print(f"  - 전체 파라미터: {total_params:,}개")
    print(f"  - 훈련 가능한 파라미터: {trainable_params:,}개")
    print(f"  - 파라미터 메모리: {param_memory:.2f} MB")
    
    print(f"\n📋 레이어별 구조:")
    print(f"{'레이어명':<25} {'타입':<15} {'출력 크기':<25} {'파라미터'}")
    print("-" * 85)
    
    for info in layer_info:
        if info['name']:  # 빈 이름 제외
            output_str = str(info['output_shape'])[:23]
            print(f"{info['name']:<25} {info['type']:<15} {output_str:<25} {info['num_params']:,}")
    
    return summary


def compare_models(
    models: Dict[str, nn.Module],
    test_dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = 'cpu'
) -> Dict[str, Dict[str, float]]:
    """
    여러 모델의 성능을 비교합니다.
    
    Args:
        models: 비교할 모델들 (이름: 모델)
        test_dataloader: 테스트 데이터 로더
        criterion: 손실 함수
        device: 연산 장치
    
    Returns:
        Dict: 모델별 성능 비교 결과
    
    교육적 목적:
    - 다양한 모델 아키텍처의 성능 비교
    - 모델 선택을 위한 객관적 기준 제공
    - 성능-복잡도 트레이드오프 분석
    - 앙상블 모델 구성을 위한 기초 자료
    """
    
    print(f"🏆 모델 성능 비교 중... ({len(models)}개 모델)")
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n📊 {model_name} 평가 중...")
        
        # 모델 평가
        model_results = evaluate_model(model, test_dataloader, criterion, device)
        
        # 모델 복잡도 정보 추가
        param_info = count_parameters(model, detailed=False)
        model_results.update({
            'total_params': param_info['total_params'],
            'trainable_params': param_info['trainable_params']
        })
        
        results[model_name] = model_results
    
    # 비교 결과 출력
    print(f"\n🏆 모델 성능 비교 결과:")
    print(f"{'모델명':<20} {'정확도':<10} {'손실':<12} {'파라미터 수':<15} {'추론시간(ms)'}")
    print("-" * 70)
    
    # 성능순으로 정렬
    sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for model_name, result in sorted_models:
        accuracy = result['accuracy']
        loss = result['loss']
        params = result['total_params']
        inference_time = result['avg_inference_time'] * 1000  # ms
        
        print(f"{model_name:<20} {accuracy:<10.4f} {loss:<12.6f} {params:<15,} {inference_time:<10.2f}")
    
    # 최고 성능 모델 식별
    best_model = sorted_models[0]
    print(f"\n🥇 최고 성능 모델: {best_model[0]} (정확도: {best_model[1]['accuracy']:.4f})")
    
    # 효율성 분석 (정확도 대비 파라미터 수)
    print(f"\n📈 효율성 분석 (정확도/파라미터 비율):")
    efficiency_scores = []
    
    for model_name, result in results.items():
        efficiency = result['accuracy'] / (result['total_params'] / 1e6)  # 정확도 per million params
        efficiency_scores.append((model_name, efficiency))
    
    efficiency_scores.sort(key=lambda x: x[1], reverse=True)
    
    for model_name, efficiency in efficiency_scores:
        print(f"  - {model_name}: {efficiency:.2f} (정확도/M파라미터)")
    
    print(f"\n🎯 가장 효율적인 모델: {efficiency_scores[0][0]}")
    
    return results