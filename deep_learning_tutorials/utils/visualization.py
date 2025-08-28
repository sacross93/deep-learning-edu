"""
시각화 유틸리티 함수 모듈

이 모듈은 딥러닝 모델의 훈련 과정과 결과를 시각화하는 함수들을 제공합니다.
- 훈련 곡선 (손실, 정확도) 시각화
- 혼동 행렬 시각화
- 특성 맵 (feature map) 시각화
- 모델 예측 결과 시각화

각 함수는 교육 목적에 맞게 상세한 설명과 해석 가이드를 포함합니다.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Optional, Tuple, Any
import warnings

# 한글 폰트 설정 (matplotlib에서 한글 표시를 위해)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

# 시각화 스타일 설정
sns.set_style("whitegrid")
plt.style.use('default')


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_accuracies: Optional[List[float]] = None,
    val_accuracies: Optional[List[float]] = None,
    title: str = "훈련 과정",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    모델 훈련 과정의 손실과 정확도 곡선을 시각화합니다.
    
    Args:
        train_losses: 에포크별 훈련 손실 리스트
        val_losses: 에포크별 검증 손실 리스트 (선택사항)
        train_accuracies: 에포크별 훈련 정확도 리스트 (선택사항)
        val_accuracies: 에포크별 검증 정확도 리스트 (선택사항)
        title: 그래프 제목
        save_path: 그래프 저장 경로 (선택사항)
        figsize: 그래프 크기
    
    교육적 목적:
    - 훈련 곡선은 모델의 학습 상태를 파악하는 가장 중요한 도구
    - 과적합(overfitting) 여부를 시각적으로 확인 가능
    - 학습률, 정규화 등 하이퍼파라미터 조정의 근거 제공
    - 조기 종료(early stopping) 시점 결정에 도움
    """
    
    print(f"📈 {title} 시각화 중...")
    
    # 서브플롯 개수 결정
    num_plots = 1 if train_accuracies is None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=figsize)
    
    if num_plots == 1:
        axes = [axes]
    
    # 에포크 번호 생성
    epochs = range(1, len(train_losses) + 1)
    
    # 1. 손실 곡선 그리기
    axes[0].plot(epochs, train_losses, 'b-', label='훈련 손실', linewidth=2, marker='o', markersize=4)
    
    if val_losses is not None:
        axes[0].plot(epochs, val_losses, 'r-', label='검증 손실', linewidth=2, marker='s', markersize=4)
        
        # 과적합 감지 및 경고
        if len(val_losses) > 5:  # 충분한 에포크가 있을 때만 검사
            # 최근 5개 에포크에서 검증 손실이 계속 증가하는지 확인
            recent_val_trend = np.polyfit(range(5), val_losses[-5:], 1)[0]
            if recent_val_trend > 0.01:  # 기울기가 양수이고 충분히 큰 경우
                axes[0].axvline(x=len(epochs)-5, color='orange', linestyle='--', alpha=0.7)
                axes[0].text(len(epochs)-5, max(val_losses), '⚠️ 과적합 의심', 
                           rotation=90, verticalalignment='bottom', fontsize=10)
    
    axes[0].set_xlabel('에포크')
    axes[0].set_ylabel('손실')
    axes[0].set_title('손실 곡선')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 최소 검증 손실 지점 표시
    if val_losses is not None:
        min_val_loss_epoch = np.argmin(val_losses) + 1
        min_val_loss = min(val_losses)
        axes[0].plot(min_val_loss_epoch, min_val_loss, 'g*', markersize=15, 
                    label=f'최소 검증 손실 (에포크 {min_val_loss_epoch})')
        axes[0].legend()
    
    # 2. 정확도 곡선 그리기 (있는 경우)
    if train_accuracies is not None:
        axes[1].plot(epochs, train_accuracies, 'b-', label='훈련 정확도', 
                    linewidth=2, marker='o', markersize=4)
        
        if val_accuracies is not None:
            axes[1].plot(epochs, val_accuracies, 'r-', label='검증 정확도', 
                        linewidth=2, marker='s', markersize=4)
            
            # 최고 검증 정확도 지점 표시
            max_val_acc_epoch = np.argmax(val_accuracies) + 1
            max_val_acc = max(val_accuracies)
            axes[1].plot(max_val_acc_epoch, max_val_acc, 'g*', markersize=15,
                        label=f'최고 검증 정확도 (에포크 {max_val_acc_epoch})')
        
        axes[1].set_xlabel('에포크')
        axes[1].set_ylabel('정확도')
        axes[1].set_title('정확도 곡선')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 그래프 저장
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 그래프 저장됨: {save_path}")
    
    plt.show()
    
    # 훈련 결과 분석 및 조언 출력
    print(f"\n📊 훈련 결과 분석:")
    print(f"  - 최종 훈련 손실: {train_losses[-1]:.4f}")
    
    if val_losses is not None:
        print(f"  - 최종 검증 손실: {val_losses[-1]:.4f}")
        print(f"  - 최소 검증 손실: {min(val_losses):.4f} (에포크 {np.argmin(val_losses)+1})")
        
        # 과적합 분석
        final_gap = val_losses[-1] - train_losses[-1]
        if final_gap > 0.5:
            print(f"  ⚠️  심각한 과적합 감지 (손실 차이: {final_gap:.4f})")
            print(f"     -> 정규화 강화, 드롭아웃 증가, 데이터 증강 고려")
        elif final_gap > 0.2:
            print(f"  ⚠️  경미한 과적합 감지 (손실 차이: {final_gap:.4f})")
            print(f"     -> 조기 종료 또는 정규화 적용 고려")
    
    if val_accuracies is not None:
        print(f"  - 최고 검증 정확도: {max(val_accuracies):.4f} (에포크 {np.argmax(val_accuracies)+1})")
        
        # 성능 개선 여지 분석
        if max(val_accuracies) < 0.7:
            print(f"  💡 성능 개선 제안:")
            print(f"     -> 모델 복잡도 증가, 학습률 조정, 데이터 전처리 개선")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "혼동 행렬",
    normalize: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    분류 모델의 혼동 행렬을 시각화합니다.
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        class_names: 클래스 이름 리스트
        title: 그래프 제목
        normalize: 정규화 여부 (비율로 표시)
        save_path: 저장 경로
        figsize: 그래프 크기
    
    교육적 목적:
    - 혼동 행렬은 분류 모델의 성능을 세부적으로 분석하는 핵심 도구
    - 어떤 클래스가 잘못 분류되는지, 어떤 클래스끼리 혼동되는지 파악 가능
    - 클래스별 정밀도, 재현율 계산의 기초 자료
    - 데이터 불균형이나 모델의 편향성 발견에 도움
    """
    
    print(f"🎯 {title} 생성 중...")
    
    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        # 행별로 정규화 (실제 클래스별 비율)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = '.2%'
        cbar_label = '비율'
    else:
        cm_display = cm
        fmt = 'd'
        cbar_label = '개수'
    
    # 그래프 생성
    plt.figure(figsize=figsize)
    
    # 히트맵 그리기
    sns.heatmap(cm_display, 
                annot=True, 
                fmt=fmt,
                cmap='Blues',
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)),
                cbar_kws={'label': cbar_label})
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('예측 라벨', fontsize=12)
    plt.ylabel('실제 라벨', fontsize=12)
    
    # 대각선 강조 (정확한 예측)
    for i in range(len(cm)):
        plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 혼동 행렬 저장됨: {save_path}")
    
    plt.show()
    
    # 분류 성능 리포트 출력
    print(f"\n📊 분류 성능 리포트:")
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names,
                                 output_dict=True)
    
    # 전체 정확도
    accuracy = report['accuracy']
    print(f"  - 전체 정확도: {accuracy:.4f}")
    
    # 클래스별 성능
    print(f"\n  클래스별 성능:")
    for class_name in (class_names if class_names else [str(i) for i in range(len(cm))]):
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            support = report[class_name]['support']
            
            print(f"    {class_name}:")
            print(f"      정밀도: {precision:.4f}, 재현율: {recall:.4f}, F1: {f1:.4f} (샘플: {support}개)")
    
    # 성능 분석 및 개선 제안
    print(f"\n💡 성능 분석:")
    
    # 가장 혼동되는 클래스 쌍 찾기
    max_confusion = 0
    confused_pair = None
    
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > max_confusion:
                max_confusion = cm[i][j]
                confused_pair = (i, j)
    
    if confused_pair and max_confusion > 0:
        class_i = class_names[confused_pair[0]] if class_names else f"클래스 {confused_pair[0]}"
        class_j = class_names[confused_pair[1]] if class_names else f"클래스 {confused_pair[1]}"
        print(f"  - 가장 혼동되는 클래스: {class_i} → {class_j} ({max_confusion}개)")
        print(f"    -> 이 두 클래스의 특징을 더 잘 구분할 수 있는 특성 추가 고려")
    
    # 성능이 낮은 클래스 식별
    worst_f1 = 1.0
    worst_class = None
    
    for class_name in (class_names if class_names else [str(i) for i in range(len(cm))]):
        if class_name in report and isinstance(report[class_name], dict):
            f1 = report[class_name]['f1-score']
            if f1 < worst_f1:
                worst_f1 = f1
                worst_class = class_name
    
    if worst_class and worst_f1 < 0.8:
        print(f"  - 성능이 낮은 클래스: {worst_class} (F1: {worst_f1:.4f})")
        print(f"    -> 해당 클래스의 데이터 증강이나 특성 엔지니어링 고려")


def plot_feature_maps(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    layer_name: str,
    num_maps: int = 16,
    figsize: Tuple[int, int] = (15, 10),
    title: str = "특성 맵 시각화"
) -> None:
    """
    CNN 모델의 특성 맵(feature map)을 시각화합니다.
    
    Args:
        model: 시각화할 CNN 모델
        input_tensor: 입력 텐서 (1개 샘플)
        layer_name: 시각화할 레이어 이름
        num_maps: 표시할 특성 맵 개수
        figsize: 그래프 크기
        title: 그래프 제목
    
    교육적 목적:
    - 특성 맵 시각화는 CNN이 무엇을 학습하는지 이해하는 핵심 방법
    - 각 레이어가 어떤 특징(엣지, 텍스처, 패턴)을 감지하는지 확인 가능
    - 모델의 해석 가능성(interpretability) 향상
    - 레이어별 특성 추출 과정의 시각적 이해
    """
    
    print(f"🔍 {title} - {layer_name} 레이어")
    
    # 모델을 평가 모드로 설정
    model.eval()
    
    # 특성 맵을 저장할 변수
    feature_maps = None
    
    # 훅(hook) 함수 정의
    def hook_fn(module, input, output):
        nonlocal feature_maps
        feature_maps = output.detach()
    
    # 지정된 레이어에 훅 등록
    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break
    
    if target_layer is None:
        print(f"❌ 레이어 '{layer_name}'을 찾을 수 없습니다.")
        print("사용 가능한 레이어:")
        for name, _ in model.named_modules():
            if name:  # 빈 이름 제외
                print(f"  - {name}")
        return
    
    # 훅 등록
    hook = target_layer.register_forward_hook(hook_fn)
    
    try:
        # 순전파 실행
        with torch.no_grad():
            _ = model(input_tensor.unsqueeze(0))  # 배치 차원 추가
        
        if feature_maps is None:
            print(f"❌ 특성 맵을 추출할 수 없습니다.")
            return
        
        # 첫 번째 샘플의 특성 맵만 사용
        feature_maps = feature_maps[0]  # (C, H, W)
        
        # 표시할 특성 맵 개수 조정
        num_channels = feature_maps.shape[0]
        num_maps = min(num_maps, num_channels)
        
        # 그리드 설정
        cols = 4
        rows = (num_maps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        print(f"  - 특성 맵 크기: {feature_maps.shape}")
        print(f"  - 표시할 맵 개수: {num_maps}/{num_channels}")
        
        # 각 특성 맵 시각화
        for i in range(num_maps):
            row = i // cols
            col = i % cols
            
            # 특성 맵 추출 및 정규화
            fmap = feature_maps[i].cpu().numpy()
            
            # 정규화 (0-1 범위)
            fmap_min, fmap_max = fmap.min(), fmap.max()
            if fmap_max > fmap_min:
                fmap = (fmap - fmap_min) / (fmap_max - fmap_min)
            
            # 시각화
            im = axes[row, col].imshow(fmap, cmap='viridis')
            axes[row, col].set_title(f'맵 {i+1}', fontsize=10)
            axes[row, col].axis('off')
            
            # 컬러바 추가 (첫 번째 행에만)
            if row == 0:
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # 빈 서브플롯 숨기기
        for i in range(num_maps, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle(f"{title} - {layer_name}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 특성 맵 분석
        print(f"\n📊 특성 맵 분석:")
        
        # 활성화 통계
        activation_mean = feature_maps.mean().item()
        activation_std = feature_maps.std().item()
        activation_max = feature_maps.max().item()
        
        print(f"  - 평균 활성화: {activation_mean:.4f}")
        print(f"  - 활성화 표준편차: {activation_std:.4f}")
        print(f"  - 최대 활성화: {activation_max:.4f}")
        
        # 활성화된 뉴런 비율
        active_ratio = (feature_maps > 0).float().mean().item()
        print(f"  - 활성화된 뉴런 비율: {active_ratio:.2%}")
        
        if active_ratio < 0.1:
            print(f"    ⚠️ 활성화 비율이 낮습니다. ReLU 죽음 현상 의심")
        elif active_ratio > 0.9:
            print(f"    ⚠️ 대부분의 뉴런이 활성화됨. 포화 상태 의심")
        
    finally:
        # 훅 제거
        hook.remove()


def plot_learning_rate_schedule(
    lr_schedule: List[float],
    title: str = "학습률 스케줄",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    학습률 스케줄을 시각화합니다.
    
    Args:
        lr_schedule: 에포크별 학습률 리스트
        title: 그래프 제목
        save_path: 저장 경로
        figsize: 그래프 크기
    
    교육적 목적:
    - 학습률 스케줄링은 모델 성능에 큰 영향을 미치는 중요한 기법
    - 적절한 학습률 감소 패턴을 시각적으로 확인
    - 다양한 스케줄러(StepLR, CosineAnnealingLR 등)의 효과 비교
    """
    
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(lr_schedule) + 1)
    plt.plot(epochs, lr_schedule, 'b-', linewidth=2, marker='o', markersize=4)
    
    plt.xlabel('에포크')
    plt.ylabel('학습률')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 로그 스케일로 표시 (학습률 변화를 더 잘 보기 위해)
    
    # 주요 변화점 표시
    if len(lr_schedule) > 1:
        lr_changes = []
        for i in range(1, len(lr_schedule)):
            if abs(lr_schedule[i] - lr_schedule[i-1]) / lr_schedule[i-1] > 0.1:  # 10% 이상 변화
                lr_changes.append(i)
        
        for change_point in lr_changes:
            plt.axvline(x=change_point+1, color='red', linestyle='--', alpha=0.7)
            plt.text(change_point+1, lr_schedule[change_point], f'변화점', 
                    rotation=90, verticalalignment='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 학습률 스케줄 저장됨: {save_path}")
    
    plt.show()
    
    print(f"📊 학습률 분석:")
    print(f"  - 초기 학습률: {lr_schedule[0]:.6f}")
    print(f"  - 최종 학습률: {lr_schedule[-1]:.6f}")
    print(f"  - 감소 비율: {lr_schedule[-1]/lr_schedule[0]:.4f}")


def plot_model_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    class_names: Optional[List[str]] = None,
    num_samples: int = 8,
    device: str = 'cpu',
    title: str = "모델 예측 결과"
) -> None:
    """
    모델의 예측 결과를 시각화합니다.
    
    Args:
        model: 예측할 모델
        dataloader: 데이터 로더
        class_names: 클래스 이름 리스트
        num_samples: 표시할 샘플 수
        device: 연산 장치
        title: 그래프 제목
    
    교육적 목적:
    - 모델이 실제로 어떤 예측을 하는지 시각적으로 확인
    - 올바른 예측과 잘못된 예측의 패턴 분석
    - 모델의 신뢰도와 실제 성능 간의 관계 파악
    """
    
    model.eval()
    model = model.to(device)
    
    # 샘플 데이터 수집
    samples = []
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(dataloader):
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            
            # 소프트맥스 적용하여 확률 계산
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            for i in range(len(data)):
                if len(samples) >= num_samples:
                    break
                
                samples.append({
                    'image': data[i].cpu(),
                    'true_label': targets[i].cpu().item(),
                    'pred_label': predictions[i].cpu().item(),
                    'confidence': probabilities[i].max().cpu().item(),
                    'correct': predictions[i].cpu().item() == targets[i].cpu().item()
                })
            
            if len(samples) >= num_samples:
                break
    
    # 시각화
    cols = 4
    rows = (num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        row = i // cols
        col = i % cols
        
        # 이미지 전처리
        img = sample['image']
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:
            img = img.permute(1, 2, 0)
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        
        # 정규화
        if img.max() <= 1.0 and img.min() >= -1.0:
            img = (img - img.min()) / (img.max() - img.min())
        
        # 이미지 표시
        if len(img.shape) == 2:
            axes[row, col].imshow(img, cmap='gray')
        else:
            axes[row, col].imshow(img)
        
        # 제목 설정
        true_name = class_names[sample['true_label']] if class_names else f"클래스 {sample['true_label']}"
        pred_name = class_names[sample['pred_label']] if class_names else f"클래스 {sample['pred_label']}"
        
        color = 'green' if sample['correct'] else 'red'
        title_text = f"실제: {true_name}\n예측: {pred_name}\n신뢰도: {sample['confidence']:.2f}"
        
        axes[row, col].set_title(title_text, color=color, fontsize=10)
        axes[row, col].axis('off')
    
    # 빈 서브플롯 숨기기
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 예측 결과 분석
    correct_count = sum(1 for s in samples if s['correct'])
    accuracy = correct_count / len(samples)
    avg_confidence = np.mean([s['confidence'] for s in samples])
    
    print(f"📊 예측 결과 분석 (샘플 {len(samples)}개):")
    print(f"  - 정확도: {accuracy:.2%}")
    print(f"  - 평균 신뢰도: {avg_confidence:.4f}")
    
    # 신뢰도별 정확도 분석
    high_conf_samples = [s for s in samples if s['confidence'] > 0.9]
    if high_conf_samples:
        high_conf_accuracy = sum(1 for s in high_conf_samples if s['correct']) / len(high_conf_samples)
        print(f"  - 고신뢰도(>0.9) 샘플 정확도: {high_conf_accuracy:.2%} ({len(high_conf_samples)}개)")
    
    low_conf_samples = [s for s in samples if s['confidence'] < 0.6]
    if low_conf_samples:
        low_conf_accuracy = sum(1 for s in low_conf_samples if s['correct']) / len(low_conf_samples)
        print(f"  - 저신뢰도(<0.6) 샘플 정확도: {low_conf_accuracy:.2%} ({len(low_conf_samples)}개)")