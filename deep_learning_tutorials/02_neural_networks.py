"""
딥러닝 강의 시리즈 2: 신경망 심화

이 튜토리얼에서는 Fashion-MNIST 데이터셋을 사용하여 
더 복잡한 신경망과 고급 기법들을 학습합니다.

학습 목표:
1. Fashion-MNIST 데이터셋의 특성 이해
2. 더 깊고 복잡한 신경망 구조 설계
3. 배치 정규화(Batch Normalization) 적용
4. 다양한 정규화 기법 비교
5. 학습률 스케줄링 구현
6. 하이퍼파라미터 튜닝 기법

데이터셋 선택 이유 - Fashion-MNIST:
- 28x28 픽셀의 흑백 의류 이미지 (10개 카테고리)
- 60,000개 훈련 + 10,000개 테스트 샘플
- MNIST보다 복잡하여 신경망 성능 개선 기법 학습에 적합
- 실제 이미지 분류 문제에 더 가까운 난이도
- 클래스 간 시각적 유사성으로 더 도전적인 분류 문제
- 정규화 기법의 효과를 명확히 확인할 수 있음
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import copy

# 우리가 만든 유틸리티 함수들 임포트
from utils.data_utils import explore_dataset, visualize_samples, create_data_split
from utils.visualization import (plot_training_curves, plot_confusion_matrix, 
                               plot_model_predictions, plot_learning_rate_schedule)
from utils.model_utils import count_parameters, save_checkpoint, evaluate_model, compare_models

print("🚀 딥러닝 강의 시리즈 2: 신경망 심화")
print("=" * 60)

# ============================================================================
# 1. 환경 설정 및 하이퍼파라미터
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  사용 장치: {device}")

# 하이퍼파라미터 설정
# Fashion-MNIST는 MNIST보다 복잡하므로 더 신중한 하이퍼파라미터 선택이 필요
BATCH_SIZE = 128       # 배치 크기를 늘려 더 안정적인 그래디언트 계산
LEARNING_RATE = 0.001  # 초기 학습률
EPOCHS = 20            # 더 많은 에포크로 충분한 학습 시간 확보
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.1  # 훈련 데이터의 10%를 검증용으로 사용

# 재현성을 위한 시드 설정
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

print(f"📊 하이퍼파라미터:")
print(f"   배치 크기: {BATCH_SIZE}")
print(f"   초기 학습률: {LEARNING_RATE}")
print(f"   에포크: {EPOCHS}")
print(f"   검증 데이터 비율: {VALIDATION_SPLIT}")

# ============================================================================
# 2. 데이터 전처리 및 로딩
# ============================================================================

print(f"\n📁 Fashion-MNIST 데이터셋 준비 중...")

# Fashion-MNIST를 위한 데이터 증강 기법
# 왜 데이터 증강이 필요한가?
# 1. 제한된 데이터로 더 많은 학습 샘플 생성
# 2. 모델의 일반화 성능 향상
# 3. 과적합 방지
# 4. 실제 환경의 다양한 변형에 대한 강건성 확보

# 훈련용 변환 (데이터 증강 포함)
train_transform = transforms.Compose([
    # RandomHorizontalFlip: 50% 확률로 좌우 반전
    # 의류 이미지에서 좌우 반전은 자연스러운 변형
    transforms.RandomHorizontalFlip(p=0.5),
    
    # RandomRotation: ±10도 범위에서 무작위 회전
    # 실제 착용 시 약간의 기울어짐을 시뮬레이션
    transforms.RandomRotation(degrees=10),
    
    # RandomAffine: 약간의 이동과 확대/축소
    # 촬영 각도나 거리의 변화를 시뮬레이션
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    
    transforms.ToTensor(),
    
    # Fashion-MNIST의 실제 통계값으로 정규화
    # 이 값들은 전체 데이터셋에서 계산된 평균과 표준편차
    transforms.Normalize((0.2860,), (0.3530,))
])

# 테스트용 변환 (데이터 증강 없음)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

# 데이터셋 로드
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=train_transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=test_transform
)

print(f"✅ 데이터셋 로드 완료")

# ============================================================================
# 3. 데이터 탐색 및 시각화
# ============================================================================

print(f"\n🔍 Fashion-MNIST 데이터셋 탐색")

# Fashion-MNIST 클래스 이름
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# 데이터셋 기본 정보 확인
explore_dataset(train_dataset, "Fashion-MNIST 훈련 데이터셋", show_samples=3)

# 샘플 데이터 시각화
print(f"\n🖼️  Fashion-MNIST 샘플 시각화")
visualize_samples(
    train_dataset, 
    num_samples=20, 
    num_cols=5,
    title="Fashion-MNIST 훈련 데이터 샘플",
    class_names=class_names
)

# Fashion-MNIST vs MNIST 복잡도 비교 설명
print(f"\n📈 Fashion-MNIST vs MNIST 복잡도 비교:")
print(f"   1. 클래스 내 변이성: 같은 의류 종류도 다양한 스타일 존재")
print(f"   2. 클래스 간 유사성: 셔츠와 티셔츠, 샌들과 부츠 등 구분이 어려움")
print(f"   3. 텍스처 복잡성: 의류의 패턴, 주름 등 복잡한 텍스처")
print(f"   4. 형태 다양성: 같은 카테고리 내에서도 다양한 형태")
print(f"   → 이러한 복잡성으로 인해 더 정교한 모델과 기법이 필요")

# ============================================================================
# 4. 데이터 분할 및 로더 생성
# ============================================================================

print(f"\n📦 데이터 분할 및 로더 생성")

# 훈련 데이터를 훈련/검증으로 분할
# 왜 검증 세트가 필요한가?
# 1. 하이퍼파라미터 튜닝을 위한 객관적 평가
# 2. 과적합 조기 감지
# 3. 모델 선택을 위한 기준
# 4. 테스트 세트의 오염 방지

train_size = int((1 - VALIDATION_SPLIT) * len(train_dataset))
val_size = len(train_dataset) - train_size

train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

# 데이터 로더 생성
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✅ 데이터 분할 완료:")
print(f"   훈련: {len(train_subset):,}개")
print(f"   검증: {len(val_subset):,}개") 
print(f"   테스트: {len(test_dataset):,}개")

# ============================================================================
# 5. 고급 신경망 모델 정의
# ============================================================================

print(f"\n🧠 고급 신경망 모델 정의")

class AdvancedNN(nn.Module):
    """
    고급 다층 퍼셉트론 with 배치 정규화
    
    개선사항:
    1. 더 깊은 네트워크 (4개 은닉층)
    2. 배치 정규화로 안정적인 학습
    3. 적응적 드롭아웃 비율
    4. 잔차 연결 (Residual Connection) 개념 도입
    
    왜 이런 구조를 선택했는가?
    - 깊은 네트워크: 더 복잡한 특징 학습 가능
    - 배치 정규화: 내부 공변량 이동 문제 해결, 빠른 학습
    - 점진적 차원 축소: 정보 손실 최소화하며 특징 압축
    """
    
    def __init__(self, dropout_rate=0.3):
        super(AdvancedNN, self).__init__()
        
        # 완전연결층들
        self.fc1 = nn.Linear(28 * 28, 512)  # 첫 번째 은닉층 (더 큰 용량)
        self.fc2 = nn.Linear(512, 256)      # 두 번째 은닉층
        self.fc3 = nn.Linear(256, 128)      # 세 번째 은닉층
        self.fc4 = nn.Linear(128, 64)       # 네 번째 은닉층
        self.fc5 = nn.Linear(64, 10)        # 출력층
        
        # 배치 정규화 레이어들
        # 왜 배치 정규화를 사용하는가?
        # 1. 내부 공변량 이동(Internal Covariate Shift) 문제 해결
        # 2. 더 높은 학습률 사용 가능
        # 3. 가중치 초기화에 덜 민감
        # 4. 정규화 효과로 과적합 방지
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(64)
        
        # 드롭아웃 레이어들 (층별로 다른 비율 적용)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate * 0.8)  # 점진적으로 감소
        self.dropout3 = nn.Dropout(dropout_rate * 0.6)
        self.dropout4 = nn.Dropout(dropout_rate * 0.4)
    
    def forward(self, x):
        # 입력 평탄화
        x = x.view(x.size(0), -1)
        
        # 첫 번째 블록: Linear → BatchNorm → ReLU → Dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # 두 번째 블록
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # 세 번째 블록
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # 네 번째 블록
        x = self.fc4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        
        # 출력층 (활성화 함수 없음)
        x = self.fc5(x)
        
        return x

class SimpleNN(nn.Module):
    """
    비교를 위한 간단한 신경망 (이전 튜토리얼과 유사)
    """
    
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 모델 인스턴스 생성
advanced_model = AdvancedNN(dropout_rate=0.3).to(device)
simple_model = SimpleNN().to(device)

print(f"✅ 모델 생성 완료")

# 모델 구조 비교
print(f"\n📋 고급 모델 구조:")
print(advanced_model)

print(f"\n📊 모델 복잡도 비교:")
advanced_params = count_parameters(advanced_model, detailed=False)
simple_params = count_parameters(simple_model, detailed=False)

print(f"   고급 모델: {advanced_params['total_params']:,}개 파라미터")
print(f"   간단 모델: {simple_params['total_params']:,}개 파라미터")
print(f"   복잡도 비율: {advanced_params['total_params'] / simple_params['total_params']:.1f}배")

# ============================================================================
# 6. 손실 함수와 옵티마이저 설정
# ============================================================================

print(f"\n⚙️  손실 함수와 옵티마이저 설정")

# 손실 함수
criterion = nn.CrossEntropyLoss()

# 옵티마이저 (Adam with weight decay)
# Weight Decay: L2 정규화와 유사한 효과로 과적합 방지
optimizer = optim.Adam(
    advanced_model.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=1e-4  # L2 정규화 계수
)

# 학습률 스케줄러
# 왜 학습률 스케줄링이 필요한가?
# 1. 초기에는 큰 학습률로 빠른 수렴
# 2. 후반에는 작은 학습률로 세밀한 조정
# 3. 지역 최솟값 탈출 도움
# 4. 더 나은 최종 성능 달성

scheduler = optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=7,    # 7 에포크마다
    gamma=0.5       # 학습률을 절반으로 감소
)

print(f"   손실 함수: {criterion.__class__.__name__}")
print(f"   옵티마이저: {optimizer.__class__.__name__}")
print(f"   초기 학습률: {LEARNING_RATE}")
print(f"   Weight Decay: {1e-4}")
print(f"   스케줄러: StepLR (7 에포크마다 0.5배)")

# ============================================================================
# 7. 향상된 훈련 함수
# ============================================================================

def train_epoch_advanced(model, train_loader, criterion, optimizer, device, epoch):
    """
    향상된 훈련 함수 (그래디언트 클리핑 포함)
    """
    model.train()
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"에포크 {epoch+1} 훈련")
    
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # 그래디언트 클리핑 (그래디언트 폭발 방지)
        # 왜 그래디언트 클리핑이 필요한가?
        # 1. 깊은 네트워크에서 그래디언트 폭발 방지
        # 2. 안정적인 훈련 보장
        # 3. 학습률을 더 크게 설정 가능
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 통계 업데이트
        running_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == targets).sum().item()
        total_samples += targets.size(0)
        
        # 진행률 바 업데이트
        current_accuracy = correct_predictions / total_samples
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_accuracy:.4f}',
            'LR': f'{current_lr:.6f}'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

def validate_epoch_advanced(model, val_loader, criterion, device):
    """
    향상된 검증 함수
    """
    model.eval()
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="검증")
        
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            
            current_accuracy = correct_predictions / total_samples
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_accuracy:.4f}'
            })
    
    avg_loss = running_loss / len(val_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

# ============================================================================
# 8. 모델 훈련 실행
# ============================================================================

print(f"\n🚀 고급 모델 훈련 시작")

# 훈련 기록
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
learning_rates = []

# 최고 성능 추적
best_val_accuracy = 0.0
best_model_state = None
patience = 5  # 조기 종료를 위한 인내심
patience_counter = 0

start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\n📅 에포크 {epoch+1}/{EPOCHS}")
    
    # 현재 학습률 기록
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    
    # 훈련
    train_loss, train_acc = train_epoch_advanced(
        advanced_model, train_loader, criterion, optimizer, device, epoch
    )
    
    # 검증
    val_loss, val_acc = validate_epoch_advanced(
        advanced_model, val_loader, criterion, device
    )
    
    # 학습률 스케줄러 업데이트
    scheduler.step()
    
    # 기록 저장
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # 결과 출력
    print(f"   훈련 - 손실: {train_loss:.4f}, 정확도: {train_acc:.4f}")
    print(f"   검증 - 손실: {val_loss:.4f}, 정확도: {val_acc:.4f}")
    print(f"   학습률: {current_lr:.6f}")
    
    # 최고 성능 모델 저장
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model_state = copy.deepcopy(advanced_model.state_dict())
        patience_counter = 0
        print(f"   🎯 새로운 최고 성능! 검증 정확도: {val_acc:.4f}")
        
        # 체크포인트 저장
        save_checkpoint(
            advanced_model, optimizer, epoch, val_loss, val_acc,
            save_path="./checkpoints/fashion_mnist_best_model.pth"
        )
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"   ⏰ 조기 종료: {patience} 에포크 동안 성능 개선 없음")
            break

training_time = time.time() - start_time
print(f"\n✅ 훈련 완료!")
print(f"   총 훈련 시간: {training_time:.2f}초")
print(f"   최고 검증 정확도: {best_val_accuracy:.4f}")

# ============================================================================
# 9. 훈련 결과 시각화
# ============================================================================

print(f"\n📈 훈련 결과 시각화")

# 훈련 곡선
plot_training_curves(
    train_losses=train_losses,
    val_losses=val_losses,
    train_accuracies=train_accuracies,
    val_accuracies=val_accuracies,
    title="Fashion-MNIST 분류 - 고급 모델 훈련 과정"
)

# 학습률 스케줄 시각화
plot_learning_rate_schedule(
    lr_schedule=learning_rates,
    title="학습률 스케줄링"
)

# ============================================================================
# 10. 모델 비교 실험
# ============================================================================

print(f"\n🏆 모델 성능 비교")

# 간단한 모델도 훈련 (비교용)
print(f"\n📊 간단한 모델 훈련 중...")

simple_optimizer = optim.Adam(simple_model.parameters(), lr=LEARNING_RATE)
simple_criterion = nn.CrossEntropyLoss()

# 간단한 모델 빠른 훈련 (5 에포크만)
for epoch in range(5):
    simple_model.train()
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        simple_optimizer.zero_grad()
        outputs = simple_model(data)
        loss = simple_criterion(outputs, targets)
        loss.backward()
        simple_optimizer.step()

# 최고 성능 모델 로드
if best_model_state is not None:
    advanced_model.load_state_dict(best_model_state)

# 모델 비교
models_to_compare = {
    "고급 모델 (BatchNorm + 깊은 구조)": advanced_model,
    "간단한 모델 (기본 구조)": simple_model
}

comparison_results = compare_models(
    models=models_to_compare,
    test_dataloader=test_loader,
    criterion=criterion,
    device=device
)

# ============================================================================
# 11. 최종 평가 및 시각화
# ============================================================================

print(f"\n🎯 최종 모델 평가")

# 상세한 성능 평가
final_results = evaluate_model(
    model=advanced_model,
    dataloader=test_loader,
    criterion=criterion,
    device=device,
    num_classes=10
)

# 예측 결과 시각화
print(f"\n🖼️  예측 결과 시각화")
plot_model_predictions(
    model=advanced_model,
    dataloader=test_loader,
    class_names=class_names,
    num_samples=16,
    device=device,
    title="Fashion-MNIST 분류 - 고급 모델 예측 결과"
)

# 혼동 행렬 생성
print(f"\n📊 혼동 행렬 생성 중...")

all_predictions = []
all_targets = []

advanced_model.eval()
with torch.no_grad():
    for data, targets in tqdm(test_loader, desc="예측 수집"):
        data, targets = data.to(device), targets.to(device)
        outputs = advanced_model(data)
        predictions = torch.argmax(outputs, dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

plot_confusion_matrix(
    y_true=np.array(all_targets),
    y_pred=np.array(all_predictions),
    class_names=class_names,
    title="Fashion-MNIST 분류 - 혼동 행렬"
)

# ============================================================================
# 12. 학습 내용 요약 및 개선 제안
# ============================================================================

print(f"\n🎓 학습 내용 요약")
print(f"=" * 60)

print(f"✅ 완료한 내용:")
print(f"   1. Fashion-MNIST 데이터셋 분석 및 전처리")
print(f"   2. 데이터 증강 기법 적용")
print(f"   3. 배치 정규화가 포함된 깊은 신경망 구현")
print(f"   4. 학습률 스케줄링 및 그래디언트 클리핑")
print(f"   5. 조기 종료 및 모델 비교")

print(f"\n📊 최종 성과:")
print(f"   - 고급 모델 최고 정확도: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
print(f"   - 총 파라미터 수: {advanced_params['total_params']:,}개")
print(f"   - 훈련 시간: {training_time:.2f}초")

print(f"\n💡 핵심 학습 포인트:")
print(f"   1. 배치 정규화의 효과: 안정적이고 빠른 학습")
print(f"   2. 데이터 증강의 중요성: 일반화 성능 향상")
print(f"   3. 학습률 스케줄링: 최적화 성능 개선")
print(f"   4. 정규화 기법들의 조합: 과적합 방지")
print(f"   5. 모델 복잡도와 성능의 트레이드오프")

print(f"\n🔍 Fashion-MNIST 특성 분석:")
print(f"   - MNIST 대비 낮은 정확도: 더 복잡한 시각적 패턴")
print(f"   - 클래스 간 혼동: 유사한 의류 카테고리 구분의 어려움")
print(f"   - 텍스처 중요성: 단순한 형태보다 복잡한 패턴 인식 필요")

print(f"\n🚀 다음 단계:")
print(f"   - 03_cnn_image_classification.py: 합성곱 신경망으로 이미지 특징 추출")
print(f"   - CIFAR-10으로 컬러 이미지 분류 도전")
print(f"   - 전이 학습 및 사전 훈련된 모델 활용")

print(f"\n🔧 추가 실험 아이디어:")
print(f"   1. 다른 정규화 기법 비교 (Layer Norm, Group Norm)")
print(f"   2. 다양한 활성화 함수 실험 (Swish, GELU)")
print(f"   3. 앙상블 기법으로 성능 향상")
print(f"   4. 하이퍼파라미터 자동 튜닝 (Optuna 등)")
print(f"   5. 모델 압축 기법 (Pruning, Quantization)")

print(f"\n" + "=" * 60)
print(f"🎉 신경망 심화 튜토리얼 완료!")
print(f"   다음 튜토리얼에서 CNN을 배워보세요!")
print(f"=" * 60)