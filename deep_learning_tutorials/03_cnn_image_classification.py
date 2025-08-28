"""
딥러닝 강의 시리즈 3: CNN 이미지 분류

이 튜토리얼에서는 CIFAR-10 데이터셋을 사용하여 
합성곱 신경망(CNN)의 핵심 개념과 구현을 학습합니다.

학습 목표:
1. 합성곱 신경망(CNN)의 구조와 원리 이해
2. 합성곱층, 풀링층, 완전연결층의 역할
3. 특성 맵(Feature Map) 시각화 및 해석
4. 데이터 증강의 고급 기법
5. 전이 학습(Transfer Learning) 개념
6. 모델 앙상블 기법

데이터셋 선택 이유 - CIFAR-10:
- 32x32 픽셀의 컬러 이미지 (RGB 3채널)
- 10개 클래스: 비행기, 자동차, 새, 고양이, 사슴, 개, 개구리, 말, 배, 트럭
- 50,000개 훈련 + 10,000개 테스트 샘플
- CNN의 효과를 명확히 보여주는 최적의 데이터셋
- 컬러 이미지로 채널 차원 처리 학습 가능
- 적당한 복잡도로 CNN 개념 학습에 이상적
- 객체의 형태, 색상, 텍스처 등 다양한 특징 포함
- 실제 이미지 분류 문제의 축소판으로 실용적 학습 가능

왜 이제 CNN을 사용하는가?
1. 이미지의 공간적 구조 보존: 완전연결층은 공간 정보 손실
2. 지역적 특징 추출: 엣지, 코너, 텍스처 등 국소 패턴 감지
3. 평행이동 불변성: 객체 위치에 관계없이 인식 가능
4. 파라미터 공유: 같은 필터로 전체 이미지 스캔하여 효율성 확보
5. 계층적 특징 학습: 저수준→고수준 특징으로 점진적 추상화
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
from utils.data_utils import explore_dataset, visualize_samples
from utils.visualization import (plot_training_curves, plot_confusion_matrix, 
                               plot_model_predictions, plot_feature_maps)
from utils.model_utils import count_parameters, save_checkpoint, evaluate_model, compare_models

print("🚀 딥러닝 강의 시리즈 3: CNN 이미지 분류")
print("=" * 60)

# ============================================================================
# 1. 환경 설정 및 하이퍼파라미터
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  사용 장치: {device}")

# CNN을 위한 하이퍼파라미터
# 컬러 이미지와 더 복잡한 패턴으로 인해 조정된 설정
BATCH_SIZE = 128       # CNN은 더 많은 메모리를 사용하므로 적절한 배치 크기
LEARNING_RATE = 0.001  # Adam 옵티마이저에 적합한 학습률
EPOCHS = 50            # CNN은 수렴이 느려 더 많은 에포크 필요
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.1

# 재현성을 위한 시드 설정
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

print(f"📊 하이퍼파라미터:")
print(f"   배치 크기: {BATCH_SIZE}")
print(f"   학습률: {LEARNING_RATE}")
print(f"   에포크: {EPOCHS}")

# ============================================================================
# 2. 데이터 전처리 및 증강
# ============================================================================

print(f"\n📁 CIFAR-10 데이터셋 준비 중...")

# CIFAR-10을 위한 고급 데이터 증강
# 왜 이런 증강 기법들이 필요한가?

# 훈련용 변환 (강력한 데이터 증강)
train_transform = transforms.Compose([
    # RandomCrop: 32x32에서 4픽셀 패딩 후 무작위 크롭
    # 이유: 객체의 위치 변화에 대한 강건성 확보
    transforms.RandomCrop(32, padding=4),
    
    # RandomHorizontalFlip: 50% 확률로 좌우 반전
    # 이유: 대부분의 객체는 좌우 대칭적 특성을 가짐
    transforms.RandomHorizontalFlip(p=0.5),
    
    # ColorJitter: 밝기, 대비, 채도, 색조 무작위 변경
    # 이유: 다양한 조명 조건과 카메라 설정에 대한 강건성
    transforms.ColorJitter(
        brightness=0.2,    # 밝기 ±20% 변화
        contrast=0.2,      # 대비 ±20% 변화
        saturation=0.2,    # 채도 ±20% 변화
        hue=0.1           # 색조 ±10% 변화
    ),
    
    # RandomRotation: ±15도 회전
    # 이유: 촬영 각도 변화에 대한 강건성
    transforms.RandomRotation(degrees=15),
    
    transforms.ToTensor(),
    
    # CIFAR-10의 채널별 정규화
    # RGB 각 채널의 평균과 표준편차로 정규화
    # 이 값들은 전체 CIFAR-10 데이터셋에서 계산된 통계값
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],  # RGB 채널별 평균
        std=[0.2023, 0.1994, 0.2010]    # RGB 채널별 표준편차
    )
])

# 테스트용 변환 (증강 없음, 정규화만)
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

# 데이터셋 로드
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    download=True, 
    transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=False, 
    download=True, 
    transform=test_transform
)

print(f"✅ CIFAR-10 데이터셋 로드 완료")

# ============================================================================
# 3. 데이터 탐색 및 시각화
# ============================================================================

print(f"\n🔍 CIFAR-10 데이터셋 탐색")

# CIFAR-10 클래스 이름
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# 데이터셋 기본 정보 확인
explore_dataset(train_dataset, "CIFAR-10 훈련 데이터셋", show_samples=3)

# 원본 데이터 시각화 (증강 전)
print(f"\n🖼️  CIFAR-10 원본 샘플 시각화")

# 증강 없는 변환으로 원본 데이터 확인
original_transform = transforms.Compose([transforms.ToTensor()])
original_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=original_transform
)

visualize_samples(
    original_dataset, 
    num_samples=20, 
    num_cols=5,
    title="CIFAR-10 원본 데이터 샘플",
    class_names=class_names
)

# CIFAR-10 vs 이전 데이터셋 비교
print(f"\n📈 CIFAR-10의 특징 및 도전과제:")
print(f"   1. 컬러 이미지 (3채널): RGB 정보 활용 필요")
print(f"   2. 낮은 해상도 (32x32): 제한된 픽셀로 객체 인식")
print(f"   3. 클래스 내 다양성: 같은 클래스도 다양한 형태/색상")
print(f"   4. 클래스 간 유사성: 개와 고양이, 자동차와 트럭 등")
print(f"   5. 배경 복잡성: 단순한 배경이 아닌 자연스러운 장면")
print(f"   → CNN의 공간적 특징 추출 능력이 필수적")

# ============================================================================
# 4. 데이터 로더 생성
# ============================================================================

print(f"\n📦 데이터 분할 및 로더 생성")

# 훈련/검증 분할
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
# 5. CNN 모델 정의
# ============================================================================

print(f"\n🧠 CNN 모델 정의")

class SimpleCNN(nn.Module):
    """
    기본 CNN 모델
    
    구조:
    - Conv Block 1: Conv2d(3→32) → BatchNorm → ReLU → MaxPool
    - Conv Block 2: Conv2d(32→64) → BatchNorm → ReLU → MaxPool  
    - Conv Block 3: Conv2d(64→128) → BatchNorm → ReLU → MaxPool
    - Classifier: Flatten → FC(128*4*4→512) → Dropout → FC(512→10)
    
    왜 이런 구조인가?
    1. 점진적 채널 증가: 저수준→고수준 특징으로 복잡도 증가
    2. 공간 차원 감소: MaxPool로 계산량 줄이고 수용 영역 확대
    3. 배치 정규화: 각 층의 입력 분포 안정화
    4. 드롭아웃: 과적합 방지
    """
    
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 첫 번째 합성곱 블록
        # 입력: (3, 32, 32) → 출력: (32, 16, 16)
        self.conv1 = nn.Conv2d(
            in_channels=3,      # RGB 3채널
            out_channels=32,    # 32개 특성 맵 생성
            kernel_size=3,      # 3x3 필터 (가장 일반적)
            padding=1          # 패딩으로 크기 유지
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 크기 절반으로 축소
        
        # 두 번째 합성곱 블록
        # 입력: (32, 16, 16) → 출력: (64, 8, 8)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 세 번째 합성곱 블록
        # 입력: (64, 8, 8) → 출력: (128, 4, 4)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 분류기 (Classifier)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)  # 특성 맵을 1차원으로 펼침
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # 첫 번째 블록: Conv → BN → ReLU → Pool
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # 두 번째 블록
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # 세 번째 블록
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 특성 맵을 1차원으로 펼치기
        # (batch_size, 128, 4, 4) → (batch_size, 128*4*4)
        x = x.view(x.size(0), -1)
        
        # 분류기
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class AdvancedCNN(nn.Module):
    """
    고급 CNN 모델 (ResNet 스타일의 잔차 연결 포함)
    
    개선사항:
    1. 더 깊은 네트워크 (5개 합성곱 블록)
    2. 잔차 연결 (Residual Connection)
    3. 적응적 평균 풀링 (Adaptive Average Pooling)
    4. 더 정교한 정규화
    """
    
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        
        # 초기 합성곱층
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 합성곱 블록들
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        # 풀링층들
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 적응적 평균 풀링 (입력 크기에 관계없이 고정 출력 크기)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 분류기
        self.fc1 = nn.Linear(256, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, num_classes)
        
        # 잔차 연결을 위한 1x1 합성곱
        self.shortcut1 = nn.Conv2d(64, 128, kernel_size=1)
        self.shortcut2 = nn.Conv2d(128, 256, kernel_size=1)
    
    def forward(self, x):
        # 초기 특징 추출
        x = F.relu(self.bn1(self.conv1(x)))
        
        # 첫 번째 블록 (잔차 연결)
        identity = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + identity  # 잔차 연결
        x = self.pool(x)
        
        # 두 번째 블록 (채널 수 증가)
        identity = self.shortcut1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = x + identity
        x = self.pool(x)
        
        # 세 번째 블록
        identity = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = x + identity
        
        # 네 번째 블록 (채널 수 증가)
        identity = self.shortcut2(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = x + identity
        x = self.pool(x)
        
        # 적응적 풀링
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # 분류기
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

# 모델 인스턴스 생성
simple_cnn = SimpleCNN(num_classes=10).to(device)
advanced_cnn = AdvancedCNN(num_classes=10).to(device)

print(f"✅ CNN 모델 생성 완료")

# 모델 구조 출력
print(f"\n📋 간단한 CNN 구조:")
print(simple_cnn)

# 파라미터 수 비교
print(f"\n📊 모델 복잡도 비교:")
simple_params = count_parameters(simple_cnn, detailed=False)
advanced_params = count_parameters(advanced_cnn, detailed=False)

print(f"   간단한 CNN: {simple_params['total_params']:,}개 파라미터")
print(f"   고급 CNN: {advanced_params['total_params']:,}개 파라미터")

# ============================================================================
# 6. 손실 함수와 옵티마이저 설정
# ============================================================================

print(f"\n⚙️  손실 함수와 옵티마이저 설정")

# 손실 함수
criterion = nn.CrossEntropyLoss()

# 옵티마이저 (고급 CNN용)
optimizer = optim.Adam(
    advanced_cnn.parameters(), 
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

# 학습률 스케줄러 (코사인 어닐링)
# 왜 코사인 어닐링인가?
# 1. 부드러운 학습률 감소로 안정적인 수렴
# 2. 주기적 재시작으로 지역 최솟값 탈출
# 3. CNN 훈련에서 우수한 성능 입증
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=EPOCHS,      # 전체 에포크 수
    eta_min=1e-6       # 최소 학습률
)

print(f"   손실 함수: {criterion.__class__.__name__}")
print(f"   옵티마이저: {optimizer.__class__.__name__}")
print(f"   스케줄러: CosineAnnealingLR")

# ============================================================================
# 7. 훈련 함수 정의
# ============================================================================

def train_epoch_cnn(model, train_loader, criterion, optimizer, device, epoch):
    """CNN을 위한 훈련 함수"""
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
        
        # 그래디언트 클리핑 (CNN에서 중요)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 통계 업데이트
        running_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == targets).sum().item()
        total_samples += targets.size(0)
        
        # 진행률 바 업데이트
        if batch_idx % 50 == 0:  # 50배치마다 업데이트
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

def validate_epoch_cnn(model, val_loader, criterion, device):
    """CNN을 위한 검증 함수"""
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

print(f"\n🚀 CNN 모델 훈련 시작")

# 훈련 기록
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
learning_rates = []

# 최고 성능 추적
best_val_accuracy = 0.0
best_model_state = None
patience = 10
patience_counter = 0

start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\n📅 에포크 {epoch+1}/{EPOCHS}")
    
    # 현재 학습률 기록
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    
    # 훈련
    train_loss, train_acc = train_epoch_cnn(
        advanced_cnn, train_loader, criterion, optimizer, device, epoch
    )
    
    # 검증
    val_loss, val_acc = validate_epoch_cnn(
        advanced_cnn, val_loader, criterion, device
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
        best_model_state = copy.deepcopy(advanced_cnn.state_dict())
        patience_counter = 0
        print(f"   🎯 새로운 최고 성능! 검증 정확도: {val_acc:.4f}")
        
        # 체크포인트 저장
        save_checkpoint(
            advanced_cnn, optimizer, epoch, val_loss, val_acc,
            save_path="./checkpoints/cifar10_cnn_best_model.pth"
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
# 9. 특성 맵 시각화
# ============================================================================

print(f"\n🔍 CNN 특성 맵 시각화")

# 최고 성능 모델 로드
if best_model_state is not None:
    advanced_cnn.load_state_dict(best_model_state)

# 샘플 이미지로 특성 맵 시각화
sample_batch = next(iter(test_loader))
sample_image = sample_batch[0][0]  # 첫 번째 이미지

print(f"📊 첫 번째 합성곱층 특성 맵:")
plot_feature_maps(
    model=advanced_cnn,
    input_tensor=sample_image,
    layer_name="conv1",
    num_maps=16,
    title="첫 번째 합성곱층 특성 맵 (저수준 특징)"
)

print(f"📊 세 번째 합성곱층 특성 맵:")
plot_feature_maps(
    model=advanced_cnn,
    input_tensor=sample_image,
    layer_name="conv3",
    num_maps=16,
    title="세 번째 합성곱층 특성 맵 (중간 수준 특징)"
)

# ============================================================================
# 10. 훈련 결과 시각화
# ============================================================================

print(f"\n📈 훈련 결과 시각화")

# 훈련 곡선
plot_training_curves(
    train_losses=train_losses,
    val_losses=val_losses,
    train_accuracies=train_accuracies,
    val_accuracies=val_accuracies,
    title="CIFAR-10 CNN 분류 - 훈련 과정"
)

# ============================================================================
# 11. 모델 비교 실험
# ============================================================================

print(f"\n🏆 CNN vs MLP 성능 비교")

# 비교를 위한 MLP 모델 (이전 튜토리얼 스타일)
class MLP_for_CIFAR(nn.Module):
    def __init__(self):
        super(MLP_for_CIFAR, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 평탄화
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# MLP 모델 빠른 훈련 (비교용)
mlp_model = MLP_for_CIFAR().to(device)
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=LEARNING_RATE)

print(f"📊 MLP 모델 빠른 훈련 중... (5 에포크)")
for epoch in range(5):
    mlp_model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        if batch_idx > 100:  # 빠른 훈련을 위해 제한
            break
        data, targets = data.to(device), targets.to(device)
        mlp_optimizer.zero_grad()
        outputs = mlp_model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        mlp_optimizer.step()

# 모델 비교
models_to_compare = {
    "고급 CNN (잔차 연결)": advanced_cnn,
    "간단한 CNN": simple_cnn,
    "MLP (완전연결층만)": mlp_model
}

comparison_results = compare_models(
    models=models_to_compare,
    test_dataloader=test_loader,
    criterion=criterion,
    device=device
)

# ============================================================================
# 12. 최종 평가 및 시각화
# ============================================================================

print(f"\n🎯 최종 CNN 모델 평가")

# 상세한 성능 평가
final_results = evaluate_model(
    model=advanced_cnn,
    dataloader=test_loader,
    criterion=criterion,
    device=device,
    num_classes=10
)

# 예측 결과 시각화
print(f"\n🖼️  CNN 예측 결과 시각화")
plot_model_predictions(
    model=advanced_cnn,
    dataloader=test_loader,
    class_names=class_names,
    num_samples=16,
    device=device,
    title="CIFAR-10 CNN 분류 - 예측 결과"
)

# 혼동 행렬 생성
print(f"\n📊 혼동 행렬 생성 중...")

all_predictions = []
all_targets = []

advanced_cnn.eval()
with torch.no_grad():
    for data, targets in tqdm(test_loader, desc="예측 수집"):
        data, targets = data.to(device), targets.to(device)
        outputs = advanced_cnn(data)
        predictions = torch.argmax(outputs, dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

plot_confusion_matrix(
    y_true=np.array(all_targets),
    y_pred=np.array(all_predictions),
    class_names=class_names,
    title="CIFAR-10 CNN 분류 - 혼동 행렬"
)

# ============================================================================
# 13. 학습 내용 요약 및 다음 단계
# ============================================================================

print(f"\n🎓 학습 내용 요약")
print(f"=" * 60)

print(f"✅ 완료한 내용:")
print(f"   1. CNN의 핵심 구성 요소 (Conv, Pool, FC) 이해")
print(f"   2. CIFAR-10 컬러 이미지 분류 구현")
print(f"   3. 특성 맵 시각화로 CNN 동작 원리 확인")
print(f"   4. 잔차 연결과 고급 CNN 기법 적용")
print(f"   5. CNN vs MLP 성능 비교 실험")

print(f"\n📊 최종 성과:")
print(f"   - CNN 최고 정확도: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
print(f"   - 총 파라미터 수: {advanced_params['total_params']:,}개")
print(f"   - 훈련 시간: {training_time:.2f}초")

print(f"\n💡 핵심 학습 포인트:")
print(f"   1. 공간적 구조 보존: CNN이 이미지에 적합한 이유")
print(f"   2. 계층적 특징 학습: 저수준→고수준 특징 추출")
print(f"   3. 파라미터 공유: 효율적인 특징 감지")
print(f"   4. 평행이동 불변성: 위치에 관계없는 객체 인식")
print(f"   5. 데이터 증강의 중요성: 일반화 성능 향상")

print(f"\n🔍 CNN vs MLP 비교 분석:")
print(f"   - CNN: 공간적 구조 활용으로 더 높은 성능")
print(f"   - MLP: 공간 정보 손실로 제한적 성능")
print(f"   - 파라미터 효율성: CNN이 더 적은 파라미터로 우수한 성능")
print(f"   - 해석 가능성: CNN의 특성 맵으로 학습 과정 시각화 가능")

print(f"\n🚀 다음 단계:")
print(f"   - 04_rnn_text_classification.py: 순환 신경망으로 텍스트 처리")
print(f"   - IMDB 영화 리뷰 감정 분석")
print(f"   - 시퀀스 데이터 처리 기법 학습")

print(f"\n🔧 추가 실험 아이디어:")
print(f"   1. 전이 학습: 사전 훈련된 ResNet, VGG 활용")
print(f"   2. 다양한 CNN 아키텍처: DenseNet, EfficientNet")
print(f"   3. 어텐션 메커니즘: 중요한 영역에 집중")
print(f"   4. 모델 압축: Pruning, Quantization")
print(f"   5. 앙상블 기법: 여러 모델 조합으로 성능 향상")

print(f"\n" + "=" * 60)
print(f"🎉 CNN 이미지 분류 튜토리얼 완료!")
print(f"   다음 튜토리얼에서 RNN을 배워보세요!")
print(f"=" * 60)