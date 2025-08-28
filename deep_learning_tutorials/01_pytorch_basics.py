"""
딥러닝 강의 시리즈 1: PyTorch 기초

이 튜토리얼에서는 PyTorch의 기본 개념과 MNIST 데이터셋을 사용한 
간단한 신경망 구현을 학습합니다.

학습 목표:
1. PyTorch 텐서 연산의 기본 이해
2. 자동 미분(autograd) 시스템 이해  
3. 데이터 로더와 데이터셋 사용법
4. 기본 신경망 구조 설계
5. 훈련 루프 구현

데이터셋 선택 이유 - MNIST:
- 28x28 픽셀의 흑백 손글씨 숫자 이미지 (0-9)
- 60,000개 훈련 + 10,000개 테스트 샘플
- 딥러닝 입문자를 위한 "Hello World" 데이터셋
- 단순한 구조로 PyTorch 기본 개념 학습에 최적
- 빠른 훈련 시간으로 즉각적인 피드백 가능
- 시각화가 쉬워 결과 해석이 직관적
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

# 우리가 만든 유틸리티 함수들 임포트
from utils.data_utils import explore_dataset, visualize_samples, get_data_statistics
from utils.visualization import plot_training_curves, plot_confusion_matrix, plot_model_predictions
from utils.model_utils import count_parameters, save_checkpoint, evaluate_model

print("🚀 딥러닝 강의 시리즈 1: PyTorch 기초")
print("=" * 60)

# ============================================================================
# 1. 환경 설정 및 하이퍼파라미터
# ============================================================================

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  사용 장치: {device}")

if device.type == 'cuda':
    print(f"   GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# 하이퍼파라미터 설정
# 이 값들은 모델 성능에 큰 영향을 미치므로 각각의 의미를 이해하는 것이 중요합니다
BATCH_SIZE = 64        # 배치 크기: 한 번에 처리할 샘플 수
                       # 작으면 메모리 절약, 크면 안정적인 그래디언트
LEARNING_RATE = 0.001  # 학습률: 가중치 업데이트 크기
                       # 너무 크면 발산, 너무 작으면 학습 속도 저하
EPOCHS = 10            # 에포크: 전체 데이터셋을 몇 번 반복할지
RANDOM_SEED = 42       # 재현 가능한 결과를 위한 랜덤 시드

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
# 2. 데이터 전처리 및 로딩
# ============================================================================

print(f"\n📁 MNIST 데이터셋 준비 중...")

# 데이터 전처리 파이프라인 정의
# 왜 이런 전처리가 필요한가?
transform = transforms.Compose([
    # ToTensor(): PIL Image나 numpy array를 PyTorch 텐서로 변환
    # 동시에 [0, 255] 범위의 픽셀값을 [0.0, 1.0] 범위로 정규화
    # 이유: 신경망은 작은 값에서 더 안정적으로 학습됨
    transforms.ToTensor(),
    
    # Normalize(): 평균 0.1307, 표준편차 0.3081로 정규화
    # 이 값들은 MNIST 데이터셋의 실제 통계값
    # 이유: 정규화된 데이터는 그래디언트 소실/폭발 문제를 완화하고
    #       더 빠르고 안정적인 학습을 가능하게 함
    transforms.Normalize((0.1307,), (0.3081,))
])

# 훈련 데이터셋 로드
# download=True: 데이터가 없으면 자동으로 다운로드
# train=True: 훈련용 데이터셋 (60,000개)
train_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

# 테스트 데이터셋 로드  
# train=False: 테스트용 데이터셋 (10,000개)
test_dataset = torchvision.datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

print(f"✅ 데이터셋 로드 완료")

# ============================================================================
# 3. 데이터 탐색 및 시각화
# ============================================================================

print(f"\n🔍 데이터셋 탐색")

# 우리가 만든 유틸리티 함수로 데이터셋 기본 정보 확인
explore_dataset(train_dataset, "MNIST 훈련 데이터셋", show_samples=3)
explore_dataset(test_dataset, "MNIST 테스트 데이터셋", show_samples=3)

# MNIST 클래스 이름 (0-9 숫자)
class_names = [str(i) for i in range(10)]

# 샘플 데이터 시각화
print(f"\n🖼️  훈련 데이터 샘플 시각화")
visualize_samples(
    train_dataset, 
    num_samples=12, 
    num_cols=4,
    title="MNIST 훈련 데이터 샘플",
    class_names=class_names
)

# ============================================================================
# 4. 데이터 로더 생성
# ============================================================================

print(f"\n📦 데이터 로더 생성 중...")

# 데이터 로더 생성
# 왜 데이터 로더가 필요한가?
# 1. 메모리 효율성: 전체 데이터를 한 번에 메모리에 올리지 않고 배치 단위로 처리
# 2. 병렬 처리: num_workers로 데이터 로딩을 병렬화하여 속도 향상
# 3. 셔플링: 매 에포크마다 데이터 순서를 바꿔 모델이 순서에 의존하지 않도록 함

train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,      # 훈련 시에는 셔플링으로 일반화 성능 향상
    num_workers=2      # 병렬 데이터 로딩 (CPU 코어 수에 따라 조정)
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False,     # 테스트 시에는 셔플링 불필요
    num_workers=2
)

print(f"✅ 데이터 로더 생성 완료")
print(f"   훈련 배치 수: {len(train_loader)}")
print(f"   테스트 배치 수: {len(test_loader)}")

# 데이터 로더 동작 확인
sample_batch = next(iter(train_loader))
sample_images, sample_labels = sample_batch
print(f"   배치 이미지 크기: {sample_images.shape}")  # [batch_size, channels, height, width]
print(f"   배치 라벨 크기: {sample_labels.shape}")    # [batch_size]

# ============================================================================
# 5. 신경망 모델 정의
# ============================================================================

print(f"\n🧠 신경망 모델 정의")

class SimpleNN(nn.Module):
    """
    간단한 다층 퍼셉트론 (Multi-Layer Perceptron, MLP)
    
    구조:
    - 입력층: 28x28 = 784개 뉴런 (MNIST 이미지를 1차원으로 펼침)
    - 은닉층 1: 128개 뉴런 + ReLU 활성화 함수
    - 은닉층 2: 64개 뉴런 + ReLU 활성화 함수  
    - 출력층: 10개 뉴런 (0-9 숫자 클래스)
    
    왜 이런 구조를 선택했는가?
    1. 완전연결층(Linear): 모든 입력이 모든 출력에 연결되어 복잡한 패턴 학습 가능
    2. ReLU 활성화: 그래디언트 소실 문제 완화, 계산 효율성
    3. 점진적 차원 축소: 784 → 128 → 64 → 10으로 정보를 압축하며 특징 추출
    """
    
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        # 완전연결층 정의
        # nn.Linear(입력_크기, 출력_크기)
        self.fc1 = nn.Linear(28 * 28, 128)  # 첫 번째 은닉층
        self.fc2 = nn.Linear(128, 64)       # 두 번째 은닉층  
        self.fc3 = nn.Linear(64, 10)        # 출력층
        
        # 드롭아웃 레이어 (과적합 방지)
        # 훈련 시 일정 비율의 뉴런을 무작위로 비활성화
        # 이유: 모델이 특정 뉴런에 과도하게 의존하는 것을 방지
        self.dropout = nn.Dropout(0.2)  # 20% 뉴런을 무작위로 비활성화
    
    def forward(self, x):
        """
        순전파(forward pass) 정의
        
        Args:
            x: 입력 텐서 [batch_size, 1, 28, 28]
        
        Returns:
            출력 텐서 [batch_size, 10] (각 클래스에 대한 로짓)
        """
        
        # 1. 입력 이미지를 1차원으로 펼치기
        # [batch_size, 1, 28, 28] → [batch_size, 784]
        # 이유: 완전연결층은 1차원 입력을 받기 때문
        x = x.view(x.size(0), -1)  # -1은 자동으로 크기 계산
        
        # 2. 첫 번째 은닉층 + ReLU 활성화
        # ReLU(x) = max(0, x): 음수는 0, 양수는 그대로
        # 이유: 비선형성 도입으로 복잡한 패턴 학습 가능
        x = F.relu(self.fc1(x))
        
        # 3. 드롭아웃 적용 (훈련 시에만)
        x = self.dropout(x)
        
        # 4. 두 번째 은닉층 + ReLU 활성화
        x = F.relu(self.fc2(x))
        
        # 5. 드롭아웃 적용
        x = self.dropout(x)
        
        # 6. 출력층 (활성화 함수 없음)
        # 이유: CrossEntropyLoss가 내부적으로 소프트맥스를 적용하므로
        x = self.fc3(x)
        
        return x

# 모델 인스턴스 생성 및 장치로 이동
model = SimpleNN().to(device)

print(f"✅ 모델 생성 완료")

# 모델 구조 출력
print(f"\n📋 모델 구조:")
print(model)

# 우리가 만든 유틸리티로 파라미터 수 분석
param_info = count_parameters(model, detailed=True)

# ============================================================================
# 6. 손실 함수와 옵티마이저 정의
# ============================================================================

print(f"\n⚙️  손실 함수와 옵티마이저 설정")

# 손실 함수: CrossEntropyLoss
# 왜 CrossEntropyLoss를 사용하는가?
# 1. 다중 클래스 분류 문제에 최적화됨
# 2. 소프트맥스 + 음의 로그 우도를 결합한 효율적인 구현
# 3. 확률 분포 간의 차이를 측정하여 직관적인 해석 가능
# 4. 그래디언트가 잘 전파되어 안정적인 학습
criterion = nn.CrossEntropyLoss()

# 옵티마이저: Adam
# 왜 Adam을 사용하는가?
# 1. 적응적 학습률: 각 파라미터마다 다른 학습률 적용
# 2. 모멘텀 효과: 이전 그래디언트 정보를 활용하여 진동 감소
# 3. 하이퍼파라미터 튜닝이 상대적으로 쉬움
# 4. 대부분의 문제에서 좋은 성능을 보임
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"   손실 함수: {criterion.__class__.__name__}")
print(f"   옵티마이저: {optimizer.__class__.__name__}")
print(f"   학습률: {LEARNING_RATE}")

# ============================================================================
# 7. 훈련 함수 정의
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    한 에포�� 동안 모델을 훈련합니다.
    
    Args:
        model: 훈련할 모델
        train_loader: 훈련 데이터 로더
        criterion: 손실 함수
        optimizer: 옵티마이저
        device: 연산 장치
    
    Returns:
        tuple: (평균 손실, 정확도)
    
    훈련 과정의 각 단계:
    1. 순전파: 입력 → 예측
    2. 손실 계산: 예측 vs 실제
    3. 역전파: 손실 → 그래디언트
    4. 가중치 업데이트: 그래디언트 → 새로운 가중치
    """
    
    model.train()  # 훈련 모드 설정 (드롭아웃, 배치 정규화 활성화)
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # tqdm으로 진행률 표시
    pbar = tqdm(train_loader, desc="훈련 중")
    
    for batch_idx, (data, targets) in enumerate(pbar):
        # 데이터를 장치로 이동
        data, targets = data.to(device), targets.to(device)
        
        # 1. 그래디언트 초기화
        # 이유: PyTorch는 그래디언트를 누적하므로 매 배치마다 초기화 필요
        optimizer.zero_grad()
        
        # 2. 순전파 (Forward pass)
        outputs = model(data)
        
        # 3. 손실 계산
        loss = criterion(outputs, targets)
        
        # 4. 역전파 (Backward pass)
        # 손실에 대한 각 파라미터의 그래디언트 계산
        loss.backward()
        
        # 5. 가중치 업데이트
        # 계산된 그래디언트를 사용하여 파라미터 업데이트
        optimizer.step()
        
        # 통계 업데이트
        running_loss += loss.item()
        
        # 정확도 계산
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == targets).sum().item()
        total_samples += targets.size(0)
        
        # 진행률 바 업데이트
        current_accuracy = correct_predictions / total_samples
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_accuracy:.4f}'
        })
    
    # 에포크 평균 계산
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    """
    검증 데이터로 모델 성능을 평가합니다.
    
    Args:
        model: 평가할 모델
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        device: 연산 장치
    
    Returns:
        tuple: (평균 손실, 정확도)
    
    검증과 훈련의 차이점:
    1. model.eval(): 드롭아웃 비활성화, 배치 정규화 고정
    2. torch.no_grad(): 그래디언트 계산 비활성화로 메모리 절약
    3. 가중치 업데이트 없음: 순수하게 성능 측정만
    """
    
    model.eval()  # 평가 모드 설정
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # 그래디언트 계산 비활성화 (메모리 절약 및 속도 향상)
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="검증 중")
        
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            
            # 순전파만 수행
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # 통계 업데이트
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == targets).sum().item()
            total_samples += targets.size(0)
            
            # 진행률 바 업데이트
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

print(f"\n🚀 모델 훈련 시작")
print(f"   총 에포크: {EPOCHS}")
print(f"   배치당 샘플 수: {BATCH_SIZE}")
print(f"   총 훈련 배치: {len(train_loader)}")

# 훈련 기록 저장용 리스트
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# 최고 성능 추적
best_val_accuracy = 0.0
best_model_state = None

# 훈련 시작 시간 기록
start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\n📅 에포크 {epoch+1}/{EPOCHS}")
    
    # 훈련
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # 검증
    val_loss, val_acc = validate_epoch(model, test_loader, criterion, device)
    
    # 기록 저장
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    # 결과 출력
    print(f"   훈련 - 손실: {train_loss:.4f}, 정확도: {train_acc:.4f}")
    print(f"   검증 - 손실: {val_loss:.4f}, 정확도: {val_acc:.4f}")
    
    # 최고 성능 모델 저장
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_model_state = model.state_dict().copy()
        print(f"   🎯 새로운 최고 성능! 검증 정확도: {val_acc:.4f}")
        
        # 체크포인트 저장
        save_checkpoint(
            model, optimizer, epoch, val_loss, val_acc,
            save_path="./checkpoints/mnist_best_model.pth",
            additional_info={
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'hyperparameters': {
                    'batch_size': BATCH_SIZE,
                    'learning_rate': LEARNING_RATE,
                    'epochs': EPOCHS
                }
            }
        )

# 훈련 완료
training_time = time.time() - start_time
print(f"\n✅ 훈련 완료!")
print(f"   총 훈련 시간: {training_time:.2f}초")
print(f"   최고 검증 정확도: {best_val_accuracy:.4f}")

# ============================================================================
# 9. 훈련 결과 시각화
# ============================================================================

print(f"\n📈 훈련 결과 시각화")

# 훈련 곡선 그리기
plot_training_curves(
    train_losses=train_losses,
    val_losses=val_losses,
    train_accuracies=train_accuracies,
    val_accuracies=val_accuracies,
    title="MNIST 분류 - 훈련 과정",
    save_path="./results/mnist_training_curves.png"
)

# ============================================================================
# 10. 최종 모델 평가
# ============================================================================

print(f"\n🎯 최종 모델 평가")

# 최고 성능 모델 로드
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"✅ 최고 성능 모델 로드 완료")

# 상세한 성능 평가
final_results = evaluate_model(
    model=model,
    dataloader=test_loader,
    criterion=criterion,
    device=device,
    num_classes=10
)

# ============================================================================
# 11. 예측 결과 시각화
# ============================================================================

print(f"\n🖼️  예측 결과 시각화")

# 모델 예측 결과 시각화
plot_model_predictions(
    model=model,
    dataloader=test_loader,
    class_names=class_names,
    num_samples=12,
    device=device,
    title="MNIST 분류 - 모델 예측 결과"
)

# 혼동 행렬 생성을 위한 전체 예측 수집
print(f"\n📊 혼동 행렬 생성 중...")

all_predictions = []
all_targets = []

model.eval()
with torch.no_grad():
    for data, targets in tqdm(test_loader, desc="예측 수집"):
        data, targets = data.to(device), targets.to(device)
        outputs = model(data)
        predictions = torch.argmax(outputs, dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# 혼동 행렬 시각화
plot_confusion_matrix(
    y_true=np.array(all_targets),
    y_pred=np.array(all_predictions),
    class_names=class_names,
    title="MNIST 분류 - 혼동 행렬",
    save_path="./results/mnist_confusion_matrix.png"
)

# ============================================================================
# 12. 학습 내용 요약 및 다음 단계 안내
# ============================================================================

print(f"\n🎓 학습 내용 요약")
print(f"=" * 60)

print(f"✅ 완료한 내용:")
print(f"   1. PyTorch 텐서와 자동 미분 시스템 이해")
print(f"   2. MNIST 데이터셋 로딩 및 전처리")
print(f"   3. 간단한 다층 퍼셉트론 구현")
print(f"   4. 훈련 루프와 검증 과정 구현")
print(f"   5. 모델 성능 평가 및 시각화")

print(f"\n📊 최종 성과:")
print(f"   - 최고 검증 정확도: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
print(f"   - 총 파라미터 수: {param_info['total_params']:,}개")
print(f"   - 훈련 시간: {training_time:.2f}초")

print(f"\n💡 핵심 학습 포인트:")
print(f"   1. 데이터 전처리의 중요성 (정규화, 텐서 변환)")
print(f"   2. 적절한 모델 구조 설계 (은닉층 크기, 활성화 함수)")
print(f"   3. 손실 함수와 옵티마이저 선택의 영향")
print(f"   4. 과적합 방지 기법 (드롭아웃)")
print(f"   5. 훈련 과정 모니터링의 중요성")

print(f"\n🚀 다음 단계:")
print(f"   - 02_neural_networks.py: 더 복잡한 신경망과 정규화 기법")
print(f"   - Fashion-MNIST로 더 어려운 분류 문제 도전")
print(f"   - 하이퍼파라미터 튜닝 기법 학습")

print(f"\n🔧 개선 실험 아이디어:")
print(f"   1. 학습률 변경 (0.01, 0.0001)")
print(f"   2. 은닉층 크기 조정 (256, 512)")
print(f"   3. 드롭아웃 비율 변경 (0.1, 0.5)")
print(f"   4. 다른 옵티마이저 시도 (SGD, RMSprop)")
print(f"   5. 배치 크기 변경 (32, 128)")

print(f"\n" + "=" * 60)
print(f"🎉 PyTorch 기초 튜토리얼 완료!")
print(f"   다음 튜토리얼에서 더 고급 기법들을 배워보세요!")
print(f"=" * 60)