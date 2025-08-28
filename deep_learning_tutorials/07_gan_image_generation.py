"""
딥러닝 강의 시리즈 7: GAN 이미지 생성

이 튜토리얼에서는 CelebA 데이터셋을 사용하여 
GAN(Generative Adversarial Networks)의 이미지 생성을 학습합니다.

학습 목표:
1. GAN의 핵심 개념과 적대적 학습 원리
2. 생성자(Generator)와 판별자(Discriminator) 구조
3. 모드 붕괴(Mode Collapse) 문제와 해결 방법
4. 생성 품질 평가 방법 (FID, IS)
5. 잠재 공간(Latent Space) 탐색과 해석
6. DCGAN과 개선된 GAN 기법들

데이터셋 선택 이유 - CelebA (Celebrity Faces):
- 200,000개 이상의 유명인 얼굴 이미지
- 고품질의 정렬된 얼굴 데이터
- 다양한 연령, 성별, 인종의 얼굴 포함
- GAN 성능을 명확히 확인할 수 있는 데이터
- 얼굴 생성은 GAN의 대표적 응용 분야
- 시각적으로 생성 품질을 쉽게 평가 가능
- 적당한 복잡도로 학습에 적합

왜 GAN을 사용하는가?
1. 생성 모델링: 새로운 데이터 샘플 생성
2. 데이터 증강: 훈련 데이터 부족 문제 해결
3. 창작 도구: 예술, 디자인 분야 활용
4. 데이터 프라이버시: 실제 데이터 대신 합성 데이터
5. 표현 학습: 의미있는 잠재 공간 학습

GAN vs 다른 생성 모델:
- VAE: 흐릿한 이미지, 안정적 학습
- GAN: 선명한 이미지, 불안정한 학습
- Diffusion: 최고 품질, 느린 생성
- Flow: 정확한 확률 계산, 제한적 표현력
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import copy
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 우리가 만든 유틸리티 함수들 임포트
from utils.visualization import plot_training_curves
from utils.model_utils import count_parameters, save_checkpoint

print("🚀 딥러닝 강의 시리즈 7: GAN 이미지 생성")
print("=" * 60)

# ============================================================================
# 1. 환경 설정 및 하이퍼파라미터
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  사용 장치: {device}")

# GAN을 위한 하이퍼파라미터
BATCH_SIZE = 64        # GAN은 큰 배치 크기가 안정성에 도움
LEARNING_RATE_G = 0.0002  # 생성자 학습률
LEARNING_RATE_D = 0.0002  # 판별자 학습률
BETA1 = 0.5            # Adam 옵티마이저 베타1 (GAN에서 일반적)
EPOCHS = 100           # GAN은 충분한 학습 시간 필요
RANDOM_SEED = 42
IMG_SIZE = 64          # 생성할 이미지 크기
LATENT_DIM = 100       # 잠재 벡터 차원
NUM_CHANNELS = 3       # RGB 채널

# 재현성을 위한 시드 설정
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"📊 하이퍼파라미터:")
print(f"   배치 크기: {BATCH_SIZE}")
print(f"   생성자 학습률: {LEARNING_RATE_G}")
print(f"   판별자 학습률: {LEARNING_RATE_D}")
print(f"   에포크: {EPOCHS}")
print(f"   이미지 크기: {IMG_SIZE}x{IMG_SIZE}")
print(f"   잠재 차원: {LATENT_DIM}")

# ============================================================================
# 2. 샘플 얼굴 데이터셋 생성 (CelebA 대신)
# ============================================================================

print(f"\n📁 샘플 얼굴 데이터셋 생성")

class SampleFaceDataset(Dataset):
    """
    교육용 샘플 얼굴 데이터셋
    
    실제 프로젝트에서는 CelebA 사용
    여기서는 학습 목적으로 간단한 합성 얼굴 데이터 생성
    """
    
    def __init__(self, num_samples=5000, img_size=64, transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transform
        
        # 샘플 데이터 생성
        self.images = self._generate_face_samples()
        
    def _generate_face_samples(self):
        """합성 얼굴 데이터 생성"""
        print("🎨 합성 얼굴 데이터 생성 중...")
        
        images = []
        np.random.seed(42)
        
        for i in tqdm(range(self.num_samples), desc="얼굴 데이터 생성"):
            # 기본 얼굴 형태 생성
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            
            # 배경색 (피부톤)
            skin_color = np.random.randint(180, 255, 3)
            image[:, :] = skin_color
            
            # 얼굴 윤곽 (타원)
            center_x, center_y = self.img_size // 2, self.img_size // 2
            face_width = np.random.randint(self.img_size // 3, self.img_size // 2)
            face_height = np.random.randint(self.img_size // 2, int(self.img_size * 0.7))
            
            # 눈 영역
            eye_y = center_y - face_height // 4
            left_eye_x = center_x - face_width // 4
            right_eye_x = center_x + face_width // 4
            eye_size = np.random.randint(3, 8)
            
            # 눈 그리기
            for dy in range(-eye_size, eye_size):
                for dx in range(-eye_size, eye_size):
                    if dx*dx + dy*dy <= eye_size*eye_size:
                        if (0 <= eye_y + dy < self.img_size and 
                            0 <= left_eye_x + dx < self.img_size):
                            image[eye_y + dy, left_eye_x + dx] = [0, 0, 0]
                        if (0 <= eye_y + dy < self.img_size and 
                            0 <= right_eye_x + dx < self.img_size):
                            image[eye_y + dy, right_eye_x + dx] = [0, 0, 0]
            
            # 코 영역
            nose_y = center_y
            nose_x = center_x
            nose_size = np.random.randint(2, 5)
            
            for dy in range(-nose_size, nose_size):
                for dx in range(-nose_size//2, nose_size//2):
                    if (0 <= nose_y + dy < self.img_size and 
                        0 <= nose_x + dx < self.img_size):
                        image[nose_y + dy, nose_x + dx] = skin_color * 0.8
            
            # 입 영역
            mouth_y = center_y + face_height // 4
            mouth_width = np.random.randint(face_width // 4, face_width // 2)
            mouth_height = np.random.randint(2, 5)
            
            for dy in range(-mouth_height, mouth_height):
                for dx in range(-mouth_width, mouth_width):
                    if (0 <= mouth_y + dy < self.img_size and 
                        0 <= center_x + dx < self.img_size):
                        image[mouth_y + dy, center_x + dx] = [100, 50, 50]
            
            # 노이즈 추가 (자연스러운 변화)
            noise = np.random.normal(0, 10, image.shape)
            image = np.clip(image + noise, 0, 255).astype(np.uint8)
            
            images.append(image)
        
        print(f"✅ {len(images)}개 얼굴 샘플 생성 완료")
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            # PIL Image로 변환 후 transform 적용
            image = Image.fromarray(image)
            image = self.transform(image)
        else:
            # 텐서로 변환
            image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        
        return image

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1] 범위로 정규화
])

# 데이터셋 및 로더 생성
dataset = SampleFaceDataset(
    num_samples=3000,
    img_size=IMG_SIZE,
    transform=transform
)

dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    num_workers=2,
    drop_last=True  # 마지막 배치 크기가 다르면 제거
)

print(f"\n✅ 데이터 로더 생성 완료:")
print(f"   총 샘플: {len(dataset):,}개")
print(f"   배치 수: {len(dataloader)}")

# 샘플 이미지 시각화
def show_sample_images(dataloader, num_samples=16):
    """샘플 이미지 시각화"""
    data_iter = iter(dataloader)
    images = next(data_iter)
    
    # 정규화 해제 [-1, 1] → [0, 1]
    images = (images + 1) / 2
    
    # 그리드로 배치
    grid = make_grid(images[:num_samples], nrow=4, padding=2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title('샘플 얼굴 데이터', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.show()

print(f"\n🖼️  샘플 데이터 시각화")
show_sample_images(dataloader)

# ============================================================================
# 3. DCGAN 모델 정의
# ============================================================================

print(f"\n🧠 DCGAN 모델 정의")

def weights_init(m):
    """
    가중치 초기화 함수
    
    DCGAN 논문에서 제안한 초기화 방법:
    - Conv, ConvTranspose: 평균 0, 표준편차 0.02의 정규분포
    - BatchNorm: 가중치 1, 편향 0으로 초기화
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    """
    DCGAN 생성자
    
    구조:
    - 잠재 벡터 → 전치 합성곱으로 이미지 생성
    - 배치 정규화와 ReLU 활성화 함수 사용
    - 마지막 층에서 Tanh로 [-1, 1] 범위 출력
    
    핵심 아이디어:
    - 전치 합성곱(Transposed Convolution)으로 업샘플링
    - 점진적으로 해상도 증가: 4x4 → 8x8 → 16x16 → 32x32 → 64x64
    - 배치 정규화로 안정적 학습
    """
    
    def __init__(self, latent_dim=100, img_channels=3, feature_dim=64):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # 첫 번째 층: 잠재 벡터 → 4x4 특성 맵
        self.initial = nn.Sequential(
            # 입력: (batch_size, latent_dim, 1, 1)
            nn.ConvTranspose2d(latent_dim, feature_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.ReLU(True)
            # 출력: (batch_size, feature_dim * 8, 4, 4)
        )
        
        # 업샘플링 층들
        self.upsample_layers = nn.Sequential(
            # 4x4 → 8x8
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),
            
            # 8x8 → 16x16
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),
            
            # 16x16 → 32x32
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),
            
            # 32x32 → 64x64
            nn.ConvTranspose2d(feature_dim, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # [-1, 1] 범위로 출력
        )
        
    def forward(self, z):
        # z: (batch_size, latent_dim)
        # 4차원으로 변환: (batch_size, latent_dim, 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        
        # 초기 특성 맵 생성
        x = self.initial(z)
        
        # 업샘플링을 통한 이미지 생성
        x = self.upsample_layers(x)
        
        return x

class Discriminator(nn.Module):
    """
    DCGAN 판별자
    
    구조:
    - 이미지 → 합성곱으로 특성 추출 → 진짜/가짜 판별
    - LeakyReLU 활성화 함수 사용
    - 배치 정규화 (첫 번째 층 제외)
    
    핵심 아이디어:
    - 합성곱으로 다운샘플링: 64x64 → 32x32 → 16x16 → 8x8 → 4x4
    - LeakyReLU로 그래디언트 흐름 개선
    - 마지막에 시그모이드로 확률 출력
    """
    
    def __init__(self, img_channels=3, feature_dim=64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # 64x64 → 32x32
            nn.Conv2d(img_channels, feature_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 → 16x16
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 → 8x8
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 → 4x4
            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 → 1x1 (최종 판별)
            nn.Conv2d(feature_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# 모델 인스턴스 생성
generator = Generator(
    latent_dim=LATENT_DIM,
    img_channels=NUM_CHANNELS,
    feature_dim=64
).to(device)

discriminator = Discriminator(
    img_channels=NUM_CHANNELS,
    feature_dim=64
).to(device)

# 가중치 초기화
generator.apply(weights_init)
discriminator.apply(weights_init)

print(f"✅ DCGAN 모델 생성 완료")

# 모델 복잡도
gen_params = count_parameters(generator, detailed=False)
disc_params = count_parameters(discriminator, detailed=False)

print(f"\n📊 모델 복잡도:")
print(f"   생성자: {gen_params['total_params']:,}개 파라미터")
print(f"   판별자: {disc_params['total_params']:,}개 파라미터")

# ============================================================================
# 4. 손실 함수와 옵티마이저
# ============================================================================

print(f"\n⚙️  손실 함수와 옵티마이저 설정")

# 손실 함수 (Binary Cross Entropy)
criterion = nn.BCELoss()

# 라벨 정의
real_label = 1.0
fake_label = 0.0

# 옵티마이저 (Adam with beta1=0.5)
# GAN에서는 beta1을 0.5로 설정하는 것이 일반적
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE_G, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE_D, betas=(BETA1, 0.999))

print(f"   손실 함수: Binary Cross Entropy")
print(f"   생성자 옵티마이저: Adam (lr={LEARNING_RATE_G}, beta1={BETA1})")
print(f"   판별자 옵티마이저: Adam (lr={LEARNING_RATE_D}, beta1={BETA1})")

# 고정된 노이즈 (생성 과정 시각화용)
fixed_noise = torch.randn(64, LATENT_DIM, device=device)

# ============================================================================
# 5. GAN 훈련 함수
# ============================================================================

def train_gan_epoch(generator, discriminator, dataloader, criterion, 
                   optimizer_G, optimizer_D, device, epoch):
    """
    GAN 훈련 함수
    
    GAN 훈련의 핵심:
    1. 판별자 훈련: 진짜 이미지는 1, 가짜 이미지는 0으로 분류
    2. 생성자 훈련: 판별자를 속이도록 학습 (가짜 이미지를 1로 분류하게)
    
    주의사항:
    - 판별자와 생성자의 균형이 중요
    - 한쪽이 너무 강해지면 학습이 불안정해짐
    """
    
    generator.train()
    discriminator.train()
    
    running_loss_D = 0.0
    running_loss_G = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"에포크 {epoch+1} 훈련")
    
    for batch_idx, real_images in enumerate(pbar):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # ========================================
        # 1. 판별자 훈련
        # ========================================
        
        discriminator.zero_grad()
        
        # 1-1. 진짜 이미지로 훈련
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        output_real = discriminator(real_images)
        loss_D_real = criterion(output_real, label)
        loss_D_real.backward()
        
        # 1-2. 가짜 이미지로 훈련
        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_images = generator(noise)
        label.fill_(fake_label)
        output_fake = discriminator(fake_images.detach())  # detach로 생성자 그래디언트 차단
        loss_D_fake = criterion(output_fake, label)
        loss_D_fake.backward()
        
        # 판별자 파라미터 업데이트
        optimizer_D.step()
        
        loss_D = loss_D_real + loss_D_fake
        
        # ========================================
        # 2. 생성자 훈련
        # ========================================
        
        generator.zero_grad()
        
        # 가짜 이미지를 진짜로 분류하도록 학습
        label.fill_(real_label)  # 생성자는 판별자를 속이려고 함
        output = discriminator(fake_images)
        loss_G = criterion(output, label)
        loss_G.backward()
        
        # 생성자 파라미터 업데이트
        optimizer_G.step()
        
        # 통계 업데이트
        running_loss_D += loss_D.item()
        running_loss_G += loss_G.item()
        num_batches += 1
        
        # 진행률 바 업데이트
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'D_Loss': f'{loss_D.item():.4f}',
                'G_Loss': f'{loss_G.item():.4f}',
                'D_Real': f'{output_real.mean().item():.4f}',
                'D_Fake': f'{output_fake.mean().item():.4f}'
            })
    
    avg_loss_D = running_loss_D / num_batches
    avg_loss_G = running_loss_G / num_batches
    
    return avg_loss_D, avg_loss_G

# ============================================================================
# 6. 생성 품질 평가 함수
# ============================================================================

def calculate_inception_score(images, batch_size=32, splits=10):
    """
    Inception Score (IS) 계산
    
    IS는 생성된 이미지의 품질과 다양성을 측정:
    - 높은 품질: 각 이미지가 명확한 클래스로 분류됨
    - 높은 다양성: 전체적으로 다양한 클래스 분포
    
    실제 구현에서는 사전 훈련된 Inception 모델 필요
    여기서는 간소화된 버전으로 구현
    """
    # 간소화된 IS 계산 (실제로는 Inception v3 모델 사용)
    # 여기서는 이미지의 통계적 특성으로 대체
    
    mean_pixel = images.mean(dim=[2, 3])  # 각 채널의 평균
    std_pixel = images.std(dim=[2, 3])    # 각 채널의 표준편차
    
    # 다양성 측정 (표준편차의 평균)
    diversity = std_pixel.mean().item()
    
    # 품질 측정 (픽셀 값의 일관성)
    quality = 1.0 / (1.0 + mean_pixel.std().item())
    
    # 간소화된 IS 점수
    is_score = quality * diversity * 10  # 스케일링
    
    return is_score

def generate_and_save_images(generator, epoch, device, num_images=64):
    """생성된 이미지 저장 및 시각화"""
    generator.eval()
    
    with torch.no_grad():
        # 고정된 노이즈로 이미지 생성
        fake_images = generator(fixed_noise[:num_images])
        
        # 정규화 해제 [-1, 1] → [0, 1]
        fake_images = (fake_images + 1) / 2
        
        # 그리드로 배치
        grid = make_grid(fake_images, nrow=8, padding=2)
        
        # 시각화
        plt.figure(figsize=(12, 12))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.title(f'생성된 얼굴 이미지 - 에포크 {epoch+1}', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.show()
        
        # IS 점수 계산
        is_score = calculate_inception_score(fake_images)
        
        return is_score

# ============================================================================
# 7. GAN 훈련 실행
# ============================================================================

print(f"\n🚀 GAN 훈련 시작")

# 훈련 기록
G_losses = []
D_losses = []
IS_scores = []

# 최고 성능 추적 (IS 점수 기준)
best_is_score = 0.0
best_generator_state = None

start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\n📅 에포크 {epoch+1}/{EPOCHS}")
    
    # 훈련
    loss_D, loss_G = train_gan_epoch(
        generator, discriminator, dataloader, criterion,
        optimizer_G, optimizer_D, device, epoch
    )
    
    # 기록 저장
    D_losses.append(loss_D)
    G_losses.append(loss_G)
    
    # 결과 출력
    print(f"   판별자 손실: {loss_D:.4f}")
    print(f"   생성자 손실: {loss_G:.4f}")
    
    # 주기적으로 이미지 생성 및 평가
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"\n🖼️  생성된 이미지 확인 (에포크 {epoch+1})")
        is_score = generate_and_save_images(generator, epoch, device)
        IS_scores.append(is_score)
        
        print(f"   Inception Score: {is_score:.4f}")
        
        # 최고 성능 모델 저장
        if is_score > best_is_score:
            best_is_score = is_score
            best_generator_state = copy.deepcopy(generator.state_dict())
            print(f"   🎯 새로운 최고 IS 점수!")
            
            # 체크포인트 저장
            save_checkpoint(
                generator, optimizer_G, epoch, loss_G, is_score,
                save_path="./checkpoints/gan_best_generator.pth"
            )
    
    # 학습 불안정성 감지
    if loss_D < 0.01 or loss_G > 10:
        print(f"   ⚠️ 학습 불안정성 감지 - D_loss: {loss_D:.4f}, G_loss: {loss_G:.4f}")

training_time = time.time() - start_time
print(f"\n✅ GAN 훈련 완료!")
print(f"   총 훈련 시간: {training_time:.2f}초")
print(f"   최고 IS 점수: {best_is_score:.4f}")

# ============================================================================
# 8. 훈련 결과 시각화
# ============================================================================

print(f"\n📈 GAN 훈련 결과 시각화")

# 손실 곡선
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(D_losses, label='판별자 손실', color='blue', linewidth=2)
plt.plot(G_losses, label='생성자 손실', color='red', linewidth=2)
plt.title('GAN 훈련 손실')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.legend()
plt.grid(True, alpha=0.3)

# IS 점수 변화
plt.subplot(1, 2, 2)
if IS_scores:
    epochs_with_is = [i * 10 for i in range(len(IS_scores))]
    plt.plot(epochs_with_is, IS_scores, 'g-', marker='o', linewidth=2, markersize=6)
    plt.title('Inception Score 변화')
    plt.xlabel('에포크')
    plt.ylabel('IS 점수')
    plt.grid(True, alpha=0.3)

plt.suptitle('GAN 훈련 과정 분석', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 9. 잠재 공간 탐색
# ============================================================================

print(f"\n🔍 잠재 공간 탐색")

def interpolate_latent_space(generator, device, num_steps=10):
    """
    잠재 공간에서의 보간 (Interpolation)
    
    두 랜덤 벡터 사이를 선형 보간하여
    생성된 이미지의 부드러운 변화 관찰
    """
    generator.eval()
    
    # 시작점과 끝점 랜덤 벡터
    z1 = torch.randn(1, LATENT_DIM, device=device)
    z2 = torch.randn(1, LATENT_DIM, device=device)
    
    interpolated_images = []
    
    with torch.no_grad():
        for i in range(num_steps):
            # 선형 보간
            alpha = i / (num_steps - 1)
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # 이미지 생성
            fake_image = generator(z_interp)
            fake_image = (fake_image + 1) / 2  # 정규화 해제
            
            interpolated_images.append(fake_image.cpu())
    
    # 시각화
    fig, axes = plt.subplots(1, num_steps, figsize=(20, 3))
    
    for i, img in enumerate(interpolated_images):
        axes[i].imshow(img.squeeze().permute(1, 2, 0))
        axes[i].set_title(f'Step {i+1}')
        axes[i].axis('off')
    
    plt.suptitle('잠재 공간 보간 - 부드러운 얼굴 변화', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def explore_latent_directions(generator, device):
    """
    잠재 공간의 의미있는 방향 탐색
    
    특정 방향으로 이동했을 때 이미지가 어떻게 변하는지 관찰
    """
    generator.eval()
    
    # 기준 잠재 벡터
    base_z = torch.randn(1, LATENT_DIM, device=device)
    
    # 랜덤 방향 벡터
    direction = torch.randn(1, LATENT_DIM, device=device)
    direction = direction / direction.norm()  # 정규화
    
    scales = [-2, -1, 0, 1, 2]  # 이동 정도
    
    with torch.no_grad():
        fig, axes = plt.subplots(1, len(scales), figsize=(15, 3))
        
        for i, scale in enumerate(scales):
            # 방향으로 이동
            z_moved = base_z + scale * direction
            
            # 이미지 생성
            fake_image = generator(z_moved)
            fake_image = (fake_image + 1) / 2
            
            axes[i].imshow(fake_image.cpu().squeeze().permute(1, 2, 0))
            axes[i].set_title(f'Scale: {scale}')
            axes[i].axis('off')
        
        plt.suptitle('잠재 공간 방향 탐색', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# 최고 성능 모델로 잠재 공간 탐색
if best_generator_state is not None:
    generator.load_state_dict(best_generator_state)

print("🔄 잠재 공간 보간:")
interpolate_latent_space(generator, device)

print("\n🧭 잠재 공간 방향 탐색:")
explore_latent_directions(generator, device)

# ============================================================================
# 10. 모드 붕괴 분석
# ============================================================================

print(f"\n🔍 모드 붕괴 (Mode Collapse) 분석")

def analyze_mode_collapse(generator, device, num_samples=100):
    """
    모드 붕괴 분석
    
    생성된 이미지들의 다양성을 측정하여
    모드 붕괴 여부 확인
    """
    generator.eval()
    
    generated_images = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, LATENT_DIM, device=device)
            fake_image = generator(z)
            fake_image = (fake_image + 1) / 2
            generated_images.append(fake_image.cpu())
    
    # 이미지들을 텐서로 결합
    all_images = torch.cat(generated_images, dim=0)
    
    # 다양성 측정
    # 1. 픽셀 값의 표준편차
    pixel_diversity = all_images.std(dim=0).mean().item()
    
    # 2. 이미지 간 평균 거리
    flattened = all_images.view(num_samples, -1)
    distances = []
    for i in range(min(50, num_samples)):  # 계산 효율성을 위해 50개만
        for j in range(i+1, min(50, num_samples)):
            dist = torch.norm(flattened[i] - flattened[j]).item()
            distances.append(dist)
    
    avg_distance = np.mean(distances) if distances else 0
    
    print(f"📊 다양성 분석:")
    print(f"   픽셀 다양성: {pixel_diversity:.4f}")
    print(f"   평균 이미지 거리: {avg_distance:.4f}")
    
    # 다양성 기준
    if pixel_diversity < 0.1:
        print(f"   ⚠️ 낮은 다양성 - 모드 붕괴 의심")
    elif pixel_diversity > 0.3:
        print(f"   ✅ 높은 다양성 - 건강한 생성")
    else:
        print(f"   📊 보통 다양성")
    
    # 샘플 이미지들 시각화
    sample_grid = make_grid(all_images[:16], nrow=4, padding=2)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(sample_grid.permute(1, 2, 0))
    plt.title(f'다양성 분석 - 생성된 샘플들\n픽셀 다양성: {pixel_diversity:.4f}', 
              fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.show()
    
    return pixel_diversity, avg_distance

diversity_score, distance_score = analyze_mode_collapse(generator, device)

# ============================================================================
# 11. 학습 내용 요약 및 실용적 활용
# ============================================================================

print(f"\n🎓 학습 내용 요약")
print(f"=" * 60)

print(f"✅ 완료한 내용:")
print(f"   1. GAN의 핵심 개념과 적대적 학습 원리")
print(f"   2. DCGAN 구조 (생성자 + 판별자) 구현")
print(f"   3. GAN 훈련의 불안정성과 해결 방법")
print(f"   4. 생성 품질 평가 (Inception Score)")
print(f"   5. 잠재 공간 탐색과 보간")
print(f"   6. 모드 붕괴 분석 및 다양성 측정")

print(f"\n📊 최종 성과:")
print(f"   - 최고 IS 점수: {best_is_score:.4f}")
print(f"   - 생성자 파라미터: {gen_params['total_params']:,}개")
print(f"   - 판별자 파라미터: {disc_params['total_params']:,}개")
print(f"   - 훈련 시간: {training_time:.2f}초")
print(f"   - 다양성 점수: {diversity_score:.4f}")

print(f"\n💡 핵심 학습 포인트:")
print(f"   1. 적대적 학습: 두 네트워크의 경쟁을 통한 학습")
print(f"   2. 생성자: 노이즈에서 현실적인 이미지 생성")
print(f"   3. 판별자: 진짜와 가짜 이미지 구별")
print(f"   4. 균형의 중요성: 생성자와 판별자의 적절한 균형")
print(f"   5. 잠재 공간: 의미있는 표현 학습")

print(f"\n🔍 GAN의 장단점:")
print(f"   장점:")
print(f"   - 고품질 이미지 생성 가능")
print(f"   - 명시적 확률 모델 불필요")
print(f"   - 다양한 응용 분야")
print(f"   - 창의적 콘텐츠 생성")
print(f"   단점:")
print(f"   - 훈련 불안정성")
print(f"   - 모드 붕괴 문제")
print(f"   - 평가 메트릭의 어려움")
print(f"   - 하이퍼파라미터 민감성")

print(f"\n🚀 실용적 활용 분야:")
print(f"   1. 예술 및 디자인: 창작 도구, 스타일 변환")
print(f"   2. 엔터테인먼트: 게임 캐릭터, 영화 특수효과")
print(f"   3. 패션: 의상 디자인, 가상 모델")
print(f"   4. 데이터 증강: 훈련 데이터 부족 해결")
print(f"   5. 의료: 의료 영상 생성, 프라이버시 보호")
print(f"   6. 보안: 딥페이크 탐지, 생체인식 강화")

print(f"\n🔧 GAN 개선 기법:")
print(f"   1. WGAN: Wasserstein 거리로 안정적 학습")
print(f"   2. Progressive GAN: 점진적 해상도 증가")
print(f"   3. StyleGAN: 스타일 기반 생성")
print(f"   4. BigGAN: 대규모 고해상도 생성")
print(f"   5. Conditional GAN: 조건부 생성")

print(f"\n🚀 다음 단계:")
print(f"   - 08_transformer_nlp.py: Transformer로 자연어 처리")
print(f"   - 어텐션 메커니즘과 현대 NLP의 핵심")
print(f"   - 기계 번역과 언어 모델링")

print(f"\n🔧 추가 실험 아이디어:")
print(f"   1. 조건부 GAN: 특정 속성을 가진 얼굴 생성")
print(f"   2. CycleGAN: 도메인 간 변환")
print(f"   3. StyleGAN: 고품질 얼굴 생성")
print(f"   4. 실제 CelebA 데이터셋 사용")
print(f"   5. FID 점수로 더 정확한 평가")

print(f"\n" + "=" * 60)
print(f"🎉 GAN 이미지 생성 튜토리얼 완료!")
print(f"   다음 튜토리얼에서 Transformer NLP를 배워보세요!")
print(f"=" * 60)
import os
from PIL import Image
import requests
import zipfile
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

# 우리가 만든 유틸리티 함수들 임포트
from utils.data_utils import download_and_extract
from utils.visualization import plot_training_curves
from utils.model_utils import count_parameters, save_checkpoint

print("🚀 딥러닝 강의 시리즈 7: GAN 이미지 생성")
print("=" * 60)#
 ============================================================================
# 1. 환경 설정 및 하이퍼파라미터
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  사용 장치: {device}")

# GAN 훈련을 위한 하이퍼파라미터
BATCH_SIZE = 64        # GAN은 큰 배치가 안정적 학습에 도움
LEARNING_RATE_G = 0.0002  # 생성자 학습률
LEARNING_RATE_D = 0.0002  # 판별자 학습률
BETA1 = 0.5            # Adam 옵티마이저 베타1 (GAN에 최적화)
EPOCHS = 100           # GAN은 충분한 학습 시간 필요
RANDOM_SEED = 42
LATENT_DIM = 100       # 잠재 벡터 차원
IMG_SIZE = 64          # 생성할 이미지 크기
NUM_CHANNELS = 3       # RGB 채널
SAMPLE_INTERVAL = 500  # 샘플 생성 간격

# 재현성을 위한 시드 설정
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

print(f"📊 하이퍼파라미터:")
print(f"   배치 크기: {BATCH_SIZE}")
print(f"   생성자 학습률: {LEARNING_RATE_G}")
print(f"   판별자 학습률: {LEARNING_RATE_D}")
print(f"   에포크: {EPOCHS}")
print(f"   잠재 차원: {LATENT_DIM}")
print(f"   이미지 크기: {IMG_SIZE}x{IMG_SIZE}")

# ============================================================================
# 2. CelebA 데이터셋 준비
# ============================================================================

print(f"\n📁 CelebA 데이터셋 준비")

class CelebADataset(Dataset):
    """
    CelebA 데이터셋 클래스
    
    CelebA는 대용량 데이터셋이므로 실제 프로젝트에서는
    공식 다운로드 링크를 사용해야 합니다.
    여기서는 교육 목적으로 샘플 데이터를 생성합니다.
    """
    
    def __init__(self, root_dir, transform=None, use_sample_data=True):
        self.root_dir = root_dir
        self.transform = transform
        self.use_sample_data = use_sample_data
        
        if use_sample_data:
            # 교육용 샘플 데이터 생성
            self.create_sample_data()
        else:
            # 실제 CelebA 데이터 로드
            self.load_celeba_data()
    
    def create_sample_data(self):
        """
        교육용 샘플 얼굴 데이터 생성
        
        실제 CelebA 대신 합성된 얼굴 패턴을 생성합니다.
        실제 프로젝트에서는 진짜 CelebA 데이터를 사용하세요.
        """
        print("🎭 교육용 샘플 얼굴 데이터 생성 중...")
        
        # 샘플 데이터 디렉토리 생성
        os.makedirs(self.root_dir, exist_ok=True)
        
        # 다양한 얼굴 패턴 생성
        self.sample_images = []
        
        for i in range(1000):  # 1000개 샘플 생성
            # 기본 얼굴 형태 생성
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            
            # 얼굴 윤곽 (타원)
            center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
            
            # 피부색 (다양한 톤)
            skin_color = np.random.randint(180, 255, 3)
            
            # 얼굴 영역 채우기
            y, x = np.ogrid[:IMG_SIZE, :IMG_SIZE]
            mask = ((x - center_x) / 20) ** 2 + ((y - center_y) / 25) ** 2 <= 1
            img[mask] = skin_color
            
            # 눈 (검은색 타원)
            eye_y = center_y - 8
            left_eye_x, right_eye_x = center_x - 8, center_x + 8
            
            # 왼쪽 눈
            eye_mask_l = ((x - left_eye_x) / 3) ** 2 + ((y - eye_y) / 2) ** 2 <= 1
            img[eye_mask_l] = [0, 0, 0]
            
            # 오른쪽 눈
            eye_mask_r = ((x - right_eye_x) / 3) ** 2 + ((y - eye_y) / 2) ** 2 <= 1
            img[eye_mask_r] = [0, 0, 0]
            
            # 입 (빨간색)
            mouth_y = center_y + 8
            mouth_mask = ((x - center_x) / 6) ** 2 + ((y - mouth_y) / 2) ** 2 <= 1
            img[mouth_mask] = [200, 50, 50]
            
            # 노이즈 추가 (자연스러움)
            noise = np.random.normal(0, 10, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            self.sample_images.append(img)
        
        print(f"✅ {len(self.sample_images)}개 샘플 얼굴 생성 완료")
    
    def load_celeba_data(self):
        """
        실제 CelebA 데이터 로드 (실제 사용 시)
        """
        # 실제 구현에서는 CelebA 데이터 로드
        # 여기서는 샘플 데이터 사용
        self.create_sample_data()
    
    def __len__(self):
        return len(self.sample_images)
    
    def __getitem__(self, idx):
        image = self.sample_images[idx]
        
        # PIL Image로 변환
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image

# 데이터 전처리 파이프라인
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] 범위로 정규화
])

# 데이터셋 생성
dataset = CelebADataset(
    root_dir='./data/celeba_sample',
    transform=transform,
    use_sample_data=True
)

# 데이터 로더 생성
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    drop_last=True  # 마지막 배치 크기가 다르면 제거
)

print(f"✅ 데이터 로더 생성 완료:")
print(f"   총 샘플 수: {len(dataset):,}개")
print(f"   배치 수: {len(dataloader)}")

# 샘플 이미지 시각화
print(f"\n🖼️  샘플 데이터 시각화")

sample_batch = next(iter(dataloader))
sample_grid = make_grid(sample_batch[:16], nrow=4, normalize=True, padding=2)

plt.figure(figsize=(10, 10))
plt.imshow(sample_grid.permute(1, 2, 0))
plt.title("CelebA 샘플 데이터 (교육용)")
plt.axis('off')
plt.show()

# ============================================================================
# 3. GAN 모델 정의
# ============================================================================

print(f"\n🧠 GAN 모델 정의")

class Generator(nn.Module):
    """
    DCGAN 스타일 생성자
    
    구조:
    - 입력: 잠재 벡터 z (latent_dim,)
    - 출력: 이미지 (3, 64, 64)
    - 전치 합성곱(Transposed Convolution)으로 업샘플링
    - 배치 정규화와 ReLU 활성화 함수 사용
    
    왜 이런 구조인가?
    1. 전치 합성곱: 저해상도 → 고해상도 변환
    2. 배치 정규화: 안정적인 학습
    3. ReLU: 양수 활성화로 자연스러운 이미지
    4. Tanh 출력: [-1, 1] 범위 (정규화된 이미지와 매칭)
    """
    
    def __init__(self, latent_dim=100, img_channels=3, feature_dim=64):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # 초기 완전연결층 (잠재 벡터 → 특성 맵)
        self.fc = nn.Linear(latent_dim, feature_dim * 8 * 4 * 4)
        
        # 전치 합성곱 레이어들 (업샘플링)
        self.conv_layers = nn.Sequential(
            # 4x4 → 8x8
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(True),
            
            # 8x8 → 16x16  
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(True),
            
            # 16x16 → 32x32
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(True),
            
            # 32x32 → 64x64
            nn.ConvTranspose2d(feature_dim, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()  # [-1, 1] 범위 출력
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        가중치 초기화 (DCGAN 논문 기준)
        
        왜 가중치 초기화가 중요한가?
        1. 안정적인 학습: 적절한 초기값으로 수렴 보장
        2. 그래디언트 흐름: 너무 크거나 작은 값 방지
        3. 대칭성 깨기: 모든 뉴런이 같은 값으로 시작하면 안됨
        """
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
    
    def forward(self, z):
        # 완전연결층
        x = self.fc(z)
        
        # 4D 텐서로 reshape
        x = x.view(x.size(0), -1, 4, 4)
        
        # 전치 합성곱으로 업샘플링
        x = self.conv_layers(x)
        
        return x

class Discriminator(nn.Module):
    """
    DCGAN 스타일 판별자
    
    구조:
    - 입력: 이미지 (3, 64, 64)
    - 출력: 실제/가짜 확률 (1,)
    - 합성곱으로 다운샘플링
    - LeakyReLU 활성화 함수 사용
    
    왜 이런 구조인가?
    1. 합성곱: 이미지의 공간적 특징 추출
    2. LeakyReLU: 음수 그래디언트도 일부 전달
    3. 배치 정규화: 안정적인 학습 (첫 층 제외)
    4. 시그모이드 출력: [0, 1] 확률값
    """
    
    def __init__(self, img_channels=3, feature_dim=64):
        super(Discriminator, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 64x64 → 32x32
            nn.Conv2d(img_channels, feature_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 32x32 → 16x16
            nn.Conv2d(feature_dim, feature_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 16x16 → 8x8
            nn.Conv2d(feature_dim * 2, feature_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 8x8 → 4x4
            nn.Conv2d(feature_dim * 4, feature_dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 4x4 → 1x1
            nn.Conv2d(feature_dim * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)
    
    def forward(self, x):
        x = self.conv_layers(x)
        return x.view(x.size(0), -1)  # (batch_size, 1)

# 모델 인스턴스 생성
generator = Generator(
    latent_dim=LATENT_DIM,
    img_channels=NUM_CHANNELS,
    feature_dim=64
).to(device)

discriminator = Discriminator(
    img_channels=NUM_CHANNELS,
    feature_dim=64
).to(device)

print(f"✅ GAN 모델 생성 완료")

# 모델 구조 출력
print(f"\n📋 생성자 구조:")
print(generator)

print(f"\n📋 판별자 구조:")
print(discriminator)

# 모델 복잡도 분석
print(f"\n📊 모델 복잡도:")
gen_params = count_parameters(generator, detailed=False)
disc_params = count_parameters(discriminator, detailed=False)

print(f"   생성자: {gen_params['total_params']:,}개 파라미터")
print(f"   판별자: {disc_params['total_params']:,}개 파라미터")
print(f"   총합: {gen_params['total_params'] + disc_params['total_params']:,}개 파라미터")

# ============================================================================
# 4. 손실 함수와 옵티마이저 설정
# ============================================================================

print(f"\n⚙️  손실 함수와 옵티마이저 설정")

# GAN 손실 함수
# 왜 Binary Cross Entropy인가?
# 1. 이진 분류: 실제(1) vs 가짜(0) 구분
# 2. 확률 출력: 시그모이드 출력과 매칭
# 3. 안정적 학습: 잘 정의된 그래디언트
criterion = nn.BCELoss()

# 옵티마이저 (생성자와 판별자 각각)
# 왜 Adam을 사용하는가?
# 1. 적응적 학습률: 각 파라미터별 최적화
# 2. 모멘텀 효과: 진동 감소
# 3. GAN에서 검증된 성능
optimizer_G = optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE_G,
    betas=(BETA1, 0.999)  # DCGAN 논문 권장값
)

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE_D,
    betas=(BETA1, 0.999)
)

print(f"   손실 함수: {criterion.__class__.__name__}")
print(f"   생성자 옵티마이저: Adam (lr={LEARNING_RATE_G}, beta1={BETA1})")
print(f"   판별자 옵티마이저: Adam (lr={LEARNING_RATE_D}, beta1={BETA1})")

# 고정된 잠재 벡터 (생성 과정 시각화용)
fixed_noise = torch.randn(16, LATENT_DIM, device=device)

print(f"\n🎲 고정 잠재 벡터 생성: {fixed_noise.shape}")

# ============================================================================
# 5. GAN 훈련 함수
# ============================================================================

def train_gan_epoch(generator, discriminator, dataloader, optimizer_G, optimizer_D, 
                   criterion, device, epoch):
    """
    GAN 한 에포크 훈련
    
    GAN 훈련 과정:
    1. 판별자 훈련: 실제 이미지는 1, 가짜 이미지는 0으로 분류
    2. 생성자 훈련: 판별자가 가짜 이미지를 1로 분류하도록 속임
    
    Args:
        generator: 생성자 모델
        discriminator: 판별자 모델
        dataloader: 데이터 로더
        optimizer_G: 생성자 옵티마이저
        optimizer_D: 판별자 옵티마이저
        criterion: 손실 함수
        device: 연산 장치
        epoch: 현재 에포크
    
    Returns:
        tuple: (생성자 손실, 판별자 손실)
    """
    
    generator.train()
    discriminator.train()
    
    running_loss_G = 0.0
    running_loss_D = 0.0
    num_batches = 0
    
    # 실제/가짜 라벨
    real_label = 1.0
    fake_label = 0.0
    
    pbar = tqdm(dataloader, desc=f"에포크 {epoch+1}")
    
    for batch_idx, real_images in enumerate(pbar):
        batch_size = real_images.size(0)
        real_images = real_images.to(device)
        
        # ====================================================================
        # 1. 판별자 훈련
        # ====================================================================
        
        optimizer_D.zero_grad()
        
        # 1-1. 실제 이미지에 대한 판별자 손실
        real_labels = torch.full((batch_size, 1), real_label, device=device)
        real_output = discriminator(real_images)
        loss_D_real = criterion(real_output, real_labels)
        
        # 1-2. 가짜 이미지에 대한 판별자 손실
        noise = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_images = generator(noise)
        fake_labels = torch.full((batch_size, 1), fake_label, device=device)
        fake_output = discriminator(fake_images.detach())  # detach로 생성자 그래디언트 차단
        loss_D_fake = criterion(fake_output, fake_labels)
        
        # 1-3. 판별자 총 손실
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()
        
        # ====================================================================
        # 2. 생성자 훈련
        # ====================================================================
        
        optimizer_G.zero_grad()
        
        # 생성자는 판별자가 가짜 이미지를 실제로 분류하도록 속임
        fake_output = discriminator(fake_images)
        loss_G = criterion(fake_output, real_labels)  # 가짜를 실제로 속이려 함
        loss_G.backward()
        optimizer_G.step()
        
        # 통계 업데이트
        running_loss_G += loss_G.item()
        running_loss_D += loss_D.item()
        num_batches += 1
        
        # 진행률 바 업데이트
        pbar.set_postfix({
            'D_Loss': f'{loss_D.item():.4f}',
            'G_Loss': f'{loss_G.item():.4f}',
            'D_Real': f'{real_output.mean().item():.3f}',
            'D_Fake': f'{fake_output.mean().item():.3f}'
        })
        
        # 주기적으로 생성 이미지 저장
        if batch_idx % SAMPLE_INTERVAL == 0:
            save_generated_samples(generator, fixed_noise, epoch, batch_idx)
    
    avg_loss_G = running_loss_G / num_batches
    avg_loss_D = running_loss_D / num_batches
    
    return avg_loss_G, avg_loss_D

def save_generated_samples(generator, fixed_noise, epoch, batch_idx):
    """생성된 샘플 이미지 저장"""
    generator.eval()
    
    with torch.no_grad():
        fake_images = generator(fixed_noise)
        
        # 이미지 그리드 생성
        grid = make_grid(fake_images, nrow=4, normalize=True, padding=2)
        
        # 저장 디렉토리 생성
        os.makedirs('./generated_samples', exist_ok=True)
        
        # 이미지 저장
        save_image(grid, f'./generated_samples/epoch_{epoch+1}_batch_{batch_idx}.png')
    
    generator.train()

# ============================================================================
# 6. GAN 훈련 실행
# ============================================================================

print(f"\n🚀 GAN 훈련 시작")

# 훈련 기록
G_losses = []
D_losses = []

# 최고 성능 추적 (생성자 기준)
best_G_loss = float('inf')
best_generator_state = None

start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\n📅 에포크 {epoch+1}/{EPOCHS}")
    
    # 한 에포크 훈련
    loss_G, loss_D = train_gan_epoch(
        generator, discriminator, dataloader,
        optimizer_G, optimizer_D, criterion, device, epoch
    )
    
    # 기록 저장
    G_losses.append(loss_G)
    D_losses.append(loss_D)
    
    # 결과 출력
    print(f"   생성자 손실: {loss_G:.4f}")
    print(f"   판별자 손실: {loss_D:.4f}")
    
    # 최고 성능 모델 저장
    if loss_G < best_G_loss:
        best_G_loss = loss_G
        best_generator_state = copy.deepcopy(generator.state_dict())
        print(f"   🎯 새로운 최고 성능! 생성자 손실: {loss_G:.4f}")
        
        # 체크포인트 저장
        save_checkpoint(
            generator, optimizer_G, epoch, loss_G, 0,
            save_path="./checkpoints/gan_generator_best.pth"
        )
    
    # 주기적으로 생성 결과 시각화
    if (epoch + 1) % 10 == 0:
        visualize_generation_progress(generator, fixed_noise, epoch)

training_time = time.time() - start_time
print(f"\n✅ 훈련 완료!")
print(f"   총 훈련 시간: {training_time:.2f}초")
print(f"   최고 생성자 손실: {best_G_loss:.4f}")

# ============================================================================
# 7. 훈련 결과 시각화
# ============================================================================

print(f"\n📈 훈련 결과 시각화")

# 손실 곡선
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(G_losses, label='생성자 손실', color='blue')
plt.plot(D_losses, label='판별자 손실', color='red')
plt.xlabel('에포크')
plt.ylabel('손실')
plt.title('GAN 훈련 손실')
plt.legend()
plt.grid(True, alpha=0.3)

# 손실 비율
plt.subplot(1, 2, 2)
loss_ratio = np.array(G_losses) / np.array(D_losses)
plt.plot(loss_ratio, color='green')
plt.xlabel('에포크')
plt.ylabel('생성자 손실 / 판별자 손실')
plt.title('손실 비율 (균형 지표)')
plt.grid(True, alpha=0.3)
plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='균형점')
plt.legend()

plt.tight_layout()
plt.show()

def visualize_generation_progress(generator, fixed_noise, epoch):
    """생성 과정 시각화"""
    generator.eval()
    
    with torch.no_grad():
        fake_images = generator(fixed_noise)
        
        # 이미지 그리드 생성
        grid = make_grid(fake_images, nrow=4, normalize=True, padding=2)
        
        # 시각화
        plt.figure(figsize=(10, 10))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title(f'생성된 얼굴 이미지 - 에포크 {epoch+1}')
        plt.axis('off')
        plt.show()
    
    generator.train()

# 최종 생성 결과
print(f"\n🎨 최종 생성 결과")

# 최고 성능 모델 로드
if best_generator_state is not None:
    generator.load_state_dict(best_generator_state)

visualize_generation_progress(generator, fixed_noise, EPOCHS-1)# =
===========================================================================
# 8. 잠재 공간 탐색
# ============================================================================

print(f"\n🌌 잠재 공간 탐색")

def explore_latent_space(generator, device, num_samples=8):
    """
    잠재 공간 탐색 및 시각화
    
    잠재 공간의 특성:
    1. 연속성: 비슷한 벡터는 비슷한 이미지 생성
    2. 의미적 구조: 특정 방향이 특정 속성 변화
    3. 보간 가능성: 두 점 사이 보간으로 자연스러운 변화
    """
    
    generator.eval()
    
    print(f"🎲 1. 랜덤 샘플 생성")
    
    # 랜덤 잠재 벡터 생성
    random_noise = torch.randn(num_samples, LATENT_DIM, device=device)
    
    with torch.no_grad():
        random_images = generator(random_noise)
        
        # 시각화
        grid = make_grid(random_images, nrow=4, normalize=True, padding=2)
        plt.figure(figsize=(10, 6))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title('랜덤 잠재 벡터로 생성된 얼굴들')
        plt.axis('off')
        plt.show()
    
    print(f"🔄 2. 잠재 공간 보간 (Interpolation)")
    
    # 두 랜덤 점 선택
    z1 = torch.randn(1, LATENT_DIM, device=device)
    z2 = torch.randn(1, LATENT_DIM, device=device)
    
    # 보간 계수
    alphas = torch.linspace(0, 1, 8, device=device)
    
    interpolated_images = []
    
    with torch.no_grad():
        for alpha in alphas:
            # 선형 보간
            z_interp = (1 - alpha) * z1 + alpha * z2
            img = generator(z_interp)
            interpolated_images.append(img)
        
        # 보간 결과 시각화
        interpolated_tensor = torch.cat(interpolated_images, dim=0)
        grid = make_grid(interpolated_tensor, nrow=8, normalize=True, padding=2)
        
        plt.figure(figsize=(16, 4))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title('잠재 공간 보간: 왼쪽에서 오른쪽으로 부드러운 변화')
        plt.axis('off')
        plt.show()
    
    print(f"🎯 3. 특정 방향 탐색")
    
    # 기준점 선택
    base_z = torch.randn(1, LATENT_DIM, device=device)
    
    # 랜덤 방향 벡터
    direction = torch.randn(1, LATENT_DIM, device=device)
    direction = direction / direction.norm()  # 정규화
    
    # 방향을 따라 이동
    scales = torch.linspace(-3, 3, 7, device=device)
    
    direction_images = []
    
    with torch.no_grad():
        for scale in scales:
            z_moved = base_z + scale * direction
            img = generator(z_moved)
            direction_images.append(img)
        
        # 방향 탐색 결과 시각화
        direction_tensor = torch.cat(direction_images, dim=0)
        grid = make_grid(direction_tensor, nrow=7, normalize=True, padding=2)
        
        plt.figure(figsize=(14, 4))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title('특정 방향을 따른 잠재 공간 탐색')
        plt.axis('off')
        plt.show()
    
    generator.train()

explore_latent_space(generator, device)

# ============================================================================
# 9. GAN 품질 평가
# ============================================================================

print(f"\n📊 GAN 생성 품질 평가")

def calculate_inception_score(generator, device, num_samples=1000, batch_size=32):
    """
    Inception Score (IS) 계산
    
    IS는 생성된 이미지의 품질을 평가하는 메트릭:
    1. 다양성: 생성된 이미지들이 다양한 클래스에 분포
    2. 선명도: 각 이미지가 특정 클래스에 명확히 분류
    
    높은 IS = 다양하면서도 선명한 이미지 생성
    """
    
    print("🎯 Inception Score 계산 중...")
    
    try:
        # 사전 훈련된 Inception 모델 로드
        from torchvision.models import inception_v3
        inception_model = inception_v3(pretrained=True, transform_input=False)
        inception_model.eval()
        inception_model.to(device)
        
        generator.eval()
        
        # 생성된 이미지들의 예측 확률 수집
        all_preds = []
        
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                current_batch_size = min(batch_size, num_samples - i)
                
                # 잠재 벡터 생성
                z = torch.randn(current_batch_size, LATENT_DIM, device=device)
                
                # 이미지 생성
                fake_images = generator(z)
                
                # Inception 모델 입력 크기로 리사이즈 (299x299)
                fake_images_resized = F.interpolate(
                    fake_images, size=(299, 299), mode='bilinear', align_corners=False
                )
                
                # 예측 확률 계산
                pred = inception_model(fake_images_resized)
                pred = F.softmax(pred, dim=1)
                
                all_preds.append(pred.cpu())
        
        # 전체 예측 확률 결합
        all_preds = torch.cat(all_preds, dim=0)
        
        # IS 계산
        # IS = exp(E[KL(p(y|x) || p(y))])
        py = all_preds.mean(dim=0)  # p(y)
        kl_div = all_preds * (torch.log(all_preds) - torch.log(py))
        kl_div = kl_div.sum(dim=1)
        is_score = torch.exp(kl_div.mean()).item()
        
        print(f"✅ Inception Score: {is_score:.2f}")
        
        generator.train()
        return is_score
        
    except Exception as e:
        print(f"❌ IS 계산 실패: {e}")
        print("💡 torchvision 버전을 확인하거나 인터넷 연결을 확인하세요.")
        return None

def analyze_mode_collapse(generator, device, num_samples=100):
    """
    모드 붕괴(Mode Collapse) 분석
    
    모드 붕괴: 생성자가 다양성을 잃고 비슷한 이미지만 생성하는 현상
    
    분석 방법:
    1. 생성된 이미지들 간의 유사도 계산
    2. 평균 유사도가 높으면 모드 붕괴 의심
    """
    
    print("🔍 모드 붕괴 분석 중...")
    
    generator.eval()
    
    # 여러 이미지 생성
    generated_images = []
    
    with torch.no_grad():
        for i in range(num_samples):
            z = torch.randn(1, LATENT_DIM, device=device)
            img = generator(z)
            generated_images.append(img.cpu())
    
    # 이미지들을 평탄화하여 벡터로 변환
    flattened_images = [img.view(-1) for img in generated_images]
    
    # 코사인 유사도 계산
    similarities = []
    
    for i in range(len(flattened_images)):
        for j in range(i+1, len(flattened_images)):
            img1, img2 = flattened_images[i], flattened_images[j]
            
            # 코사인 유사도
            similarity = F.cosine_similarity(img1.unsqueeze(0), img2.unsqueeze(0))
            similarities.append(similarity.item())
    
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    
    print(f"📊 모드 붕괴 분석 결과:")
    print(f"   평균 유사도: {avg_similarity:.4f}")
    print(f"   유사도 표준편차: {std_similarity:.4f}")
    
    if avg_similarity > 0.8:
        print(f"   ⚠️  높은 유사도 - 모드 붕괴 의심")
        print(f"      해결 방법: 학습률 조정, 정규화 기법 적용")
    elif avg_similarity < 0.3:
        print(f"   ✅ 낮은 유사도 - 다양한 이미지 생성")
    else:
        print(f"   📊 보통 수준의 다양성")
    
    # 유사도 분포 시각화
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(avg_similarity, color='red', linestyle='--', 
                label=f'평균: {avg_similarity:.3f}')
    plt.xlabel('코사인 유사도')
    plt.ylabel('빈도')
    plt.title('생성된 이미지들 간의 유사도 분포')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    generator.train()
    
    return avg_similarity, std_similarity

# 품질 평가 실행
is_score = calculate_inception_score(generator, device, num_samples=500)
similarity_stats = analyze_mode_collapse(generator, device, num_samples=50)

# ============================================================================
# 10. GAN 개선 기법 소개
# ============================================================================

print(f"\n🚀 GAN 개선 기법 소개")

def explain_gan_improvements():
    """
    GAN의 주요 개선 기법들 설명
    """
    
    print(f"\n📚 1. 훈련 안정화 기법")
    print(f"   🎯 Spectral Normalization:")
    print(f"      - 판별자의 립시츠 상수 제한")
    print(f"      - 그래디언트 폭발 방지")
    print(f"      - 더 안정적인 훈련")
    
    print(f"\n   🎯 Progressive Growing:")
    print(f"      - 낮은 해상도부터 시작하여 점진적 증가")
    print(f"      - 고해상도 이미지 생성 가능")
    print(f"      - 안정적인 학습 과정")
    
    print(f"\n   🎯 Self-Attention:")
    print(f"      - 장거리 의존성 모델링")
    print(f"      - 더 일관된 이미지 생성")
    print(f"      - 세부 디테일 향상")
    
    print(f"\n📚 2. 손실 함수 개선")
    print(f"   🎯 Wasserstein GAN (WGAN):")
    print(f"      - Earth Mover's Distance 사용")
    print(f"      - 더 의미있는 손실값")
    print(f"      - 모드 붕괴 완화")
    
    print(f"\n   🎯 Least Squares GAN (LSGAN):")
    print(f"      - MSE 손실 사용")
    print(f"      - 그래디언트 소실 완화")
    print(f"      - 더 안정적인 훈련")
    
    print(f"\n📚 3. 아키텍처 개선")
    print(f"   🎯 StyleGAN:")
    print(f"      - 스타일 기반 생성")
    print(f"      - 세밀한 제어 가능")
    print(f"      - 최고 품질의 얼굴 생성")
    
    print(f"\n   🎯 BigGAN:")
    print(f"      - 대규모 모델과 데이터")
    print(f"      - 클래스 조건부 생성")
    print(f"      - ImageNet 수준 품질")
    
    print(f"\n📚 4. 조건부 생성")
    print(f"   🎯 Conditional GAN (cGAN):")
    print(f"      - 라벨 조건부 생성")
    print(f"      - 원하는 클래스 이미지 생성")
    print(f"      - 제어 가능한 생성")
    
    print(f"\n   🎯 Pix2Pix:")
    print(f"      - 이미지 간 변환")
    print(f"      - 스케치 → 사진 변환")
    print(f"      - 다양한 응용 가능")

explain_gan_improvements()

# ============================================================================
# 11. 학습 내용 요약 및 다음 단계
# ============================================================================

print(f"\n🎓 학습 내용 요약")
print(f"=" * 60)

print(f"✅ 완료한 내용:")
print(f"   1. GAN의 핵심 개념과 적대적 학습 원리")
print(f"   2. DCGAN 구조로 얼굴 이미지 생성")
print(f"   3. 생성자와 판별자의 균형잡힌 훈련")
print(f"   4. 잠재 공간 탐색과 보간 기법")
print(f"   5. 생성 품질 평가 (IS, 모드 붕괴 분석)")
print(f"   6. GAN 개선 기법들 학습")

print(f"\n📊 최종 성과:")
if is_score:
    print(f"   - Inception Score: {is_score:.2f}")
if similarity_stats:
    print(f"   - 평균 이미지 유사도: {similarity_stats[0]:.4f}")
print(f"   - 생성자 파라미터: {gen_params['total_params']:,}개")
print(f"   - 판별자 파라미터: {disc_params['total_params']:,}개")
print(f"   - 훈련 시간: {training_time:.2f}초")

print(f"\n💡 핵심 학습 포인트:")
print(f"   1. 적대적 학습: 두 네트워크의 경쟁을 통한 개선")
print(f"   2. 생성 모델링: 데이터 분포 학습과 샘플링")
print(f"   3. 잠재 공간: 연속적이고 의미있는 표현 공간")
print(f"   4. 훈련 균형: 생성자와 판별자의 적절한 균형")
print(f"   5. 품질 평가: 정량적 메트릭과 정성적 평가")

print(f"\n🔍 GAN의 장단점:")
print(f"   장점:")
print(f"   - 선명한 이미지 생성 (VAE 대비)")
print(f"   - 다양한 응용 가능 (이미지 변환, 스타일 전이)")
print(f"   - 창작 도구로서의 가치")
print(f"   - 데이터 증강 효과")
print(f"   단점:")
print(f"   - 불안정한 훈련 (모드 붕괴, 그래디언트 소실)")
print(f"   - 평가의 어려움 (정량적 메트릭 부족)")
print(f"   - 하이퍼파라미터 민감성")

print(f"\n🚀 다음 단계:")
print(f"   - 08_transformer_nlp.py: Transformer로 자연어 처리")
print(f"   - Multi30k 데이터셋으로 기계 번역")
print(f"   - 어텐션 메커니즘과 셀프 어텐션")

print(f"\n🔧 추가 실험 아이디어:")
print(f"   1. 조건부 GAN: 특정 속성을 가진 얼굴 생성")
print(f"   2. StyleGAN: 스타일 기반 고품질 생성")
print(f"   3. CycleGAN: 도메인 간 이미지 변환")
print(f"   4. Pix2Pix: 스케치에서 사진으로 변환")
print(f"   5. 3D GAN: 3차원 객체 생성")

print(f"\n🎯 실제 응용 분야:")
print(f"   - 엔터테인먼트: 가상 캐릭터, 게임 에셋 생성")
print(f"   - 패션: 의류 디자인, 가상 피팅")
print(f"   - 의료: 의료 영상 데이터 증강")
print(f"   - 예술: 디지털 아트, 창작 도구")
print(f"   - 데이터 프라이버시: 합성 데이터 생성")

print(f"\n" + "=" * 60)
print(f"🎉 GAN 이미지 생성 튜토리얼 완료!")
print(f"   다음 튜토리얼에서 Transformer를 배워보세요!")
print(f"=" * 60)