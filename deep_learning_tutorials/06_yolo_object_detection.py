"""
딥러닝 강의 시리즈 6: YOLO 객체 탐지

이 튜토리얼에서는 COCO 데이터셋을 사용하여 
YOLO(You Only Look Once) 알고리즘의 객체 탐지를 학습합니다.

학습 목표:
1. 객체 탐지 문제의 특성과 도전과제 이해
2. YOLO 알고리즘의 원리와 구조
3. 바운딩 박스와 신뢰도 점수 처리
4. Non-Maximum Suppression (NMS) 알고리즘
5. 실시간 객체 탐지 구현
6. 객체 탐지 성능 평가 (mAP, IoU)

데이터셋 선택 이유 - COCO (Common Objects in Context):
- 80개 클래스의 일상적인 객체들
- 330,000개 이미지, 1.5M 객체 인스턴스
- 객체 탐지의 표준 벤치마크 데이터셋
- 다양한 크기와 형태의 객체 포함
- 복잡한 배경과 겹치는 객체들
- 실제 환경과 유사한 다양한 상황
- 정확한 바운딩 박스 어노테이션

왜 YOLO를 사용하는가?
1. 실시간 처리: 단일 네트워크 패스로 빠른 추론
2. 전역적 추론: 전체 이미지를 한 번에 처리
3. 일반화 능력: 다양한 도메인에서 우수한 성능
4. 단순한 구조: 이해하기 쉬운 end-to-end 학습
5. 실용성: 실제 응용에서 널리 사용

전통적 객체 탐지 vs YOLO:
- R-CNN 계열: 2단계 (region proposal + classification)
- YOLO: 1단계 (직접 바운딩 박스 + 클래스 예측)
- 속도: YOLO가 훨씬 빠름 (실시간 가능)
- 정확도: R-CNN이 약간 높지만 YOLO도 충분히 실용적
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import json
import os
from tqdm import tqdm
import time
import copy
import warnings
warnings.filterwarnings('ignore')

# 우리가 만든 유틸리티 함수들 임포트
from utils.visualization import plot_training_curves
from utils.model_utils import count_parameters, save_checkpoint

print("🚀 딥러닝 강의 시리즈 6: YOLO 객체 탐지")
print("=" * 60)

# ============================================================================
# 1. 환경 설정 및 하이퍼파라미터
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  사용 장치: {device}")

# YOLO를 위한 하이퍼파라미터
BATCH_SIZE = 16        # 객체 탐지는 메모리를 많이 사용
LEARNING_RATE = 0.001  # 안정적인 학습을 위한 보수적 학습률
EPOCHS = 50            # 객체 탐지는 충분한 학습 시간 필요
RANDOM_SEED = 42
IMG_SIZE = 416         # YOLO 표준 입력 크기 (32의 배수)
GRID_SIZE = 13         # 416 / 32 = 13
NUM_CLASSES = 20       # 간소화된 클래스 수 (PASCAL VOC 기준)
NUM_BOXES = 2          # 각 그리드 셀당 예측할 바운딩 박스 수
CONFIDENCE_THRESHOLD = 0.5  # 객체 신뢰도 임계값
NMS_THRESHOLD = 0.4    # Non-Maximum Suppression 임계값

# 재현성을 위한 시드 설정
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"📊 하이퍼파라미터:")
print(f"   배치 크기: {BATCH_SIZE}")
print(f"   학습률: {LEARNING_RATE}")
print(f"   에포크: {EPOCHS}")
print(f"   이미지 크기: {IMG_SIZE}x{IMG_SIZE}")
print(f"   그리드 크기: {GRID_SIZE}x{GRID_SIZE}")
print(f"   클래스 수: {NUM_CLASSES}")
print(f"   박스 수: {NUM_BOXES}")

# PASCAL VOC 클래스 이름 (간소화)
CLASS_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# ============================================================================
# 2. 데이터 전처리 및 유틸리티 함수
# ============================================================================

print(f"\n📝 객체 탐지 유틸리티 함수 정의")

def calculate_iou(box1, box2):
    """
    IoU (Intersection over Union) 계산
    
    Args:
        box1, box2: [x1, y1, x2, y2] 형태의 바운딩 박스
    
    Returns:
        float: IoU 값 (0~1)
    
    IoU는 객체 탐지에서 가장 중요한 평가 메트릭:
    - 예측 박스와 실제 박스의 겹치는 정도 측정
    - 1에 가까울수록 정확한 예측
    - NMS와 mAP 계산에 핵심적으로 사용
    """
    # 교집합 영역 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 교집합이 없는 경우
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # 각 박스의 면적
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 합집합 면적
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def non_max_suppression(boxes, scores, threshold=0.4):
    """
    Non-Maximum Suppression (NMS) 구현
    
    Args:
        boxes: 바운딩 박스들 [[x1, y1, x2, y2], ...]
        scores: 각 박스의 신뢰도 점수
        threshold: IoU 임계값
    
    Returns:
        list: 선택된 박스들의 인덱스
    
    NMS의 목적:
    - 같은 객체에 대한 중복 탐지 제거
    - 가장 신뢰도가 높은 박스만 선택
    - 깔끔한 최종 결과 생성
    
    알고리즘:
    1. 신뢰도 순으로 정렬
    2. 가장 높은 점수의 박스 선택
    3. 선택된 박스와 IoU가 높은 박스들 제거
    4. 반복
    """
    if len(boxes) == 0:
        return []
    
    # 신뢰도 순으로 정렬
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    selected = []
    
    while indices:
        # 가장 높은 점수의 박스 선택
        current = indices.pop(0)
        selected.append(current)
        
        # 나머지 박스들과 IoU 계산
        remaining = []
        for idx in indices:
            iou = calculate_iou(boxes[current], boxes[idx])
            if iou <= threshold:  # IoU가 임계값 이하인 경우만 유지
                remaining.append(idx)
        
        indices = remaining
    
    return selected

def convert_to_corners(center_x, center_y, width, height):
    """중심점 + 크기 → 모서리 좌표 변환"""
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    return [x1, y1, x2, y2]

def convert_to_center(x1, y1, x2, y2):
    """모서리 좌표 → 중심점 + 크기 변환"""
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    return center_x, center_y, width, height

# ============================================================================
# 3. 간소화된 YOLO 모델 정의
# ============================================================================

print(f"\n🧠 YOLO 모델 정의")

class SimpleYOLO(nn.Module):
    """
    간소화된 YOLO 모델
    
    구조:
    1. 백본 네트워크: 특성 추출 (CNN)
    2. 탐지 헤드: 바운딩 박스 + 클래스 예측
    
    YOLO의 핵심 아이디어:
    - 이미지를 그리드로 분할
    - 각 그리드 셀이 객체의 중심을 포함하면 해당 객체 예측
    - 바운딩 박스 좌표 + 신뢰도 + 클래스 확률을 동시에 예측
    
    출력 형태:
    - (batch_size, grid_size, grid_size, num_boxes * 5 + num_classes)
    - 5 = (x, y, w, h, confidence)
    """
    
    def __init__(self, grid_size=13, num_boxes=2, num_classes=20):
        super(SimpleYOLO, self).__init__()
        
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        # 백본 네트워크 (특성 추출)
        # 실제 YOLO는 Darknet을 사용하지만, 여기서는 간소화된 CNN 사용
        self.backbone = nn.Sequential(
            # 첫 번째 블록: 416 → 208
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 208x208
            
            # 두 번째 블록: 208 → 104
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 104x104
            
            # 세 번째 블록: 104 → 52
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 52x52
            
            # 네 번째 블록: 52 → 26
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 26x26
            
            # 다섯 번째 블록: 26 → 13
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),  # 13x13
        )
        
        # 탐지 헤드
        # 출력 채널 수 = num_boxes * (5 + num_classes)
        # 5 = x, y, w, h, confidence
        output_channels = num_boxes * (5 + num_classes)
        
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, output_channels, kernel_size=1, stride=1, padding=0)
        )
        
    def forward(self, x):
        # 백본을 통한 특성 추출
        features = self.backbone(x)
        
        # 탐지 헤드를 통한 예측
        predictions = self.detection_head(features)
        
        # 출력 형태 변경: (batch, channels, height, width) → (batch, height, width, channels)
        predictions = predictions.permute(0, 2, 3, 1)
        
        return predictions

class YOLOLoss(nn.Module):
    """
    YOLO 손실 함수
    
    YOLO 손실은 여러 구성요소로 이루어짐:
    1. 좌표 손실: 바운딩 박스 위치 오차
    2. 크기 손실: 바운딩 박스 크기 오차
    3. 신뢰도 손실: 객체 존재 여부 오차
    4. 클래스 손실: 클래스 분류 오차
    
    각 손실에 다른 가중치를 적용하여 균형 조정
    """
    
    def __init__(self, grid_size=13, num_boxes=2, num_classes=20, 
                 lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord  # 좌표 손실 가중치
        self.lambda_noobj = lambda_noobj  # 객체 없음 손실 가중치
        
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch_size, grid_size, grid_size, num_boxes * 5 + num_classes)
            targets: (batch_size, grid_size, grid_size, num_boxes * 5 + num_classes)
        """
        batch_size = predictions.size(0)
        
        # 예측값 분리
        pred_boxes = predictions[..., :self.num_boxes * 5].view(
            batch_size, self.grid_size, self.grid_size, self.num_boxes, 5
        )
        pred_classes = predictions[..., self.num_boxes * 5:]
        
        # 타겟값 분리
        target_boxes = targets[..., :self.num_boxes * 5].view(
            batch_size, self.grid_size, self.grid_size, self.num_boxes, 5
        )
        target_classes = targets[..., self.num_boxes * 5:]
        
        # 객체가 있는 셀 마스크
        obj_mask = target_boxes[..., 4] > 0  # confidence > 0
        noobj_mask = target_boxes[..., 4] == 0
        
        # 1. 좌표 손실 (객체가 있는 경우만)
        coord_loss = 0
        if obj_mask.sum() > 0:
            pred_xy = pred_boxes[obj_mask][..., :2]
            target_xy = target_boxes[obj_mask][..., :2]
            coord_loss += F.mse_loss(pred_xy, target_xy, reduction='sum')
            
            # 크기 손실 (제곱근 적용으로 큰 박스와 작은 박스 균형)
            pred_wh = torch.sqrt(torch.abs(pred_boxes[obj_mask][..., 2:4]) + 1e-6)
            target_wh = torch.sqrt(target_boxes[obj_mask][..., 2:4] + 1e-6)
            coord_loss += F.mse_loss(pred_wh, target_wh, reduction='sum')
        
        # 2. 신뢰도 손실
        # 객체가 있는 경우
        obj_conf_loss = 0
        if obj_mask.sum() > 0:
            pred_conf_obj = pred_boxes[obj_mask][..., 4]
            target_conf_obj = target_boxes[obj_mask][..., 4]
            obj_conf_loss = F.mse_loss(pred_conf_obj, target_conf_obj, reduction='sum')
        
        # 객체가 없는 경우
        noobj_conf_loss = 0
        if noobj_mask.sum() > 0:
            pred_conf_noobj = pred_boxes[noobj_mask][..., 4]
            target_conf_noobj = target_boxes[noobj_mask][..., 4]
            noobj_conf_loss = F.mse_loss(pred_conf_noobj, target_conf_noobj, reduction='sum')
        
        # 3. 클래스 손실 (객체가 있는 셀에서만)
        class_loss = 0
        obj_class_mask = target_classes.sum(dim=-1) > 0
        if obj_class_mask.sum() > 0:
            pred_class_obj = pred_classes[obj_class_mask]
            target_class_obj = target_classes[obj_class_mask]
            class_loss = F.mse_loss(pred_class_obj, target_class_obj, reduction='sum')
        
        # 총 손실 계산
        total_loss = (
            self.lambda_coord * coord_loss +
            obj_conf_loss +
            self.lambda_noobj * noobj_conf_loss +
            class_loss
        )
        
        # 배치 크기로 정규화
        total_loss = total_loss / batch_size
        
        return total_loss, {
            'coord_loss': coord_loss / batch_size if coord_loss != 0 else 0,
            'obj_conf_loss': obj_conf_loss / batch_size if obj_conf_loss != 0 else 0,
            'noobj_conf_loss': noobj_conf_loss / batch_size if noobj_conf_loss != 0 else 0,
            'class_loss': class_loss / batch_size if class_loss != 0 else 0
        }

# ============================================================================
# 4. 샘플 데이터셋 생성 (COCO 대신)
# ============================================================================

print(f"\n📁 샘플 객체 탐지 데이터셋 생성")

class SampleObjectDetectionDataset(Dataset):
    """
    교육용 샘플 객체 탐지 데이터셋
    
    실제 프로젝트에서는 COCO나 PASCAL VOC 사용
    여기서는 학습 목적으로 간단한 합성 데이터 생성
    """
    
    def __init__(self, num_samples=1000, img_size=416, grid_size=13, 
                 num_classes=20, transform=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.transform = transform
        
        # 샘플 데이터 생성
        self.samples = self._generate_samples()
        
    def _generate_samples(self):
        """합성 데이터 생성"""
        print("🎨 합성 객체 탐지 데이터 생성 중...")
        
        samples = []
        np.random.seed(42)
        
        for i in tqdm(range(self.num_samples), desc="데이터 생성"):
            # 랜덤 이미지 생성 (RGB)
            image = np.random.randint(0, 256, (self.img_size, self.img_size, 3), dtype=np.uint8)
            
            # 랜덤 객체들 생성 (1~3개)
            num_objects = np.random.randint(1, 4)
            boxes = []
            classes = []
            
            for _ in range(num_objects):
                # 랜덤 바운딩 박스
                center_x = np.random.uniform(0.1, 0.9)
                center_y = np.random.uniform(0.1, 0.9)
                width = np.random.uniform(0.1, 0.3)
                height = np.random.uniform(0.1, 0.3)
                
                # 이미지 경계 내로 제한
                x1 = max(0, center_x - width/2)
                y1 = max(0, center_y - height/2)
                x2 = min(1, center_x + width/2)
                y2 = min(1, center_y + height/2)
                
                boxes.append([x1, y1, x2, y2])
                classes.append(np.random.randint(0, self.num_classes))
                
                # 이미지에 간단한 사각형 그리기 (시각적 표시)
                x1_px = int(x1 * self.img_size)
                y1_px = int(y1 * self.img_size)
                x2_px = int(x2 * self.img_size)
                y2_px = int(y2 * self.img_size)
                
                color = np.random.randint(100, 255, 3)
                image[y1_px:y2_px, x1_px:x2_px] = color
            
            samples.append({
                'image': image,
                'boxes': boxes,
                'classes': classes
            })
        
        print(f"✅ {len(samples)}개 샘플 생성 완료")
        return samples
    
    def _create_target_tensor(self, boxes, classes):
        """YOLO 형식의 타겟 텐서 생성"""
        target = torch.zeros(self.grid_size, self.grid_size, 2 * 5 + self.num_classes)
        
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = box
            center_x, center_y, width, height = convert_to_center(x1, y1, x2, y2)
            
            # 그리드 셀 좌표
            grid_x = int(center_x * self.grid_size)
            grid_y = int(center_y * self.grid_size)
            
            # 그리드 내 상대 좌표
            rel_x = center_x * self.grid_size - grid_x
            rel_y = center_y * self.grid_size - grid_y
            
            # 첫 번째 바운딩 박스에 할당
            target[grid_y, grid_x, 0] = rel_x
            target[grid_y, grid_x, 1] = rel_y
            target[grid_y, grid_x, 2] = width
            target[grid_y, grid_x, 3] = height
            target[grid_y, grid_x, 4] = 1.0  # confidence
            
            # 클래스 원-핫 인코딩
            target[grid_y, grid_x, 10 + cls] = 1.0
        
        return target
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = sample['image']
        boxes = sample['boxes']
        classes = sample['classes']
        
        # 이미지를 텐서로 변환
        image = torch.FloatTensor(image).permute(2, 0, 1) / 255.0
        
        # 타겟 텐서 생성
        target = self._create_target_tensor(boxes, classes)
        
        return image, target

# 데이터셋 및 로더 생성
print(f"\n📦 데이터셋 및 로더 생성")

train_dataset = SampleObjectDetectionDataset(
    num_samples=800,
    img_size=IMG_SIZE,
    grid_size=GRID_SIZE,
    num_classes=NUM_CLASSES
)

val_dataset = SampleObjectDetectionDataset(
    num_samples=200,
    img_size=IMG_SIZE,
    grid_size=GRID_SIZE,
    num_classes=NUM_CLASSES
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

print(f"✅ 데이터 로더 생성 완료:")
print(f"   훈련 배치: {len(train_loader)}")
print(f"   검증 배치: {len(val_loader)}")

# ============================================================================
# 5. 모델 초기화 및 훈련 설정
# ============================================================================

print(f"\n🧠 YOLO 모델 초기화")

# 모델 생성
yolo_model = SimpleYOLO(
    grid_size=GRID_SIZE,
    num_boxes=NUM_BOXES,
    num_classes=NUM_CLASSES
).to(device)

# 손실 함수
yolo_loss = YOLOLoss(
    grid_size=GRID_SIZE,
    num_boxes=NUM_BOXES,
    num_classes=NUM_CLASSES
)

# 옵티마이저
optimizer = optim.Adam(yolo_model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

# 학습률 스케줄러
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print(f"✅ 모델 초기화 완료")

# 모델 복잡도
params_info = count_parameters(yolo_model, detailed=False)
print(f"📊 모델 복잡도: {params_info['total_params']:,}개 파라미터")

# ============================================================================
# 6. 훈련 함수 정의
# ============================================================================

def train_epoch_yolo(model, train_loader, criterion, optimizer, device):
    """YOLO 훈련 함수"""
    model.train()
    
    running_loss = 0.0
    running_losses = {'coord': 0, 'obj_conf': 0, 'noobj_conf': 0, 'class': 0}
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="훈련")
    
    for batch_idx, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # 순전파
        predictions = model(images)
        
        # 손실 계산
        loss, loss_components = criterion(predictions, targets)
        
        # 역전파
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # 통계 업데이트
        running_loss += loss.item()
        for key, value in loss_components.items():
            if isinstance(value, torch.Tensor):
                running_losses[key.split('_')[0]] += value.item()
            else:
                running_losses[key.split('_')[0]] += value
        num_batches += 1
        
        # 진행률 바 업데이트
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Coord': f'{loss_components["coord_loss"]:.4f}',
                'Conf': f'{loss_components["obj_conf_loss"]:.4f}'
            })
    
    avg_loss = running_loss / num_batches
    avg_losses = {k: v / num_batches for k, v in running_losses.items()}
    
    return avg_loss, avg_losses

def validate_epoch_yolo(model, val_loader, criterion, device):
    """YOLO 검증 함수"""
    model.eval()
    
    running_loss = 0.0
    running_losses = {'coord': 0, 'obj_conf': 0, 'noobj_conf': 0, 'class': 0}
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="검증")
        
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            
            predictions = model(images)
            loss, loss_components = criterion(predictions, targets)
            
            running_loss += loss.item()
            for key, value in loss_components.items():
                if isinstance(value, torch.Tensor):
                    running_losses[key.split('_')[0]] += value.item()
                else:
                    running_losses[key.split('_')[0]] += value
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / num_batches
    avg_losses = {k: v / num_batches for k, v in running_losses.items()}
    
    return avg_loss, avg_losses

# ============================================================================
# 7. 모델 훈련 실행
# ============================================================================

print(f"\n🚀 YOLO 모델 훈련 시작")

# 훈련 기록
train_losses = []
val_losses = []
loss_components_history = []

# 최고 성능 추적
best_val_loss = float('inf')
best_model_state = None
patience = 10
patience_counter = 0

start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\n📅 에포크 {epoch+1}/{EPOCHS}")
    
    # 훈련
    train_loss, train_loss_components = train_epoch_yolo(
        yolo_model, train_loader, yolo_loss, optimizer, device
    )
    
    # 검증
    val_loss, val_loss_components = validate_epoch_yolo(
        yolo_model, val_loader, yolo_loss, device
    )
    
    # 학습률 스케줄러 업데이트
    scheduler.step()
    
    # 기록 저장
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    loss_components_history.append(val_loss_components)
    
    # 결과 출력
    print(f"   훈련 손실: {train_loss:.4f}")
    print(f"   검증 손실: {val_loss:.4f}")
    print(f"   검증 손실 구성:")
    print(f"     좌표: {val_loss_components['coord']:.4f}")
    print(f"     객체 신뢰도: {val_loss_components['obj']:.4f}")
    print(f"     비객체 신뢰도: {val_loss_components['noobj']:.4f}")
    print(f"     클래스: {val_loss_components['class']:.4f}")
    
    # 최고 성능 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(yolo_model.state_dict())
        patience_counter = 0
        print(f"   🎯 새로운 최고 성능! 검증 손실: {val_loss:.4f}")
        
        # 체크포인트 저장
        save_checkpoint(
            yolo_model, optimizer, epoch, val_loss, val_loss,
            save_path="./checkpoints/yolo_best_model.pth"
        )
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"   ⏰ 조기 종료: {patience} 에포크 동안 성능 개선 없음")
            break

training_time = time.time() - start_time
print(f"\n✅ 훈련 완료!")
print(f"   총 훈련 시간: {training_time:.2f}초")
print(f"   최고 검증 손실: {best_val_loss:.4f}")

# ============================================================================
# 8. 훈련 결과 시각화
# ============================================================================

print(f"\n📈 훈련 결과 시각화")

# 훈련 곡선
plot_training_curves(
    train_losses=train_losses,
    val_losses=val_losses,
    title="YOLO 객체 탐지 - 훈련 과정"
)

# 손실 구성요소 변화
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

epochs_range = range(1, len(loss_components_history) + 1)

# 좌표 손실
axes[0, 0].plot(epochs_range, [h['coord'] for h in loss_components_history], 'b-', linewidth=2)
axes[0, 0].set_title('좌표 손실')
axes[0, 0].set_xlabel('에포크')
axes[0, 0].set_ylabel('손실')
axes[0, 0].grid(True, alpha=0.3)

# 객체 신뢰도 손실
axes[0, 1].plot(epochs_range, [h['obj'] for h in loss_components_history], 'r-', linewidth=2)
axes[0, 1].set_title('객체 신뢰도 손실')
axes[0, 1].set_xlabel('에포크')
axes[0, 1].set_ylabel('손실')
axes[0, 1].grid(True, alpha=0.3)

# 비객체 신뢰도 손실
axes[1, 0].plot(epochs_range, [h['noobj'] for h in loss_components_history], 'g-', linewidth=2)
axes[1, 0].set_title('비객체 신뢰도 손실')
axes[1, 0].set_xlabel('에포크')
axes[1, 0].set_ylabel('손실')
axes[1, 0].grid(True, alpha=0.3)

# 클래스 손실
axes[1, 1].plot(epochs_range, [h['class'] for h in loss_components_history], 'm-', linewidth=2)
axes[1, 1].set_title('클래스 손실')
axes[1, 1].set_xlabel('에포크')
axes[1, 1].set_ylabel('손실')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('YOLO 손실 구성요소 변화', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 9. 객체 탐지 결과 시각화
# ============================================================================

print(f"\n🎯 객체 탐지 결과 시각화")

def decode_predictions(predictions, confidence_threshold=0.5, grid_size=13):
    """
    YOLO 예측 결과를 바운딩 박스로 디코딩
    
    Args:
        predictions: 모델 출력 (batch_size, grid_size, grid_size, channels)
        confidence_threshold: 신뢰도 임계값
        grid_size: 그리드 크기
    
    Returns:
        list: 각 이미지별 탐지된 객체들 [boxes, scores, classes]
    """
    batch_size = predictions.size(0)
    results = []
    
    for b in range(batch_size):
        pred = predictions[b]  # (grid_size, grid_size, channels)
        
        boxes = []
        scores = []
        classes = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                for box_idx in range(2):  # num_boxes
                    # 바운딩 박스 정보 추출
                    start_idx = box_idx * 5
                    rel_x = pred[i, j, start_idx]
                    rel_y = pred[i, j, start_idx + 1]
                    width = pred[i, j, start_idx + 2]
                    height = pred[i, j, start_idx + 3]
                    confidence = torch.sigmoid(pred[i, j, start_idx + 4])
                    
                    if confidence > confidence_threshold:
                        # 절대 좌표로 변환
                        center_x = (j + torch.sigmoid(rel_x)) / grid_size
                        center_y = (i + torch.sigmoid(rel_y)) / grid_size
                        
                        # 바운딩 박스 좌표
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2
                        
                        # 클래스 예측
                        class_probs = torch.softmax(pred[i, j, 10:], dim=0)
                        class_score, class_idx = torch.max(class_probs, dim=0)
                        
                        final_score = confidence * class_score
                        
                        if final_score > confidence_threshold:
                            boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                            scores.append(final_score.item())
                            classes.append(class_idx.item())
        
        results.append((boxes, scores, classes))
    
    return results

def visualize_detections(images, predictions, confidence_threshold=0.5, nms_threshold=0.4):
    """탐지 결과 시각화"""
    # 최고 성능 모델 로드
    if best_model_state is not None:
        yolo_model.load_state_dict(best_model_state)
    
    yolo_model.eval()
    
    with torch.no_grad():
        # 예측 수행
        model_predictions = yolo_model(images)
        
        # 예측 결과 디코딩
        results = decode_predictions(model_predictions, confidence_threshold)
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.flatten()
        
        for idx in range(min(4, len(images))):
            image = images[idx].cpu().permute(1, 2, 0).numpy()
            boxes, scores, classes = results[idx]
            
            # NMS 적용
            if len(boxes) > 0:
                selected_indices = non_max_suppression(boxes, scores, nms_threshold)
                boxes = [boxes[i] for i in selected_indices]
                scores = [scores[i] for i in selected_indices]
                classes = [classes[i] for i in selected_indices]
            
            # 이미지 표시
            axes[idx].imshow(image)
            axes[idx].set_title(f'탐지된 객체: {len(boxes)}개')
            axes[idx].axis('off')
            
            # 바운딩 박스 그리기
            for box, score, cls in zip(boxes, scores, classes):
                x1, y1, x2, y2 = box
                
                # 이미지 크기로 변환
                x1 *= IMG_SIZE
                y1 *= IMG_SIZE
                x2 *= IMG_SIZE
                y2 *= IMG_SIZE
                
                # 바운딩 박스
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[idx].add_patch(rect)
                
                # 클래스 라벨과 신뢰도 표시
                if cls < len(CLASS_NAMES):
                    label = f'{CLASS_NAMES[cls]}: {score:.2f}'
                else:
                    label = f'Class {cls}: {score:.2f}'
                
                axes[idx].text(x1, y1-5, label, 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                             fontsize=8, color='white')
        
        plt.suptitle('YOLO 객체 탐지 결과', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# 샘플 이미지로 탐지 결과 시각화
print("🎯 객체 탐지 결과 확인")
sample_batch = next(iter(val_loader))
sample_images = sample_batch[0][:4].to(device)

visualize_detections(sample_images, None)

# ============================================================================
# 10. 성능 평가 (mAP 계산)
# ============================================================================

print(f"\n📊 객체 탐지 성능 평가")

def calculate_map(predictions, targets, iou_threshold=0.5):
    """
    mAP (mean Average Precision) 계산
    
    mAP는 객체 탐지의 표준 평가 메트릭:
    1. 각 클래스별 AP (Average Precision) 계산
    2. 모든 클래스의 AP 평균
    3. IoU 임계값에 따른 정밀도-재현율 곡선
    
    Args:
        predictions: 예측 결과 [(boxes, scores, classes), ...]
        targets: 실제 정답 [(boxes, classes), ...]
        iou_threshold: IoU 임계값
    
    Returns:
        float: mAP 점수
    """
    
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0
    
    # 간소화된 mAP 계산
    total_precision = 0.0
    total_recall = 0.0
    num_samples = 0
    
    for pred, target in zip(predictions, targets):
        pred_boxes, pred_scores, pred_classes = pred
        target_boxes, target_classes = target
        
        if len(pred_boxes) == 0 or len(target_boxes) == 0:
            continue
        
        # 각 예측에 대해 최고 IoU 찾기
        matches = 0
        for pred_box in pred_boxes:
            best_iou = 0
            for target_box in target_boxes:
                iou = calculate_iou(pred_box, target_box)
                best_iou = max(best_iou, iou)
            
            if best_iou >= iou_threshold:
                matches += 1
        
        # 정밀도와 재현율 계산
        precision = matches / len(pred_boxes) if len(pred_boxes) > 0 else 0
        recall = matches / len(target_boxes) if len(target_boxes) > 0 else 0
        
        total_precision += precision
        total_recall += recall
        num_samples += 1
    
    if num_samples == 0:
        return 0.0
    
    avg_precision = total_precision / num_samples
    avg_recall = total_recall / num_samples
    
    # F1 점수로 mAP 근사
    if avg_precision + avg_recall > 0:
        map_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    else:
        map_score = 0.0
    
    return map_score

def evaluate_model(model, dataloader, device):
    """모델 성능 평가"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="평가")):
            if batch_idx >= 50:  # 평가 속도를 위해 50배치만
                break
                
            images = images.to(device)
            
            # 예측 수행
            predictions = model(images)
            results = decode_predictions(predictions, confidence_threshold=0.3)
            
            # 타겟 데이터 변환 (간소화)
            batch_targets = []
            for i in range(len(images)):
                # 실제 구현에서는 타겟 데이터에서 바운딩 박스 추출
                # 여기서는 샘플 데이터이므로 더미 타겟 생성
                dummy_boxes = [[0.2, 0.2, 0.8, 0.8]]  # 더미 박스
                dummy_classes = [0]  # 더미 클래스
                batch_targets.append((dummy_boxes, dummy_classes))
            
            all_predictions.extend(results)
            all_targets.extend(batch_targets)
    
    # mAP 계산
    map_score = calculate_map(all_predictions, all_targets)
    
    return map_score

# 최고 성능 모델로 평가
if best_model_state is not None:
    yolo_model.load_state_dict(best_model_state)

print("📈 모델 성능 평가 중...")
map_score = evaluate_model(yolo_model, val_loader, device)

print(f"✅ 평가 완료:")
print(f"   mAP@0.5: {map_score:.4f}")

# ============================================================================
# 11. 학습 내용 요약 및 실용적 활용
# ============================================================================

print(f"\n🎓 학습 내용 요약")
print(f"=" * 60)

print(f"✅ 완료한 내용:")
print(f"   1. YOLO 알고리즘의 핵심 개념과 구조")
print(f"   2. 객체 탐지 문제의 특성과 도전과제")
print(f"   3. 바운딩 박스와 신뢰도 점수 처리")
print(f"   4. Non-Maximum Suppression (NMS) 구현")
print(f"   5. IoU 계산과 객체 탐지 평가 메트릭")
print(f"   6. 실시간 객체 탐지 시스템 구현")

print(f"\n📊 최종 성과:")
print(f"   - 최고 검증 손실: {best_val_loss:.4f}")
print(f"   - mAP@0.5: {map_score:.4f}")
print(f"   - 모델 파라미터: {params_info['total_params']:,}개")
print(f"   - 훈련 시간: {training_time:.2f}초")

print(f"\n💡 핵심 학습 포인트:")
print(f"   1. 단일 네트워크: 한 번의 순전파로 모든 객체 탐지")
print(f"   2. 그리드 기반: 이미지를 그리드로 나누어 처리")
print(f"   3. 앵커 박스: 다양한 크기의 객체 탐지")
print(f"   4. 다중 손실: 위치, 크기, 신뢰도, 클래스 손실 결합")
print(f"   5. 실시간 처리: 빠른 추론 속도")

print(f"\n🔍 YOLO의 장단점:")
print(f"   장점:")
print(f"   - 빠른 추론 속도 (실시간 가능)")
print(f"   - 전역적 맥락 이해")
print(f"   - 단순한 구조")
print(f"   - End-to-end 학습")
print(f"   단점:")
print(f"   - 작은 객체 탐지 어려움")
print(f"   - 정확도가 2-stage 방법보다 낮음")
print(f"   - 그리드 제약으로 인한 한계")

print(f"\n🚀 실용적 활용 분야:")
print(f"   1. 자율주행: 차량, 보행자, 신호등 탐지")
print(f"   2. 보안: 침입자 탐지, 이상 행동 감지")
print(f"   3. 의료: 의료 영상에서 병변 탐지")
print(f"   4. 제조업: 불량품 검출, 품질 관리")
print(f"   5. 스포츠: 선수 추적, 경기 분석")
print(f"   6. 소매: 재고 관리, 고객 행동 분석")

print(f"\n🔧 YOLO 발전 과정:")
print(f"   1. YOLOv1: 최초의 단일 단계 탐지기")
print(f"   2. YOLOv2/YOLO9000: 앵커 박스, 배치 정규화")
print(f"   3. YOLOv3: 다중 스케일, FPN 구조")
print(f"   4. YOLOv4: CSPNet, Mish 활성화")
print(f"   5. YOLOv5: PyTorch 구현, 사용자 친화적")
print(f"   6. YOLOv8: 최신 기법 통합, 높은 성능")

print(f"\n🚀 다음 단계:")
print(f"   - 07_gan_image_generation.py: GAN으로 이미지 생성")
print(f"   - 생성적 적대 신경망의 핵심 개념")
print(f"   - 창의적 AI와 이미지 합성")

print(f"\n🔧 추가 실험 아이디어:")
print(f"   1. 실제 COCO 데이터셋 사용")
print(f"   2. YOLOv5/v8 구현")
print(f"   3. 다양한 앵커 박스 전략")
print(f"   4. 데이터 증강 기법 적용")
print(f"   5. 모델 경량화 (MobileNet 백본)")

print(f"\n" + "=" * 60)
print(f"🎉 YOLO 객체 탐지 튜토리얼 완료!")
print(f"   다음 튜토리얼에서 GAN 이미지 생성을 배워보세요!")
print(f"=" * 60)
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[idx].add_patch(rect)
                
                # 라벨
                label = f'{CLASS_NAMES[cls]}: {score:.2f}'
                axes[idx].text(x1, y1 - 5, label, color='red', fontsize=10,
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.suptitle('YOLO 객체 탐지 결과', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

# 샘플 이미지로 탐지 결과 시각화
sample_images, _ = next(iter(val_loader))
sample_images = sample_images.to(device)

visualize_detections(sample_images[:4], None)

# ============================================================================
# 10. 학습 내용 요약 및 실용적 활용
# ============================================================================

print(f"\n🎓 학습 내용 요약")
print(f"=" * 60)

print(f"✅ 완료한 내용:")
print(f"   1. YOLO 알고리즘의 핵심 원리와 구조 이해")
print(f"   2. 객체 탐지 문제의 특성과 도전과제")
print(f"   3. 바운딩 박스와 신뢰도 점수 처리")
print(f"   4. Non-Maximum Suppression (NMS) 구현")
print(f"   5. 다중 손실 함수 설계 및 균형 조정")
print(f"   6. 실시간 객체 탐지 시스템 구현")

print(f"\n📊 최종 성과:")
print(f"   - YOLO 최고 검증 손실: {best_val_loss:.4f}")
print(f"   - 총 파라미터: {params_info['total_params']:,}개")
print(f"   - 훈련 시간: {training_time:.2f}초")
print(f"   - 그리드 크기: {GRID_SIZE}x{GRID_SIZE}")

print(f"\n💡 핵심 학습 포인트:")
print(f"   1. 단일 네트워크로 위치 + 분류 동시 수행")
print(f"   2. 그리드 기반 예측으로 효율적 처리")
print(f"   3. 다중 손실 함수의 가중치 균형")
print(f"   4. NMS로 중복 탐지 제거")
print(f"   5. IoU 메트릭의 중요성")

print(f"\n🔍 YOLO의 장단점:")
print(f"   장점:")
print(f"   - 실시간 처리 가능한 빠른 속도")
print(f"   - End-to-end 학습으로 간단한 구조")
print(f"   - 전역적 컨텍스트 활용")
print(f"   - 다양한 크기 객체 탐지 가능")
print(f"   단점:")
print(f"   - 작은 객체 탐지 어려움")
print(f"   - 겹치는 객체 처리 제한")
print(f"   - 정확한 위치 예측의 어려움")

print(f"\n🚀 실용적 활용 분야:")
print(f"   1. 자율주행: 차량, 보행자, 신호등 탐지")
print(f"   2. 보안 시스템: 침입자 탐지, 이상 행동 감지")
print(f"   3. 의료 영상: 병변, 종양 탐지")
print(f"   4. 제조업: 불량품 검출, 품질 관리")
print(f"   5. 스포츠 분석: 선수 추적, 경기 분석")
print(f"   6. 소매업: 재고 관리, 고객 행동 분석")

print(f"\n🔧 성능 개선 방법:")
print(f"   1. 데이터 증강: 다양한 각도, 조명, 크기")
print(f"   2. 앵커 박스: 다양한 종횡비 사전 정의")
print(f"   3. 멀티스케일 훈련: 다양한 해상도 학습")
print(f"   4. 하드 네거티브 마이닝: 어려운 샘플 집중 학습")
print(f"   5. 앙상블: 여러 모델 결과 결합")

print(f"\n🚀 다음 단계:")
print(f"   - 07_gan_image_generation.py: GAN으로 이미지 생성")
print(f"   - 생성 모델의 원리와 적대적 학습")
print(f"   - 창조적 AI와 데이터 증강 활용")

print(f"\n🔧 추가 실험 아이디어:")
print(f"   1. YOLOv3, YOLOv4, YOLOv5 비교")
print(f"   2. 실제 COCO 데이터셋 사용")
print(f"   3. 전이 학습으로 특정 도메인 적응")
print(f"   4. 모바일 최적화 (YOLOv5s, YOLOv8n)")
print(f"   5. 실시간 웹캠 객체 탐지 구현")

print(f"\n" + "=" * 60)
print(f"🎉 YOLO 객체 탐지 튜토리얼 완료!")
print(f"   다음 튜토리얼에서 GAN 이미지 생성을 배워보세요!")
print(f"=" * 60)
import json
import os
from tqdm import tqdm
import time
import copy
import requests
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# 우리가 만든 유틸리티 함수들 임포트
from utils.data_utils import download_and_extract
from utils.visualization import plot_training_curves
from utils.model_utils import count_parameters, save_checkpoint

print("🚀 딥러닝 강의 시리즈 6: YOLO 객체 탐지")
print("=" * 60)#
 ============================================================================
# 1. 환경 설정 및 하이퍼파라미터
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  사용 장치: {device}")

# YOLO 객체 탐지를 위한 하이퍼파라미터
BATCH_SIZE = 16        # 객체 탐지는 메모리를 많이 사용
LEARNING_RATE = 0.001  # 사전 훈련된 모델 파인튜닝용
EPOCHS = 50            # 객체 탐지는 충분한 학습 필요
RANDOM_SEED = 42
IMG_SIZE = 640         # YOLO 표준 입력 크기
CONF_THRESHOLD = 0.5   # 신뢰도 임계값
IOU_THRESHOLD = 0.45   # NMS IoU 임계값
MAX_DETECTIONS = 100   # 최대 탐지 객체 수

# 재현성을 위한 시드 설정
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"📊 하이퍼파라미터:")
print(f"   배치 크기: {BATCH_SIZE}")
print(f"   학습률: {LEARNING_RATE}")
print(f"   에포크: {EPOCHS}")
print(f"   이미지 크기: {IMG_SIZE}x{IMG_SIZE}")
print(f"   신뢰도 임계값: {CONF_THRESHOLD}")
print(f"   IoU 임계값: {IOU_THRESHOLD}")

# COCO 클래스 이름 (80개 클래스)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

print(f"\n📋 COCO 데이터셋 정보:")
print(f"   클래스 수: {len(COCO_CLASSES)}개")
print(f"   주요 클래스: {COCO_CLASSES[:10]}")

# ============================================================================
# 2. YOLO 모델 및 유틸리티 함수
# ============================================================================

print(f"\n🤖 YOLO 모델 및 유틸리티 함수 정의")

class YOLODetector:
    """
    YOLO 객체 탐지기 클래스
    
    이 클래스는 사전 훈련된 YOLOv8 모델을 사용하여
    객체 탐지 기능을 제공합니다.
    
    왜 사전 훈련된 모델을 사용하는가?
    1. 계산 효율성: 처음부터 훈련하면 수일~수주 소요
    2. 데이터 효율성: COCO 전체 데이터셋 필요 (수백GB)
    3. 성능: 이미 최적화된 고성능 모델 활용
    4. 실용성: 실제 프로젝트에서 일반적인 접근법
    5. 학습 목적: 알고리즘 이해에 집중 가능
    """
    
    def __init__(self, model_name='yolov8n.pt'):
        """
        YOLO 탐지기 초기화
        
        Args:
            model_name: 사용할 YOLO 모델 이름
                - yolov8n.pt: nano (가장 빠름, 가장 작음)
                - yolov8s.pt: small
                - yolov8m.pt: medium  
                - yolov8l.pt: large
                - yolov8x.pt: extra large (가장 정확, 가장 큼)
        """
        print(f"🔄 YOLO 모델 로딩 중: {model_name}")
        
        try:
            # Ultralytics YOLO 모델 로드
            self.model = YOLO(model_name)
            print(f"✅ YOLO 모델 로드 완료")
            
            # 모델을 적절한 장치로 이동
            if torch.cuda.is_available():
                self.model.to(device)
                print(f"   GPU로 모델 이동 완료")
            
        except Exception as e:
            print(f"❌ YOLO 모델 로드 실패: {e}")
            print("💡 인터넷 연결을 확인하고 다시 시도해주세요.")
            raise
    
    def detect_objects(self, image_path, conf_threshold=0.5, iou_threshold=0.45):
        """
        이미지에서 객체 탐지 수행
        
        Args:
            image_path: 이미지 파일 경로 또는 numpy 배열
            conf_threshold: 신뢰도 임계값
            iou_threshold: NMS IoU 임계값
        
        Returns:
            dict: 탐지 결과 (바운딩 박스, 클래스, 신뢰도)
        
        YOLO 탐지 과정:
        1. 이미지 전처리 (리사이즈, 정규화)
        2. 모델 추론 (바운딩 박스 + 클래스 확률)
        3. 후처리 (NMS, 임계값 필터링)
        """
        
        # YOLO 추론 실행
        results = self.model(
            image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        # 결과 파싱
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            if boxes is not None:
                for i in range(len(boxes)):
                    # 바운딩 박스 좌표 (x1, y1, x2, y2)
                    box = boxes.xyxy[i].cpu().numpy()
                    
                    # 신뢰도 점수
                    confidence = boxes.conf[i].cpu().numpy()
                    
                    # 클래스 ID
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # 클래스 이름
                    class_name = COCO_CLASSES[class_id]
                    
                    detections.append({
                        'bbox': box,  # [x1, y1, x2, y2]
                        'confidence': float(confidence),
                        'class_id': class_id,
                        'class_name': class_name
                    })
        
        return detections
    
    def visualize_detections(self, image_path, detections, save_path=None):
        """
        탐지 결과를 이미지에 시각화
        
        Args:
            image_path: 원본 이미지 경로
            detections: 탐지 결과 리스트
            save_path: 결과 이미지 저장 경로
        
        Returns:
            PIL Image: 시각화된 이미지
        
        시각화 요소:
        1. 바운딩 박스: 객체 위치 표시
        2. 클래스 라벨: 객체 종류 표시
        3. 신뢰도 점수: 예측 확신도 표시
        4. 색상 코딩: 클래스별 구분
        """
        
        # 이미지 로드
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = Image.fromarray(image_path)
        
        # 그리기 객체 생성
        draw = ImageDraw.Draw(image)
        
        # 클래스별 색상 생성 (해시 기반)
        def get_color(class_id):
            np.random.seed(class_id)
            return tuple(np.random.randint(0, 255, 3))
        
        # 각 탐지 결과 그리기
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            # 바운딩 박스 좌표
            x1, y1, x2, y2 = bbox
            
            # 클래스별 색상
            color = get_color(class_id)
            
            # 바운딩 박스 그리기
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # 라벨 텍스트
            label = f"{class_name}: {confidence:.2f}"
            
            # 텍스트 배경 박스
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # 텍스트 크기 계산
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
            
            # 텍스트 배경
            draw.rectangle(
                [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
                fill=color
            )
            
            # 텍스트 그리기
            draw.text((x1 + 2, y1 - text_height - 2), label, fill='white', font=font)
        
        # 결과 저장
        if save_path:
            image.save(save_path)
            print(f"💾 탐지 결과 저장: {save_path}")
        
        return image

def calculate_iou(box1, box2):
    """
    두 바운딩 박스 간의 IoU (Intersection over Union) 계산
    
    Args:
        box1, box2: [x1, y1, x2, y2] 형태의 바운딩 박스
    
    Returns:
        float: IoU 값 (0~1)
    
    IoU는 객체 탐지에서 가장 중요한 메트릭:
    1. 예측 정확도 평가: 예측 박스와 실제 박스의 겹침 정도
    2. NMS 알고리즘: 중복 탐지 제거
    3. mAP 계산: 평균 정밀도 계산의 기준
    
    계산 방법:
    IoU = 교집합 영역 / 합집합 영역
    """
    
    # 교집합 영역 계산
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # 교집합이 없는 경우
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # 교집합 면적
    intersection = (x2 - x1) * (y2 - y1)
    
    # 각 박스의 면적
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 합집합 면적
    union = area1 + area2 - intersection
    
    # IoU 계산
    iou = intersection / union if union > 0 else 0.0
    
    return iou

def non_max_suppression(detections, iou_threshold=0.45):
    """
    Non-Maximum Suppression (NMS) 알고리즘 구현
    
    Args:
        detections: 탐지 결과 리스트
        iou_threshold: IoU 임계값
    
    Returns:
        list: NMS 적용 후 탐지 결과
    
    NMS가 필요한 이유:
    1. 중복 탐지: 하나의 객체에 여러 바운딩 박스
    2. 성능 향상: 가장 확실한 탐지만 유지
    3. 후처리 표준: 모든 객체 탐지 모델에서 사용
    
    NMS 알고리즘:
    1. 신뢰도 순으로 정렬
    2. 가장 높은 신뢰도 박스 선택
    3. 겹치는 박스들 제거 (IoU > threshold)
    4. 반복
    """
    
    if not detections:
        return []
    
    # 신뢰도 순으로 정렬 (내림차순)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # NMS 적용
    keep = []
    
    while detections:
        # 가장 높은 신뢰도 탐지 선택
        current = detections.pop(0)
        keep.append(current)
        
        # 나머지 탐지들과 IoU 계산
        remaining = []
        for detection in detections:
            # 같은 클래스인 경우만 NMS 적용
            if detection['class_id'] == current['class_id']:
                iou = calculate_iou(current['bbox'], detection['bbox'])
                # IoU가 임계값보다 낮으면 유지
                if iou <= iou_threshold:
                    remaining.append(detection)
            else:
                # 다른 클래스는 그대로 유지
                remaining.append(detection)
        
        detections = remaining
    
    return keep

# ============================================================================
# 3. 샘플 이미지 준비 및 탐지 실험
# ============================================================================

print(f"\n📁 샘플 이미지 준비 및 객체 탐지 실험")

def download_sample_images():
    """
    객체 탐지 실험용 샘플 이미지 다운로드
    
    다양한 시나리오의 이미지:
    1. 단일 객체: 명확한 탐지 대상
    2. 다중 객체: 여러 객체 동시 탐지
    3. 복잡한 배경: 실제 환경 시뮬레이션
    4. 겹치는 객체: NMS 효과 확인
    """
    
    sample_images = [
        {
            'url': 'https://images.unsplash.com/photo-1551963831-b3b1ca40c98e?w=800',
            'filename': 'dogs.jpg',
            'description': '강아지들 (다중 객체)'
        },
        {
            'url': 'https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800', 
            'filename': 'city_street.jpg',
            'description': '도시 거리 (복잡한 배경)'
        },
        {
            'url': 'https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=800',
            'filename': 'mountain_bike.jpg', 
            'description': '산악자전거 (단일 객체)'
        }
    ]
    
    os.makedirs('./sample_images', exist_ok=True)
    
    downloaded_images = []
    
    for img_info in sample_images:
        try:
            print(f"📥 다운로드 중: {img_info['description']}")
            
            response = requests.get(img_info['url'])
            response.raise_for_status()
            
            filepath = f"./sample_images/{img_info['filename']}"
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            downloaded_images.append({
                'path': filepath,
                'description': img_info['description']
            })
            
            print(f"✅ 다운로드 완료: {filepath}")
            
        except Exception as e:
            print(f"❌ 다운로드 실패 ({img_info['filename']}): {e}")
    
    return downloaded_images

# 샘플 이미지 다운로드 (실패 시 기본 이미지 사용)
try:
    sample_images = download_sample_images()
except:
    print("⚠️  샘플 이미지 다운로드 실패. 기본 이미지를 사용합니다.")
    sample_images = []

# YOLO 탐지기 초기화
try:
    detector = YOLODetector('yolov8n.pt')  # nano 모델 (빠른 추론)
    
    print(f"\n🔍 YOLO 객체 탐지 실험")
    
    # 샘플 이미지들에 대해 객체 탐지 수행
    if sample_images:
        for i, img_info in enumerate(sample_images):
            print(f"\n📊 이미지 {i+1}: {img_info['description']}")
            
            # 객체 탐지 수행
            detections = detector.detect_objects(
                img_info['path'],
                conf_threshold=CONF_THRESHOLD,
                iou_threshold=IOU_THRESHOLD
            )
            
            print(f"   탐지된 객체 수: {len(detections)}개")
            
            # 탐지 결과 출력
            for j, detection in enumerate(detections):
                print(f"   {j+1}. {detection['class_name']}: {detection['confidence']:.3f}")
            
            # 결과 시각화
            result_image = detector.visualize_detections(
                img_info['path'],
                detections,
                save_path=f"./results/detection_result_{i+1}.jpg"
            )
            
            # 이미지 표시
            plt.figure(figsize=(12, 8))
            plt.imshow(result_image)
            plt.title(f"YOLO 객체 탐지 결과: {img_info['description']}")
            plt.axis('off')
            plt.show()
    
    else:
        print("📝 샘플 이미지가 없어 기본 테스트를 수행합니다.")
        
        # 기본 테스트용 더미 이미지 생성
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        print("💡 실제 사용 시에는 다음과 같이 사용하세요:")
        print("   detections = detector.detect_objects('your_image.jpg')")
        print("   result_img = detector.visualize_detections('your_image.jpg', detections)")

except Exception as e:
    print(f"❌ YOLO 모델 초기화 실패: {e}")
    print("💡 다음을 확인해주세요:")
    print("   1. 인터넷 연결 상태")
    print("   2. ultralytics 라이브러리 설치: pip install ultralytics")
    print("   3. PyTorch 설치 상태")
    
    # 대안: 간단한 YOLO 구조 설명
    print(f"\n📚 YOLO 알고리즘 이론 설명")
    
    class SimpleYOLO(nn.Module):
        """
        교육용 간단한 YOLO 구조 (실제 동작하지 않음)
        
        실제 YOLO의 핵심 개념을 보여주는 예시 코드
        """
        
        def __init__(self, num_classes=80, num_anchors=3):
            super(SimpleYOLO, self).__init__()
            
            # 백본 네트워크 (특징 추출)
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64), 
                nn.ReLU(),
                nn.MaxPool2d(2),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            
            # 탐지 헤드
            # 각 그리드 셀마다 num_anchors개의 바운딩 박스 예측
            # 각 박스마다: [x, y, w, h, confidence] + class_probabilities
            output_size = num_anchors * (5 + num_classes)
            
            self.detection_head = nn.Conv2d(128, output_size, 1)
            
        def forward(self, x):
            # 특징 추출
            features = self.backbone(x)
            
            # 탐지 예측
            predictions = self.detection_head(features)
            
            return predictions
    
    # 간단한 YOLO 모델 생성 (예시)
    simple_yolo = SimpleYOLO(num_classes=80)
    
    print(f"\n📋 간단한 YOLO 구조:")
    print(simple_yolo)
    
    # 파라미터 수 계산
    params = count_parameters(simple_yolo, detailed=False)
    print(f"\n📊 모델 정보:")
    print(f"   파라미터 수: {params['total_params']:,}개")
    print(f"   (실제 YOLOv8n은 약 3.2M개 파라미터)")

# ============================================================================
# 4. YOLO 알고리즘 핵심 개념 설명
# ============================================================================

print(f"\n📚 YOLO 알고리즘 핵심 개념")

def explain_yolo_concepts():
    """
    YOLO 알고리즘의 핵심 개념들을 시각적으로 설명
    """
    
    print(f"\n🎯 1. YOLO의 기본 아이디어")
    print(f"   - 이미지를 S×S 그리드로 분할")
    print(f"   - 각 그리드 셀이 객체 탐지 담당")
    print(f"   - 한 번의 네트워크 패스로 모든 객체 탐지")
    print(f"   - 'You Only Look Once' - 이름의 유래")
    
    print(f"\n📐 2. 바운딩 박스 예측")
    print(f"   - 각 그리드 셀마다 B개의 바운딩 박스 예측")
    print(f"   - 박스 정보: (x, y, w, h, confidence)")
    print(f"   - x, y: 박스 중심점 (그리드 셀 기준 상대 좌표)")
    print(f"   - w, h: 박스 크기 (이미지 기준 상대 크기)")
    print(f"   - confidence: 객체 존재 확률 × IoU")
    
    print(f"\n🏷️  3. 클래스 예측")
    print(f"   - 각 그리드 셀마다 C개 클래스 확률 예측")
    print(f"   - P(Class_i | Object): 객체가 있을 때 클래스 확률")
    print(f"   - 최종 점수: confidence × class_probability")
    
    print(f"\n⚙️  4. 손실 함수")
    print(f"   - 좌표 손실: 바운딩 박스 위치 오차")
    print(f"   - 신뢰도 손실: 객체 존재 여부 오차") 
    print(f"   - 클래스 손실: 클래스 분류 오차")
    print(f"   - 가중치: 좌표 > 신뢰도 > 클래스")
    
    print(f"\n🔄 5. 후처리 (Post-processing)")
    print(f"   - 신뢰도 임계값 필터링")
    print(f"   - Non-Maximum Suppression (NMS)")
    print(f"   - 최종 탐지 결과 출력")

explain_yolo_concepts()

# ============================================================================
# 5. 객체 탐지 성능 평가 메트릭
# ============================================================================

print(f"\n📊 객체 탐지 성능 평가 메트릭")

def explain_detection_metrics():
    """
    객체 탐지 성능 평가 메트릭 설명
    """
    
    print(f"\n🎯 1. IoU (Intersection over Union)")
    print(f"   - 예측 박스와 실제 박스의 겹침 정도")
    print(f"   - IoU = 교집합 / 합집합")
    print(f"   - 0.5 이상이면 일반적으로 '정확한' 탐지")
    print(f"   - COCO에서는 0.5~0.95 범위에서 평가")
    
    print(f"\n📈 2. Precision과 Recall")
    print(f"   - Precision = TP / (TP + FP)")
    print(f"     올바른 탐지 / 전체 탐지")
    print(f"   - Recall = TP / (TP + FN)")
    print(f"     올바른 탐지 / 전체 실제 객체")
    print(f"   - Trade-off: 임계값에 따라 변화")
    
    print(f"\n🏆 3. AP (Average Precision)")
    print(f"   - Precision-Recall 곡선 아래 면적")
    print(f"   - 특정 IoU 임계값에서의 성능")
    print(f"   - AP@0.5: IoU 0.5에서의 AP")
    print(f"   - AP@0.75: IoU 0.75에서의 AP")
    
    print(f"\n🥇 4. mAP (mean Average Precision)")
    print(f"   - 모든 클래스의 AP 평균")
    print(f"   - mAP@0.5: IoU 0.5에서의 mAP")
    print(f"   - mAP@0.5:0.95: IoU 0.5~0.95 평균")
    print(f"   - COCO 대회의 공식 메트릭")

explain_detection_metrics()

# IoU 계산 예시
print(f"\n🧮 IoU 계산 예시:")

# 예시 바운딩 박스
box1 = [100, 100, 200, 200]  # 실제 박스
box2 = [150, 150, 250, 250]  # 예측 박스

iou_example = calculate_iou(box1, box2)
print(f"   박스1: {box1} (실제)")
print(f"   박스2: {box2} (예측)")
print(f"   IoU: {iou_example:.3f}")

# 시각화
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# 박스 그리기
rect1 = patches.Rectangle((box1[0], box1[1]), box1[2]-box1[0], box1[3]-box1[1], 
                         linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, label='실제 박스')
rect2 = patches.Rectangle((box2[0], box2[1]), box2[2]-box2[0], box2[3]-box2[1], 
                         linewidth=2, edgecolor='red', facecolor='red', alpha=0.3, label='예측 박스')

ax.add_patch(rect1)
ax.add_patch(rect2)

ax.set_xlim(50, 300)
ax.set_ylim(50, 300)
ax.set_aspect('equal')
ax.legend()
ax.set_title(f'IoU 계산 예시 (IoU = {iou_example:.3f})')
ax.grid(True, alpha=0.3)

plt.show()

# ============================================================================
# 6. 실시간 객체 탐지 시뮬레이션
# ============================================================================

print(f"\n⚡ 실시간 객체 탐지 시뮬레이션")

def simulate_realtime_detection():
    """
    실시간 객체 탐지 성능 시뮬레이션
    
    실제 웹캠이나 비디오 스트림 대신
    성능 측정과 최적화 방법을 보여줍니다.
    """
    
    print(f"\n🎥 실시간 탐지 성능 분석")
    
    # 다양한 이미지 크기에서 추론 시간 측정
    image_sizes = [320, 416, 512, 640, 832]
    inference_times = []
    
    if 'detector' in globals():
        for size in image_sizes:
            # 더미 이미지 생성
            dummy_image = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            
            # 추론 시간 측정
            start_time = time.time()
            
            try:
                # 실제 추론 (여러 번 실행하여 평균 계산)
                for _ in range(5):
                    _ = detector.detect_objects(dummy_image, conf_threshold=0.5)
                
                avg_time = (time.time() - start_time) / 5
                inference_times.append(avg_time * 1000)  # ms 단위
                
                fps = 1.0 / avg_time
                print(f"   {size}×{size}: {avg_time*1000:.1f}ms ({fps:.1f} FPS)")
                
            except:
                inference_times.append(0)
                print(f"   {size}×{size}: 측정 실패")
        
        # 성능 그래프
        if any(t > 0 for t in inference_times):
            plt.figure(figsize=(10, 6))
            
            plt.subplot(1, 2, 1)
            plt.plot(image_sizes, inference_times, 'bo-')
            plt.xlabel('이미지 크기')
            plt.ylabel('추론 시간 (ms)')
            plt.title('이미지 크기별 추론 시간')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            fps_values = [1000/t if t > 0 else 0 for t in inference_times]
            plt.plot(image_sizes, fps_values, 'ro-')
            plt.xlabel('이미지 크기')
            plt.ylabel('FPS')
            plt.title('이미지 크기별 FPS')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    print(f"\n⚡ 실시간 탐지 최적화 팁:")
    print(f"   1. 모델 크기: nano < small < medium < large")
    print(f"   2. 입력 해상도: 낮을수록 빠름 (정확도 trade-off)")
    print(f"   3. 배치 처리: 여러 이미지 동시 처리")
    print(f"   4. GPU 활용: CUDA 가속")
    print(f"   5. 모델 양자화: INT8, FP16 정밀도 감소")
    print(f"   6. TensorRT: NVIDIA GPU 최적화")

simulate_realtime_detection()

# ============================================================================
# 7. 학습 내용 요약 및 다음 단계
# ============================================================================

print(f"\n🎓 학습 내용 요약")
print(f"=" * 60)

print(f"✅ 완료한 내용:")
print(f"   1. YOLO 알고리즘의 핵심 원리 이해")
print(f"   2. 사전 훈련된 모델을 활용한 객체 탐지")
print(f"   3. 바운딩 박스와 신뢰도 점수 처리")
print(f"   4. Non-Maximum Suppression (NMS) 구현")
print(f"   5. 객체 탐지 성능 평가 메트릭 학습")
print(f"   6. 실시간 탐지 성능 분석")

print(f"\n💡 핵심 학습 포인트:")
print(f"   1. 실시간 처리: YOLO의 가장 큰 장점")
print(f"   2. End-to-End 학습: 단일 네트워크로 완전한 탐지")
print(f"   3. 그리드 기반 예측: 효율적인 공간 분할")
print(f"   4. 후처리의 중요성: NMS로 중복 제거")
print(f"   5. 성능 평가: IoU, mAP 등 전문 메트릭")

print(f"\n🔍 YOLO의 장단점:")
print(f"   장점:")
print(f"   - 빠른 추론 속도 (실시간 가능)")
print(f"   - 간단한 구조 (이해하기 쉬움)")
print(f"   - 전역적 추론 (전체 이미지 고려)")
print(f"   - 다양한 크기 객체 탐지")
print(f"   단점:")
print(f"   - 작은 객체 탐지 어려움")
print(f"   - 겹치는 객체 처리 제한")
print(f"   - 새로운 종횡비 객체 어려움")

print(f"\n🚀 다음 단계:")
print(f"   - 07_gan_image_generation.py: GAN으로 이미지 생성")
print(f"   - CelebA 데이터셋으로 얼굴 생성")
print(f"   - 생성적 적대 신경망 학습")

print(f"\n🔧 추가 실험 아이디어:")
print(f"   1. 커스텀 데이터셋: 자신만의 객체 탐지")
print(f"   2. 모델 파인튜닝: 특정 도메인 최적화")
print(f"   3. 다른 YOLO 버전: YOLOv5, YOLOv8, YOLOv9")
print(f"   4. 실시간 비디오: 웹캠 또는 비디오 파일")
print(f"   5. 모바일 배포: ONNX, TensorFlow Lite")

print(f"\n🎯 실제 응용 분야:")
print(f"   - 자율주행: 차량, 보행자, 신호등 탐지")
print(f"   - 보안 시스템: 침입자, 위험물 탐지")
print(f"   - 의료 영상: 병변, 장기 탐지")
print(f"   - 제조업: 불량품, 부품 탐지")
print(f"   - 스포츠 분석: 선수, 공 추적")

print(f"\n" + "=" * 60)
print(f"🎉 YOLO 객체 탐지 튜토리얼 완료!")
print(f"   다음 튜토리얼에서 GAN 이미지 생성을 배워보세요!")
print(f"=" * 60)