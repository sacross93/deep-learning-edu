"""
딥러닝 강의 시리즈 4: RNN 텍스트 분류

이 튜토리얼에서는 IMDB 영화 리뷰 데이터셋을 사용하여 
순환 신경망(RNN)의 핵심 개념과 텍스트 분류를 학습합니다.

학습 목표:
1. 순환 신경망(RNN)의 구조와 원리 이해
2. 텍스트 데이터 전처리 (토큰화, 패딩, 임베딩)
3. 단어 임베딩(Word Embedding)의 개념과 활용
4. 시퀀스 데이터 처리 기법
5. 그래디언트 소실 문제와 해결 방법
6. 양방향 RNN과 다층 RNN 구현

데이터셋 선택 이유 - IMDB 영화 리뷰:
- 50,000개의 영화 리뷰 (25,000 훈련 + 25,000 테스트)
- 이진 분류: 긍정(positive) vs 부정(negative) 감정
- 자연어 처리의 대표적인 벤치마크 데이터셋
- 다양한 길이의 텍스트로 시퀀스 처리 학습에 적합
- 실제 사용자 리뷰로 현실적인 언어 패턴 포함
- 감정 분석은 NLP의 핵심 응용 분야

왜 이제 RNN을 사용하는가?
1. 시퀀스 데이터 처리: 단어 순서가 의미에 중요한 영향
2. 가변 길이 입력: 리뷰마다 다른 길이 처리 가능
3. 문맥 정보 활용: 이전 단어들의 정보를 현재 예측에 반영
4. 메모리 메커니즘: 은닉 상태로 과거 정보 기억
5. 순차적 처리: 텍스트의 자연스러운 순서 보존

CNN과의 차이점:
- CNN: 공간적 패턴 (이미지의 지역적 특징)
- RNN: 시간적 패턴 (시퀀스의 순차적 의존성)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torchtext
from torchtext.data.utils import get_tokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import copy
import re
import string
from collections import Counter, defaultdict
import pickle
import os

# 우리가 만든 유틸리티 함수들 임포트
from utils.data_utils import explore_dataset
from utils.visualization import plot_training_curves, plot_confusion_matrix, plot_model_predictions
from utils.model_utils import count_parameters, save_checkpoint, evaluate_model, compare_models

print("🚀 딥러닝 강의 시리즈 4: RNN 텍스트 분류")
print("=" * 60)

# ============================================================================
# 1. 환경 설정 및 하이퍼파라미터
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  사용 장치: {device}")

# RNN을 위한 하이퍼파라미터
# 텍스트 데이터의 특성을 고려한 설정
BATCH_SIZE = 64        # 텍스트는 메모리를 많이 사용하므로 적당한 크기
LEARNING_RATE = 0.001  # RNN은 그래디언트 소실로 인해 신중한 학습률 필요
EPOCHS = 20            # 텍스트 분류는 상대적으로 빠른 수렴
RANDOM_SEED = 42
MAX_VOCAB_SIZE = 10000 # 어휘 사전 크기 제한
MAX_SEQ_LENGTH = 500   # 최대 시퀀스 길이 (메모리 효율성)
EMBEDDING_DIM = 100    # 단어 임베딩 차원
HIDDEN_DIM = 128       # RNN 은닉 상태 차원

# 재현성을 위한 시드 설정
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"📊 하이퍼파라미터:")
print(f"   배치 크기: {BATCH_SIZE}")
print(f"   학습률: {LEARNING_RATE}")
print(f"   에포크: {EPOCHS}")
print(f"   어휘 사전 크기: {MAX_VOCAB_SIZE}")
print(f"   최대 시퀀스 길이: {MAX_SEQ_LENGTH}")
print(f"   임베딩 차원: {EMBEDDING_DIM}")
print(f"   은닉 차원: {HIDDEN_DIM}")

# ============================================================================
# 2. 텍스트 전처리 클래스
# ============================================================================

print(f"\n📝 텍스트 전처리 시스템 구축")

class TextPreprocessor:
    """
    텍스트 전처리를 위한 클래스
    
    주요 기능:
    1. 텍스트 정제 (소문자 변환, 특수문자 제거)
    2. 토큰화 (문장을 단어로 분할)
    3. 어휘 사전 구축
    4. 텍스트를 숫자 시퀀스로 변환
    5. 패딩 (동일한 길이로 맞춤)
    
    왜 이런 전처리가 필요한가?
    - 신경망은 숫자만 처리 가능 → 텍스트를 숫자로 변환
    - 일관된 입력 형태 → 배치 처리를 위한 동일한 길이
    - 노이즈 제거 → 모델이 핵심 정보에 집중
    """
    
    def __init__(self, max_vocab_size=10000, max_seq_length=500):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.tokenizer = get_tokenizer('basic_english')
        
        # 특수 토큰 정의
        self.PAD_TOKEN = '<PAD>'  # 패딩용
        self.UNK_TOKEN = '<UNK>'  # 미지의 단어용
        self.SOS_TOKEN = '<SOS>'  # 문장 시작
        self.EOS_TOKEN = '<EOS>'  # 문장 끝
        
        # 어휘 사전 (단어 → 인덱스)
        self.word2idx = {}
        self.idx2word = {}
        self.word_counts = Counter()
        
    def clean_text(self, text):
        """
        텍스트 정제
        
        Args:
            text: 원본 텍스트
        
        Returns:
            정제된 텍스트
        
        정제 과정:
        1. 소문자 변환 → 대소문자 통일
        2. HTML 태그 제거 → 웹 데이터 정제
        3. 특수문자 처리 → 의미있는 구두점만 보존
        4. 연속 공백 제거 → 일관된 형태
        """
        # 소문자 변환
        text = text.lower()
        
        # HTML 태그 제거 (IMDB 데이터에 포함됨)
        text = re.sub(r'<[^>]+>', '', text)
        
        # 구두점 앞뒤에 공백 추가 (토큰화 개선)
        text = re.sub(r'([.!?])', r' \1 ', text)
        
        # 연속된 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def build_vocab(self, texts):
        """
        어휘 사전 구축
        
        Args:
            texts: 텍스트 리스트
        
        어휘 사전 구축 과정:
        1. 모든 텍스트에서 단어 빈도 계산
        2. 빈도순으로 정렬하여 상위 N개 선택
        3. 특수 토큰 추가
        4. 단어 ↔ 인덱스 매핑 생성
        
        왜 어휘 크기를 제한하는가?
        - 메모리 효율성: 임베딩 테이블 크기 제한
        - 일반화 성능: 희귀 단어는 노이즈일 가능성
        - 계산 효율성: 소프트맥스 계산량 감소
        """
        print("📚 어휘 사전 구축 중...")
        
        # 모든 텍스트에서 단어 빈도 계산
        for text in tqdm(texts, desc="단어 빈도 계산"):
            cleaned_text = self.clean_text(text)
            tokens = self.tokenizer(cleaned_text)
            self.word_counts.update(tokens)
        
        # 특수 토큰 먼저 추가
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]
        
        # 빈도순으로 정렬하여 상위 단어들 선택
        most_common = self.word_counts.most_common(self.max_vocab_size - len(special_tokens))
        
        # 어휘 사전 구축
        vocab_words = special_tokens + [word for word, count in most_common]
        
        self.word2idx = {word: idx for idx, word in enumerate(vocab_words)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        print(f"✅ 어휘 사전 구축 완료:")
        print(f"   총 단어 수: {len(self.word_counts):,}개")
        print(f"   어휘 사전 크기: {len(self.word2idx):,}개")
        print(f"   가장 빈번한 단어: {most_common[:10]}")
        
        # 어휘 사전 저장
        os.makedirs('./data', exist_ok=True)
        with open('./data/vocab.pkl', 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_counts': self.word_counts
            }, f)
    
    def text_to_sequence(self, text):
        """
        텍스트를 숫자 시퀀스로 변환
        
        Args:
            text: 입력 텍스트
        
        Returns:
            숫자 시퀀스 (리스트)
        
        변환 과정:
        1. 텍스트 정제
        2. 토큰화
        3. 각 단어를 인덱스로 변환
        4. 길이 제한 적용
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenizer(cleaned_text)
        
        # 단어를 인덱스로 변환
        sequence = []
        for token in tokens:
            if token in self.word2idx:
                sequence.append(self.word2idx[token])
            else:
                sequence.append(self.word2idx[self.UNK_TOKEN])  # 미지의 단어
        
        # 길이 제한
        if len(sequence) > self.max_seq_length:
            sequence = sequence[:self.max_seq_length]
        
        return sequence
    
    def pad_sequences(self, sequences):
        """
        시퀀스들을 동일한 길이로 패딩
        
        Args:
            sequences: 숫자 시퀀스들의 리스트
        
        Returns:
            패딩된 텐서
        
        패딩이 필요한 이유:
        - 배치 처리: 동일한 크기의 텐서 필요
        - 효율성: GPU 병렬 처리 최적화
        - 일관성: 모델 입력 형태 통일
        """
        # 텐서로 변환
        tensor_sequences = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
        
        # 패딩 (짧은 시퀀스는 PAD_TOKEN으로 채움)
        padded = pad_sequence(
            tensor_sequences, 
            batch_first=True, 
            padding_value=self.word2idx[self.PAD_TOKEN]
        )
        
        return padded

# ============================================================================
# 3. IMDB 데이터셋 클래스
# ============================================================================

class IMDBDataset(Dataset):
    """
    IMDB 영화 리뷰 데이터셋 클래스
    
    PyTorch Dataset을 상속하여 커스텀 데이터셋 구현
    """
    
    def __init__(self, texts, labels, preprocessor):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        
        # 텍스트를 미리 시퀀스로 변환 (효율성)
        print("📝 텍스트를 시퀀스로 변환 중...")
        self.sequences = []
        for text in tqdm(texts, desc="시퀀스 변환"):
            sequence = self.preprocessor.text_to_sequence(text)
            self.sequences.append(sequence)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    """
    배치 데이터를 처리하는 함수
    
    Args:
        batch: (시퀀스, 라벨) 튜플들의 리스트
    
    Returns:
        패딩된 시퀀스 텐서와 라벨 텐서
    """
    sequences, labels = zip(*batch)
    
    # 시퀀스 길이 계산 (패딩 마스크용)
    lengths = [len(seq) for seq in sequences]
    
    # 패딩
    padded_sequences = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in sequences],
        batch_first=True,
        padding_value=0  # PAD_TOKEN의 인덱스
    )
    
    labels = torch.tensor(labels, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return padded_sequences, labels, lengths

# ============================================================================
# 4. 데이터 로딩 및 전처리
# ============================================================================

print(f"\n📁 IMDB 데이터셋 로딩 및 전처리")

# IMDB 데이터셋 다운로드 및 로드
try:
    # torchtext를 사용한 IMDB 데이터 로드
    from torchtext.datasets import IMDB
    
    print("📥 IMDB 데이터셋 다운로드 중...")
    
    # 훈련 및 테스트 데이터 로드
    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')
    
    # 데이터를 리스트로 변환
    train_data = list(train_iter)
    test_data = list(test_iter)
    
    # 라벨 변환 (1: 긍정, 2: 부정 → 0: 부정, 1: 긍정)
    train_texts = [text for label, text in train_data]
    train_labels = [0 if label == 1 else 1 for label, text in train_data]  # 1->0, 2->1
    
    test_texts = [text for label, text in test_data]
    test_labels = [0 if label == 1 else 1 for label, text in test_data]
    
    print(f"✅ IMDB 데이터 로드 완료:")
    print(f"   훈련 샘플: {len(train_texts):,}개")
    print(f"   테스트 샘플: {len(test_texts):,}개")
    
except Exception as e:
    print(f"❌ IMDB 데이터 로드 실패: {e}")
    print("💡 대안: 샘플 데이터로 진행합니다.")
    
    # 샘플 데이터 생성 (실제 프로젝트에서는 실제 데이터 사용)
    train_texts = [
        "This movie is absolutely fantastic! Great acting and plot.",
        "Terrible movie, waste of time. Poor acting and boring story.",
        "Amazing cinematography and wonderful performances by all actors.",
        "Not recommended. Very disappointing and poorly executed.",
        "One of the best movies I've ever seen. Highly recommended!",
    ] * 1000  # 샘플 확장
    
    train_labels = [1, 0, 1, 0, 1] * 1000  # 긍정/부정 라벨
    test_texts = train_texts[:500]
    test_labels = train_labels[:500]

# 데이터 탐색
print(f"\n🔍 IMDB 데이터 탐색")
print(f"📊 라벨 분포:")
positive_count = sum(train_labels)
negative_count = len(train_labels) - positive_count
print(f"   긍정 리뷰: {positive_count:,}개 ({positive_count/len(train_labels)*100:.1f}%)")
print(f"   부정 리뷰: {negative_count:,}개 ({negative_count/len(train_labels)*100:.1f}%)")

print(f"\n📝 샘플 리뷰:")
for i in range(3):
    label_text = "긍정" if train_labels[i] == 1 else "부정"
    print(f"   {label_text}: {train_texts[i][:100]}...")

# 텍스트 길이 분석
text_lengths = [len(text.split()) for text in train_texts]
print(f"\n📏 텍스트 길이 통계:")
print(f"   평균 길이: {np.mean(text_lengths):.1f} 단어")
print(f"   최대 길이: {max(text_lengths)} 단어")
print(f"   최소 길이: {min(text_lengths)} 단어")
print(f"   중간값: {np.median(text_lengths):.1f} 단어")

# 전처리기 초기화 및 어휘 사전 구축
preprocessor = TextPreprocessor(
    max_vocab_size=MAX_VOCAB_SIZE,
    max_seq_length=MAX_SEQ_LENGTH
)

# 어휘 사전 구축 (훈련 데이터만 사용)
preprocessor.build_vocab(train_texts)

# 데이터셋 생성
train_dataset = IMDBDataset(train_texts, train_labels, preprocessor)
test_dataset = IMDBDataset(test_texts, test_labels, preprocessor)

# 검증 데이터 분할
val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size

train_subset, val_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

# 데이터 로더 생성
train_loader = DataLoader(
    train_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=2
)

val_loader = DataLoader(
    val_subset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=collate_fn,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=collate_fn,
    num_workers=2
)

print(f"\n✅ 데이터 로더 생성 완료:")
print(f"   훈련 배치: {len(train_loader)}")
print(f"   검증 배치: {len(val_loader)}")
print(f"   테스트 배치: {len(test_loader)}")

# ============================================================================
# 5. RNN 모델 정의
# ============================================================================

print(f"\n🧠 RNN 모델 정의")

class SimpleRNN(nn.Module):
    """
    기본 RNN 모델
    
    구조:
    - Embedding Layer: 단어 → 벡터 변환
    - RNN Layer: 시퀀스 처리
    - Classifier: 최종 분류
    
    RNN의 핵심 개념:
    1. 은닉 상태(Hidden State): 이전 정보를 기억
    2. 순차 처리: 단어를 하나씩 순서대로 처리
    3. 파라미터 공유: 모든 시간 단계에서 같은 가중치 사용
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes=2):
        super(SimpleRNN, self).__init__()
        
        # 임베딩 레이어
        # 왜 임베딩이 필요한가?
        # 1. 희소 표현 → 밀집 표현: 원-핫 벡터를 실수 벡터로
        # 2. 의미적 유사성: 비슷한 단어는 비슷한 벡터
        # 3. 차원 효율성: 어휘 크기보다 작은 차원으로 표현
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # PAD_TOKEN의 인덱스
        )
        
        # RNN 레이어
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,  # (batch, seq, feature) 순서
            dropout=0.2
        )
        
        # 분류기
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x, lengths=None):
        # 임베딩
        # (batch_size, seq_len) → (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # RNN 처리
        # output: (batch_size, seq_len, hidden_dim)
        # hidden: (1, batch_size, hidden_dim)
        output, hidden = self.rnn(embedded)
        
        # 마지막 시간 단계의 출력 사용
        # 실제 시퀀스 길이를 고려하여 마지막 유효한 출력 선택
        if lengths is not None:
            # 각 시퀀스의 실제 마지막 위치에서 출력 추출
            batch_size = output.size(0)
            last_outputs = []
            for i in range(batch_size):
                last_idx = min(lengths[i] - 1, output.size(1) - 1)
                last_outputs.append(output[i, last_idx])
            last_output = torch.stack(last_outputs)
        else:
            # 길이 정보가 없으면 마지막 위치 사용
            last_output = output[:, -1, :]
        
        # 분류
        last_output = self.dropout(last_output)
        logits = self.classifier(last_output)
        
        return logits

class BiLSTM(nn.Module):
    """
    양방향 LSTM 모델
    
    개선사항:
    1. LSTM: 그래디언트 소실 문제 해결
    2. 양방향: 앞뒤 문맥 모두 활용
    3. 다층 구조: 더 복잡한 패턴 학습
    
    왜 LSTM인가?
    - 장기 의존성: 멀리 떨어진 단어들 간의 관계 학습
    - 게이트 메커니즘: 정보의 선택적 기억/망각
    - 그래디언트 소실 완화: 안정적인 역전파
    
    왜 양방향인가?
    - 완전한 문맥: "not good"에서 "not"의 영향을 "good" 이후에도 반영
    - 더 풍부한 표현: 순방향 + 역방향 정보 결합
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=2, num_classes=2):
        super(BiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # 양방향 LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True  # 양방향 설정
        )
        
        # 양방향이므로 hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x, lengths=None):
        # 임베딩
        embedded = self.embedding(x)
        
        # 패킹 (효율적인 처리를 위해)
        if lengths is not None:
            # 길이순으로 정렬
            sorted_lengths, sorted_idx = lengths.sort(0, descending=True)
            sorted_embedded = embedded[sorted_idx]
            
            # 패킹
            packed_embedded = nn.utils.rnn.pack_padded_sequence(
                sorted_embedded, sorted_lengths.cpu(), batch_first=True, enforce_sorted=True
            )
            
            # LSTM 처리
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            
            # 언패킹
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            
            # 원래 순서로 복원
            _, unsorted_idx = sorted_idx.sort(0)
            output = output[unsorted_idx]
            
            # 마지막 유효한 출력 추출
            batch_size = output.size(0)
            last_outputs = []
            for i in range(batch_size):
                last_idx = min(lengths[i] - 1, output.size(1) - 1)
                last_outputs.append(output[i, last_idx])
            last_output = torch.stack(last_outputs)
        else:
            # 패킹 없이 처리
            output, (hidden, cell) = self.lstm(embedded)
            last_output = output[:, -1, :]
        
        # 분류
        logits = self.classifier(last_output)
        
        return logits

# 모델 인스턴스 생성
vocab_size = len(preprocessor.word2idx)

simple_rnn = SimpleRNN(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=2
).to(device)

bilstm = BiLSTM(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=2,
    num_classes=2
).to(device)

print(f"✅ RNN 모델 생성 완료")

# 모델 복잡도 비교
print(f"\n📊 모델 복잡도 비교:")
simple_params = count_parameters(simple_rnn, detailed=False)
bilstm_params = count_parameters(bilstm, detailed=False)

print(f"   간단한 RNN: {simple_params['total_params']:,}개 파라미터")
print(f"   양방향 LSTM: {bilstm_params['total_params']:,}개 파라미터")

# ============================================================================
# 6. 손실 함수와 옵티마이저 설정
# ============================================================================

print(f"\n⚙️  손실 함수와 옵티마이저 설정")

# 손실 함수
criterion = nn.CrossEntropyLoss()

# 옵티마이저 (BiLSTM용)
optimizer = optim.Adam(
    bilstm.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-5  # 텍스트 모델에서는 약한 정규화
)

# 학습률 스케줄러
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',        # 검증 정확도 기준
    factor=0.5,        # 학습률을 절반으로
    patience=3,        # 3 에포크 동안 개선 없으면
    verbose=True
)

print(f"   손실 함수: {criterion.__class__.__name__}")
print(f"   옵티마이저: {optimizer.__class__.__name__}")
print(f"   스케줄러: ReduceLROnPlateau")

# ============================================================================
# 7. 훈련 함수 정의
# ============================================================================

def train_epoch_rnn(model, train_loader, criterion, optimizer, device, epoch):
    """RNN을 위한 훈련 함수"""
    model.train()
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    pbar = tqdm(train_loader, desc=f"에포크 {epoch+1} 훈련")
    
    for batch_idx, (sequences, labels, lengths) in enumerate(pbar):
        sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 그래디언트 클리핑 (RNN에서 매우 중요)
        # RNN은 그래디언트 폭발에 취약하므로 클리핑 필수
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        # 통계 업데이트
        running_loss += loss.item()
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        
        # 진행률 바 업데이트
        if batch_idx % 20 == 0:
            current_accuracy = correct_predictions / total_samples
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_accuracy:.4f}'
            })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

def validate_epoch_rnn(model, val_loader, criterion, device):
    """RNN을 위한 검증 함수"""
    model.eval()
    
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="검증")
        
        for sequences, labels, lengths in pbar:
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
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

print(f"\n🚀 BiLSTM 모델 훈련 시작")

# 훈련 기록
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# 최고 성능 추적
best_val_accuracy = 0.0
best_model_state = None
patience = 5
patience_counter = 0

start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\n📅 에포크 {epoch+1}/{EPOCHS}")
    
    # 훈련
    train_loss, train_acc = train_epoch_rnn(
        bilstm, train_loader, criterion, optimizer, device, epoch
    )
    
    # 검증
    val_loss, val_acc = validate_epoch_rnn(
        bilstm, val_loader, criterion, device
    )
    
    # 학습률 스케줄러 업데이트
    scheduler.step(val_acc)
    
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
        best_model_state = copy.deepcopy(bilstm.state_dict())
        patience_counter = 0
        print(f"   🎯 새로운 최고 성능! 검증 정확도: {val_acc:.4f}")
        
        # 체크포인트 저장
        save_checkpoint(
            bilstm, optimizer, epoch, val_loss, val_acc,
            save_path="./checkpoints/imdb_bilstm_best_model.pth"
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
    title="IMDB 감정 분석 - BiLSTM 훈련 과정"
)

# ============================================================================
# 10. 모델 비교 실험
# ============================================================================

print(f"\n🏆 RNN 모델 비교")

# 간단한 RNN도 빠르게 훈련 (비교용)
print(f"📊 간단한 RNN 훈련 중...")

simple_optimizer = optim.Adam(simple_rnn.parameters(), lr=LEARNING_RATE)

for epoch in range(3):  # 빠른 훈련
    simple_rnn.train()
    for batch_idx, (sequences, labels, lengths) in enumerate(train_loader):
        if batch_idx > 50:  # 제한적 훈련
            break
        sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
        simple_optimizer.zero_grad()
        outputs = simple_rnn(sequences, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(simple_rnn.parameters(), max_norm=5.0)
        simple_optimizer.step()

# 최고 성능 모델 로드
if best_model_state is not None:
    bilstm.load_state_dict(best_model_state)

# 모델 비교
models_to_compare = {
    "양방향 LSTM": bilstm,
    "간단한 RNN": simple_rnn
}

comparison_results = compare_models(
    models=models_to_compare,
    test_dataloader=test_loader,
    criterion=criterion,
    device=device
)

# ============================================================================
# 11. 최종 평가 및 예측 분석
# ============================================================================

print(f"\n🎯 최종 BiLSTM 모델 평가")

# 상세한 성능 평가
final_results = evaluate_model(
    model=bilstm,
    dataloader=test_loader,
    criterion=criterion,
    device=device,
    num_classes=2
)

# 예측 결과 분석
print(f"\n🔍 예측 결과 분석")

def analyze_predictions(model, test_loader, preprocessor, device, num_samples=10):
    """예측 결과를 텍스트와 함께 분석"""
    model.eval()
    
    class_names = ['부정', '긍정']
    samples_analyzed = 0
    
    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
            outputs = model(sequences, lengths)
            predictions = torch.argmax(outputs, dim=1)
            probabilities = F.softmax(outputs, dim=1)
            
            for i in range(len(sequences)):
                if samples_analyzed >= num_samples:
                    return
                
                # 시퀀스를 텍스트로 복원
                sequence = sequences[i].cpu().numpy()
                words = []
                for idx in sequence:
                    if idx == 0:  # PAD_TOKEN
                        break
                    if idx in preprocessor.idx2word:
                        words.append(preprocessor.idx2word[idx])
                
                text = ' '.join(words)
                true_label = labels[i].item()
                pred_label = predictions[i].item()
                confidence = probabilities[i][pred_label].item()
                
                print(f"\n샘플 {samples_analyzed + 1}:")
                print(f"  텍스트: {text[:100]}...")
                print(f"  실제: {class_names[true_label]}")
                print(f"  예측: {class_names[pred_label]} (신뢰도: {confidence:.3f})")
                print(f"  정확: {'✓' if true_label == pred_label else '✗'}")
                
                samples_analyzed += 1

analyze_predictions(bilstm, test_loader, preprocessor, device, num_samples=5)

# 혼동 행렬 생성
print(f"\n📊 혼동 행렬 생성 중...")

all_predictions = []
all_targets = []

bilstm.eval()
with torch.no_grad():
    for sequences, labels, lengths in tqdm(test_loader, desc="예측 수집"):
        sequences, labels, lengths = sequences.to(device), labels.to(device), lengths.to(device)
        outputs = bilstm(sequences, lengths)
        predictions = torch.argmax(outputs, dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

plot_confusion_matrix(
    y_true=np.array(all_targets),
    y_pred=np.array(all_predictions),
    class_names=['부정', '긍정'],
    title="IMDB 감정 분석 - 혼동 행렬"
)

# ============================================================================
# 12. 학습 내용 요약 및 다음 단계
# ============================================================================

print(f"\n🎓 학습 내용 요약")
print(f"=" * 60)

print(f"✅ 완료한 내용:")
print(f"   1. RNN의 핵심 개념 (순차 처리, 은닉 상태) 이해")
print(f"   2. 텍스트 전처리 파이프라인 구축")
print(f"   3. 단어 임베딩과 어휘 사전 구축")
print(f"   4. 양방향 LSTM으로 감정 분석 구현")
print(f"   5. 그래디언트 클리핑과 시퀀스 패킹 기법")

print(f"\n📊 최종 성과:")
print(f"   - BiLSTM 최고 정확도: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
print(f"   - 총 파라미터 수: {bilstm_params['total_params']:,}개")
print(f"   - 훈련 시간: {training_time:.2f}초")

print(f"\n💡 핵심 학습 포인트:")
print(f"   1. 시퀀스 데이터 처리: 순서가 중요한 데이터의 특성")
print(f"   2. 텍스트 전처리: 토큰화, 어휘 사전, 패딩의 중요성")
print(f"   3. 임베딩 레이어: 희소 → 밀집 표현의 효과")
print(f"   4. RNN vs LSTM: 그래디언트 소실 문제와 해결")
print(f"   5. 양방향 처리: 완전한 문맥 정보 활용")

print(f"\n🔍 RNN의 장단점:")
print(f"   장점:")
print(f"   - 가변 길이 시퀀스 처리 가능")
print(f"   - 순차적 의존성 모델링")
print(f"   - 메모리 효율적 (고정 크기 은닉 상태)")
print(f"   단점:")
print(f"   - 순차 처리로 인한 병렬화 제한")
print(f"   - 장거리 의존성 학습 어려움")
print(f"   - 그래디언트 소실/폭발 문제")

print(f"\n🚀 다음 단계:")
print(f"   - 05_lstm_sequence_prediction.py: LSTM으로 시계열 예측")
print(f"   - 주식 가격 데이터로 시계열 분석")
print(f"   - 장기 의존성과 시계열 패턴 학습")

print(f"\n🔧 추가 실험 아이디어:")
print(f"   1. 어텐션 메커니즘: 중요한 단어에 집중")
print(f"   2. 사전 훈련된 임베딩: Word2Vec, GloVe 활용")
print(f"   3. 다양한 RNN 변형: GRU, ConvLSTM")
print(f"   4. 계층적 어텐션: 문장 → 문서 수준 분석")
print(f"   5. 멀티태스크 학습: 감정 + 주제 분류 동시 학습")

print(f"\n" + "=" * 60)
print(f"🎉 RNN 텍스트 분류 튜토리얼 완료!")
print(f"   다음 튜토리얼에서 LSTM 시계열 예측을 배워보세요!")
print(f"=" * 60)