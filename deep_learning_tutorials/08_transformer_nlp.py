"""
딥러닝 강의 시리즈 8: Transformer NLP

이 튜토리얼에서는 Multi30k 번역 데이터셋을 사용하여 
Transformer 모델의 자연어 처리를 학습합니다.

학습 목표:
1. Transformer 아키텍처의 핵심 개념
2. Self-Attention과 Multi-Head Attention 메커니즘
3. 위치 인코딩(Positional Encoding)의 역할
4. 인코더-디코더 구조와 마스킹
5. 기계 번역 구현 및 BLEU 점수 평가
6. 어텐션 가중치 시각화와 해석

데이터셋 선택 이유 - Multi30k (영어-독일어 번역):
- 29,000개의 영어-독일어 문장 쌍
- 적당한 크기로 교육에 적합
- 기계 번역의 표준 벤치마크
- 다양한 일상 표현과 어휘 포함
- Transformer 효과를 명확히 확인 가능
- 어텐션 메커니즘 시각화에 적합

왜 Transformer를 사용하는가?
1. 병렬 처리: RNN과 달리 순차 처리 불필요
2. 장거리 의존성: 어텐션으로 직접 연결
3. 확장성: 대규모 모델과 데이터에 적합
4. 범용성: NLP 전 분야에서 SOTA 달성
5. 해석 가능성: 어텐션 가중치로 모델 동작 이해

RNN/LSTM vs Transformer:
- RNN: 순차 처리, 장거리 의존성 제한, 느린 훈련
- Transformer: 병렬 처리, 직접적 의존성, 빠른 훈련
- 성능: Transformer가 대부분 작업에서 우수
- 메모리: Transformer가 더 많은 메모리 사용
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import copy
import re
import warnings
from collections import Counter
import pickle
import os
warnings.filterwarnings('ignore')

# 우리가 만든 유틸리티 함수들 임포트
from utils.visualization import plot_training_curves
from utils.model_utils import count_parameters, save_checkpoint

print("🚀 딥러닝 강의 시리즈 8: Transformer NLP")
print("=" * 60)

# ============================================================================
# 1. 환경 설정 및 하이퍼파라미터
# =====================================================
import pickle
import os
from collections import Counter
import requests
import zipfile
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
import warnings
warnings.filterwarnings('ignore')

# 우리가 만든 유틸리티 함수들 임포트
from utils.visualization import plot_training_curves
from utils.model_utils import count_parameters, save_checkpoint

print("🚀 딥러닝 강의 시리즈 8: Transformer NLP")
print("=" * 60)# ==
==========================================================================
# 1. 환경 설정 및 하이퍼파라미터
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  사용 장치: {device}")

# Transformer를 위한 하이퍼파라미터
BATCH_SIZE = 32        # Transformer는 메모리를 많이 사용
LEARNING_RATE = 0.0001 # Transformer는 낮은 학습률 선호
EPOCHS = 50            # 충분한 학습 시간
RANDOM_SEED = 42
MAX_SEQ_LENGTH = 100   # 최대 시퀀스 길이
VOCAB_SIZE = 10000     # 어휘 사전 크기
D_MODEL = 512          # 모델 차원 (임베딩 차원)
N_HEADS = 8            # 멀티헤드 어텐션 헤드 수
N_LAYERS = 6           # 인코더/디코더 레이어 수
D_FF = 2048            # 피드포워드 네트워크 차원
DROPOUT = 0.1          # 드롭아웃 비율

# 재현성을 위한 시드 설정
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print(f"📊 하이퍼파라미터:")
print(f"   배치 크기: {BATCH_SIZE}")
print(f"   학습률: {LEARNING_RATE}")
print(f"   에포크: {EPOCHS}")
print(f"   최대 시퀀스 길이: {MAX_SEQ_LENGTH}")
print(f"   모델 차원: {D_MODEL}")
print(f"   어텐션 헤드: {N_HEADS}")
print(f"   레이어 수: {N_LAYERS}")

# 특수 토큰 정의
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'  # Start of Sentence
EOS_TOKEN = '<EOS>'  # End of Sentence
UNK_TOKEN = '<UNK>'  # Unknown

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

print(f"\n🏷️  특수 토큰:")
print(f"   PAD: {PAD_IDX}, SOS: {SOS_IDX}, EOS: {EOS_IDX}, UNK: {UNK_IDX}")

# ============================================================================
# 2. 데이터 전처리 클래스
# ============================================================================

print(f"\n📝 번역 데이터 전처리 시스템")

class TranslationDataProcessor:
    """
    기계 번역을 위한 데이터 전처리 클래스
    
    주요 기능:
    1. 텍스트 정제 및 토큰화
    2. 어휘 사전 구축 (소스/타겟 언어별)
    3. 시퀀스 변환 및 패딩
    4. 배치 생성
    """
    
    def __init__(self, max_vocab_size=10000, max_seq_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        
        # 어휘 사전 (소스/타겟 언어별)
        self.src_word2idx = {}
        self.src_idx2word = {}
        self.tgt_word2idx = {}
        self.tgt_idx2word = {}
        
        # 단어 빈도
        self.src_word_counts = Counter()
        self.tgt_word_counts = Counter()
    
    def clean_text(self, text):
        """
        텍스트 정제
        
        번역 데이터 전처리:
        1. 소문자 변환
        2. 구두점 분리
        3. 연속 공백 제거
        4. 특수 문자 처리
        """
        # 소문자 변환
        text = text.lower().strip()
        
        # 구두점 앞뒤에 공백 추가
        text = re.sub(r'([.!?,:;])', r' \1 ', text)
        
        # 연속된 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def tokenize(self, text):
        """간단한 공백 기반 토큰화"""
        return self.clean_text(text).split()
    
    def build_vocab(self, src_sentences, tgt_sentences):
        """
        소스/타겟 언어의 어휘 사전 구축
        
        Args:
            src_sentences: 소스 언어 문장 리스트
            tgt_sentences: 타겟 언어 문장 리스트
        """
        print("📚 어휘 사전 구축 중...")
        
        # 단어 빈도 계산
        for sentence in tqdm(src_sentences, desc="소스 언어 분석"):
            tokens = self.tokenize(sentence)
            self.src_word_counts.update(tokens)
        
        for sentence in tqdm(tgt_sentences, desc="타겟 언어 분석"):
            tokens = self.tokenize(sentence)
            self.tgt_word_counts.update(tokens)
        
        # 소스 언어 어휘 사전
        self._build_single_vocab(
            self.src_word_counts, 
            self.src_word2idx, 
            self.src_idx2word,
            "소스 (영어)"
        )
        
        # 타겟 언어 어휘 사전
        self._build_single_vocab(
            self.tgt_word_counts,
            self.tgt_word2idx,
            self.tgt_idx2word, 
            "타겟 (독일어)"
        )
    
    def _build_single_vocab(self, word_counts, word2idx, idx2word, lang_name):
        """단일 언어 어휘 사전 구축"""
        
        # 특수 토큰 먼저 추가
        vocab_words = SPECIAL_TOKENS.copy()
        
        # 빈도순으로 정렬하여 상위 단어들 선택
        most_common = word_counts.most_common(self.max_vocab_size - len(SPECIAL_TOKENS))
        vocab_words.extend([word for word, count in most_common])
        
        # 인덱스 매핑 생성
        for idx, word in enumerate(vocab_words):
            word2idx[word] = idx
            idx2word[idx] = word
        
        print(f"✅ {lang_name} 어휘 사전:")
        print(f"   총 단어 수: {len(word_counts):,}개")
        print(f"   어휘 사전 크기: {len(word2idx):,}개")
        print(f"   상위 단어: {[word for word, _ in most_common[:10]]}")
    
    def sentence_to_indices(self, sentence, word2idx, add_eos=False):
        """
        문장을 인덱스 시퀀스로 변환
        
        Args:
            sentence: 입력 문장
            word2idx: 단어-인덱스 매핑
            add_eos: EOS 토큰 추가 여부
        
        Returns:
            list: 인덱스 시퀀스
        """
        tokens = self.tokenize(sentence)
        
        # 길이 제한
        if len(tokens) > self.max_seq_length - (2 if add_eos else 1):
            tokens = tokens[:self.max_seq_length - (2 if add_eos else 1)]
        
        # 인덱스 변환
        indices = []
        for token in tokens:
            if token in word2idx:
                indices.append(word2idx[token])
            else:
                indices.append(word2idx[UNK_TOKEN])
        
        # EOS 토큰 추가 (타겟 시퀀스용)
        if add_eos:
            indices.append(EOS_IDX)
        
        return indices

# ============================================================================
# 3. Multi30k 데이터셋 클래스
# ============================================================================

class Multi30kDataset(Dataset):
    """
    Multi30k 번역 데이터셋 클래스
    
    실제 Multi30k 데이터 대신 교육용 샘플 데이터를 생성합니다.
    실제 프로젝트에서는 공식 Multi30k 데이터를 사용하세요.
    """
    
    def __init__(self, processor, use_sample_data=True):
        self.processor = processor
        
        if use_sample_data:
            self.create_sample_data()
        else:
            self.load_multi30k_data()
        
        # 데이터 전처리
        self.prepare_data()
    
    def create_sample_data(self):
        """
        교육용 샘플 번역 데이터 생성
        
        영어-독일어 번역 쌍을 시뮬레이션합니다.
        실제 사용 시에는 진짜 Multi30k 데이터를 사용하세요.
        """
        print("🌍 교육용 번역 샘플 데이터 생성 중...")
        
        # 기본 어휘
        english_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'man', 'woman', 'child', 'people', 'person', 'boy', 'girl', 'dog', 'cat', 'house',
            'car', 'book', 'table', 'chair', 'water', 'food', 'good', 'bad', 'big', 'small',
            'red', 'blue', 'green', 'black', 'white', 'walk', 'run', 'eat', 'drink', 'see',
            'go', 'come', 'have', 'be', 'do', 'make', 'get', 'take', 'give', 'say', 'know'
        ]
        
        german_words = [
            'der', 'die', 'das', 'ein', 'eine', 'und', 'oder', 'aber', 'in', 'auf', 'zu', 'für', 'von', 'mit',
            'mann', 'frau', 'kind', 'leute', 'person', 'junge', 'mädchen', 'hund', 'katze', 'haus',
            'auto', 'buch', 'tisch', 'stuhl', 'wasser', 'essen', 'gut', 'schlecht', 'groß', 'klein',
            'rot', 'blau', 'grün', 'schwarz', 'weiß', 'gehen', 'laufen', 'essen', 'trinken', 'sehen',
            'gehen', 'kommen', 'haben', 'sein', 'machen', 'bekommen', 'nehmen', 'geben', 'sagen', 'wissen'
        ]
        
        # 샘플 문장 패턴
        patterns = [
            ("the {noun} is {adj}", "der {noun} ist {adj}"),
            ("a {adj} {noun}", "ein {adj} {noun}"),
            ("I {verb} the {noun}", "ich {verb} den {noun}"),
            ("the {noun} {verb}", "der {noun} {verb}"),
            ("{adj} {noun} and {adj} {noun}", "{adj} {noun} und {adj} {noun}")
        ]
        
        self.src_sentences = []
        self.tgt_sentences = []
        
        # 샘플 문장 생성
        np.random.seed(RANDOM_SEED)
        
        for _ in range(2000):  # 2000개 샘플 생성
            # 패턴 선택
            en_pattern, de_pattern = np.random.choice(patterns)
            
            # 단어 선택
            replacements = {}
            if '{noun}' in en_pattern:
                noun_idx = np.random.randint(13, 24)  # 명사 범위
                replacements['noun'] = english_words[noun_idx]
            if '{adj}' in en_pattern:
                adj_idx = np.random.randint(29, 39)   # 형용사 범위
                replacements['adj'] = english_words[adj_idx]
            if '{verb}' in en_pattern:
                verb_idx = np.random.randint(39, 50)  # 동사 범위
                replacements['verb'] = english_words[verb_idx]
            
            # 영어 문장 생성
            en_sentence = en_pattern
            for key, value in replacements.items():
                en_sentence = en_sentence.replace(f'{{{key}}}', value)
            
            # 독일어 문장 생성 (단순 매핑)
            de_sentence = de_pattern
            for key, value in replacements.items():
                # 영어-독일어 단어 매핑 (단순화)
                if key == 'noun':
                    de_word = german_words[english_words.index(value)]
                elif key == 'adj':
                    de_word = german_words[english_words.index(value)]
                elif key == 'verb':
                    de_word = german_words[english_words.index(value)]
                else:
                    de_word = value
                
                de_sentence = de_sentence.replace(f'{{{key}}}', de_word)
            
            self.src_sentences.append(en_sentence)
            self.tgt_sentences.append(de_sentence)
        
        print(f"✅ {len(self.src_sentences)}개 번역 쌍 생성 완료")
        
        # 샘플 출력
        print(f"\n📝 샘플 번역 쌍:")
        for i in range(5):
            print(f"   EN: {self.src_sentences[i]}")
            print(f"   DE: {self.tgt_sentences[i]}")
            print()
    
    def load_multi30k_data(self):
        """실제 Multi30k 데이터 로드 (실제 사용 시)"""
        # 실제 구현에서는 Multi30k 데이터 로드
        # 여기서는 샘플 데이터 사용
        self.create_sample_data()
    
    def prepare_data(self):
        """데이터 전처리 및 인덱스 변환"""
        print("🔧 데이터 전처리 중...")
        
        # 어휘 사전 구축
        self.processor.build_vocab(self.src_sentences, self.tgt_sentences)
        
        # 문장을 인덱스 시퀀스로 변환
        self.src_sequences = []
        self.tgt_sequences = []
        
        for src_sent, tgt_sent in tqdm(zip(self.src_sentences, self.tgt_sentences), 
                                      desc="시퀀스 변환", total=len(self.src_sentences)):
            
            # 소스 시퀀스 (EOS 없음)
            src_seq = self.processor.sentence_to_indices(
                src_sent, self.processor.src_word2idx, add_eos=False
            )
            
            # 타겟 시퀀스 (EOS 포함)
            tgt_seq = self.processor.sentence_to_indices(
                tgt_sent, self.processor.tgt_word2idx, add_eos=True
            )
            
            self.src_sequences.append(src_seq)
            self.tgt_sequences.append(tgt_seq)
        
        print(f"✅ 전처리 완료: {len(self.src_sequences)}개 시퀀스")
    
    def __len__(self):
        return len(self.src_sequences)
    
    def __getitem__(self, idx):
        return {
            'src': torch.tensor(self.src_sequences[idx], dtype=torch.long),
            'tgt': torch.tensor(self.tgt_sequences[idx], dtype=torch.long)
        }

def collate_fn(batch):
    """
    배치 데이터 처리 함수
    
    Args:
        batch: 배치 데이터 리스트
    
    Returns:
        dict: 패딩된 배치 데이터
    """
    src_sequences = [item['src'] for item in batch]
    tgt_sequences = [item['tgt'] for item in batch]
    
    # 패딩
    src_padded = pad_sequence(src_sequences, batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_sequences, batch_first=True, padding_value=PAD_IDX)
    
    return {
        'src': src_padded,
        'tgt': tgt_padded,
        'src_lengths': torch.tensor([len(seq) for seq in src_sequences]),
        'tgt_lengths': torch.tensor([len(seq) for seq in tgt_sequences])
    }

# 데이터 준비
processor = TranslationDataProcessor(
    max_vocab_size=VOCAB_SIZE,
    max_seq_length=MAX_SEQ_LENGTH
)

dataset = Multi30kDataset(processor, use_sample_data=True)

# 데이터 분할
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

# 데이터 로더 생성
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
    collate_fn=collate_fn, num_workers=2
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, num_workers=2
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    collate_fn=collate_fn, num_workers=2
)

print(f"\n✅ 데이터 로더 생성 완료:")
print(f"   훈련: {len(train_dataset):,}개")
print(f"   검증: {len(val_dataset):,}개")
print(f"   테스트: {len(test_dataset):,}개")

# 실제 어휘 사전 크기 업데이트
SRC_VOCAB_SIZE = len(processor.src_word2idx)
TGT_VOCAB_SIZE = len(processor.tgt_word2idx)

print(f"   소스 어휘: {SRC_VOCAB_SIZE:,}개")
print(f"   타겟 어휘: {TGT_VOCAB_SIZE:,}개")#
 ============================================================================
# 4. Transformer 모델 구현
# ============================================================================

print(f"\n🤖 Transformer 모델 구현")

class PositionalEncoding(nn.Module):
    """
    위치 인코딩 (Positional Encoding)
    
    Transformer는 순환 구조가 없어 위치 정보가 없습니다.
    위치 인코딩으로 각 토큰의 위치 정보를 제공합니다.
    
    수식: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
          PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    왜 sin/cos 함수인가?
    1. 주기성: 다양한 길이의 시퀀스 처리
    2. 상대적 위치: 위치 간 관계 학습 가능
    3. 외삽: 훈련보다 긴 시퀀스도 처리 가능
    """
    
    def __init__(self, d_model, max_length=5000):
        super(PositionalEncoding, self).__init__()
        
        # 위치 인코딩 테이블 생성
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length).unsqueeze(1).float()
        
        # 주파수 계산
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        # sin/cos 적용
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스
        
        # 배치 차원 추가
        pe = pe.unsqueeze(0)
        
        # 학습되지 않는 파라미터로 등록
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class MultiHeadAttention(nn.Module):
    """
    멀티헤드 어텐션 (Multi-Head Attention)
    
    핵심 아이디어:
    1. 여러 개의 어텐션 헤드로 다양한 관점에서 정보 수집
    2. 각 헤드는 서로 다른 표현 부공간에 집중
    3. 헤드들의 결과를 결합하여 풍부한 표현 생성
    
    Self-Attention 수식:
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    
    왜 √d_k로 나누는가?
    - 내적값이 커지면 softmax가 포화되어 그래디언트 소실
    - 스케일링으로 안정적인 학습 보장
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Query, Key, Value 변환 행렬
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 출력 변환 행렬
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        스케일드 닷 프로덕트 어텐션
        
        Args:
            Q: Query (batch_size, n_heads, seq_len, d_k)
            K: Key (batch_size, n_heads, seq_len, d_k)
            V: Value (batch_size, n_heads, seq_len, d_k)
            mask: 어텐션 마스크
        
        Returns:
            output: 어텐션 출력
            attention_weights: 어텐션 가중치
        """
        
        # 어텐션 점수 계산
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 마스킹 적용 (패딩이나 미래 토큰 차단)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 소프트맥스로 확률 변환
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Value와 가중합
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, query, key, value, mask=None):
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 1. Query, Key, Value 변환
        Q = self.w_q(query)  # (batch_size, seq_len, d_model)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # 2. 멀티헤드로 분할
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. 어텐션 계산
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )
        
        # 4. 헤드 결합
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # 5. 출력 변환
        output = self.w_o(attention_output)
        
        return output, attention_weights

class FeedForward(nn.Module):
    """
    피드포워드 네트워크
    
    구조: Linear → ReLU → Dropout → Linear
    
    역할:
    1. 비선형 변환으로 표현력 증가
    2. 각 위치별로 독립적 처리
    3. 어텐션으로 수집된 정보를 정제
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    """
    Transformer 인코더 레이어
    
    구조:
    1. Multi-Head Self-Attention
    2. Add & Norm (잔차 연결 + 레이어 정규화)
    3. Feed-Forward Network
    4. Add & Norm
    
    잔차 연결의 중요성:
    - 그래디언트 흐름 개선
    - 깊은 네트워크 훈련 가능
    - 정보 손실 방지
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 1. Self-Attention + 잔차 연결
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Feed-Forward + 잔차 연결
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights

class TransformerDecoderLayer(nn.Module):
    """
    Transformer 디코더 레이어
    
    구조:
    1. Masked Multi-Head Self-Attention (미래 토큰 차단)
    2. Add & Norm
    3. Multi-Head Cross-Attention (인코더 출력과 어텐션)
    4. Add & Norm
    5. Feed-Forward Network
    6. Add & Norm
    
    마스킹의 중요성:
    - 훈련 시 미래 정보 누수 방지
    - 추론 시와 동일한 조건 유지
    - 자기회귀적 생성 보장
    """
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 1. Masked Self-Attention
        self_attn_output, self_attn_weights = self.self_attention(
            x, x, x, tgt_mask
        )
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # 2. Cross-Attention (인코더 출력과)
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # 3. Feed-Forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, self_attn_weights, cross_attn_weights

class Transformer(nn.Module):
    """
    완전한 Transformer 모델 (기계 번역용)
    
    구조:
    - 인코더: 소스 언어 이해
    - 디코더: 타겟 언어 생성
    - 임베딩: 토큰을 벡터로 변환
    - 위치 인코딩: 위치 정보 추가
    """
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8, 
                 n_layers=6, d_ff=2048, max_length=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        
        # 임베딩 레이어
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        
        # 인코더 레이어들
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 디코더 레이어들
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # 출력 투영
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        
        # 가중치 초기화
        self.init_weights()
    
    def init_weights(self):
        """Xavier 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, seq, pad_idx=PAD_IDX):
        """패딩 마스크 생성"""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size):
        """미래 토큰 차단 마스크 생성"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def encode(self, src, src_mask=None):
        """인코더 순전파"""
        # 임베딩 + 위치 인코딩
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 인코더 레이어들 통과
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn_weights = layer(x, src_mask)
            attention_weights.append(attn_weights)
        
        return x, attention_weights
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """디코더 순전파"""
        # 임베딩 + 위치 인코딩
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 디코더 레이어들 통과
        self_attention_weights = []
        cross_attention_weights = []
        
        for layer in self.decoder_layers:
            x, self_attn, cross_attn = layer(x, encoder_output, src_mask, tgt_mask)
            self_attention_weights.append(self_attn)
            cross_attention_weights.append(cross_attn)
        
        return x, self_attention_weights, cross_attention_weights
    
    def forward(self, src, tgt):
        """전체 순전파"""
        batch_size, src_len = src.size()
        _, tgt_len = tgt.size()
        
        # 마스크 생성
        src_mask = self.create_padding_mask(src).to(src.device)
        
        tgt_padding_mask = self.create_padding_mask(tgt).to(tgt.device)
        tgt_look_ahead_mask = self.create_look_ahead_mask(tgt_len).to(tgt.device)
        tgt_mask = tgt_padding_mask & tgt_look_ahead_mask
        
        # 인코딩
        encoder_output, encoder_attention = self.encode(src, src_mask)
        
        # 디코딩
        decoder_output, decoder_self_attention, decoder_cross_attention = self.decode(
            tgt, encoder_output, src_mask, tgt_mask
        )
        
        # 출력 투영
        output = self.output_projection(decoder_output)
        
        return output, {
            'encoder_attention': encoder_attention,
            'decoder_self_attention': decoder_self_attention,
            'decoder_cross_attention': decoder_cross_attention
        }

# 모델 인스턴스 생성
model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE,
    tgt_vocab_size=TGT_VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    dropout=DROPOUT
).to(device)

print(f"✅ Transformer 모델 생성 완료")

# 모델 복잡도 분석
params_info = count_parameters(model, detailed=False)
print(f"\n📊 모델 정보:")
print(f"   총 파라미터: {params_info['total_params']:,}개")
print(f"   훈련 가능한 파라미터: {params_info['trainable_params']:,}개")

# 모델 구조 요약
print(f"\n📋 Transformer 구조:")
print(f"   소스 어휘: {SRC_VOCAB_SIZE:,}개")
print(f"   타겟 어휘: {TGT_VOCAB_SIZE:,}개")
print(f"   모델 차원: {D_MODEL}")
print(f"   어텐션 헤드: {N_HEADS}개")
print(f"   인코더/디코더 레이어: {N_LAYERS}개")
print(f"   피드포워드 차원: {D_FF}")

# ============================================================================
# 5. 손실 함수와 옵티마이저 설정
# ============================================================================

print(f"\n⚙️  손실 함수와 옵티마이저 설정")

# 라벨 스무딩을 적용한 CrossEntropyLoss
class LabelSmoothingLoss(nn.Module):
    """
    라벨 스무딩 손실 함수
    
    라벨 스무딩의 효과:
    1. 과신뢰 방지: 모델이 너무 확신하지 않도록
    2. 일반화 향상: 더 부드러운 확률 분포 학습
    3. 정규화 효과: 과적합 방지
    
    수식: y_smooth = (1-α)y_true + α/K
    여기서 α는 스무딩 계수, K는 클래스 수
    """
    
    def __init__(self, vocab_size, smoothing=0.1, ignore_index=PAD_IDX):
        super(LabelSmoothingLoss, self).__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        # pred: (batch_size * seq_len, vocab_size)
        # target: (batch_size * seq_len)
        
        batch_size, seq_len, vocab_size = pred.size()
        pred = pred.view(-1, vocab_size)
        target = target.view(-1)
        
        # 유효한 토큰만 선택 (패딩 제외)
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]
        
        if pred.size(0) == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 라벨 스무딩 적용
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        
        # KL Divergence 계산
        log_pred = F.log_softmax(pred, dim=1)
        loss = F.kl_div(log_pred, true_dist, reduction='batchmean')
        
        return loss

# 손실 함수
criterion = LabelSmoothingLoss(
    vocab_size=TGT_VOCAB_SIZE,
    smoothing=0.1,
    ignore_index=PAD_IDX
)

# 옵티마이저 (Adam with warmup)
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.98),
    eps=1e-9
)

# 학습률 스케줄러 (Warmup + Cosine Annealing)
class WarmupScheduler:
    """
    Transformer를 위한 Warmup 스케줄러
    
    Warmup의 필요성:
    1. 초기 큰 그래디언트로 인한 불안정성 방지
    2. 점진적 학습률 증가로 안정적 학습
    3. Transformer의 표준 학습 방법
    
    수식: lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

# 스케줄러 생성
scheduler = WarmupScheduler(optimizer, D_MODEL, warmup_steps=4000)

print(f"   손실 함수: Label Smoothing CrossEntropy")
print(f"   옵티마이저: Adam with Warmup")
print(f"   스케줄러: Warmup + Cosine Annealing")

# ============================================================================
# 6. BLEU 점수 계산 함수
# ============================================================================

print(f"\n📊 BLEU 점수 평가 시스템")

def calculate_bleu_score(predictions, references, max_n=4):
    """
    BLEU (Bilingual Evaluation Understudy) 점수 계산
    
    BLEU는 기계 번역의 표준 평가 메트릭:
    1. n-gram 정밀도 계산 (1-gram ~ 4-gram)
    2. 길이 패널티 적용
    3. 기하 평균으로 최종 점수 계산
    
    Args:
        predictions: 예측 문장들 (토큰 리스트의 리스트)
        references: 참조 문장들 (토큰 리스트의 리스트)
        max_n: 최대 n-gram 크기
    
    Returns:
        float: BLEU 점수 (0~1)
    """
    
    if len(predictions) != len(references):
        return 0.0
    
    # 간소화된 BLEU 계산
    total_score = 0.0
    valid_pairs = 0
    
    for pred, ref in zip(predictions, references):
        if len(pred) == 0 or len(ref) == 0:
            continue
        
        # 1-gram 정밀도 (단어 일치율)
        pred_words = set(pred)
        ref_words = set(ref)
        
        if len(pred_words) == 0:
            continue
        
        precision = len(pred_words & ref_words) / len(pred_words)
        
        # 길이 패널티
        length_penalty = min(1.0, len(pred) / len(ref)) if len(ref) > 0 else 0.0
        
        # BLEU 점수 (간소화)
        bleu = precision * length_penalty
        
        total_score += bleu
        valid_pairs += 1
    
    return total_score / valid_pairs if valid_pairs > 0 else 0.0

# ============================================================================
# 7. 훈련 및 검증 함수
# ============================================================================

def train_epoch_transformer(model, train_loader, criterion, scheduler, device):
    """Transformer 훈련 함수"""
    model.train()
    
    running_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="훈련")
    
    for batch in pbar:
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        # 타겟 입력과 출력 분리
        tgt_input = tgt[:, :-1]  # <SOS> ~ 마지막 전 토큰
        tgt_output = tgt[:, 1:]  # 첫 토큰 ~ <EOS>
        
        scheduler.zero_grad()
        
        # 순전파
        output, attention_weights = model(src, tgt_input)
        
        # 손실 계산
        loss = criterion(output, tgt_output)
        
        # 역전파
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 파라미터 업데이트
        scheduler.step()
        
        # 통계 업데이트
        running_loss += loss.item()
        num_batches += 1
        
        # 진행률 바 업데이트
        if num_batches % 10 == 0:
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return running_loss / num_batches

def validate_epoch_transformer(model, val_loader, criterion, device):
    """Transformer 검증 함수"""
    model.eval()
    
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="검증")
        
        for batch in pbar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output, _ = model(src, tgt_input)
            loss = criterion(output, tgt_output)
            
            running_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    return running_loss / num_batches

# ============================================================================
# 8. 번역 생성 함수
# ============================================================================

def translate_sentence(model, src_sentence, src_word2idx, tgt_idx2word, 
                      device, max_length=50):
    """
    문장 번역 함수
    
    Args:
        model: 훈련된 Transformer 모델
        src_sentence: 소스 문장 (문자열)
        src_word2idx: 소스 언어 단어-인덱스 매핑
        tgt_idx2word: 타겟 언어 인덱스-단어 매핑
        device: 디바이스
        max_length: 최대 생성 길이
    
    Returns:
        str: 번역된 문장
    """
    model.eval()
    
    with torch.no_grad():
        # 소스 문장 전처리
        src_tokens = processor.tokenize(src_sentence)
        src_indices = []
        
        for token in src_tokens:
            if token in src_word2idx:
                src_indices.append(src_word2idx[token])
            else:
                src_indices.append(src_word2idx[UNK_TOKEN])
        
        # 텐서로 변환
        src = torch.tensor([src_indices], device=device)
        
        # 인코딩
        encoder_output, _ = model.encode(src)
        
        # 디코딩 (자기회귀적 생성)
        tgt_indices = [SOS_IDX]  # <SOS>로 시작
        
        for _ in range(max_length):
            tgt = torch.tensor([tgt_indices], device=device)
            
            # 디코딩
            decoder_output, _, _ = model.decode(tgt, encoder_output)
            
            # 다음 토큰 예측
            next_token_logits = model.output_projection(decoder_output[:, -1, :])
            next_token = torch.argmax(next_token_logits, dim=-1).item()
            
            # <EOS>면 종료
            if next_token == EOS_IDX:
                break
            
            tgt_indices.append(next_token)
        
        # 인덱스를 단어로 변환
        translated_tokens = []
        for idx in tgt_indices[1:]:  # <SOS> 제외
            if idx in tgt_idx2word:
                translated_tokens.append(tgt_idx2word[idx])
            else:
                translated_tokens.append(UNK_TOKEN)
        
        return ' '.join(translated_tokens)

# ============================================================================
# 9. 모델 훈련 실행
# ============================================================================

print(f"\n🚀 Transformer 훈련 시작")

# 훈련 기록
train_losses = []
val_losses = []
bleu_scores = []

# 최고 성능 추적
best_val_loss = float('inf')
best_bleu_score = 0.0
best_model_state = None
patience = 10
patience_counter = 0

start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\n📅 에포크 {epoch+1}/{EPOCHS}")
    
    # 훈련
    train_loss = train_epoch_transformer(model, train_loader, criterion, scheduler, device)
    
    # 검증
    val_loss = validate_epoch_transformer(model, val_loader, criterion, device)
    
    # 기록 저장
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # 결과 출력
    print(f"   훈련 손실: {train_loss:.4f}")
    print(f"   검증 손실: {val_loss:.4f}")
    
    # 주기적으로 BLEU 점수 계산
    if (epoch + 1) % 5 == 0:
        print(f"\n📊 BLEU 점수 계산 중...")
        
        # 샘플 번역 수행
        sample_predictions = []
        sample_references = []
        
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 10:  # 10개 배치만 평가
                    break
                
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                
                for j in range(min(5, src.size(0))):  # 배치당 5개 샘플
                    # 소스 문장 복원
                    src_indices = src[j].cpu().tolist()
                    src_tokens = [processor.src_idx2word.get(idx, UNK_TOKEN) 
                                for idx in src_indices if idx != PAD_IDX]
                    src_sentence = ' '.join(src_tokens)
                    
                    # 참조 번역 복원
                    tgt_indices = tgt[j].cpu().tolist()
                    ref_tokens = [processor.tgt_idx2word.get(idx, UNK_TOKEN) 
                                for idx in tgt_indices if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]]
                    
                    # 번역 생성
                    translated = translate_sentence(
                        model, src_sentence, processor.src_word2idx, 
                        processor.tgt_idx2word, device
                    )
                    
                    pred_tokens = translated.split()
                    
                    sample_predictions.append(pred_tokens)
                    sample_references.append(ref_tokens)
        
        # BLEU 점수 계산
        bleu_score = calculate_bleu_score(sample_predictions, sample_references)
        bleu_scores.append(bleu_score)
        
        print(f"   BLEU 점수: {bleu_score:.4f}")
        
        # 샘플 번역 출력
        print(f"\n🔤 샘플 번역:")
        for i in range(min(3, len(sample_predictions))):
            src_text = ' '.join([processor.src_idx2word.get(idx, UNK_TOKEN) 
                               for idx in val_loader.dataset[i]['src'].tolist() 
                               if idx != PAD_IDX])
            print(f"   소스: {src_text}")
            print(f"   참조: {' '.join(sample_references[i])}")
            print(f"   번역: {' '.join(sample_predictions[i])}")
            print()
    
    # 최고 성능 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        print(f"   🎯 새로운 최고 성능! 검증 손실: {val_loss:.4f}")
        
        # 체크포인트 저장
        save_checkpoint(
            model, optimizer, epoch, val_loss, bleu_scores[-1] if bleu_scores else 0,
            save_path="./checkpoints/transformer_best_model.pth"
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
print(f"   최고 BLEU 점수: {max(bleu_scores) if bleu_scores else 0:.4f}")

# ============================================================================
# 10. 훈련 결과 시각화
# ============================================================================

print(f"\n📈 훈련 결과 시각화")

# 훈련 곡선
plot_training_curves(
    train_losses=train_losses,
    val_losses=val_losses,
    title="Transformer 기계 번역 - 훈련 과정"
)

# BLEU 점수 변화
if bleu_scores:
    plt.figure(figsize=(10, 6))
    epochs_with_bleu = [i * 5 + 5 for i in range(len(bleu_scores))]
    plt.plot(epochs_with_bleu, bleu_scores, 'g-', marker='o', linewidth=2, markersize=6)
    plt.title('BLEU 점수 변화', fontsize=16, fontweight='bold')
    plt.xlabel('에포크')
    plt.ylabel('BLEU 점수')
    plt.grid(True, alpha=0.3)
    plt.show()

# ============================================================================
# 11. 어텐션 가중치 시각화
# ============================================================================

print(f"\n🔍 어텐션 가중치 시각화")

def visualize_attention(model, src_sentence, tgt_sentence, processor, device):
    """
    어텐션 가중치 시각화
    
    Self-Attention과 Cross-Attention의 가중치를 히트맵으로 표시하여
    모델이 어떤 부분에 집중하는지 시각적으로 확인
    """
    model.eval()
    
    with torch.no_grad():
        # 문장 전처리
        src_tokens = processor.tokenize(src_sentence)
        tgt_tokens = processor.tokenize(tgt_sentence)
        
        src_indices = processor.sentence_to_indices(src_sentence, processor.src_word2idx)
        tgt_indices = processor.sentence_to_indices(tgt_sentence, processor.tgt_word2idx, add_eos=True)
        
        # 텐서로 변환
        src = torch.tensor([src_indices], device=device)
        tgt = torch.tensor([tgt_indices[:-1]], device=device)  # EOS 제외
        
        # 순전파 (어텐션 가중치 포함)
        output, attention_weights = model(src, tgt)
        
        # Cross-Attention 가중치 (마지막 레이어, 첫 번째 헤드)
        cross_attention = attention_weights['decoder_cross_attention'][-1][0, 0]  # (tgt_len, src_len)
        cross_attention = cross_attention.cpu().numpy()
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Cross-Attention 히트맵
        im1 = ax1.imshow(cross_attention, cmap='Blues', aspect='auto')
        ax1.set_title('Cross-Attention (Decoder → Encoder)', fontweight='bold')
        ax1.set_xlabel('소스 토큰')
        ax1.set_ylabel('타겟 토큰')
        
        # 토큰 라벨 설정
        ax1.set_xticks(range(len(src_tokens)))
        ax1.set_xticklabels(src_tokens, rotation=45)
        ax1.set_yticks(range(len(tgt_tokens)))
        ax1.set_yticklabels(tgt_tokens)
        
        plt.colorbar(im1, ax=ax1)
        
        # Self-Attention 가중치 (인코더 마지막 레이어, 첫 번째 헤드)
        if attention_weights['encoder_attention']:
            self_attention = attention_weights['encoder_attention'][-1][0, 0]  # (src_len, src_len)
            self_attention = self_attention.cpu().numpy()
            
            im2 = ax2.imshow(self_attention, cmap='Reds', aspect='auto')
            ax2.set_title('Self-Attention (Encoder)', fontweight='bold')
            ax2.set_xlabel('소스 토큰')
            ax2.set_ylabel('소스 토큰')
            
            ax2.set_xticks(range(len(src_tokens)))
            ax2.set_xticklabels(src_tokens, rotation=45)
            ax2.set_yticks(range(len(src_tokens)))
            ax2.set_yticklabels(src_tokens)
            
            plt.colorbar(im2, ax=ax2)
        
        plt.tight_layout()
        plt.show()

# 최고 성능 모델로 어텐션 시각화
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# 샘플 문장으로 어텐션 시각화
sample_src = "the man is walking"
sample_tgt = "der mann geht"

print(f"🔍 어텐션 가중치 분석:")
print(f"   소스: {sample_src}")
print(f"   타겟: {sample_tgt}")

visualize_attention(model, sample_src, sample_tgt, processor, device)

# ============================================================================
# 12. 대화형 번역 시스템
# ============================================================================

print(f"\n💬 대화형 번역 시스템")

def interactive_translation():
    """대화형 번역 인터페이스"""
    print("🌍 영어 → 독일어 번역기")
    print("'quit'를 입력하면 종료됩니다.")
    print("-" * 40)
    
    while True:
        try:
            # 사용자 입력
            english_text = input("\n영어 문장을 입력하세요: ").strip()
            
            if english_text.lower() == 'quit':
                print("번역기를 종료합니다.")
                break
            
            if not english_text:
                continue
            
            # 번역 수행
            german_translation = translate_sentence(
                model, english_text, processor.src_word2idx, 
                processor.tgt_idx2word, device
            )
            
            print(f"독일어 번역: {german_translation}")
            
        except KeyboardInterrupt:
            print("\n번역기를 종료합니다.")
            break
        except Exception as e:
            print(f"오류 발생: {e}")

# 대화형 번역 실행 (주석 처리 - 자동 실행 방지)
# interactive_translation()

# 샘플 번역 테스트
print(f"🔤 샘플 번역 테스트:")
test_sentences = [
    "the cat is sleeping",
    "a good book",
    "people are walking",
    "the red car",
    "I have water"
]

for sentence in test_sentences:
    translation = translate_sentence(
        model, sentence, processor.src_word2idx, 
        processor.tgt_idx2word, device
    )
    print(f"   EN: {sentence}")
    print(f"   DE: {translation}")
    print()

# ============================================================================
# 13. 학습 내용 요약 및 실용적 활용
# ============================================================================

print(f"\n🎓 학습 내용 요약")
print(f"=" * 60)

print(f"✅ 완료한 내용:")
print(f"   1. Transformer 아키텍처의 핵심 구성요소")
print(f"   2. Self-Attention과 Multi-Head Attention 메커니즘")
print(f"   3. 위치 인코딩과 마스킹 기법")
print(f"   4. 인코더-디코더 구조와 기계 번역")
print(f"   5. BLEU 점수를 통한 번역 품질 평가")
print(f"   6. 어텐션 가중치 시각화와 해석")

print(f"\n📊 최종 성과:")
print(f"   - 최고 검증 손실: {best_val_loss:.4f}")
print(f"   - 최고 BLEU 점수: {max(bleu_scores) if bleu_scores else 0:.4f}")
print(f"   - 모델 파라미터: {params_info['total_params']:,}개")
print(f"   - 훈련 시간: {training_time:.2f}초")

print(f"\n💡 핵심 학습 포인트:")
print(f"   1. 어텐션 메커니즘: 모든 위치 간 직접 연결")
print(f"   2. 병렬 처리: RNN과 달리 순차 처리 불필요")
print(f"   3. 위치 인코딩: 순서 정보 제공")
print(f"   4. 멀티헤드: 다양한 관점에서 정보 수집")
print(f"   5. 잔차 연결: 깊은 네트워크 안정적 학습")

print(f"\n🔍 Transformer의 장단점:")
print(f"   장점:")
print(f"   - 병렬 처리로 빠른 훈련")
print(f"   - 장거리 의존성 효과적 처리")
print(f"   - 확장성 (대규모 모델 가능)")
print(f"   - 해석 가능성 (어텐션 가중치)")
print(f"   단점:")
print(f"   - 높은 메모리 사용량")
print(f"   - 시퀀스 길이에 따른 계산 복잡도 증가")
print(f"   - 위치 정보 명시적 제공 필요")

print(f"\n🚀 실용적 활용 분야:")
print(f"   1. 기계 번역: Google Translate, DeepL")
print(f"   2. 언어 모델: GPT, BERT, T5")
print(f"   3. 텍스트 요약: 문서 자동 요약")
print(f"   4. 질의응답: 챗봇, 검색 시스템")
print(f"   5. 코드 생성: GitHub Copilot")
print(f"   6. 이미지 캡셔닝: Vision Transformer")

print(f"\n🔧 Transformer 발전 과정:")
print(f"   1. Transformer (2017): Attention Is All You Need")
print(f"   2. BERT (2018): 양방향 인코더")
print(f"   3. GPT (2018): 자기회귀 디코더")
print(f"   4. T5 (2019): Text-to-Text Transfer")
print(f"   5. GPT-3 (2020): 대규모 언어 모델")
print(f"   6. ChatGPT (2022): 대화형 AI")

print(f"\n🚀 다음 단계:")
print(f"   - 통합 테스트 및 문서화")
print(f"   - 전체 튜토리얼 시리즈 완성")
print(f"   - README 파일 업데이트")

print(f"\n🔧 추가 실험 아이디어:")
print(f"   1. 실제 Multi30k 데이터셋 사용")
print(f"   2. Beam Search 디코딩 구현")
print(f"   3. 다양한 어텐션 헤드 수 실험")
print(f"   4. 사전 훈련된 모델 파인튜닝")
print(f"   5. 다른 언어 쌍으로 확장")

print(f"\n" + "=" * 60)
print(f"🎉 Transformer NLP 튜토리얼 완료!")
print(f"   모든 딥러닝 튜토리얼이 완성되었습니다!")
print(f"=" * 60)
    betas=(0.9, 0.98),
    eps=1e-9
)

# 학습률 스케줄러 (Warmup + Cosine Annealing)
class WarmupScheduler:
    """
    Warmup + Cosine Annealing 스케줄러
    
    Warmup의 필요성:
    1. 초기 안정화: 큰 학습률로 인한 불안정성 방지
    2. 점진적 증가: 모델이 적응할 시간 제공
    3. Transformer 특성: 어텐션 메커니즘의 안정적 초기화
    """
    
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
    
    def step(self):
        self.step_num += 1
        lr = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5),
            self.step_num * self.warmup_steps ** (-1.5)
        )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.optimizer.step()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

scheduler = WarmupScheduler(optimizer, D_MODEL, warmup_steps=4000)

print(f"   손실 함수: LabelSmoothingLoss (smoothing=0.1)")
print(f"   옵티마이저: Adam (lr={LEARNING_RATE})")
print(f"   스케줄러: Warmup + Cosine Annealing")#
 ============================================================================
# 6. 훈련 함수 정의
# ============================================================================

def train_transformer_epoch(model, train_loader, criterion, scheduler, device, epoch):
    """Transformer 한 에포크 훈련"""
    model.train()
    
    running_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"에포크 {epoch+1} 훈련")
    
    for batch_idx, batch in enumerate(pbar):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        # 디코더 입력 (SOS 토큰 추가)
        tgt_input = torch.cat([
            torch.full((tgt.size(0), 1), SOS_IDX, device=device),
            tgt[:, :-1]
        ], dim=1)
        
        # 타겟 (EOS 토큰 포함)
        tgt_output = tgt
        
        scheduler.zero_grad()
        
        # 순전파
        output, attention_weights = model(src, tgt_input)
        
        # 손실 계산
        loss = criterion(output, tgt_output)
        
        # 역전파
        loss.backward()
        
        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 옵티마이저 스텝
        scheduler.step()
        
        running_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / num_batches
    return avg_loss

def validate_transformer_epoch(model, val_loader, criterion, device):
    """Transformer 검증"""
    model.eval()
    
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="검증")
        
        for batch in pbar:
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            tgt_input = torch.cat([
                torch.full((tgt.size(0), 1), SOS_IDX, device=device),
                tgt[:, :-1]
            ], dim=1)
            
            tgt_output = tgt
            
            output, _ = model(src, tgt_input)
            loss = criterion(output, tgt_output)
            
            running_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = running_loss / num_batches
    return avg_loss

# ============================================================================
# 7. 모델 훈련 실행
# ============================================================================

print(f"\n🚀 Transformer 훈련 시작")

# 훈련 기록
train_losses = []
val_losses = []

# 최고 성능 추적
best_val_loss = float('inf')
best_model_state = None
patience = 10
patience_counter = 0

start_time = time.time()

for epoch in range(EPOCHS):
    print(f"\n📅 에포크 {epoch+1}/{EPOCHS}")
    
    # 훈련
    train_loss = train_transformer_epoch(
        model, train_loader, criterion, scheduler, device, epoch
    )
    
    # 검증
    val_loss = validate_transformer_epoch(
        model, val_loader, criterion, device
    )
    
    # 기록 저장
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    
    # 결과 출력
    print(f"   훈련 손실: {train_loss:.4f}")
    print(f"   검증 손실: {val_loss:.4f}")
    
    # 최고 성능 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        patience_counter = 0
        print(f"   🎯 새로운 최고 성능! 검증 손실: {val_loss:.4f}")
        
        # 체크포인트 저장
        save_checkpoint(
            model, optimizer, epoch, val_loss, 0,
            save_path="./checkpoints/transformer_best.pth"
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
    title="Transformer 기계 번역 - 훈련 과정"
)

# ============================================================================
# 9. 번역 및 평가
# ============================================================================

print(f"\n🌍 번역 성능 평가")

def translate_sentence(model, sentence, processor, device, max_length=50):
    """
    단일 문장 번역
    
    Args:
        model: 훈련된 Transformer 모델
        sentence: 번역할 문장 (소스 언어)
        processor: 데이터 전처리기
        device: 연산 장치
        max_length: 최대 생성 길이
    
    Returns:
        str: 번역된 문장 (타겟 언어)
    """
    model.eval()
    
    # 소스 문장 전처리
    src_tokens = processor.sentence_to_indices(
        sentence, processor.src_word2idx, add_eos=False
    )
    src_tensor = torch.tensor([src_tokens], device=device)
    
    # 인코딩
    with torch.no_grad():
        encoder_output, _ = model.encode(src_tensor)
    
    # 디코딩 (자기회귀적 생성)
    tgt_tokens = [SOS_IDX]
    
    for _ in range(max_length):
        tgt_tensor = torch.tensor([tgt_tokens], device=device)
        
        with torch.no_grad():
            output, _ = model(src_tensor, tgt_tensor)
            
        # 다음 토큰 예측
        next_token_logits = output[0, -1, :]
        next_token = torch.argmax(next_token_logits).item()
        
        tgt_tokens.append(next_token)
        
        # EOS 토큰이면 종료
        if next_token == EOS_IDX:
            break
    
    # 토큰을 단어로 변환
    translated_words = []
    for token_id in tgt_tokens[1:]:  # SOS 제외
        if token_id == EOS_IDX:
            break
        if token_id in processor.tgt_idx2word:
            word = processor.tgt_idx2word[token_id]
            if word not in SPECIAL_TOKENS:
                translated_words.append(word)
    
    return ' '.join(translated_words)

def calculate_bleu_score(model, test_loader, processor, device):
    """
    BLEU 점수 계산
    
    BLEU (Bilingual Evaluation Understudy):
    - 기계 번역 품질 평가의 표준 메트릭
    - n-gram 정밀도의 기하 평균
    - 간결성 페널티 적용
    - 0~1 범위 (높을수록 좋음)
    """
    model.eval()
    
    references = []  # 정답 번역
    candidates = []  # 모델 번역
    
    print("🔍 BLEU 점수 계산을 위한 번역 중...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="번역")):
            if batch_idx >= 50:  # 계산 시간 단축을 위해 제한
                break
                
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            
            batch_size = src.size(0)
            
            for i in range(batch_size):
                # 소스 문장 복원
                src_tokens = src[i].cpu().numpy()
                src_words = []
                for token_id in src_tokens:
                    if token_id == PAD_IDX:
                        break
                    if token_id in processor.src_idx2word:
                        word = processor.src_idx2word[token_id]
                        if word not in SPECIAL_TOKENS:
                            src_words.append(word)
                
                src_sentence = ' '.join(src_words)
                
                # 정답 번역 복원
                tgt_tokens = tgt[i].cpu().numpy()
                ref_words = []
                for token_id in tgt_tokens:
                    if token_id in [PAD_IDX, EOS_IDX]:
                        break
                    if token_id in processor.tgt_idx2word:
                        word = processor.tgt_idx2word[token_id]
                        if word not in SPECIAL_TOKENS:
                            ref_words.append(word)
                
                # 모델 번역
                translated = translate_sentence(model, src_sentence, processor, device)
                
                if ref_words and translated:
                    references.append([ref_words])  # BLEU는 리스트의 리스트 형태
                    candidates.append(translated.split())
    
    if not references or not candidates:
        print("❌ 번역 결과가 없어 BLEU 점수를 계산할 수 없습니다.")
        return 0.0
    
    # BLEU 점수 계산
    try:
        bleu_score = corpus_bleu(references, candidates)
        return bleu_score
    except:
        print("❌ BLEU 점수 계산 중 오류 발생")
        return 0.0

# 최고 성능 모델 로드
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# 샘플 번역 테스트
print(f"\n📝 샘플 번역 테스트:")

sample_sentences = [
    "the man is good",
    "a big house",
    "the cat and the dog",
    "I see the book",
    "red car"
]

for sentence in sample_sentences:
    translation = translate_sentence(model, sentence, processor, device)
    print(f"   EN: {sentence}")
    print(f"   DE: {translation}")
    print()

# BLEU 점수 계산
bleu_score = calculate_bleu_score(model, test_loader, processor, device)
print(f"📊 BLEU 점수: {bleu_score:.4f}")

# ============================================================================
# 10. 어텐션 시각화
# ============================================================================

print(f"\n👁️  어텐션 가중치 시각화")

def visualize_attention(model, sentence, processor, device):
    """
    어텐션 가중치 시각화
    
    어텐션 시각화의 의미:
    1. 모델 해석: 어떤 단어에 집중하는지 확인
    2. 정렬 확인: 소스-타겟 단어 간 대응 관계
    3. 디버깅: 모델의 잘못된 동작 파악
    """
    model.eval()
    
    # 문장 전처리
    src_tokens = processor.sentence_to_indices(
        sentence, processor.src_word2idx, add_eos=False
    )
    src_tensor = torch.tensor([src_tokens], device=device)
    
    # 번역 생성 (어텐션 가중치 수집)
    tgt_tokens = [SOS_IDX]
    attention_weights = []
    
    src_words = []
    for token_id in src_tokens:
        if token_id in processor.src_idx2word:
            word = processor.src_idx2word[token_id]
            if word not in SPECIAL_TOKENS:
                src_words.append(word)
    
    tgt_words = []
    
    with torch.no_grad():
        for step in range(20):  # 최대 20 토큰
            tgt_tensor = torch.tensor([tgt_tokens], device=device)
            
            output, attn_weights = model(src_tensor, tgt_tensor)
            
            # 다음 토큰 예측
            next_token_logits = output[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
            
            tgt_tokens.append(next_token)
            
            # 어텐션 가중치 저장 (마지막 레이어, 첫 번째 헤드)
            cross_attn = attn_weights['decoder_cross_attention'][-1]  # 마지막 레이어
            attn_weight = cross_attn[0, 0, -1, :len(src_tokens)].cpu().numpy()  # 첫 번째 헤드
            attention_weights.append(attn_weight)
            
            # 단어 추가
            if next_token in processor.tgt_idx2word:
                word = processor.tgt_idx2word[next_token]
                if word == EOS_TOKEN:
                    break
                elif word not in SPECIAL_TOKENS:
                    tgt_words.append(word)
            
            if next_token == EOS_IDX:
                break
    
    # 어텐션 히트맵 시각화
    if attention_weights and src_words and tgt_words:
        attention_matrix = np.array(attention_weights)
        
        plt.figure(figsize=(12, 8))
        
        # 히트맵 그리기
        sns.heatmap(
            attention_matrix,
            xticklabels=src_words,
            yticklabels=tgt_words,
            cmap='Blues',
            cbar=True,
            square=False
        )
        
        plt.title(f'Cross-Attention 가중치\n소스: "{sentence}"')
        plt.xlabel('소스 단어 (영어)')
        plt.ylabel('타겟 단어 (독일어)')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        print(f"✅ 어텐션 시각화 완료")
        print(f"   소스: {' '.join(src_words)}")
        print(f"   타겟: {' '.join(tgt_words)}")
    else:
        print(f"❌ 어텐션 시각화 실패: 충분한 데이터가 없습니다.")

# 어텐션 시각화 실행
sample_sentence = "the man is good"
visualize_attention(model, sample_sentence, processor, device)

# ============================================================================
# 11. 학습 내용 요약 및 다음 단계
# ============================================================================

print(f"\n🎓 학습 내용 요약")
print(f"=" * 60)

print(f"✅ 완료한 내용:")
print(f"   1. Transformer 아키텍처의 핵심 구성 요소 구현")
print(f"   2. Self-Attention과 Multi-Head Attention 메커니즘")
print(f"   3. 위치 인코딩과 마스킹 기법")
print(f"   4. 인코더-디코더 구조로 기계 번역 구현")
print(f"   5. BLEU 점수를 통한 번역 품질 평가")
print(f"   6. 어텐션 가중치 시각화와 해석")

print(f"\n📊 최종 성과:")
print(f"   - 최고 검증 손실: {best_val_loss:.4f}")
print(f"   - BLEU 점수: {bleu_score:.4f}")
print(f"   - 총 파라미터: {params_info['total_params']:,}개")
print(f"   - 훈련 시간: {training_time:.2f}초")

print(f"\n💡 핵심 학습 포인트:")
print(f"   1. 어텐션 메커니즘: 장거리 의존성의 직접적 모델링")
print(f"   2. 병렬 처리: RNN 대비 훨씬 빠른 훈련과 추론")
print(f"   3. 위치 인코딩: 순서 정보 없는 구조에 위치 정보 주입")
print(f"   4. 마스킹: 훈련과 추론 시 정보 누수 방지")
print(f"   5. 잔차 연결: 깊은 네트워크의 안정적 훈련")

print(f"\n🔍 Transformer의 혁신:")
print(f"   1. 'Attention is All You Need': 순환/합성곱 없이 어텐션만으로")
print(f"   2. 확장성: 모델과 데이터 크기에 따른 성능 향상")
print(f"   3. 범용성: NLP 전 분야에서 SOTA 달성")
print(f"   4. 전이 학습: 사전 훈련된 모델의 파인튜닝")
print(f"   5. 해석 가능성: 어텐션 가중치로 모델 동작 이해")

print(f"\n🚀 Transformer 발전사:")
print(f"   - 2017: 원조 Transformer (기계 번역)")
print(f"   - 2018: BERT (양방향 인코더)")
print(f"   - 2019: GPT-2 (대규모 언어 모델)")
print(f"   - 2020: GPT-3 (1750억 파라미터)")
print(f"   - 2022: ChatGPT (대화형 AI)")
print(f"   - 2023: GPT-4 (멀티모달)")

print(f"\n🔧 추가 실험 아이디어:")
print(f"   1. 사전 훈련된 모델: BERT, GPT 파인튜닝")
print(f"   2. 다른 NLP 태스크: 감정 분석, 질의응답, 요약")
print(f"   3. 멀티모달: 이미지-텍스트 결합 모델")
print(f"   4. 효율적 어텐션: Sparse Attention, Linear Attention")
print(f"   5. 모델 압축: 지식 증류, 양자화")

print(f"\n🎯 실제 응용 분야:")
print(f"   - 기계 번역: Google 번역, DeepL")
print(f"   - 검색 엔진: 의미 기반 검색")
print(f"   - 챗봇: ChatGPT, Claude, Bard")
print(f"   - 코드 생성: GitHub Copilot, CodeT5")
print(f"   - 창작 도구: 글쓰기 보조, 시나리오 생성")

print(f"\n" + "=" * 60)
print(f"🎉 Transformer NLP 튜토리얼 완료!")
print(f"   모든 딥러닝 강의 시리즈를 완주하셨습니다!")
print(f"=" * 60)

print(f"\n🏆 전체 시리즈 완주 축하합니다!")
print(f"   1. ✅ PyTorch 기초 (MNIST)")
print(f"   2. ✅ 신경망 심화 (Fashion-MNIST)")
print(f"   3. ✅ CNN 이미지 분류 (CIFAR-10)")
print(f"   4. ✅ RNN 텍스트 분류 (IMDB)")
print(f"   5. ✅ LSTM 시계열 예측 (주식 데이터)")
print(f"   6. ✅ YOLO 객체 탐지 (COCO)")
print(f"   7. ✅ GAN 이미지 생성 (CelebA)")
print(f"   8. ✅ Transformer NLP (Multi30k)")

print(f"\n🌟 이제 여러분은 딥러닝의 핵심 기법들을 모두 마스터했습니다!")
print(f"   계속해서 최신 연구와 기술을 탐구하며 성장하세요!")