# Transformer 완전 이론 가이드

## 1. 개요 및 핵심 개념

### Transformer의 정의와 목적
Transformer는 2017년 "Attention Is All You Need" 논문에서 소개된 신경망 아키텍처로, 순환 신경망(RNN)이나 합성곱 신경망(CNN) 없이 오직 어텐션 메커니즘만을 사용하여 시퀀스-투-시퀀스 작업을 수행합니다. 주로 자연어 처리 분야에서 혁신적인 성과를 보여주었으며, 현재 GPT, BERT 등 대부분의 최신 언어 모델의 기반이 되고 있습니다.

### 해결하고자 하는 문제
1. **순차 처리의 한계**: RNN은 시퀀스를 순차적으로 처리해야 하므로 병렬화가 어렵고 훈련 시간이 오래 걸립니다.
2. **장거리 의존성**: RNN에서 그래디언트 소실 문제로 인해 멀리 떨어진 토큰 간의 관계를 학습하기 어렵습니다.
3. **정보 병목**: 인코더-디코더 구조에서 고정 크기 벡터로 모든 정보를 압축해야 하는 한계가 있습니다.

### 다른 기법과의 차이점
- **RNN vs Transformer**: RNN은 순차 처리, Transformer는 병렬 처리 가능
- **CNN vs Transformer**: CNN은 지역적 패턴, Transformer는 전역적 관계 학습
- **기존 Attention vs Self-Attention**: 기존은 인코더-디코더 간, Self-Attention은 시퀀스 내부 관계

## 2. 수학적 원리

### Self-Attention 메커니즘
Self-Attention은 입력 시퀀스의 각 위치에서 다른 모든 위치와의 관련성을 계산합니다.

**핵심 수식:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

여기서:
- Q (Query): 질의 행렬, 현재 위치에서 찾고자 하는 정보
- K (Key): 키 행렬, 각 위치의 특성을 나타내는 벡터
- V (Value): 값 행렬, 실제로 전달될 정보
- d_k: 키 벡터의 차원 (스케일링 팩터)

**단계별 계산 과정:**
1. **선형 변환**: 입력 X를 Q, K, V로 변환
   ```
   Q = XW_Q, K = XW_K, V = XW_V
   ```

2. **유사도 계산**: Query와 Key의 내적으로 어텐션 스코어 계산
   ```
   scores = QK^T / √d_k
   ```

3. **정규화**: Softmax로 확률 분포 생성
   ```
   attention_weights = softmax(scores)
   ```

4. **가중합**: Value에 어텐션 가중치 적용
   ```
   output = attention_weights × V
   ```

### Multi-Head Attention
여러 개의 어텐션 헤드를 병렬로 사용하여 다양한 관점에서 정보를 추출합니다.

**수식:**
```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W_O
head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
```

**직관적 해석:**
- 각 헤드는 서로 다른 종류의 관계를 학습 (구문적, 의미적, 위치적 등)
- 병렬 처리로 계산 효율성 향상
- 다양한 표현 공간에서의 정보 추출

## 3. 아키텍처 상세 분석

### 전체 Transformer 구조
```
입력 임베딩 + 위치 인코딩
    ↓
인코더 스택 (N개 레이어)
    ↓
디코더 스택 (N개 레이어)
    ↓
선형 변환 + Softmax
    ↓
출력 확률 분포
```

### 인코더 (Encoder) 구조
각 인코더 레이어는 다음 두 개의 서브레이어로 구성됩니다:

1. **Multi-Head Self-Attention**
   - 입력 시퀀스 내 모든 위치 간의 관계 학습
   - 병렬 처리로 효율적인 계산

2. **Position-wise Feed-Forward Network**
   ```
   FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
   ```
   - 각 위치에 독립적으로 적용되는 완전연결층
   - ReLU 활성화 함수 사용

**잔차 연결 및 층 정규화:**
```
output = LayerNorm(x + Sublayer(x))
```

### 디코더 (Decoder) 구조
각 디코더 레이어는 세 개의 서브레이어로 구성됩니다:

1. **Masked Multi-Head Self-Attention**
   - 미래 토큰을 보지 못하도록 마스킹 적용
   - 자기회귀적 생성을 위한 구조

2. **Multi-Head Cross-Attention**
   - 인코더 출력과 디코더 간의 어텐션
   - Query는 디코더, Key와 Value는 인코더에서

3. **Position-wise Feed-Forward Network**
   - 인코더와 동일한 구조

### 위치 인코딩 (Positional Encoding)
Transformer는 순환 구조가 없어 위치 정보를 별도로 제공해야 합니다.

**사인/코사인 위치 인코딩:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

여기서:
- pos: 위치 인덱스
- i: 차원 인덱스
- d_model: 모델의 차원

**특징:**
- 상대적 위치 관계 학습 가능
- 훈련 시보다 긴 시퀀스에도 적용 가능
- 주기적 패턴으로 위치 구분

## 4. 용어 사전

### 핵심 용어 정의

**Attention (어텐션)**
- 입력의 특정 부분에 집중하는 메커니즘
- 관련성이 높은 정보에 더 큰 가중치 부여

**Self-Attention (셀프 어텐션)**
- 같은 시퀀스 내의 다른 위치들과의 관계를 계산하는 어텐션
- 문맥적 표현 학습에 핵심적 역할

**Multi-Head Attention (멀티헤드 어텐션)**
- 여러 개의 어텐션 헤드를 병렬로 사용하는 구조
- 다양한 표현 공간에서 정보 추출

**Query, Key, Value (쿼리, 키, 값)**
- Query: 찾고자 하는 정보의 표현
- Key: 각 위치의 특성을 나타내는 벡터
- Value: 실제로 전달될 정보

**Positional Encoding (위치 인코딩)**
- 시퀀스에서 각 토큰의 위치 정보를 나타내는 벡터
- 사인/코사인 함수 또는 학습 가능한 임베딩 사용

**Layer Normalization (층 정규화)**
- 각 레이어의 출력을 정규화하는 기법
- 훈련 안정성과 수렴 속도 개선

**Residual Connection (잔차 연결)**
- 입력을 출력에 직접 더하는 연결
- 그래디언트 소실 문제 완화

**Masked Attention (마스크드 어텐션)**
- 특정 위치의 정보를 가리는 어텐션
- 디코더에서 미래 토큰 참조 방지

### 약어 및 기호

- **d_model**: 모델의 차원 (일반적으로 512 또는 768)
- **d_k, d_v**: Key, Value 벡터의 차원
- **h**: 어텐션 헤드의 개수
- **N**: 인코더/디코더 레이어의 개수
- **FFN**: Feed-Forward Network
- **PE**: Positional Encoding

## 5. 실제 구현 연결점

### PyTorch 구현과 이론의 매핑

**1. Multi-Head Attention 구현**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 이론의 W_Q, W_K, W_V에 해당
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
    
    def forward(self, query, key, value, mask=None):
        # 1. 선형 변환: Q, K, V 계산
        Q = self.w_q(query)  # (batch, seq_len, d_model)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # 2. 멀티헤드로 분할
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 3. Scaled Dot-Product Attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # 4. 헤드 결합 및 출력 변환
        output = self.w_o(attention_output)
        return output
```

**2. 위치 인코딩 구현**
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        
        # 이론의 PE 수식 구현
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

### 주요 함수 및 클래스 설명

**nn.MultiheadAttention**
- PyTorch에서 제공하는 멀티헤드 어텐션 구현
- `embed_dim`: d_model에 해당
- `num_heads`: 어텐션 헤드 개수
- `dropout`: 어텐션 가중치에 적용할 드롭아웃 비율

**nn.TransformerEncoder/Decoder**
- 완전한 Transformer 인코더/디코더 구현
- `num_layers`: N (레이어 개수)
- `norm`: 층 정규화 적용 여부

**nn.LayerNorm**
- 층 정규화 구현
- `normalized_shape`: 정규화할 차원
- `eps`: 수치적 안정성을 위한 작은 값

### 파라미터 설정 가이드

**모델 크기별 권장 설정:**

1. **Base 모델**
   - d_model: 512
   - num_heads: 8
   - num_layers: 6
   - d_ff: 2048

2. **Large 모델**
   - d_model: 1024
   - num_heads: 16
   - num_layers: 24
   - d_ff: 4096

**하이퍼파라미터 조정 가이드:**
- **학습률**: 0.0001 ~ 0.001 (Warmup 스케줄링 권장)
- **배치 크기**: GPU 메모리에 따라 조정 (8 ~ 128)
- **드롭아웃**: 0.1 ~ 0.3 (과적합 방지)
- **라벨 스무딩**: 0.1 (분류 성능 향상)

## 6. 실용적 고려사항

### 장점
1. **병렬 처리**: 모든 위치를 동시에 계산 가능
2. **장거리 의존성**: 직접적인 연결로 멀리 떨어진 토큰 간 관계 학습
3. **해석 가능성**: 어텐션 가중치를 통한 모델 동작 이해
4. **전이 학습**: 사전 훈련된 모델의 효과적 활용

### 단점
1. **메모리 사용량**: O(n²) 복잡도로 긴 시퀀스에서 메모리 부족
2. **위치 정보**: 명시적 위치 인코딩 필요
3. **귀납적 편향 부족**: CNN의 지역성, RNN의 순차성 같은 구조적 편향 없음
4. **계산 복잡도**: 시퀀스 길이의 제곱에 비례

### 적용 분야
1. **자연어 처리**: 기계 번역, 텍스트 요약, 질의응답
2. **컴퓨터 비전**: Vision Transformer (ViT), 이미지 분류
3. **음성 처리**: 음성 인식, 음성 합성
4. **시계열 분석**: 예측, 이상 탐지

### 한계 및 해결 방안
1. **긴 시퀀스 처리**
   - 문제: O(n²) 메모리 복잡도
   - 해결: Sparse Attention, Linformer, Performer 등

2. **위치 정보 학습**
   - 문제: 절대/상대 위치 구분 어려움
   - 해결: 상대적 위치 인코딩, RoPE (Rotary Position Embedding)

3. **계산 효율성**
   - 문제: 작은 시퀀스에서 오버헤드
   - 해결: 하이브리드 모델, 적응적 계산

### 성능 최적화 팁

**1. 메모리 최적화**
```python
# 그래디언트 체크포인팅 사용
model = torch.utils.checkpoint.checkpoint_sequential(model, segments=2)

# 혼합 정밀도 훈련
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    output = model(input)
```

**2. 훈련 안정성**
```python
# 학습률 워밍업 스케줄러
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, 
    lr_lambda=lambda step: min(1.0, step / warmup_steps)
)

# 그래디언트 클리핑
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**3. 추론 최적화**
```python
# KV 캐시 사용 (디코더에서)
def forward_with_cache(self, x, cache=None):
    if cache is not None:
        # 이전 계산된 Key, Value 재사용
        k_cache, v_cache = cache
        k = torch.cat([k_cache, k_new], dim=1)
        v = torch.cat([v_cache, v_new], dim=1)
    return output, (k, v)
```

## 7. Transformer 기반 모델들

### BERT (Bidirectional Encoder Representations from Transformers)
**구조 특징:**
- 인코더만 사용하는 구조
- 양방향 문맥 학습 (Masked Language Model)
- 사전 훈련 + 파인튜닝 패러다임

**핵심 혁신:**
- **Masked LM**: 입력의 일부를 마스킹하고 예측
- **Next Sentence Prediction**: 문장 간 관계 학습
- **양방향 어텐션**: 전체 문맥 정보 활용

**적용 분야:**
- 텍스트 분류, 개체명 인식, 질의응답
- 문장 유사도, 감정 분석

### GPT (Generative Pre-trained Transformer)
**구조 특징:**
- 디코더만 사용하는 구조
- 자기회귀적 언어 모델
- 왼쪽에서 오른쪽으로 순차 생성

**핵심 혁신:**
- **Causal Attention**: 미래 토큰 마스킹
- **In-context Learning**: 예시를 통한 학습
- **Scaling Laws**: 모델 크기와 성능의 관계

**발전 과정:**
- GPT-1: 117M 파라미터, 개념 증명
- GPT-2: 1.5B 파라미터, 텍스트 생성 품질 향상
- GPT-3: 175B 파라미터, Few-shot 학습 능력
- GPT-4: 멀티모달, 추론 능력 대폭 향상

### T5 (Text-to-Text Transfer Transformer)
**구조 특징:**
- 완전한 인코더-디코더 구조
- 모든 NLP 작업을 텍스트 생성으로 통일
- 상대적 위치 인코딩 사용

**핵심 아이디어:**
- "Text-to-Text": 입력과 출력 모두 텍스트
- 통합된 프레임워크로 다양한 작업 수행
- 작업별 프롬프트 설계

### Vision Transformer (ViT)
**구조 특징:**
- 이미지를 패치로 분할하여 시퀀스로 처리
- 위치 임베딩으로 공간 정보 제공
- 분류 토큰 ([CLS]) 사용

**혁신점:**
- CNN 없이 순수 Transformer로 이미지 처리
- 대규모 데이터셋에서 CNN 성능 초과
- 전이 학습 효과 우수

### 기타 주요 모델들

**RoBERTa (Robustly Optimized BERT Pretraining Approach)**
- BERT의 훈련 방법 개선
- Next Sentence Prediction 제거
- 더 큰 배치, 더 긴 훈련

**ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)**
- 생성자-판별자 구조
- 모든 토큰에 대한 학습
- 효율적인 사전 훈련

**DeBERTa (Decoding-enhanced BERT with Disentangled Attention)**
- 분리된 어텐션 메커니즘
- 상대적 위치 정보 강화
- 향상된 성능

## 8. 확장 학습 자료

### 추천 논문
1. **원본 논문**
   - "Attention Is All You Need" (Vaswani et al., 2017)
   - Transformer 아키텍처의 최초 제안

2. **BERT 관련**
   - "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (Devlin et al., 2018)
   - "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (Liu et al., 2019)

3. **GPT 시리즈**
   - "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
   - "Language Models are Few-Shot Learners" (Brown et al., 2020)

4. **효율성 개선**
   - "Linformer: Self-Attention with Linear Complexity" (Wang et al., 2020)
   - "Performer: Rethinking Attention with Performers" (Choromanski et al., 2020)

### 관련 기법 소개
1. **Attention 변형들**
   - Sparse Attention: 희소 어텐션 패턴
   - Local Attention: 지역적 어텐션 윈도우
   - Cross Attention: 서로 다른 시퀀스 간 어텐션

2. **위치 인코딩 개선**
   - Relative Position Encoding: 상대적 위치 정보
   - RoPE (Rotary Position Embedding): 회전 위치 임베딩
   - ALiBi (Attention with Linear Biases): 선형 편향 어텐션

3. **아키텍처 변형**
   - Transformer-XL: 긴 시퀀스 처리
   - Reformer: 메모리 효율적 Transformer
   - Switch Transformer: 희소 전문가 모델

### 실습 문제 및 과제

**기초 실습**
1. 간단한 Self-Attention 메커니즘 구현
2. 위치 인코딩 시각화
3. 어텐션 가중치 분석

**중급 실습**
1. 미니 Transformer 모델 구현
2. 기계 번역 모델 훈련
3. 어텐션 패턴 분석 및 해석

**고급 과제**
1. 효율적인 Attention 메커니즘 구현
2. 멀티모달 Transformer 설계
3. 대규모 언어 모델 파인튜닝

**프로젝트 아이디어**
1. 도메인 특화 언어 모델 개발
2. 코드 생성 Transformer 구현
3. 시계열 예측을 위한 Transformer 적용

### 추가 학습 리소스
1. **온라인 강의**
   - CS224N (Stanford NLP Course)
   - Hugging Face Transformers Course
   - Fast.ai NLP Course

2. **구현 라이브러리**
   - Hugging Face Transformers
   - FairSeq (Facebook)
   - T5X (Google)

3. **시각화 도구**
   - BertViz: BERT 어텐션 시각화
   - Transformer Explainer: 인터랙티브 설명
   - Attention Visualizer: 어텐션 패턴 분석

이 이론 가이드를 통해 Transformer의 모든 핵심 개념을 이해하고, 실제 구현과 연결하여 깊이 있는 학습을 진행할 수 있습니다.