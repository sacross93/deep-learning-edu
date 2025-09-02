# Apriori 연관 규칙 마이닝 이론

## 1. 연관 규칙 마이닝 개요

### 1.1 연관 규칙 마이닝이란?
**연관 규칙 마이닝(Association Rule Mining)**은 대용량 데이터베이스에서 항목들 간의 흥미로운 관계나 패턴을 발견하는 데이터 마이닝 기법입니다. "만약 A를 구매하면 B도 구매할 가능성이 높다"와 같은 규칙을 찾아냅니다.

### 1.2 핵심 아이디어
- **장바구니 분석**: 고객이 함께 구매하는 상품들의 패턴 발견
- **연관성 발견**: 데이터 항목들 간의 숨겨진 관계 탐지
- **예측과 추천**: 발견된 패턴을 이용한 예측 및 추천 시스템

### 1.3 주요 특징
- 비지도 학습 방법
- 대용량 트랜잭션 데이터 처리
- 해석 가능한 규칙 생성
- 다양한 도메인에 적용 가능

## 2. 기본 개념과 용어

### 2.1 기본 용어 정의
```
• 항목(Item): 개별 상품이나 속성 (예: 우유, 빵)
• 항목집합(Itemset): 하나 이상의 항목들의 집합 {우유, 빵}
• 트랜잭션(Transaction): 한 번의 구매에서 발생한 항목집합
• 트랜잭션 데이터베이스: 모든 트랜잭션들의 집합
• k-항목집합: k개의 항목을 포함하는 항목집합
```

### 2.2 연관 규칙의 형태
```
X → Y (X implies Y)

여기서:
- X: 선행부(Antecedent) 또는 조건부
- Y: 후행부(Consequent) 또는 결과부
- X ∩ Y = ∅ (X와 Y는 서로소)

예시: {우유, 빵} → {버터}
```

### 2.3 트랜잭션 데이터 예시
```
TID | 구매 항목
----|----------
T1  | {우유, 빵, 버터}
T2  | {우유, 빵}
T3  | {우유, 버터, 치즈}
T4  | {빵, 버터, 잼}
T5  | {우유, 빵, 버터, 치즈}
```

## 3. 핵심 측정 지표

### 3.1 지지도 (Support)
**정의**: 전체 트랜잭션 중 특정 항목집합이 나타나는 비율

```
Support(X) = |X가 포함된 트랜잭션 수| / |전체 트랜잭션 수|

Support(X → Y) = Support(X ∪ Y)
```

**예시**: Support({우유, 빵}) = 3/5 = 0.6 (60%)

### 3.2 신뢰도 (Confidence)
**정의**: X가 주어졌을 때 Y가 함께 나타날 조건부 확률

```
Confidence(X → Y) = Support(X ∪ Y) / Support(X)
                  = P(Y|X)
```

**예시**: Confidence({우유} → {빵}) = Support({우유, 빵}) / Support({우유}) = 0.6 / 0.8 = 0.75 (75%)

### 3.3 리프트 (Lift)
**정의**: X와 Y의 독립성을 측정하는 지표

```
Lift(X → Y) = Confidence(X → Y) / Support(Y)
            = Support(X ∪ Y) / (Support(X) × Support(Y))
```

**해석**:
- Lift = 1: X와 Y는 독립적
- Lift > 1: X와 Y는 양의 상관관계
- Lift < 1: X와 Y는 음의 상관관계

### 3.4 확신도 (Conviction)
**정의**: 규칙의 예외가 얼마나 드문지를 측정

```
Conviction(X → Y) = (1 - Support(Y)) / (1 - Confidence(X → Y))
```

**해석**:
- Conviction = 1: X와 Y는 독립적
- Conviction > 1: 강한 연관성
- Conviction = ∞: 완벽한 규칙 (예외 없음)

## 4. Apriori 알고리즘

### 4.1 Apriori 원리
**핵심 아이디어**: "빈발하지 않은 항목집합의 부분집합은 모두 빈발하지 않다"

```
만약 {A, B, C}가 빈발하지 않다면,
{A, B}, {A, C}, {B, C}, {A}, {B}, {C} 모두 빈발하지 않다.

역으로, 만약 {A}가 빈발하지 않다면,
{A}를 포함하는 모든 상위집합도 빈발하지 않다.
```

### 4.2 알고리즘 단계

#### 단계 1: 빈발 1-항목집합 찾기
```python
def find_frequent_1_itemsets(transactions, min_support):
    item_counts = {}
    for transaction in transactions:
        for item in transaction:
            item_counts[item] = item_counts.get(item, 0) + 1
    
    frequent_1_itemsets = []
    for item, count in item_counts.items():
        if count / len(transactions) >= min_support:
            frequent_1_itemsets.append({item})
    
    return frequent_1_itemsets
```

#### 단계 2: 후보 k-항목집합 생성
```python
def generate_candidates(frequent_k_minus_1_itemsets, k):
    candidates = []
    n = len(frequent_k_minus_1_itemsets)
    
    for i in range(n):
        for j in range(i + 1, n):
            # 첫 k-2개 항목이 같은 경우에만 결합
            itemset1 = sorted(list(frequent_k_minus_1_itemsets[i]))
            itemset2 = sorted(list(frequent_k_minus_1_itemsets[j]))
            
            if itemset1[:-1] == itemset2[:-1]:
                candidate = set(itemset1) | set(itemset2)
                if is_valid_candidate(candidate, frequent_k_minus_1_itemsets):
                    candidates.append(candidate)
    
    return candidates
```

#### 단계 3: 빈발 k-항목집합 선별
```python
def prune_candidates(candidates, transactions, min_support):
    frequent_itemsets = []
    
    for candidate in candidates:
        count = 0
        for transaction in transactions:
            if candidate.issubset(transaction):
                count += 1
        
        support = count / len(transactions)
        if support >= min_support:
            frequent_itemsets.append(candidate)
    
    return frequent_itemsets
```

### 4.3 전체 Apriori 알고리즘
```python
def apriori(transactions, min_support):
    # 1단계: 빈발 1-항목집합 찾기
    frequent_itemsets = [find_frequent_1_itemsets(transactions, min_support)]
    
    k = 2
    while frequent_itemsets[k-2]:  # 이전 단계에서 빈발 항목집합이 있는 동안
        # 2단계: 후보 생성
        candidates = generate_candidates(frequent_itemsets[k-2], k)
        
        # 3단계: 빈발 항목집합 선별
        frequent_k_itemsets = prune_candidates(candidates, transactions, min_support)
        frequent_itemsets.append(frequent_k_itemsets)
        
        k += 1
    
    # 모든 빈발 항목집합 결합
    all_frequent_itemsets = []
    for itemsets in frequent_itemsets:
        all_frequent_itemsets.extend(itemsets)
    
    return all_frequent_itemsets
```

## 5. 연관 규칙 생성

### 5.1 규칙 생성 과정
빈발 항목집합에서 연관 규칙을 생성하는 과정:

```python
def generate_association_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    
    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue
        
        # 모든 가능한 분할 시도
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = set(antecedent)
                consequent = itemset - antecedent
                
                # 신뢰도 계산
                support_itemset = calculate_support(itemset, transactions)
                support_antecedent = calculate_support(antecedent, transactions)
                confidence = support_itemset / support_antecedent
                
                if confidence >= min_confidence:
                    # 추가 지표 계산
                    support_consequent = calculate_support(consequent, transactions)
                    lift = confidence / support_consequent
                    
                    rules.append({
                        'antecedent': antecedent,
                        'consequent': consequent,
                        'support': support_itemset,
                        'confidence': confidence,
                        'lift': lift
                    })
    
    return rules
```

### 5.2 규칙 평가 기준
1. **최소 지지도**: 규칙이 충분히 자주 나타나는가?
2. **최소 신뢰도**: 규칙이 충분히 신뢰할 만한가?
3. **리프트**: 규칙이 우연보다 의미 있는가?

## 6. Apriori 알고리즘의 최적화

### 6.1 성능 개선 기법

#### 해시 기반 기법
```python
def hash_based_pruning(candidates, transactions, hash_table_size):
    hash_table = [0] * hash_table_size
    
    # 해시 테이블 구축
    for transaction in transactions:
        for candidate in generate_2_subsets(transaction):
            hash_value = hash_function(candidate) % hash_table_size
            hash_table[hash_value] += 1
    
    # 해시 값이 최소 지지도를 만족하지 않는 후보 제거
    pruned_candidates = []
    for candidate in candidates:
        hash_value = hash_function(candidate) % hash_table_size
        if hash_table[hash_value] >= min_support_count:
            pruned_candidates.append(candidate)
    
    return pruned_candidates
```

#### 트랜잭션 축소
```python
def reduce_transactions(transactions, frequent_itemsets):
    reduced_transactions = []
    
    for transaction in transactions:
        # 빈발하지 않은 항목 제거
        reduced_transaction = set()
        for item in transaction:
            if any(item in itemset for itemset in frequent_itemsets):
                reduced_transaction.add(item)
        
        if len(reduced_transaction) >= 2:  # 최소 2개 항목 필요
            reduced_transactions.append(reduced_transaction)
    
    return reduced_transactions
```

### 6.2 메모리 최적화
- **비트맵 인덱스**: 트랜잭션을 비트 벡터로 표현
- **압축 기법**: 희소한 데이터 구조 활용
- **배치 처리**: 메모리에 맞는 크기로 데이터 분할

## 7. Apriori의 장단점

### 7.1 장점
1. **직관적 이해**: 알고리즘이 이해하기 쉬움
2. **완전성**: 모든 빈발 항목집합을 찾아냄
3. **해석 가능성**: 생성된 규칙이 명확함
4. **검증된 방법**: 오랜 기간 사용되어 신뢰성 높음

### 7.2 단점
1. **계산 복잡도**: 후보 생성과 지지도 계산이 비효율적
2. **메모리 사용량**: 대량의 후보 항목집합 저장 필요
3. **다중 스캔**: 데이터베이스를 여러 번 스캔해야 함
4. **희소 데이터**: 최소 지지도가 낮을 때 성능 저하

### 7.3 시간 복잡도
- **최악의 경우**: O(2^n) (n은 항목 수)
- **실제 성능**: 최소 지지도 설정으로 크게 개선
- **공간 복잡도**: O(후보 항목집합 수)

## 8. Apriori 변형 알고리즘

### 8.1 FP-Growth
- **FP-Tree 구조**: 압축된 트리 구조로 데이터 저장
- **패턴 성장**: 트리를 이용한 효율적인 패턴 발견
- **장점**: 후보 생성 없이 빈발 패턴 발견

### 8.2 Eclat
- **수직적 데이터 형식**: 항목별로 트랜잭션 ID 저장
- **교집합 연산**: 빠른 지지도 계산
- **깊이 우선 탐색**: 메모리 효율적

### 8.3 PCY (Park-Chen-Yu)
- **해시 기법**: 2-항목집합 후보 축소
- **메모리 효율성**: 해시 테이블로 메모리 절약
- **성능 개선**: Apriori보다 빠른 처리

## 9. 실제 적용 사례

### 9.1 소매업 (Market Basket Analysis)
- **상품 배치**: 연관성 높은 상품을 가까이 배치
- **교차 판매**: 관련 상품 추천
- **프로모션**: 번들 상품 기획

### 9.2 웹 사용 패턴 분석
- **페이지 추천**: 함께 방문하는 페이지 패턴
- **사이트 구조**: 네비게이션 개선
- **개인화**: 사용자별 맞춤 콘텐츠

### 9.3 의료 데이터 분석
- **증상 연관성**: 함께 나타나는 증상 패턴
- **약물 상호작용**: 동시 처방되는 약물 분석
- **진단 지원**: 증상 기반 질병 예측

### 9.4 금융 서비스
- **사기 탐지**: 비정상적인 거래 패턴
- **상품 추천**: 고객 성향 기반 금융상품 추천
- **리스크 관리**: 연관된 위험 요소 분석

## 10. 구현 시 고려사항

### 10.1 데이터 전처리
```python
def preprocess_transactions(raw_data):
    # 1. 결측값 처리
    cleaned_data = raw_data.dropna()
    
    # 2. 트랜잭션 형태로 변환
    transactions = []
    for _, row in cleaned_data.iterrows():
        transaction = set()
        for item in row:
            if pd.notna(item) and item != '':
                transaction.add(str(item).strip())
        if transaction:
            transactions.append(transaction)
    
    return transactions
```

### 10.2 파라미터 선택
- **최소 지지도**: 너무 높으면 중요한 패턴 놓침, 너무 낮으면 노이즈 증가
- **최소 신뢰도**: 일반적으로 0.5-0.8 사용
- **도메인 지식**: 업무 특성을 반영한 임계값 설정

### 10.3 결과 해석
```python
def interpret_rules(rules, top_n=10):
    # 리프트 기준으로 정렬
    sorted_rules = sorted(rules, key=lambda x: x['lift'], reverse=True)
    
    print(f"상위 {top_n}개 연관 규칙:")
    for i, rule in enumerate(sorted_rules[:top_n], 1):
        antecedent = ', '.join(rule['antecedent'])
        consequent = ', '.join(rule['consequent'])
        
        print(f"{i}. {antecedent} → {consequent}")
        print(f"   지지도: {rule['support']:.3f}")
        print(f"   신뢰도: {rule['confidence']:.3f}")
        print(f"   리프트: {rule['lift']:.3f}")
        print()
```

## 요약

Apriori 알고리즘은 연관 규칙 마이닝의 기본이 되는 중요한 알고리즘입니다:

1. **핵심 개념**: 지지도, 신뢰도, 리프트를 이용한 연관성 측정
2. **Apriori 원리**: 빈발하지 않은 항목집합의 상위집합도 빈발하지 않음
3. **단계적 접근**: 1-항목집합부터 시작하여 점진적으로 확장
4. **실용적 가치**: 다양한 도메인에서 의미 있는 패턴 발견

적절한 파라미터 설정과 도메인 지식을 결합하면 비즈니스에 유용한 인사이트를 얻을 수 있습니다.