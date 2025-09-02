"""
Apriori 연관 규칙 마이닝 실습

이 실습에서는 Apriori 알고리즘을 사용하여 장바구니 데이터에서 
연관 규칙을 발견하고 비즈니스 인사이트를 도출하는 방법을 학습합니다.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class AprioriAlgorithm:
    """Apriori 알고리즘 구현 클래스"""
    
    def __init__(self, min_support=0.1, min_confidence=0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = []
        self.association_rules = []
    
    def fit(self, transactions):
        """Apriori 알고리즘 실행"""
        self.transactions = transactions
        self.n_transactions = len(transactions)
        
        print(f"Apriori 알고리즘 실행 중...")
        print(f"- 트랜잭션 수: {self.n_transactions}")
        print(f"- 최소 지지도: {self.min_support}")
        print(f"- 최소 신뢰도: {self.min_confidence}")
        
        # 1단계: 빈발 1-항목집합 찾기
        frequent_1_itemsets = self._find_frequent_1_itemsets()
        if not frequent_1_itemsets:
            print("빈발 1-항목집합이 없습니다.")
            return
        
        all_frequent_itemsets = [frequent_1_itemsets]
        print(f"빈발 1-항목집합: {len(frequent_1_itemsets)}개")
        
        # k-항목집합 반복 생성
        k = 2
        while all_frequent_itemsets[k-2]:
            # 후보 생성
            candidates = self._generate_candidates(all_frequent_itemsets[k-2], k)
            
            # 빈발 항목집합 선별
            frequent_k_itemsets = self._prune_candidates(candidates)
            
            if frequent_k_itemsets:
                all_frequent_itemsets.append(frequent_k_itemsets)
                print(f"빈발 {k}-항목집합: {len(frequent_k_itemsets)}개")
            else:
                break
            
            k += 1
        
        # 모든 빈발 항목집합 저장
        self.frequent_itemsets = []
        for itemsets in all_frequent_itemsets:
            self.frequent_itemsets.extend(itemsets)
        
        print(f"총 빈발 항목집합: {len(self.frequent_itemsets)}개")
        
        # 연관 규칙 생성
        self._generate_association_rules()
        print(f"생성된 연관 규칙: {len(self.association_rules)}개")
    
    def _find_frequent_1_itemsets(self):
        """빈발 1-항목집합 찾기"""
        item_counts = Counter()
        
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1
        
        frequent_1_itemsets = []
        for item, count in item_counts.items():
            support = count / self.n_transactions
            if support >= self.min_support:
                frequent_1_itemsets.append(frozenset([item]))
        
        return frequent_1_itemsets
    
    def _generate_candidates(self, frequent_k_minus_1_itemsets, k):
        """후보 k-항목집합 생성"""
        candidates = []
        n = len(frequent_k_minus_1_itemsets)
        
        for i in range(n):
            for j in range(i + 1, n):
                # 두 (k-1)-항목집합을 결합
                itemset1 = frequent_k_minus_1_itemsets[i]
                itemset2 = frequent_k_minus_1_itemsets[j]
                
                # 합집합이 k개 항목을 가지는지 확인
                union = itemset1 | itemset2
                if len(union) == k:
                    # Apriori 원리 확인: 모든 (k-1) 부분집합이 빈발한지 확인
                    if self._is_valid_candidate(union, frequent_k_minus_1_itemsets):
                        candidates.append(union)
        
        return candidates
    
    def _is_valid_candidate(self, candidate, frequent_k_minus_1_itemsets):
        """후보가 Apriori 원리를 만족하는지 확인"""
        k_minus_1_subsets = list(combinations(candidate, len(candidate) - 1))
        
        for subset in k_minus_1_subsets:
            if frozenset(subset) not in frequent_k_minus_1_itemsets:
                return False
        
        return True
    
    def _prune_candidates(self, candidates):
        """후보에서 빈발 항목집합 선별"""
        frequent_itemsets = []
        
        for candidate in candidates:
            count = 0
            for transaction in self.transactions:
                if candidate.issubset(transaction):
                    count += 1
            
            support = count / self.n_transactions
            if support >= self.min_support:
                frequent_itemsets.append(candidate)
        
        return frequent_itemsets
    
    def _generate_association_rules(self):
        """연관 규칙 생성"""
        self.association_rules = []
        
        for itemset in self.frequent_itemsets:
            if len(itemset) < 2:
                continue
            
            # 모든 가능한 분할 시도
            for i in range(1, len(itemset)):
                for antecedent in combinations(itemset, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    # 지지도와 신뢰도 계산
                    support_itemset = self._calculate_support(itemset)
                    support_antecedent = self._calculate_support(antecedent)
                    confidence = support_itemset / support_antecedent
                    
                    if confidence >= self.min_confidence:
                        # 추가 지표 계산
                        support_consequent = self._calculate_support(consequent)
                        lift = confidence / support_consequent if support_consequent > 0 else 0
                        
                        # 확신도 계산
                        if confidence < 1.0:
                            conviction = (1 - support_consequent) / (1 - confidence)
                        else:
                            conviction = float('inf')
                        
                        rule = {
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': support_itemset,
                            'confidence': confidence,
                            'lift': lift,
                            'conviction': conviction
                        }
                        
                        self.association_rules.append(rule)
    
    def _calculate_support(self, itemset):
        """항목집합의 지지도 계산"""
        count = 0
        for transaction in self.transactions:
            if itemset.issubset(transaction):
                count += 1
        return count / self.n_transactions
    
    def get_rules(self, sort_by='lift', ascending=False):
        """연관 규칙 반환 (정렬 옵션 포함)"""
        if not self.association_rules:
            return []
        
        sorted_rules = sorted(self.association_rules, 
                            key=lambda x: x[sort_by], 
                            reverse=not ascending)
        return sorted_rules
    
    def print_rules(self, top_n=10, sort_by='lift'):
        """연관 규칙 출력"""
        rules = self.get_rules(sort_by=sort_by)
        
        print(f"\\n상위 {min(top_n, len(rules))}개 연관 규칙 ({sort_by} 기준):")
        print("="*80)
        
        for i, rule in enumerate(rules[:top_n], 1):
            antecedent = ', '.join(sorted(rule['antecedent']))
            consequent = ', '.join(sorted(rule['consequent']))
            
            print(f"{i:2d}. {antecedent} → {consequent}")
            print(f"    지지도: {rule['support']:.3f} | "
                  f"신뢰도: {rule['confidence']:.3f} | "
                  f"리프트: {rule['lift']:.3f} | "
                  f"확신도: {rule['conviction']:.3f}")
            print()

def create_sample_market_data():
    """샘플 장바구니 데이터 생성"""
    
    # 상품 카테고리별 아이템 정의
    categories = {
        'dairy': ['milk', 'cheese', 'yogurt', 'butter'],
        'bakery': ['bread', 'croissant', 'muffin', 'cake'],
        'beverages': ['coffee', 'tea', 'juice', 'soda'],
        'snacks': ['chips', 'cookies', 'nuts', 'chocolate'],
        'fruits': ['apple', 'banana', 'orange', 'grapes'],
        'vegetables': ['lettuce', 'tomato', 'onion', 'carrot']
    }
    
    # 연관성이 있는 상품 그룹 정의
    associated_groups = [
        ['milk', 'cookies', 'chocolate'],  # 간식 세트
        ['bread', 'butter', 'cheese'],     # 아침식사 세트
        ['coffee', 'croissant', 'muffin'], # 카페 세트
        ['lettuce', 'tomato', 'onion'],    # 샐러드 세트
        ['apple', 'banana', 'yogurt'],     # 건강 세트
        ['chips', 'soda', 'nuts']          # 파티 세트
    ]
    
    np.random.seed(42)
    transactions = []
    
    # 1000개의 트랜잭션 생성
    for _ in range(1000):
        transaction = set()
        
        # 80% 확률로 연관 그룹에서 아이템 선택
        if np.random.random() < 0.8:
            group = np.random.choice(len(associated_groups))
            selected_group = associated_groups[group]
            
            # 그룹에서 2-4개 아이템 선택
            n_items = np.random.randint(2, min(5, len(selected_group) + 1))
            selected_items = np.random.choice(selected_group, n_items, replace=False)
            transaction.update(selected_items)
        
        # 추가로 랜덤 아이템 선택 (1-3개)
        all_items = [item for category in categories.values() for item in category]
        n_random = np.random.randint(1, 4)
        random_items = np.random.choice(all_items, n_random, replace=False)
        transaction.update(random_items)
        
        # 최소 2개 아이템이 있는 트랜잭션만 추가
        if len(transaction) >= 2:
            transactions.append(transaction)
    
    return transactions, categories

def create_online_shopping_data():
    """온라인 쇼핑 데이터 생성"""
    
    # 전자제품 카테고리
    electronics = {
        'mobile': ['smartphone', 'phone_case', 'screen_protector', 'charger'],
        'computer': ['laptop', 'mouse', 'keyboard', 'monitor'],
        'audio': ['headphones', 'speakers', 'microphone', 'earbuds'],
        'accessories': ['usb_cable', 'power_bank', 'adapter', 'stand']
    }
    
    # 강한 연관성을 가진 상품 조합
    strong_associations = [
        ['smartphone', 'phone_case', 'screen_protector'],
        ['laptop', 'mouse', 'keyboard'],
        ['headphones', 'microphone', 'adapter'],
        ['power_bank', 'usb_cable', 'charger']
    ]
    
    np.random.seed(123)
    transactions = []
    
    for _ in range(800):
        transaction = set()
        
        # 90% 확률로 강한 연관성 그룹에서 선택
        if np.random.random() < 0.9:
            group = np.random.choice(len(strong_associations))
            selected_group = strong_associations[group]
            
            # 그룹에서 대부분 아이템 선택
            n_items = np.random.randint(2, len(selected_group) + 1)
            selected_items = np.random.choice(selected_group, n_items, replace=False)
            transaction.update(selected_items)
        
        # 추가 아이템 (낮은 확률)
        if np.random.random() < 0.3:
            all_items = [item for category in electronics.values() for item in category]
            additional_item = np.random.choice(all_items)
            transaction.add(additional_item)
        
        if len(transaction) >= 2:
            transactions.append(transaction)
    
    return transactions, electronics

def analyze_transaction_data(transactions, title="Transaction Data"):
    """트랜잭션 데이터 기본 분석"""
    
    print(f"\\n{title} 분석")
    print("="*50)
    
    # 기본 통계
    n_transactions = len(transactions)
    transaction_lengths = [len(t) for t in transactions]
    avg_length = np.mean(transaction_lengths)
    
    print(f"총 트랜잭션 수: {n_transactions}")
    print(f"평균 트랜잭션 길이: {avg_length:.2f}")
    print(f"최소/최대 트랜잭션 길이: {min(transaction_lengths)}/{max(transaction_lengths)}")
    
    # 아이템 빈도 분석
    item_counts = Counter()
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
    
    print(f"\\n고유 아이템 수: {len(item_counts)}")
    print("\\n상위 10개 인기 아이템:")
    for item, count in item_counts.most_common(10):
        support = count / n_transactions
        print(f"  {item}: {count}회 ({support:.3f})")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 트랜잭션 길이 분포
    axes[0].hist(transaction_lengths, bins=20, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Transaction Length')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Transaction Length Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # 상위 아이템 빈도
    top_items = item_counts.most_common(15)
    items, counts = zip(*top_items)
    
    axes[1].barh(range(len(items)), counts, alpha=0.7)
    axes[1].set_yticks(range(len(items)))
    axes[1].set_yticklabels(items)
    axes[1].set_xlabel('Frequency')
    axes[1].set_title('Top 15 Items by Frequency')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} - Basic Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return item_counts

def compare_support_thresholds(transactions, support_values):
    """다양한 지지도 임계값 비교"""
    
    print("\\n지지도 임계값별 결과 비교")
    print("="*50)
    
    results = []
    
    for min_support in support_values:
        apriori = AprioriAlgorithm(min_support=min_support, min_confidence=0.5)
        apriori.fit(transactions)
        
        results.append({
            'min_support': min_support,
            'frequent_itemsets': len(apriori.frequent_itemsets),
            'association_rules': len(apriori.association_rules)
        })
        
        print(f"지지도 {min_support:.2f}: "
              f"빈발항목집합 {len(apriori.frequent_itemsets)}개, "
              f"연관규칙 {len(apriori.association_rules)}개")
    
    # 시각화
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 빈발 항목집합 수
    axes[0].plot(df_results['min_support'], df_results['frequent_itemsets'], 
                'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Minimum Support')
    axes[0].set_ylabel('Number of Frequent Itemsets')
    axes[0].set_title('Frequent Itemsets vs Support Threshold')
    axes[0].grid(True, alpha=0.3)
    
    # 연관 규칙 수
    axes[1].plot(df_results['min_support'], df_results['association_rules'], 
                'ro-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Minimum Support')
    axes[1].set_ylabel('Number of Association Rules')
    axes[1].set_title('Association Rules vs Support Threshold')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Impact of Support Threshold', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return df_results

def visualize_association_rules(rules, top_n=15):
    """연관 규칙 시각화"""
    
    if not rules:
        print("시각화할 연관 규칙이 없습니다.")
        return
    
    # 상위 규칙 선택
    top_rules = rules[:top_n]
    
    # 데이터 준비
    rule_labels = []
    supports = []
    confidences = []
    lifts = []
    
    for rule in top_rules:
        antecedent = ', '.join(sorted(rule['antecedent']))
        consequent = ', '.join(sorted(rule['consequent']))
        rule_label = f"{antecedent} → {consequent}"
        
        rule_labels.append(rule_label)
        supports.append(rule['support'])
        confidences.append(rule['confidence'])
        lifts.append(rule['lift'])
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 지지도 vs 신뢰도 산점도
    scatter = axes[0, 0].scatter(supports, confidences, c=lifts, 
                               s=100, alpha=0.7, cmap='viridis')
    axes[0, 0].set_xlabel('Support')
    axes[0, 0].set_ylabel('Confidence')
    axes[0, 0].set_title('Support vs Confidence (colored by Lift)')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0], label='Lift')
    
    # 2. 리프트 분포
    axes[0, 1].hist(lifts, bins=15, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=1, color='red', linestyle='--', label='Lift = 1')
    axes[0, 1].set_xlabel('Lift')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Lift Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 상위 규칙의 신뢰도
    y_pos = range(len(rule_labels[:10]))
    axes[1, 0].barh(y_pos, confidences[:10], alpha=0.7)
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels([label[:30] + '...' if len(label) > 30 else label 
                               for label in rule_labels[:10]], fontsize=8)
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_title('Top 10 Rules by Confidence')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 상위 규칙의 리프트
    axes[1, 1].barh(y_pos, lifts[:10], alpha=0.7, color='orange')
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([label[:30] + '...' if len(label) > 30 else label 
                               for label in rule_labels[:10]], fontsize=8)
    axes[1, 1].set_xlabel('Lift')
    axes[1, 1].set_title('Top 10 Rules by Lift')
    axes[1, 1].axvline(x=1, color='red', linestyle='--', alpha=0.7)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Association Rules Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def business_insights_analysis(rules, transactions):
    """비즈니스 인사이트 분석"""
    
    print("\\n비즈니스 인사이트 분석")
    print("="*50)
    
    if not rules:
        print("분석할 연관 규칙이 없습니다.")
        return
    
    # 1. 가장 강한 연관성 (리프트 기준)
    print("\\n1. 가장 강한 연관성을 가진 규칙들:")
    high_lift_rules = [rule for rule in rules if rule['lift'] > 2.0]
    high_lift_rules.sort(key=lambda x: x['lift'], reverse=True)
    
    for i, rule in enumerate(high_lift_rules[:5], 1):
        antecedent = ', '.join(sorted(rule['antecedent']))
        consequent = ', '.join(sorted(rule['consequent']))
        print(f"  {i}. {antecedent} → {consequent}")
        print(f"     리프트: {rule['lift']:.2f} (독립적일 때보다 {rule['lift']:.1f}배 높은 확률)")
    
    # 2. 높은 신뢰도 규칙
    print("\\n2. 높은 신뢰도를 가진 규칙들:")
    high_conf_rules = [rule for rule in rules if rule['confidence'] > 0.8]
    high_conf_rules.sort(key=lambda x: x['confidence'], reverse=True)
    
    for i, rule in enumerate(high_conf_rules[:5], 1):
        antecedent = ', '.join(sorted(rule['antecedent']))
        consequent = ', '.join(sorted(rule['consequent']))
        print(f"  {i}. {antecedent} → {consequent}")
        print(f"     신뢰도: {rule['confidence']:.1%} ({antecedent}를 구매한 고객의 {rule['confidence']:.1%}가 {consequent}도 구매)")
    
    # 3. 빈발 항목 조합
    print("\\n3. 자주 함께 구매되는 상품 조합:")
    frequent_pairs = [rule for rule in rules if len(rule['antecedent']) == 1 and len(rule['consequent']) == 1]
    frequent_pairs.sort(key=lambda x: x['support'], reverse=True)
    
    for i, rule in enumerate(frequent_pairs[:5], 1):
        antecedent = list(rule['antecedent'])[0]
        consequent = list(rule['consequent'])[0]
        print(f"  {i}. {antecedent} + {consequent}")
        print(f"     지지도: {rule['support']:.1%} (전체 거래의 {rule['support']:.1%}에서 함께 구매)")
    
    # 4. 추천 시스템 활용
    print("\\n4. 추천 시스템 활용 방안:")
    recommendation_rules = [rule for rule in rules if rule['confidence'] > 0.6 and rule['lift'] > 1.5]
    
    print(f"  - 총 {len(recommendation_rules)}개의 추천 규칙 활용 가능")
    print("  - 고객이 특정 상품을 장바구니에 담으면 연관 상품 추천")
    print("  - 교차 판매 및 상향 판매 전략 수립")
    
    # 5. 상품 배치 전략
    print("\\n5. 상품 배치 전략:")
    layout_rules = [rule for rule in rules if rule['lift'] > 1.2]
    
    print(f"  - {len(layout_rules)}개 규칙을 매장 레이아웃에 활용")
    print("  - 연관성 높은 상품들을 인접하게 배치")
    print("  - 고객 동선 최적화로 매출 증대 기대")

def main():
    """메인 실습 함수"""
    
    print("Apriori 연관 규칙 마이닝 실습")
    print("="*50)
    
    # 1. 샘플 데이터 생성 및 분석
    print("\\n1. 샘플 마켓 데이터 생성")
    market_transactions, market_categories = create_sample_market_data()
    market_item_counts = analyze_transaction_data(market_transactions, "Market Basket Data")
    
    # 2. Apriori 알고리즘 적용 (마켓 데이터)
    print("\\n2. 마켓 데이터에 Apriori 적용")
    market_apriori = AprioriAlgorithm(min_support=0.05, min_confidence=0.6)
    market_apriori.fit(market_transactions)
    
    # 결과 출력
    market_rules = market_apriori.get_rules(sort_by='lift')
    market_apriori.print_rules(top_n=10)
    
    # 3. 온라인 쇼핑 데이터
    print("\\n3. 온라인 쇼핑 데이터 분석")
    online_transactions, online_categories = create_online_shopping_data()
    online_item_counts = analyze_transaction_data(online_transactions, "Online Shopping Data")
    
    # 4. 온라인 데이터에 Apriori 적용
    print("\\n4. 온라인 데이터에 Apriori 적용")
    online_apriori = AprioriAlgorithm(min_support=0.1, min_confidence=0.7)
    online_apriori.fit(online_transactions)
    
    online_rules = online_apriori.get_rules(sort_by='lift')
    online_apriori.print_rules(top_n=10)
    
    # 5. 지지도 임계값 비교
    print("\\n5. 지지도 임계값 영향 분석")
    support_values = [0.02, 0.05, 0.1, 0.15, 0.2]
    support_comparison = compare_support_thresholds(market_transactions, support_values)
    
    # 6. 연관 규칙 시각화
    print("\\n6. 연관 규칙 시각화")
    if market_rules:
        visualize_association_rules(market_rules, top_n=20)
    
    # 7. 비즈니스 인사이트 분석
    print("\\n7. 비즈니스 인사이트 도출")
    business_insights_analysis(market_rules, market_transactions)
    
    # 8. 성능 비교 및 최적화
    print("\\n8. 알고리즘 성능 분석")
    
    # 다양한 파라미터로 성능 측정
    import time
    
    performance_results = []
    
    for min_sup in [0.05, 0.1, 0.15]:
        for min_conf in [0.5, 0.7, 0.9]:
            start_time = time.time()
            
            apriori = AprioriAlgorithm(min_support=min_sup, min_confidence=min_conf)
            apriori.fit(market_transactions)
            
            end_time = time.time()
            
            performance_results.append({
                'min_support': min_sup,
                'min_confidence': min_conf,
                'execution_time': end_time - start_time,
                'frequent_itemsets': len(apriori.frequent_itemsets),
                'rules': len(apriori.association_rules)
            })
    
    # 성능 결과 출력
    print("\\n파라미터별 성능 비교:")
    print("지지도 | 신뢰도 | 실행시간(초) | 빈발항목집합 | 연관규칙")
    print("-" * 55)
    
    for result in performance_results:
        print(f"{result['min_support']:6.2f} | {result['min_confidence']:6.2f} | "
              f"{result['execution_time']:10.3f} | {result['frequent_itemsets']:11d} | "
              f"{result['rules']:8d}")
    
    # 9. 주요 학습 포인트 정리
    print("\\n" + "="*50)
    print("주요 학습 포인트")
    print("="*50)
    print("1. Apriori 알고리즘은 빈발 항목집합을 단계적으로 찾습니다.")
    print("2. 지지도, 신뢰도, 리프트는 연관 규칙의 품질을 측정합니다.")
    print("3. 최소 지지도가 낮을수록 더 많은 패턴을 발견하지만 계산 시간이 증가합니다.")
    print("4. 리프트 > 1인 규칙이 실제로 의미 있는 연관성을 나타냅니다.")
    print("5. 비즈니스 도메인 지식과 결합하여 실용적 인사이트를 도출할 수 있습니다.")
    print("6. 추천 시스템, 상품 배치, 마케팅 전략 등에 활용 가능합니다.")
    
    print("\\n실습을 완료했습니다!")

if __name__ == "__main__":
    main()