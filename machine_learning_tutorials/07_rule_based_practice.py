"""
규칙 기반 학습 실습

이 실습에서는 Mushroom 데이터셋을 사용하여 독성 버섯을 분류하는 규칙을 학습합니다.
규칙 품질 지표(지지도, 신뢰도, 리프트)를 계산하고 의사결정나무와 비교합니다.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 유틸리티 함수 임포트
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from data_utils import load_sample_dataset
from evaluation_utils import calculate_classification_metrics
from visualization_utils import plot_confusion_matrix

class RuleBasedClassifier:
    """규칙 기반 분류기"""
    
    def __init__(self, min_support=0.1, min_confidence=0.8, min_lift=1.0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.min_lift = min_lift
        self.rules = []
        self.feature_names = None
        self.class_names = None
        
    def _calculate_support(self, itemset, data):
        """지지도 계산"""
        count = 0
        for _, row in data.iterrows():
            if all(row[attr] == value for attr, value in itemset.items()):
                count += 1
        return count / len(data)
    
    def _calculate_confidence(self, antecedent, consequent, data):
        """신뢰도 계산"""
        # P(consequent | antecedent) = P(antecedent ∪ consequent) / P(antecedent)
        antecedent_count = 0
        both_count = 0
        
        for _, row in data.iterrows():
            antecedent_match = all(row[attr] == value for attr, value in antecedent.items())
            consequent_match = all(row[attr] == value for attr, value in consequent.items())
            
            if antecedent_match:
                antecedent_count += 1
                if consequent_match:
                    both_count += 1
        
        return both_count / antecedent_count if antecedent_count > 0 else 0
    
    def _calculate_lift(self, antecedent, consequent, data):
        """리프트 계산"""
        confidence = self._calculate_confidence(antecedent, consequent, data)
        consequent_support = self._calculate_support(consequent, data)
        
        return confidence / consequent_support if consequent_support > 0 else 0
    
    def _generate_candidate_rules(self, data, target_column):
        """후보 규칙 생성"""
        rules = []
        feature_columns = [col for col in data.columns if col != target_column]
        
        # 단일 조건 규칙 생성
        for feature in feature_columns:
            for value in data[feature].unique():
                for target_class in data[target_column].unique():
                    antecedent = {feature: value}
                    consequent = {target_column: target_class}
                    
                    support = self._calculate_support({**antecedent, **consequent}, data)
                    confidence = self._calculate_confidence(antecedent, consequent, data)
                    lift = self._calculate_lift(antecedent, consequent, data)
                    
                    if (support >= self.min_support and 
                        confidence >= self.min_confidence and 
                        lift >= self.min_lift):
                        
                        rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': support,
                            'confidence': confidence,
                            'lift': lift
                        })
        
        # 이중 조건 규칙 생성 (계산 효율성을 위해 제한)
        for i, feature1 in enumerate(feature_columns[:5]):  # 처음 5개 특성만 사용
            for feature2 in feature_columns[i+1:6]:  # 조합 수 제한
                for value1 in data[feature1].unique()[:3]:  # 값도 제한
                    for value2 in data[feature2].unique()[:3]:
                        for target_class in data[target_column].unique():
                            antecedent = {feature1: value1, feature2: value2}
                            consequent = {target_column: target_class}
                            
                            support = self._calculate_support({**antecedent, **consequent}, data)
                            confidence = self._calculate_confidence(antecedent, consequent, data)
                            lift = self._calculate_lift(antecedent, consequent, data)
                            
                            if (support >= self.min_support and 
                                confidence >= self.min_confidence and 
                                lift >= self.min_lift):
                                
                                rules.append({
                                    'antecedent': antecedent,
                                    'consequent': consequent,
                                    'support': support,
                                    'confidence': confidence,
                                    'lift': lift
                                })
        
        return rules
    
    def fit(self, X, y):
        """규칙 학습"""
        # 데이터 준비
        data = X.copy()
        data['target'] = y
        self.feature_names = X.columns.tolist()
        self.class_names = np.unique(y)
        
        # 규칙 생성
        self.rules = self._generate_candidate_rules(data, 'target')
        
        # 규칙을 신뢰도 순으로 정렬
        self.rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"총 {len(self.rules)}개의 규칙이 생성되었습니다.")
        
    def predict(self, X):
        """예측 수행"""
        predictions = []
        
        for _, row in X.iterrows():
            prediction = None
            
            # 규칙을 순서대로 적용
            for rule in self.rules:
                antecedent = rule['antecedent']
                consequent = rule['consequent']
                
                # 조건 확인
                if all(row[attr] == value for attr, value in antecedent.items()):
                    prediction = list(consequent.values())[0]
                    break
            
            # 매칭되는 규칙이 없으면 가장 빈번한 클래스로 예측
            if prediction is None:
                prediction = self.class_names[0]  # 기본값
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def get_top_rules(self, n=10):
        """상위 n개 규칙 반환"""
        return self.rules[:n]

def load_mushroom_data():
    """버섯 데이터셋 로딩 및 전처리"""
    print("=== 버섯 데이터셋 로딩 ===")
    
    # 실제 환경에서는 UCI 버섯 데이터셋을 사용하지만, 
    # 여기서는 시뮬레이션된 데이터를 생성합니다.
    np.random.seed(42)
    
    # 버섯의 특성들
    cap_shape = ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken']
    cap_surface = ['fibrous', 'grooves', 'scaly', 'smooth']
    cap_color = ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow']
    bruises = ['bruises', 'no']
    odor = ['almond', 'anise', 'creosote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy']
    gill_spacing = ['close', 'crowded', 'distant']
    gill_size = ['broad', 'narrow']
    gill_color = ['black', 'brown', 'buff', 'chocolate', 'gray', 'green', 'orange', 'pink', 'purple', 'red', 'white', 'yellow']
    stalk_shape = ['enlarging', 'tapering']
    stalk_root = ['bulbous', 'club', 'cup', 'equal', 'rhizomorphs', 'rooted', 'missing']
    
    n_samples = 2000
    data = []
    
    for i in range(n_samples):
        # 특성 값들을 랜덤하게 선택
        features = {
            'cap_shape': np.random.choice(cap_shape),
            'cap_surface': np.random.choice(cap_surface),
            'cap_color': np.random.choice(cap_color),
            'bruises': np.random.choice(bruises),
            'odor': np.random.choice(odor),
            'gill_spacing': np.random.choice(gill_spacing),
            'gill_size': np.random.choice(gill_size),
            'gill_color': np.random.choice(gill_color),
            'stalk_shape': np.random.choice(stalk_shape),
            'stalk_root': np.random.choice(stalk_root)
        }
        
        # 독성 여부를 특성에 따라 결정 (규칙 기반)
        is_poisonous = False
        
        # 독성 규칙들
        if features['odor'] in ['foul', 'pungent', 'fishy', 'spicy']:
            is_poisonous = True
        elif features['cap_color'] in ['green', 'purple'] and features['bruises'] == 'no':
            is_poisonous = True
        elif features['gill_color'] in ['green', 'purple'] and features['gill_spacing'] == 'close':
            is_poisonous = True
        elif features['stalk_root'] == 'bulbous' and features['cap_surface'] == 'scaly':
            is_poisonous = True
        
        # 일부 노이즈 추가
        if np.random.random() < 0.05:  # 5% 노이즈
            is_poisonous = not is_poisonous
        
        features['class'] = 'poisonous' if is_poisonous else 'edible'
        data.append(features)
    
    df = pd.DataFrame(data)
    
    print(f"데이터셋 크기: {df.shape}")
    print(f"클래스 분포:")
    print(df['class'].value_counts())
    print(f"특성 수: {len(df.columns) - 1}")
    
    return df

def analyze_data_characteristics(df):
    """데이터 특성 분석"""
    print("\n=== 데이터 특성 분석 ===")
    
    # 기본 정보
    print(f"전체 샘플 수: {len(df)}")
    print(f"특성 수: {len(df.columns) - 1}")
    print(f"클래스: {df['class'].unique()}")
    
    # 클래스 분포
    class_dist = df['class'].value_counts()
    print(f"\n클래스 분포:")
    for class_name, count in class_dist.items():
        print(f"  {class_name}: {count} ({count/len(df)*100:.1f}%)")
    
    # 각 특성의 고유값 수
    print(f"\n특성별 고유값 수:")
    for col in df.columns:
        if col != 'class':
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count}개")
    
    # 클래스별 특성 분포 (일부만)
    print(f"\n주요 특성의 클래스별 분포:")
    for feature in ['odor', 'cap_color', 'gill_color']:
        print(f"\n{feature}:")
        crosstab = pd.crosstab(df[feature], df['class'], normalize='columns')
        print(crosstab.round(3))

def extract_and_evaluate_rules(df):
    """규칙 추출 및 평가"""
    print("\n=== 규칙 기반 분류기 학습 ===")
    
    # 데이터 분할
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 규칙 기반 분류기 학습
    rule_classifier = RuleBasedClassifier(
        min_support=0.05,    # 최소 지지도 5%
        min_confidence=0.8,  # 최소 신뢰도 80%
        min_lift=1.2        # 최소 리프트 1.2
    )
    
    print("규칙 학습 중...")
    rule_classifier.fit(X_train, y_train)
    
    # 예측 수행
    y_pred_rules = rule_classifier.predict(X_test)
    
    # 성능 평가
    accuracy_rules = accuracy_score(y_test, y_pred_rules)
    print(f"\n규칙 기반 분류기 정확도: {accuracy_rules:.4f}")
    
    # 상위 규칙들 출력
    print(f"\n=== 상위 10개 규칙 ===")
    top_rules = rule_classifier.get_top_rules(10)
    
    for i, rule in enumerate(top_rules, 1):
        antecedent_str = " AND ".join([f"{k}={v}" for k, v in rule['antecedent'].items()])
        consequent_str = list(rule['consequent'].values())[0]
        
        print(f"\n규칙 {i}:")
        print(f"  IF {antecedent_str}")
        print(f"  THEN class = {consequent_str}")
        print(f"  지지도: {rule['support']:.4f}")
        print(f"  신뢰도: {rule['confidence']:.4f}")
        print(f"  리프트: {rule['lift']:.4f}")
    
    return rule_classifier, y_test, y_pred_rules

def compare_with_decision_tree(X_train, X_test, y_train, y_test):
    """의사결정나무와 비교"""
    print("\n=== 의사결정나무와 비교 ===")
    
    # 레이블 인코딩 (의사결정나무를 위해)
    le_dict = {}
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()
    
    for col in X_train.columns:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train[col])
        X_test_encoded[col] = le.transform(X_test[col])
        le_dict[col] = le
    
    # 의사결정나무 학습
    dt_classifier = DecisionTreeClassifier(
        max_depth=10, 
        min_samples_split=20, 
        min_samples_leaf=10, 
        random_state=42
    )
    dt_classifier.fit(X_train_encoded, y_train)
    
    # 예측
    y_pred_dt = dt_classifier.predict(X_test_encoded)
    
    # 성능 비교
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    
    print(f"의사결정나무 정확도: {accuracy_dt:.4f}")
    
    # 특성 중요도
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': dt_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n특성 중요도 (의사결정나무):")
    for _, row in feature_importance.head().iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return dt_classifier, y_pred_dt

def analyze_rule_quality_metrics():
    """규칙 품질 지표 분석"""
    print("\n=== 규칙 품질 지표 상세 분석 ===")
    
    # 예시 데이터로 지표 계산 설명
    print("예시: 'odor=foul' → 'class=poisonous' 규칙")
    
    # 가상의 데이터로 계산 예시
    total_samples = 1000
    foul_odor_samples = 150
    poisonous_samples = 400
    foul_and_poisonous = 140
    
    # 지지도 계산
    support = foul_and_poisonous / total_samples
    print(f"\n1. 지지도 (Support):")
    print(f"   공식: P(odor=foul AND class=poisonous)")
    print(f"   계산: {foul_and_poisonous}/{total_samples} = {support:.4f}")
    print(f"   해석: 전체 데이터의 {support*100:.1f}%가 이 규칙을 만족")
    
    # 신뢰도 계산
    confidence = foul_and_poisonous / foul_odor_samples
    print(f"\n2. 신뢰도 (Confidence):")
    print(f"   공식: P(class=poisonous | odor=foul)")
    print(f"   계산: {foul_and_poisonous}/{foul_odor_samples} = {confidence:.4f}")
    print(f"   해석: 냄새가 foul인 버섯의 {confidence*100:.1f}%가 독성")
    
    # 리프트 계산
    poisonous_prob = poisonous_samples / total_samples
    lift = confidence / poisonous_prob
    print(f"\n3. 리프트 (Lift):")
    print(f"   공식: Confidence / P(class=poisonous)")
    print(f"   계산: {confidence:.4f} / {poisonous_prob:.4f} = {lift:.4f}")
    print(f"   해석: 냄새 정보가 독성 예측을 {lift:.1f}배 개선")
    
    # 리프트 해석
    if lift > 1:
        print(f"   → 양의 상관관계: 유용한 규칙")
    elif lift == 1:
        print(f"   → 독립적 관계: 무의미한 규칙")
    else:
        print(f"   → 음의 상관관계: 역효과 규칙")

def visualize_results(y_test, y_pred_rules, y_pred_dt):
    """결과 시각화"""
    print("\n=== 결과 시각화 ===")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 규칙 기반 분류기 혼동 행렬
    cm_rules = confusion_matrix(y_test, y_pred_rules)
    sns.heatmap(cm_rules, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['edible', 'poisonous'], 
                yticklabels=['edible', 'poisonous'],
                ax=axes[0])
    axes[0].set_title('규칙 기반 분류기')
    axes[0].set_xlabel('예측')
    axes[0].set_ylabel('실제')
    
    # 의사결정나무 혼동 행렬
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens',
                xticklabels=['edible', 'poisonous'], 
                yticklabels=['edible', 'poisonous'],
                ax=axes[1])
    axes[1].set_title('의사결정나무')
    axes[1].set_xlabel('예측')
    axes[1].set_ylabel('실제')
    
    plt.tight_layout()
    plt.show()
    
    # 성능 비교 막대 그래프
    accuracies = [accuracy_score(y_test, y_pred_rules), accuracy_score(y_test, y_pred_dt)]
    methods = ['규칙 기반', '의사결정나무']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, accuracies, color=['skyblue', 'lightgreen'])
    plt.title('분류 방법별 정확도 비교')
    plt.ylabel('정확도')
    plt.ylim(0, 1)
    
    # 막대 위에 정확도 값 표시
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.4f}', ha='center', va='bottom')
    
    plt.show()

def demonstrate_rule_interpretation():
    """규칙 해석 시연"""
    print("\n=== 규칙 해석 및 실용적 적용 ===")
    
    print("생성된 규칙들의 실용적 의미:")
    
    practical_rules = [
        {
            'rule': "IF odor=foul THEN class=poisonous",
            'interpretation': "악취가 나는 버섯은 독성일 가능성이 매우 높음",
            'practical_use': "야생 버섯 채취 시 냄새를 먼저 확인"
        },
        {
            'rule': "IF gill_color=green THEN class=poisonous", 
            'interpretation': "아가미가 녹색인 버섯은 독성일 가능성이 높음",
            'practical_use': "버섯의 아가미 색깔을 주의 깊게 관찰"
        },
        {
            'rule': "IF cap_color=white AND bruises=no THEN class=edible",
            'interpretation': "흰색 갓을 가지고 상처가 없는 버섯은 식용 가능성이 높음",
            'practical_use': "안전한 버섯 식별의 보조 지표로 활용"
        }
    ]
    
    for i, rule_info in enumerate(practical_rules, 1):
        print(f"\n{i}. {rule_info['rule']}")
        print(f"   해석: {rule_info['interpretation']}")
        print(f"   활용: {rule_info['practical_use']}")
    
    print(f"\n규칙 기반 학습의 장점:")
    print("- 투명성: 의사결정 과정이 명확하게 드러남")
    print("- 검증 가능성: 도메인 전문가가 규칙의 타당성을 검토 가능")
    print("- 설명 가능성: 각 예측에 대한 명확한 근거 제시")
    print("- 수정 용이성: 잘못된 규칙을 개별적으로 수정 가능")

def main():
    """메인 실행 함수"""
    print("규칙 기반 학습 실습을 시작합니다.")
    print("=" * 50)
    
    # 1. 데이터 로딩
    df = load_mushroom_data()
    
    # 2. 데이터 특성 분석
    analyze_data_characteristics(df)
    
    # 3. 데이터 분할
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # 4. 규칙 추출 및 평가
    rule_classifier, y_test, y_pred_rules = extract_and_evaluate_rules(df)
    
    # 5. 의사결정나무와 비교
    dt_classifier, y_pred_dt = compare_with_decision_tree(X_train, X_test, y_train, y_test)
    
    # 6. 규칙 품질 지표 분석
    analyze_rule_quality_metrics()
    
    # 7. 결과 시각화
    visualize_results(y_test, y_pred_rules, y_pred_dt)
    
    # 8. 규칙 해석 시연
    demonstrate_rule_interpretation()
    
    print("\n" + "=" * 50)
    print("규칙 기반 학습 실습이 완료되었습니다.")
    print("\n주요 학습 내용:")
    print("1. 규칙 기반 분류기의 구현과 동작 원리")
    print("2. 규칙 품질 지표(지지도, 신뢰도, 리프트) 계산")
    print("3. 의사결정나무와의 성능 및 특성 비교")
    print("4. 생성된 규칙의 실용적 해석과 적용")

if __name__ == "__main__":
    main()