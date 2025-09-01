"""
앙상블 학습 실습

이 실습에서는 Wine Quality 데이터셋을 사용하여 다양한 앙상블 기법을 구현하고
개별 모델과 앙상블 모델의 성능을 비교 분석합니다.

주요 내용:
1. 데이터 로딩 및 전처리
2. 기본 모델들 구현 (의사결정나무, 로지스틱 회귀, SVM)
3. 배깅 (Random Forest) 구현
4. 부스팅 (AdaBoost, Gradient Boosting) 구현
5. 스태킹 구현
6. 성능 비교 및 분석
7. 앙상블 효과 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                            GradientBoostingClassifier, VotingClassifier,
                            BaggingClassifier)
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data():
    """
    Wine Quality 데이터셋 로딩 및 탐색
    """
    print("=== 1. 데이터 로딩 및 탐색 ===")
    
    # Wine 데이터셋 로딩 (sklearn 내장 데이터셋 사용)
    wine = load_wine()
    X, y = wine.data, wine.target
    feature_names = wine.feature_names
    target_names = wine.target_names
    
    print(f"데이터셋 크기: {X.shape}")
    print(f"특성 수: {X.shape[1]}")
    print(f"클래스 수: {len(np.unique(y))}")
    print(f"클래스 분포: {np.bincount(y)}")
    print(f"클래스 이름: {target_names}")
    
    # 데이터프레임 생성
    df = pd.DataFrame(X, columns=feature_names)
    df['quality'] = y
    
    print("\n데이터 기본 정보:")
    print(df.info())
    
    print("\n기술 통계:")
    print(df.describe())
    
    # 클래스 분포 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.bar(target_names, np.bincount(y))
    plt.title('Wine Quality Class Distribution')
    plt.xlabel('Wine Class')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.pie(np.bincount(y), labels=target_names, autopct='%1.1f%%')
    plt.title('Wine Quality Class Proportion')
    
    plt.tight_layout()
    plt.show()
    
    return X, y, feature_names, target_names

def preprocess_data(X, y):
    """
    데이터 전처리
    """
    print("\n=== 2. 데이터 전처리 ===")
    
    # 훈련/테스트 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 특성 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"훈련 데이터 크기: {X_train.shape}")
    print(f"테스트 데이터 크기: {X_test.shape}")
    print(f"훈련 데이터 클래스 분포: {np.bincount(y_train)}")
    print(f"테스트 데이터 클래스 분포: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler

def create_base_models():
    """
    기본 모델들 생성
    """
    print("\n=== 3. 기본 모델들 생성 ===")
    
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True),
        'Random Forest (Individual)': RandomForestClassifier(n_estimators=1, random_state=42)
    }
    
    print("생성된 기본 모델들:")
    for name, model in models.items():
        print(f"- {name}: {type(model).__name__}")
    
    return models

def evaluate_base_models(models, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """
    기본 모델들 성능 평가
    """
    print("\n=== 4. 기본 모델들 성능 평가 ===")
    
    results = {}
    cv_scores = {}
    
    # 교차 검증 설정
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\n{name} 평가 중...")
        
        # 스케일링이 필요한 모델 구분
        if name in ['Logistic Regression', 'SVM']:
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        # 모델 훈련
        model.fit(X_train_use, y_train)
        
        # 예측
        y_pred = model.predict(X_test_use)
        
        # 성능 계산
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        
        # 교차 검증
        cv_score = cross_val_score(model, X_train_use, y_train, cv=cv, scoring='accuracy')
        cv_scores[name] = cv_score
        
        print(f"테스트 정확도: {accuracy:.4f}")
        print(f"교차 검증 평균: {cv_score.mean():.4f} (±{cv_score.std()*2:.4f})")
    
    return results, cv_scores

def implement_bagging(X_train, X_test, y_train, y_test):
    """
    배깅 (Random Forest) 구현
    """
    print("\n=== 5. 배깅 (Random Forest) 구현 ===")
    
    # Random Forest 모델들
    rf_models = {
        'Random Forest (10)': RandomForestClassifier(n_estimators=10, random_state=42),
        'Random Forest (50)': RandomForestClassifier(n_estimators=50, random_state=42),
        'Random Forest (100)': RandomForestClassifier(n_estimators=100, random_state=42),
        'Random Forest (200)': RandomForestClassifier(n_estimators=200, random_state=42)
    }
    
    rf_results = {}
    
    for name, model in rf_models.items():
        print(f"\n{name} 훈련 중...")
        
        # 모델 훈련
        model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        rf_results[name] = accuracy
        
        print(f"테스트 정확도: {accuracy:.4f}")
        
        # 특성 중요도 출력 (100개 트리 모델만)
        if name == 'Random Forest (100)':
            feature_importance = model.feature_importances_
            print(f"상위 5개 중요 특성:")
            wine = load_wine()
            feature_names = wine.feature_names
            top_features = np.argsort(feature_importance)[-5:][::-1]
            for i, idx in enumerate(top_features):
                print(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    return rf_results, rf_models['Random Forest (100)']

def implement_boosting(X_train, X_test, y_train, y_test):
    """
    부스팅 (AdaBoost, Gradient Boosting) 구현
    """
    print("\n=== 6. 부스팅 구현 ===")
    
    # 부스팅 모델들
    boosting_models = {
        'AdaBoost': AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50,
            random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }
    
    boosting_results = {}
    
    for name, model in boosting_models.items():
        print(f"\n{name} 훈련 중...")
        
        # 모델 훈련
        model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        boosting_results[name] = accuracy
        
        print(f"테스트 정확도: {accuracy:.4f}")
        
        # 특성 중요도 출력
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
            wine = load_wine()
            feature_names = wine.feature_names
            top_features = np.argsort(feature_importance)[-5:][::-1]
            print(f"상위 5개 중요 특성:")
            for i, idx in enumerate(top_features):
                print(f"  {i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    return boosting_results, boosting_models

def implement_voting(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """
    투표 앙상블 구현
    """
    print("\n=== 7. 투표 앙상블 구현 ===")
    
    # 기본 모델들 정의
    dt = DecisionTreeClassifier(random_state=42, max_depth=10)
    lr = LogisticRegression(random_state=42, max_iter=1000)
    svm = SVC(random_state=42, probability=True)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    
    # 하드 투표 앙상블
    hard_voting = VotingClassifier(
        estimators=[('dt', dt), ('rf', rf)],
        voting='hard'
    )
    
    # 소프트 투표 앙상블 (확률 지원 모델들만)
    soft_voting = VotingClassifier(
        estimators=[('lr', lr), ('svm', svm), ('rf', rf)],
        voting='soft'
    )
    
    voting_results = {}
    
    # 하드 투표 평가
    print("\n하드 투표 앙상블 훈련 중...")
    hard_voting.fit(X_train, y_train)
    y_pred_hard = hard_voting.predict(X_test)
    accuracy_hard = accuracy_score(y_test, y_pred_hard)
    voting_results['Hard Voting'] = accuracy_hard
    print(f"하드 투표 정확도: {accuracy_hard:.4f}")
    
    # 소프트 투표 평가 (스케일된 데이터 사용)
    print("\n소프트 투표 앙상블 훈련 중...")
    
    # 각 모델을 개별적으로 훈련 (스케일링 고려)
    lr_scaled = LogisticRegression(random_state=42, max_iter=1000)
    svm_scaled = SVC(random_state=42, probability=True)
    rf_original = RandomForestClassifier(n_estimators=50, random_state=42)
    
    lr_scaled.fit(X_train_scaled, y_train)
    svm_scaled.fit(X_train_scaled, y_train)
    rf_original.fit(X_train, y_train)
    
    # 소프트 투표를 수동으로 구현
    lr_proba = lr_scaled.predict_proba(X_test_scaled)
    svm_proba = svm_scaled.predict_proba(X_test_scaled)
    rf_proba = rf_original.predict_proba(X_test)
    
    # 확률 평균
    avg_proba = (lr_proba + svm_proba + rf_proba) / 3
    y_pred_soft = np.argmax(avg_proba, axis=1)
    accuracy_soft = accuracy_score(y_test, y_pred_soft)
    voting_results['Soft Voting'] = accuracy_soft
    print(f"소프트 투표 정확도: {accuracy_soft:.4f}")
    
    return voting_results

def implement_stacking(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
    """
    스태킹 앙상블 구현
    """
    print("\n=== 8. 스태킹 앙상블 구현 ===")
    
    from sklearn.model_selection import cross_val_predict
    
    # 기본 모델들
    base_models = {
        'dt': DecisionTreeClassifier(random_state=42, max_depth=10),
        'lr': LogisticRegression(random_state=42, max_iter=1000),
        'rf': RandomForestClassifier(n_estimators=50, random_state=42)
    }
    
    # 1단계: 기본 모델들의 교차 검증 예측 생성
    print("1단계: 기본 모델들의 교차 검증 예측 생성")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_features_train = np.zeros((X_train.shape[0], len(base_models)))
    meta_features_test = np.zeros((X_test.shape[0], len(base_models)))
    
    for i, (name, model) in enumerate(base_models.items()):
        print(f"  {name} 처리 중...")
        
        # 스케일링이 필요한 모델 구분
        if name == 'lr':
            X_train_use = X_train_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_test_use = X_test
        
        # 교차 검증 예측 (훈련 데이터용)
        cv_predictions = cross_val_predict(model, X_train_use, y_train, cv=cv, method='predict_proba')
        meta_features_train[:, i] = cv_predictions[:, 1] if cv_predictions.shape[1] == 2 else np.max(cv_predictions, axis=1)
        
        # 전체 훈련 데이터로 모델 훈련 후 테스트 예측
        model.fit(X_train_use, y_train)
        test_predictions = model.predict_proba(X_test_use)
        meta_features_test[:, i] = test_predictions[:, 1] if test_predictions.shape[1] == 2 else np.max(test_predictions, axis=1)
    
    # 2단계: 메타 모델 훈련
    print("\n2단계: 메타 모델 훈련")
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(meta_features_train, y_train)
    
    # 최종 예측
    y_pred_stacking = meta_model.predict(meta_features_test)
    accuracy_stacking = accuracy_score(y_test, y_pred_stacking)
    
    print(f"스태킹 앙상블 정확도: {accuracy_stacking:.4f}")
    
    # 메타 특성 중요도
    print("\n메타 모델 계수 (기본 모델 중요도):")
    for i, (name, coef) in enumerate(zip(base_models.keys(), meta_model.coef_[0])):
        print(f"  {name}: {coef:.4f}")
    
    return accuracy_stacking, meta_features_train, meta_features_test

def compare_all_models(base_results, rf_results, boosting_results, voting_results, stacking_accuracy):
    """
    모든 모델 성능 비교
    """
    print("\n=== 9. 전체 모델 성능 비교 ===")
    
    # 모든 결과 통합
    all_results = {}
    all_results.update(base_results)
    all_results.update(rf_results)
    all_results.update(boosting_results)
    all_results.update(voting_results)
    all_results['Stacking'] = stacking_accuracy
    
    # 결과 정렬
    sorted_results = dict(sorted(all_results.items(), key=lambda x: x[1], reverse=True))
    
    print("모델별 테스트 정확도:")
    print("-" * 40)
    for i, (name, accuracy) in enumerate(sorted_results.items(), 1):
        print(f"{i:2d}. {name:<25}: {accuracy:.4f}")
    
    # 시각화
    plt.figure(figsize=(15, 8))
    
    # 성능 비교 막대 그래프
    plt.subplot(2, 2, 1)
    models = list(sorted_results.keys())
    accuracies = list(sorted_results.values())
    colors = ['red' if 'Random Forest' in model or 'AdaBoost' in model or 'Gradient' in model 
              or 'Voting' in model or 'Stacking' in model else 'lightblue' for model in models]
    
    bars = plt.bar(range(len(models)), accuracies, color=colors)
    plt.xlabel('Models')
    plt.ylabel('Test Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylim(0.8, 1.0)
    
    # 값 표시
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 앙상블 vs 단일 모델 비교
    plt.subplot(2, 2, 2)
    single_models = ['Decision Tree', 'Logistic Regression', 'SVM']
    ensemble_models = ['Random Forest (100)', 'AdaBoost', 'Gradient Boosting', 'Soft Voting', 'Stacking']
    
    single_acc = [all_results[model] for model in single_models if model in all_results]
    ensemble_acc = [all_results[model] for model in ensemble_models if model in all_results]
    
    plt.boxplot([single_acc, ensemble_acc], labels=['Single Models', 'Ensemble Models'])
    plt.ylabel('Test Accuracy')
    plt.title('Single vs Ensemble Models')
    plt.grid(True, alpha=0.3)
    
    # Random Forest 트리 개수별 성능
    plt.subplot(2, 2, 3)
    rf_names = [name for name in rf_results.keys()]
    rf_accs = [rf_results[name] for name in rf_names]
    n_estimators = [10, 50, 100, 200]
    
    plt.plot(n_estimators, rf_accs, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Trees')
    plt.ylabel('Test Accuracy')
    plt.title('Random Forest: Effect of Number of Trees')
    plt.grid(True, alpha=0.3)
    
    # 앙상블 기법별 성능
    plt.subplot(2, 2, 4)
    ensemble_types = ['Bagging\n(Random Forest)', 'Boosting\n(AdaBoost)', 'Boosting\n(Gradient)', 'Voting\n(Soft)', 'Stacking']
    ensemble_scores = [
        all_results['Random Forest (100)'],
        all_results['AdaBoost'],
        all_results['Gradient Boosting'],
        all_results['Soft Voting'],
        all_results['Stacking']
    ]
    
    bars = plt.bar(ensemble_types, ensemble_scores, color=['skyblue', 'lightcoral', 'lightcoral', 'lightgreen', 'gold'])
    plt.ylabel('Test Accuracy')
    plt.title('Ensemble Techniques Comparison')
    plt.ylim(0.85, 1.0)
    
    for bar, score in zip(bars, ensemble_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return sorted_results

def analyze_ensemble_effects(X_train, X_test, y_train, y_test):
    """
    앙상블 효과 분석
    """
    print("\n=== 10. 앙상블 효과 분석 ===")
    
    # 다양한 크기의 Random Forest로 분산 감소 효과 분석
    n_estimators_list = [1, 5, 10, 20, 50, 100, 200]
    n_runs = 10
    
    results_matrix = np.zeros((len(n_estimators_list), n_runs))
    
    print("Random Forest 크기별 성능 변동성 분석 중...")
    
    for i, n_est in enumerate(n_estimators_list):
        for run in range(n_runs):
            rf = RandomForestClassifier(n_estimators=n_est, random_state=run)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results_matrix[i, run] = accuracy
    
    # 결과 분석
    mean_accuracies = np.mean(results_matrix, axis=1)
    std_accuracies = np.std(results_matrix, axis=1)
    
    print("\nRandom Forest 크기별 성능 통계:")
    print("Trees\tMean Acc\tStd Dev\tMin Acc\tMax Acc")
    print("-" * 50)
    for i, n_est in enumerate(n_estimators_list):
        print(f"{n_est:3d}\t{mean_accuracies[i]:.4f}\t\t{std_accuracies[i]:.4f}\t{np.min(results_matrix[i]):.4f}\t{np.max(results_matrix[i]):.4f}")
    
    # 시각화
    plt.figure(figsize=(15, 5))
    
    # 평균 성능과 표준편차
    plt.subplot(1, 3, 1)
    plt.errorbar(n_estimators_list, mean_accuracies, yerr=std_accuracies, 
                marker='o', capsize=5, capthick=2, linewidth=2)
    plt.xlabel('Number of Trees')
    plt.ylabel('Test Accuracy')
    plt.title('Random Forest: Mean Performance ± Std Dev')
    plt.grid(True, alpha=0.3)
    
    # 표준편차 변화
    plt.subplot(1, 3, 2)
    plt.plot(n_estimators_list, std_accuracies, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Trees')
    plt.ylabel('Standard Deviation')
    plt.title('Variance Reduction Effect')
    plt.grid(True, alpha=0.3)
    
    # 박스플롯으로 분포 비교
    plt.subplot(1, 3, 3)
    plt.boxplot([results_matrix[i] for i in [0, 2, 4, 6]], 
               labels=[f'{n_estimators_list[i]} trees' for i in [0, 2, 4, 6]])
    plt.ylabel('Test Accuracy')
    plt.title('Performance Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results_matrix

def feature_importance_analysis(rf_model, boosting_models):
    """
    특성 중요도 분석
    """
    print("\n=== 11. 특성 중요도 분석 ===")
    
    wine = load_wine()
    feature_names = wine.feature_names
    
    # Random Forest 특성 중요도
    rf_importance = rf_model.feature_importances_
    
    # Gradient Boosting 특성 중요도
    gb_importance = boosting_models['Gradient Boosting'].feature_importances_
    
    # 특성 중요도 비교
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Random Forest': rf_importance,
        'Gradient Boosting': gb_importance
    })
    
    # 정렬
    importance_df = importance_df.sort_values('Random Forest', ascending=False)
    
    print("특성 중요도 비교 (상위 10개):")
    print(importance_df.head(10).to_string(index=False))
    
    # 시각화
    plt.figure(figsize=(15, 6))
    
    # Random Forest 특성 중요도
    plt.subplot(1, 2, 1)
    top_10_rf = importance_df.head(10)
    plt.barh(range(len(top_10_rf)), top_10_rf['Random Forest'])
    plt.yticks(range(len(top_10_rf)), top_10_rf['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.gca().invert_yaxis()
    
    # Gradient Boosting 특성 중요도
    plt.subplot(1, 2, 2)
    top_10_gb = importance_df.sort_values('Gradient Boosting', ascending=False).head(10)
    plt.barh(range(len(top_10_gb)), top_10_gb['Gradient Boosting'])
    plt.yticks(range(len(top_10_gb)), top_10_gb['Feature'])
    plt.xlabel('Feature Importance')
    plt.title('Gradient Boosting Feature Importance')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()
    
    # 특성 중요도 상관관계
    correlation = np.corrcoef(rf_importance, gb_importance)[0, 1]
    print(f"\nRandom Forest와 Gradient Boosting 특성 중요도 상관계수: {correlation:.4f}")

def main():
    """
    메인 실행 함수
    """
    print("앙상블 학습 실습을 시작합니다!")
    print("=" * 50)
    
    # 1. 데이터 로딩 및 탐색
    X, y, feature_names, target_names = load_and_explore_data()
    
    # 2. 데이터 전처리
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler = preprocess_data(X, y)
    
    # 3. 기본 모델들 생성 및 평가
    base_models = create_base_models()
    base_results, cv_scores = evaluate_base_models(
        base_models, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    )
    
    # 4. 배깅 (Random Forest) 구현
    rf_results, best_rf_model = implement_bagging(X_train, X_test, y_train, y_test)
    
    # 5. 부스팅 구현
    boosting_results, boosting_models = implement_boosting(X_train, X_test, y_train, y_test)
    
    # 6. 투표 앙상블 구현
    voting_results = implement_voting(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
    # 7. 스태킹 구현
    stacking_accuracy, meta_train, meta_test = implement_stacking(
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    )
    
    # 8. 전체 모델 성능 비교
    final_results = compare_all_models(
        base_results, rf_results, boosting_results, voting_results, stacking_accuracy
    )
    
    # 9. 앙상블 효과 분석
    variance_analysis = analyze_ensemble_effects(X_train, X_test, y_train, y_test)
    
    # 10. 특성 중요도 분석
    feature_importance_analysis(best_rf_model, boosting_models)
    
    print("\n" + "=" * 50)
    print("앙상블 학습 실습이 완료되었습니다!")
    print("\n주요 결과:")
    print(f"- 최고 성능 모델: {max(final_results.items(), key=lambda x: x[1])}")
    print(f"- 앙상블 평균 성능: {np.mean([acc for name, acc in final_results.items() if any(keyword in name for keyword in ['Random Forest', 'AdaBoost', 'Gradient', 'Voting', 'Stacking'])]):.4f}")
    print(f"- 단일 모델 평균 성능: {np.mean([acc for name, acc in final_results.items() if name in ['Decision Tree', 'Logistic Regression', 'SVM']]):.4f}")

if __name__ == "__main__":
    main()