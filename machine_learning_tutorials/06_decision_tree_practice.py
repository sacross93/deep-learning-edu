#!/usr/bin/env python3
"""
의사결정나무 실습
Iris 데이터셋을 활용한 다중 클래스 분류 및 트리 시각화
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score, validation_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_and_explore_data():
    """Iris 데이터셋 로딩 및 탐색"""
    print("=" * 60)
    print("1. 데이터셋 로딩 및 탐색")
    print("=" * 60)
    
    # Iris 데이터셋 로딩
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # DataFrame 생성
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['species'] = [target_names[i] for i in y]
    
    print(f"데이터셋 크기: {X.shape}")
    print(f"특성 개수: {X.shape[1]}")
    print(f"클래스 개수: {len(target_names)}")
    print(f"클래스 이름: {target_names}")
    
    print("\n데이터 미리보기:")
    print(df.head())
    
    print("\n클래스별 분포:")
    print(df['species'].value_counts())
    
    print("\n특성별 기본 통계:")
    print(df.describe())
    
    return X, y, feature_names, target_names, df

def visualize_data(df, feature_names):
    """데이터 시각화"""
    print("\n" + "=" * 60)
    print("2. 데이터 시각화")
    print("=" * 60)
    
    # 특성별 분포 시각화
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Iris Dataset - Feature Distributions by Species', fontsize=16)
    
    for i, feature in enumerate(feature_names):
        row, col = i // 2, i % 2
        for species in df['species'].unique():
            species_data = df[df['species'] == species][feature]
            axes[row, col].hist(species_data, alpha=0.7, label=species, bins=15)
        
        axes[row, col].set_title(feature)
        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Frequency')
        axes[row, col].legend()
    
    plt.tight_layout()
    plt.show()
    
    # 특성 간 상관관계 시각화
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[feature_names].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.show()
    
    # 페어플롯
    plt.figure(figsize=(12, 10))
    pd.plotting.scatter_matrix(df[feature_names + ['species']], 
                              c=df['target'], 
                              figsize=(12, 10),
                              alpha=0.7)
    plt.suptitle('Iris Dataset - Pairwise Feature Relationships')
    plt.show()

def build_decision_tree(X, y, feature_names, target_names):
    """의사결정나무 모델 구축"""
    print("\n" + "=" * 60)
    print("3. 의사결정나무 모델 구축")
    print("=" * 60)
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"훈련 데이터: {X_train.shape}")
    print(f"테스트 데이터: {X_test.shape}")
    
    # 기본 의사결정나무 모델
    dt_basic = DecisionTreeClassifier(random_state=42)
    dt_basic.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred_basic = dt_basic.predict(X_test)
    accuracy_basic = accuracy_score(y_test, y_pred_basic)
    
    print(f"\n기본 모델 정확도: {accuracy_basic:.4f}")
    print("\n분류 보고서:")
    print(classification_report(y_test, y_pred_basic, target_names=target_names))
    
    return dt_basic, X_train, X_test, y_train, y_test

def visualize_tree(dt_model, feature_names, target_names):
    """의사결정나무 시각화"""
    print("\n" + "=" * 60)
    print("4. 의사결정나무 시각화")
    print("=" * 60)
    
    # 트리 구조 시각화
    plt.figure(figsize=(20, 12))
    plot_tree(dt_model, 
              feature_names=feature_names,
              class_names=target_names,
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title('Decision Tree Visualization (Full Tree)', fontsize=16)
    plt.show()
    
    # 간단한 트리 (깊이 제한)
    dt_simple = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_simple.fit(dt_model.feature_importances_.reshape(1, -1), [0])  # 더미 학습
    
    # 텍스트 형태 트리 출력
    print("\n트리 구조 (텍스트 형태):")
    tree_rules = export_text(dt_model, feature_names=feature_names)
    print(tree_rules[:1000] + "..." if len(tree_rules) > 1000 else tree_rules)

def analyze_feature_importance(dt_model, feature_names):
    """특성 중요도 분석"""
    print("\n" + "=" * 60)
    print("5. 특성 중요도 분석")
    print("=" * 60)
    
    # 특성 중요도 계산
    importances = dt_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("특성 중요도 순위:")
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # 특성 중요도 시각화
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Feature Importance in Decision Tree')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.show()
    
    return importances

def compare_split_criteria(X_train, X_test, y_train, y_test, feature_names):
    """분할 기준 비교 (지니 vs 엔트로피)"""
    print("\n" + "=" * 60)
    print("6. 분할 기준 비교")
    print("=" * 60)
    
    criteria = ['gini', 'entropy']
    results = {}
    
    for criterion in criteria:
        dt = DecisionTreeClassifier(criterion=criterion, random_state=42)
        dt.fit(X_train, y_train)
        
        train_score = dt.score(X_train, y_train)
        test_score = dt.score(X_test, y_test)
        
        results[criterion] = {
            'model': dt,
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'tree_depth': dt.get_depth(),
            'n_leaves': dt.get_n_leaves()
        }
        
        print(f"\n{criterion.upper()} 기준:")
        print(f"  훈련 정확도: {train_score:.4f}")
        print(f"  테스트 정확도: {test_score:.4f}")
        print(f"  트리 깊이: {dt.get_depth()}")
        print(f"  리프 노드 수: {dt.get_n_leaves()}")
    
    # 결과 비교 시각화
    criteria_names = list(results.keys())
    train_scores = [results[c]['train_accuracy'] for c in criteria_names]
    test_scores = [results[c]['test_accuracy'] for c in criteria_names]
    
    x = np.arange(len(criteria_names))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, train_scores, width, label='Train Accuracy', alpha=0.8)
    plt.bar(x + width/2, test_scores, width, label='Test Accuracy', alpha=0.8)
    
    plt.xlabel('Split Criterion')
    plt.ylabel('Accuracy')
    plt.title('Comparison of Split Criteria')
    plt.xticks(x, criteria_names)
    plt.legend()
    plt.ylim(0.8, 1.0)
    
    for i, (train, test) in enumerate(zip(train_scores, test_scores)):
        plt.text(i - width/2, train + 0.01, f'{train:.3f}', ha='center')
        plt.text(i + width/2, test + 0.01, f'{test:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return results

def hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """하이퍼파라미터 튜닝"""
    print("\n" + "=" * 60)
    print("7. 하이퍼파라미터 튜닝")
    print("=" * 60)
    
    # max_depth 튜닝
    max_depths = range(1, 11)
    train_scores, test_scores = validation_curve(
        DecisionTreeClassifier(random_state=42), X_train, y_train,
        param_name='max_depth', param_range=max_depths,
        cv=5, scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(12, 5))
    
    # max_depth 검증 곡선
    plt.subplot(1, 2, 1)
    plt.plot(max_depths, train_mean, 'o-', label='Training Score', alpha=0.8)
    plt.fill_between(max_depths, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(max_depths, test_mean, 'o-', label='Validation Score', alpha=0.8)
    plt.fill_between(max_depths, test_mean - test_std, test_mean + test_std, alpha=0.2)
    
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Validation Curve (Max Depth)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # min_samples_split 튜닝
    min_samples_splits = range(2, 21, 2)
    train_scores2, test_scores2 = validation_curve(
        DecisionTreeClassifier(random_state=42), X_train, y_train,
        param_name='min_samples_split', param_range=min_samples_splits,
        cv=5, scoring='accuracy'
    )
    
    train_mean2 = np.mean(train_scores2, axis=1)
    test_mean2 = np.mean(test_scores2, axis=1)
    
    plt.subplot(1, 2, 2)
    plt.plot(min_samples_splits, train_mean2, 'o-', label='Training Score', alpha=0.8)
    plt.plot(min_samples_splits, test_mean2, 'o-', label='Validation Score', alpha=0.8)
    
    plt.xlabel('Min Samples Split')
    plt.ylabel('Accuracy')
    plt.title('Validation Curve (Min Samples Split)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 최적 파라미터 찾기
    best_depth_idx = np.argmax(test_mean)
    best_depth = max_depths[best_depth_idx]
    
    best_split_idx = np.argmax(test_mean2)
    best_split = min_samples_splits[best_split_idx]
    
    print(f"최적 max_depth: {best_depth} (검증 점수: {test_mean[best_depth_idx]:.4f})")
    print(f"최적 min_samples_split: {best_split} (검증 점수: {test_mean2[best_split_idx]:.4f})")
    
    return best_depth, best_split

def pruning_analysis(X_train, X_test, y_train, y_test, feature_names, target_names):
    """가지치기 효과 분석"""
    print("\n" + "=" * 60)
    print("8. 가지치기 효과 분석")
    print("=" * 60)
    
    # 가지치기 전후 비교
    models = {
        'No Pruning': DecisionTreeClassifier(random_state=42),
        'Max Depth=3': DecisionTreeClassifier(max_depth=3, random_state=42),
        'Min Samples Split=10': DecisionTreeClassifier(min_samples_split=10, random_state=42),
        'Combined': DecisionTreeClassifier(max_depth=3, min_samples_split=10, 
                                         min_samples_leaf=5, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        
        results[name] = {
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'depth': model.get_depth(),
            'leaves': model.get_n_leaves(),
            'overfitting': train_acc - test_acc
        }
        
        print(f"\n{name}:")
        print(f"  훈련 정확도: {train_acc:.4f}")
        print(f"  테스트 정확도: {test_acc:.4f}")
        print(f"  과적합 정도: {train_acc - test_acc:.4f}")
        print(f"  트리 깊이: {model.get_depth()}")
        print(f"  리프 노드 수: {model.get_n_leaves()}")
    
    # 결과 시각화
    model_names = list(results.keys())
    train_accs = [results[m]['train_accuracy'] for m in model_names]
    test_accs = [results[m]['test_accuracy'] for m in model_names]
    overfitting = [results[m]['overfitting'] for m in model_names]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 정확도 비교
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, train_accs, width, label='Train Accuracy', alpha=0.8)
    ax1.bar(x + width/2, test_accs, width, label='Test Accuracy', alpha=0.8)
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training vs Test Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 과적합 정도
    colors = ['red' if x > 0.05 else 'green' for x in overfitting]
    ax2.bar(model_names, overfitting, color=colors, alpha=0.7)
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Overfitting (Train - Test)')
    ax2.set_title('Overfitting Analysis')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Overfitting Threshold')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return results

def decision_boundary_visualization(X, y, feature_names, target_names):
    """결정 경계 시각화 (2D)"""
    print("\n" + "=" * 60)
    print("9. 결정 경계 시각화")
    print("=" * 60)
    
    # 가장 중요한 두 특성 선택 (sepal length, petal length)
    X_2d = X[:, [0, 2]]  # sepal length, petal length
    feature_pair = [feature_names[0], feature_names[2]]
    
    # 모델 학습
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_2d, y)
    
    # 결정 경계 그리기
    h = 0.02
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 5))
    
    # 결정 경계
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.xlabel(feature_pair[0])
    plt.ylabel(feature_pair[1])
    plt.title('Decision Tree - Decision Boundary')
    plt.colorbar(scatter)
    
    # 트리 구조 (간단한 버전)
    plt.subplot(1, 2, 2)
    plot_tree(dt, feature_names=feature_pair, class_names=target_names,
              filled=True, rounded=True, fontsize=8)
    plt.title('Decision Tree Structure (2D)')
    
    plt.tight_layout()
    plt.show()

def cross_validation_analysis(X, y):
    """교차 검증 분석"""
    print("\n" + "=" * 60)
    print("10. 교차 검증 분석")
    print("=" * 60)
    
    # 다양한 설정으로 교차 검증
    models = {
        'Default': DecisionTreeClassifier(random_state=42),
        'Pruned': DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42),
        'Gini': DecisionTreeClassifier(criterion='gini', random_state=42),
        'Entropy': DecisionTreeClassifier(criterion='entropy', random_state=42)
    }
    
    cv_results = {}
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        cv_results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        print(f"{name}:")
        print(f"  평균 정확도: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        print(f"  개별 점수: {scores}")
    
    # 교차 검증 결과 시각화
    plt.figure(figsize=(12, 6))
    
    model_names = list(cv_results.keys())
    means = [cv_results[m]['mean'] for m in model_names]
    stds = [cv_results[m]['std'] for m in model_names]
    
    plt.bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('Cross-Validation Results Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 평균값 표시
    for i, (mean, std) in enumerate(zip(means, stds)):
        plt.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return cv_results

def main():
    """메인 실행 함수"""
    print("의사결정나무 실습 - Iris 데이터셋")
    print("=" * 60)
    
    # 1. 데이터 로딩 및 탐색
    X, y, feature_names, target_names, df = load_and_explore_data()
    
    # 2. 데이터 시각화
    visualize_data(df, feature_names)
    
    # 3. 의사결정나무 모델 구축
    dt_model, X_train, X_test, y_train, y_test = build_decision_tree(
        X, y, feature_names, target_names
    )
    
    # 4. 트리 시각화
    visualize_tree(dt_model, feature_names, target_names)
    
    # 5. 특성 중요도 분석
    importances = analyze_feature_importance(dt_model, feature_names)
    
    # 6. 분할 기준 비교
    criteria_results = compare_split_criteria(X_train, X_test, y_train, y_test, feature_names)
    
    # 7. 하이퍼파라미터 튜닝
    best_depth, best_split = hyperparameter_tuning(X_train, X_test, y_train, y_test)
    
    # 8. 가지치기 효과 분석
    pruning_results = pruning_analysis(X_train, X_test, y_train, y_test, 
                                     feature_names, target_names)
    
    # 9. 결정 경계 시각화
    decision_boundary_visualization(X, y, feature_names, target_names)
    
    # 10. 교차 검증 분석
    cv_results = cross_validation_analysis(X, y)
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("실습 요약")
    print("=" * 60)
    print("1. Iris 데이터셋으로 의사결정나무 모델 구축 완료")
    print("2. 트리 구조 시각화 및 해석 완료")
    print("3. 특성 중요도 분석 완료")
    print("4. 분할 기준(지니 vs 엔트로피) 비교 완료")
    print("5. 하이퍼파라미터 튜닝 완료")
    print("6. 가지치기 효과 분석 완료")
    print("7. 결정 경계 시각화 완료")
    print("8. 교차 검증 분석 완료")
    
    print(f"\n주요 결과:")
    print(f"- 가장 중요한 특성: {feature_names[np.argmax(importances)]}")
    print(f"- 최적 트리 깊이: {best_depth}")
    print(f"- 최적 최소 분할 샘플 수: {best_split}")
    
    best_model_name = max(cv_results.keys(), key=lambda x: cv_results[x]['mean'])
    print(f"- 최고 성능 모델: {best_model_name} (CV 점수: {cv_results[best_model_name]['mean']:.4f})")

if __name__ == "__main__":
    main()