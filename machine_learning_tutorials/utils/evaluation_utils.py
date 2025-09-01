"""
평가 유틸리티 모듈

분류, 회귀, 클러스터링 성능 지표 계산 및 모델 비교 기능을 제공합니다.
요구사항 2.3, 8.2를 충족합니다.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    calinski_harabasz_score, davies_bouldin_score
)
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold, KFold
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_pred_proba: np.ndarray = None, 
                                   average: str = 'weighted') -> Dict[str, float]:
    """
    분류 성능 지표를 계산합니다.
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        y_pred_proba: 예측 확률 (ROC-AUC 계산용)
        average: 다중 클래스 평균 방법
    
    Returns:
        성능 지표 딕셔너리
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    # ROC-AUC (이진 분류 또는 확률 예측이 있는 경우)
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # 이진 분류
                if y_pred_proba.ndim == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            else:  # 다중 클래스
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=average)
        except Exception as e:
            print(f"ROC-AUC 계산 중 오류: {e}")
            metrics['roc_auc'] = None
    
    return metrics


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    회귀 성능 지표를 계산합니다.
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
    
    Returns:
        성능 지표 딕셔너리
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2_score': r2_score(y_true, y_pred)
    }
    
    # 추가 지표들
    residuals = y_true - y_pred
    metrics['mean_residual'] = np.mean(residuals)
    metrics['std_residual'] = np.std(residuals)
    
    # MAPE (Mean Absolute Percentage Error)
    if not np.any(y_true == 0):
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    else:
        metrics['mape'] = None
    
    return metrics


def calculate_clustering_metrics(X: np.ndarray, labels_true: np.ndarray = None, 
                               labels_pred: np.ndarray = None) -> Dict[str, float]:
    """
    클러스터링 성능 지표를 계산합니다.
    
    Args:
        X: 입력 데이터
        labels_true: 실제 클러스터 라벨 (있는 경우)
        labels_pred: 예측된 클러스터 라벨
    
    Returns:
        성능 지표 딕셔너리
    """
    metrics = {}
    
    if labels_pred is not None:
        # 내부 평가 지표 (실제 라벨 불필요)
        if len(np.unique(labels_pred)) > 1:  # 클러스터가 2개 이상인 경우
            metrics['silhouette_score'] = silhouette_score(X, labels_pred)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels_pred)
            metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels_pred)
        
        # 외부 평가 지표 (실제 라벨 필요)
        if labels_true is not None:
            metrics['adjusted_rand_score'] = adjusted_rand_score(labels_true, labels_pred)
            metrics['normalized_mutual_info_score'] = normalized_mutual_info_score(labels_true, labels_pred)
            
            # 정확도 (클러스터 라벨을 최적 매핑)
            metrics['cluster_accuracy'] = _calculate_cluster_accuracy(labels_true, labels_pred)
    
    return metrics


def _calculate_cluster_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    클러스터링 정확도를 계산합니다 (헝가리안 알고리즘 사용).
    """
    from scipy.optimize import linear_sum_assignment
    
    # 혼동 행렬 생성
    unique_true = np.unique(labels_true)
    unique_pred = np.unique(labels_pred)
    
    cost_matrix = np.zeros((len(unique_true), len(unique_pred)))
    
    for i, true_label in enumerate(unique_true):
        for j, pred_label in enumerate(unique_pred):
            cost_matrix[i, j] = -np.sum((labels_true == true_label) & (labels_pred == pred_label))
    
    # 헝가리안 알고리즘으로 최적 매핑 찾기
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # 정확도 계산
    total_correct = -cost_matrix[row_indices, col_indices].sum()
    accuracy = total_correct / len(labels_true)
    
    return accuracy


def cross_validation_evaluation(model, X: np.ndarray, y: np.ndarray, 
                              cv: int = 5, scoring: Union[str, List[str]] = None,
                              task_type: str = 'classification') -> Dict[str, Any]:
    """
    교차 검증을 통한 모델 평가를 수행합니다.
    
    Args:
        model: 평가할 모델
        X: 입력 특성
        y: 타겟 변수
        cv: 교차 검증 폴드 수
        scoring: 평가 지표
        task_type: 작업 유형 ('classification', 'regression')
    
    Returns:
        교차 검증 결과 딕셔너리
    """
    # 기본 스코어링 설정
    if scoring is None:
        if task_type == 'classification':
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:  # regression
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    
    # 교차 검증 수행
    if task_type == 'classification':
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
    
    cv_results = cross_validate(model, X, y, cv=cv_splitter, scoring=scoring, 
                               return_train_score=True, n_jobs=-1)
    
    # 결과 정리
    results = {}
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        results[metric] = {
            'test_mean': np.mean(test_scores),
            'test_std': np.std(test_scores),
            'train_mean': np.mean(train_scores),
            'train_std': np.std(train_scores),
            'test_scores': test_scores,
            'train_scores': train_scores
        }
    
    return results


def compare_models(models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray, 
                  task_type: str = 'classification') -> pd.DataFrame:
    """
    여러 모델의 성능을 비교합니다.
    
    Args:
        models: 모델 딕셔너리 {모델명: 모델객체}
        X_train, y_train: 훈련 데이터
        X_test, y_test: 테스트 데이터
        task_type: 작업 유형
    
    Returns:
        모델 비교 결과 데이터프레임
    """
    results = []
    
    for model_name, model in models.items():
        try:
            # 모델 훈련
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_test)
            
            # 성능 계산
            if task_type == 'classification':
                y_pred_proba = None
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                
                metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
            else:  # regression
                metrics = calculate_regression_metrics(y_test, y_pred)
            
            # 결과 저장
            result = {'model': model_name}
            result.update(metrics)
            results.append(result)
            
        except Exception as e:
            print(f"모델 {model_name} 평가 중 오류: {e}")
            continue
    
    return pd.DataFrame(results)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str] = None, 
                         figsize: Tuple[int, int] = (8, 6)):
    """
    혼동 행렬을 시각화합니다.
    
    Args:
        y_true: 실제 라벨
        y_pred: 예측 라벨
        class_names: 클래스 이름 리스트
        figsize: 그래프 크기
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('혼동 행렬 (Confusion Matrix)')
    plt.xlabel('예측 라벨')
    plt.ylabel('실제 라벨')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray, 
                  figsize: Tuple[int, int] = (8, 6)):
    """
    ROC 곡선을 시각화합니다.
    
    Args:
        y_true: 실제 라벨 (이진 분류)
        y_pred_proba: 예측 확률
        figsize: 그래프 크기
    """
    if y_pred_proba.ndim == 2:
        y_pred_proba = y_pred_proba[:, 1]
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC 곡선')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, 
                  figsize: Tuple[int, int] = (12, 5)):
    """
    회귀 모델의 잔차를 시각화합니다.
    
    Args:
        y_true: 실제 값
        y_pred: 예측 값
        figsize: 그래프 크기
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 잔차 vs 예측값
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('예측값')
    axes[0].set_ylabel('잔차')
    axes[0].set_title('잔차 vs 예측값')
    axes[0].grid(True)
    
    # 잔차 히스토그램
    axes[1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('잔차')
    axes[1].set_ylabel('빈도')
    axes[1].set_title('잔차 분포')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_learning_curve(model, X: np.ndarray, y: np.ndarray, 
                       train_sizes: np.ndarray = None,
                       cv: int = 5, figsize: Tuple[int, int] = (10, 6)):
    """
    학습 곡선을 시각화합니다.
    
    Args:
        model: 평가할 모델
        X: 입력 특성
        y: 타겟 변수
        train_sizes: 훈련 세트 크기 배열
        cv: 교차 검증 폴드 수
        figsize: 그래프 크기
    """
    from sklearn.model_selection import learning_curve
    
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, train_sizes=train_sizes, cv=cv, n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, 'o-', label='훈련 점수')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    
    plt.plot(train_sizes, val_mean, 'o-', label='검증 점수')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
    
    plt.xlabel('훈련 세트 크기')
    plt.ylabel('점수')
    plt.title('학습 곡선')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def interpret_classification_results(metrics: Dict[str, float], 
                                   class_names: List[str] = None) -> str:
    """
    분류 결과를 해석합니다.
    
    Args:
        metrics: 성능 지표 딕셔너리
        class_names: 클래스 이름 리스트
    
    Returns:
        해석 결과 문자열
    """
    interpretation = "=== 분류 성능 해석 ===\n\n"
    
    # 정확도 해석
    accuracy = metrics.get('accuracy', 0)
    if accuracy >= 0.9:
        interpretation += f"• 정확도 ({accuracy:.3f}): 매우 우수한 성능\n"
    elif accuracy >= 0.8:
        interpretation += f"• 정확도 ({accuracy:.3f}): 우수한 성능\n"
    elif accuracy >= 0.7:
        interpretation += f"• 정확도 ({accuracy:.3f}): 양호한 성능\n"
    else:
        interpretation += f"• 정확도 ({accuracy:.3f}): 개선이 필요한 성능\n"
    
    # F1-score 해석
    f1 = metrics.get('f1_score', 0)
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    
    interpretation += f"• F1-score ({f1:.3f}): 정밀도({precision:.3f})와 재현율({recall:.3f})의 조화평균\n"
    
    if precision > recall + 0.1:
        interpretation += "  → 정밀도가 재현율보다 높음 (False Positive가 적음)\n"
    elif recall > precision + 0.1:
        interpretation += "  → 재현율이 정밀도보다 높음 (False Negative가 적음)\n"
    else:
        interpretation += "  → 정밀도와 재현율이 균형적\n"
    
    # ROC-AUC 해석
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        auc = metrics['roc_auc']
        if auc >= 0.9:
            interpretation += f"• ROC-AUC ({auc:.3f}): 탁월한 분류 성능\n"
        elif auc >= 0.8:
            interpretation += f"• ROC-AUC ({auc:.3f}): 우수한 분류 성능\n"
        elif auc >= 0.7:
            interpretation += f"• ROC-AUC ({auc:.3f}): 양호한 분류 성능\n"
        else:
            interpretation += f"• ROC-AUC ({auc:.3f}): 개선이 필요한 성능\n"
    
    return interpretation


def interpret_regression_results(metrics: Dict[str, float]) -> str:
    """
    회귀 결과를 해석합니다.
    
    Args:
        metrics: 성능 지표 딕셔너리
    
    Returns:
        해석 결과 문자열
    """
    interpretation = "=== 회귀 성능 해석 ===\n\n"
    
    # R² 해석
    r2 = metrics.get('r2_score', 0)
    if r2 >= 0.9:
        interpretation += f"• R² ({r2:.3f}): 매우 우수한 설명력 (분산의 {r2*100:.1f}% 설명)\n"
    elif r2 >= 0.7:
        interpretation += f"• R² ({r2:.3f}): 우수한 설명력 (분산의 {r2*100:.1f}% 설명)\n"
    elif r2 >= 0.5:
        interpretation += f"• R² ({r2:.3f}): 양호한 설명력 (분산의 {r2*100:.1f}% 설명)\n"
    else:
        interpretation += f"• R² ({r2:.3f}): 낮은 설명력 (분산의 {r2*100:.1f}% 설명)\n"
    
    # 오차 지표 해석
    mse = metrics.get('mse', 0)
    rmse = metrics.get('rmse', 0)
    mae = metrics.get('mae', 0)
    
    interpretation += f"• RMSE ({rmse:.3f}): 예측 오차의 표준편차\n"
    interpretation += f"• MAE ({mae:.3f}): 절대 오차의 평균\n"
    
    if rmse > mae * 1.5:
        interpretation += "  → 큰 오차가 있는 이상치 존재 가능성\n"
    else:
        interpretation += "  → 오차가 비교적 균등하게 분포\n"
    
    # MAPE 해석
    if 'mape' in metrics and metrics['mape'] is not None:
        mape = metrics['mape']
        if mape <= 5:
            interpretation += f"• MAPE ({mape:.1f}%): 매우 정확한 예측\n"
        elif mape <= 10:
            interpretation += f"• MAPE ({mape:.1f}%): 정확한 예측\n"
        elif mape <= 20:
            interpretation += f"• MAPE ({mape:.1f}%): 양호한 예측\n"
        else:
            interpretation += f"• MAPE ({mape:.1f}%): 개선이 필요한 예측\n"
    
    return interpretation


def interpret_clustering_results(metrics: Dict[str, float]) -> str:
    """
    클러스터링 결과를 해석합니다.
    
    Args:
        metrics: 성능 지표 딕셔너리
    
    Returns:
        해석 결과 문자열
    """
    interpretation = "=== 클러스터링 성능 해석 ===\n\n"
    
    # 실루엣 점수 해석
    if 'silhouette_score' in metrics:
        silhouette = metrics['silhouette_score']
        if silhouette >= 0.7:
            interpretation += f"• 실루엣 점수 ({silhouette:.3f}): 매우 우수한 클러스터 구조\n"
        elif silhouette >= 0.5:
            interpretation += f"• 실루엣 점수 ({silhouette:.3f}): 우수한 클러스터 구조\n"
        elif silhouette >= 0.25:
            interpretation += f"• 실루엣 점수 ({silhouette:.3f}): 양호한 클러스터 구조\n"
        else:
            interpretation += f"• 실루엣 점수 ({silhouette:.3f}): 약한 클러스터 구조\n"
    
    # Calinski-Harabasz 지수 해석
    if 'calinski_harabasz_score' in metrics:
        ch_score = metrics['calinski_harabasz_score']
        interpretation += f"• Calinski-Harabasz 지수 ({ch_score:.1f}): 클러스터 간 분리도 (높을수록 좋음)\n"
    
    # Davies-Bouldin 지수 해석
    if 'davies_bouldin_score' in metrics:
        db_score = metrics['davies_bouldin_score']
        interpretation += f"• Davies-Bouldin 지수 ({db_score:.3f}): 클러스터 내 응집도 (낮을수록 좋음)\n"
    
    # 외부 평가 지표
    if 'adjusted_rand_score' in metrics:
        ari = metrics['adjusted_rand_score']
        interpretation += f"• 조정된 랜드 지수 ({ari:.3f}): 실제 라벨과의 일치도\n"
    
    if 'cluster_accuracy' in metrics:
        accuracy = metrics['cluster_accuracy']
        interpretation += f"• 클러스터 정확도 ({accuracy:.3f}): 최적 매핑 후 정확도\n"
    
    return interpretation