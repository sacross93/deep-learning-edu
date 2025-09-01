"""
시각화 유틸리티 모듈

알고리즘 비교, 파라미터 분석, 클러스터링 결과, 결정 경계 시각화 기능을 제공합니다.
요구사항 3.1, 3.2를 충족합니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 색상 팔레트 설정
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def plot_algorithm_comparison(results_df: pd.DataFrame, 
                            metrics: List[str] = None,
                            figsize: Tuple[int, int] = (15, 10)):
    """
    여러 알고리즘의 성능을 비교 시각화합니다.
    
    Args:
        results_df: 알고리즘별 성능 결과 데이터프레임
        metrics: 비교할 지표 리스트
        figsize: 그래프 크기
    """
    if metrics is None:
        # 수치형 컬럼에서 'model' 제외
        metrics = [col for col in results_df.columns if col != 'model' and pd.api.types.is_numeric_dtype(results_df[col])]
    
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if i < len(axes):
            # 막대 그래프
            bars = axes[i].bar(results_df['model'], results_df[metric], 
                              color=COLORS[:len(results_df)], alpha=0.7)
            axes[i].set_title(f'{metric} 비교')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
            
            # 값 표시
            for bar, value in zip(bars, results_df[metric]):
                if not pd.isna(value):
                    axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
    
    # 빈 서브플롯 제거
    for i in range(n_metrics, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()


def plot_parameter_sensitivity(param_name: str, param_values: List[Any], 
                             scores: List[float], 
                             figsize: Tuple[int, int] = (10, 6)):
    """
    파라미터 민감도를 시각화합니다.
    
    Args:
        param_name: 파라미터 이름
        param_values: 파라미터 값 리스트
        scores: 각 파라미터 값에 대한 성능 점수
        figsize: 그래프 크기
    """
    plt.figure(figsize=figsize)
    
    # 선 그래프
    plt.plot(param_values, scores, 'o-', linewidth=2, markersize=8, color=COLORS[0])
    
    # 최적값 표시
    best_idx = np.argmax(scores)
    plt.scatter(param_values[best_idx], scores[best_idx], 
               color='red', s=100, zorder=5, label=f'최적값: {param_values[best_idx]}')
    
    plt.xlabel(param_name)
    plt.ylabel('성능 점수')
    plt.title(f'{param_name} 민감도 분석')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # x축 로그 스케일 (파라미터 값이 넓은 범위인 경우)
    if len(param_values) > 5 and max(param_values) / min(param_values) > 100:
        plt.xscale('log')
    
    plt.tight_layout()
    plt.show()


def plot_clustering_results(X: np.ndarray, labels: np.ndarray, 
                          centers: np.ndarray = None,
                          title: str = "클러스터링 결과",
                          figsize: Tuple[int, int] = (10, 8)):
    """
    클러스터링 결과를 시각화합니다.
    
    Args:
        X: 입력 데이터
        labels: 클러스터 라벨
        centers: 클러스터 중심점 (있는 경우)
        title: 그래프 제목
        figsize: 그래프 크기
    """
    # 2D 시각화를 위한 차원 축소
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        if centers is not None:
            centers_2d = pca.transform(centers)
    else:
        X_2d = X
        centers_2d = centers
    
    plt.figure(figsize=figsize)
    
    # 클러스터별 색상
    unique_labels = np.unique(labels)
    colors = COLORS[:len(unique_labels)]
    
    # 각 클러스터 시각화
    for i, label in enumerate(unique_labels):
        if label == -1:  # 노이즈 포인트 (DBSCAN)
            color = 'black'
            marker = 'x'
            alpha = 0.5
        else:
            color = colors[i % len(colors)]
            marker = 'o'
            alpha = 0.7
        
        mask = labels == label
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=color, marker=marker, alpha=alpha, s=50,
                   label=f'클러스터 {label}' if label != -1 else '노이즈')
    
    # 클러스터 중심점 표시
    if centers_2d is not None:
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], 
                   c='red', marker='X', s=200, linewidths=2,
                   label='중심점')
    
    plt.xlabel('첫 번째 주성분' if X.shape[1] > 2 else '특성 1')
    plt.ylabel('두 번째 주성분' if X.shape[1] > 2 else '특성 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_decision_boundary(model, X: np.ndarray, y: np.ndarray,
                         feature_names: List[str] = None,
                         title: str = "결정 경계",
                         figsize: Tuple[int, int] = (10, 8)):
    """
    2D 결정 경계를 시각화합니다.
    
    Args:
        model: 훈련된 모델
        X: 입력 데이터 (2D)
        y: 타겟 라벨
        feature_names: 특성 이름 리스트
        title: 그래프 제목
        figsize: 그래프 크기
    """
    if X.shape[1] != 2:
        print("결정 경계 시각화는 2차원 데이터만 지원합니다.")
        return
    
    plt.figure(figsize=figsize)
    
    # 그리드 생성
    h = 0.02  # 그리드 간격
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # 예측
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # 결정 경계 그리기
    unique_labels = np.unique(y)
    colors = COLORS[:len(unique_labels)]
    cmap = ListedColormap(colors[:len(unique_labels)])
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    
    # 데이터 포인트 그리기
    for i, label in enumerate(unique_labels):
        mask = y == label
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[i], marker='o', s=50, alpha=0.8,
                   label=f'클래스 {label}')
    
    plt.xlabel(feature_names[0] if feature_names else '특성 1')
    plt.ylabel(feature_names[1] if feature_names else '특성 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_data_distribution(df: pd.DataFrame, 
                         target_column: str = None,
                         figsize: Tuple[int, int] = (15, 10)):
    """
    데이터 분포를 시각화합니다.
    
    Args:
        df: 입력 데이터프레임
        target_column: 타겟 컬럼명
        figsize: 그래프 크기
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if target_column and target_column in numeric_columns:
        numeric_columns = numeric_columns.drop(target_column)
    
    n_numeric = len(numeric_columns)
    
    if n_numeric > 0:
        n_cols = min(4, n_numeric)
        n_rows = (n_numeric + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(numeric_columns):
            if i < len(axes):
                if target_column and target_column in df.columns:
                    # 타겟별 분포
                    unique_targets = df[target_column].unique()
                    for j, target in enumerate(unique_targets):
                        subset = df[df[target_column] == target][col]
                        axes[i].hist(subset, bins=20, alpha=0.6, 
                                   label=f'{target_column}={target}',
                                   color=COLORS[j % len(COLORS)])
                    axes[i].legend()
                else:
                    # 전체 분포
                    axes[i].hist(df[col], bins=30, alpha=0.7, 
                               color=COLORS[0], edgecolor='black')
                
                axes[i].set_title(f'{col} 분포')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('빈도')
                axes[i].grid(True, alpha=0.3)
        
        # 빈 서브플롯 제거
        for i in range(n_numeric, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, 
                           figsize: Tuple[int, int] = (12, 10)):
    """
    상관관계 히트맵을 시각화합니다.
    
    Args:
        df: 입력 데이터프레임
        figsize: 그래프 크기
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) > 1:
        correlation_matrix = numeric_df.corr()
        
        plt.figure(figsize=figsize)
        
        # 마스크 생성 (상삼각 행렬 숨기기)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, 
                   cmap='coolwarm', center=0, square=True, 
                   linewidths=0.5, cbar_kws={"shrink": .8})
        
        plt.title('특성 간 상관관계 히트맵')
        plt.tight_layout()
        plt.show()
    else:
        print("상관관계 분석을 위한 충분한 수치형 변수가 없습니다.")


def plot_feature_importance(feature_names: List[str], 
                          importance_scores: np.ndarray,
                          title: str = "특성 중요도",
                          figsize: Tuple[int, int] = (10, 8)):
    """
    특성 중요도를 시각화합니다.
    
    Args:
        feature_names: 특성 이름 리스트
        importance_scores: 중요도 점수
        title: 그래프 제목
        figsize: 그래프 크기
    """
    # 중요도 순으로 정렬
    indices = np.argsort(importance_scores)[::-1]
    
    plt.figure(figsize=figsize)
    
    # 막대 그래프
    bars = plt.bar(range(len(importance_scores)), 
                  importance_scores[indices],
                  color=COLORS[0], alpha=0.7)
    
    plt.xlabel('특성')
    plt.ylabel('중요도')
    plt.title(title)
    plt.xticks(range(len(importance_scores)), 
              [feature_names[i] for i in indices], rotation=45)
    
    # 값 표시
    for i, (bar, score) in enumerate(zip(bars, importance_scores[indices])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pca_analysis(X: np.ndarray, y: np.ndarray = None,
                     feature_names: List[str] = None,
                     figsize: Tuple[int, int] = (15, 5)):
    """
    PCA 분석 결과를 시각화합니다.
    
    Args:
        X: 입력 데이터
        y: 타겟 라벨 (있는 경우)
        feature_names: 특성 이름 리스트
        figsize: 그래프 크기
    """
    # PCA 수행
    pca = PCA()
    X_pca = pca.fit_transform(StandardScaler().fit_transform(X))
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. 설명된 분산 비율
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    axes[0].bar(range(1, len(explained_variance_ratio) + 1), 
               explained_variance_ratio, alpha=0.7, color=COLORS[0])
    axes[0].plot(range(1, len(cumulative_variance) + 1), 
                cumulative_variance, 'ro-', color=COLORS[1])
    axes[0].set_xlabel('주성분')
    axes[0].set_ylabel('설명된 분산 비율')
    axes[0].set_title('주성분별 설명된 분산')
    axes[0].grid(True, alpha=0.3)
    
    # 2. 첫 두 주성분 시각화
    if y is not None:
        unique_labels = np.unique(y)
        for i, label in enumerate(unique_labels):
            mask = y == label
            axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], 
                          c=COLORS[i % len(COLORS)], alpha=0.7, s=50,
                          label=f'클래스 {label}')
        axes[1].legend()
    else:
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1], 
                       c=COLORS[0], alpha=0.7, s=50)
    
    axes[1].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})')
    axes[1].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%})')
    axes[1].set_title('첫 두 주성분 시각화')
    axes[1].grid(True, alpha=0.3)
    
    # 3. 주성분 로딩 (첫 번째 주성분)
    if feature_names is not None and len(feature_names) == X.shape[1]:
        loadings = pca.components_[0]
        indices = np.argsort(np.abs(loadings))[::-1]
        
        axes[2].barh(range(len(loadings)), loadings[indices], 
                    color=COLORS[2], alpha=0.7)
        axes[2].set_yticks(range(len(loadings)))
        axes[2].set_yticklabels([feature_names[i] for i in indices])
        axes[2].set_xlabel('로딩 값')
        axes[2].set_title('PC1 특성 로딩')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, '특성 이름이\n제공되지 않음', 
                    ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('PC1 특성 로딩')
    
    plt.tight_layout()
    plt.show()


def plot_tsne_visualization(X: np.ndarray, y: np.ndarray = None,
                          perplexity: int = 30, n_iter: int = 1000,
                          figsize: Tuple[int, int] = (10, 8)):
    """
    t-SNE 시각화를 수행합니다.
    
    Args:
        X: 입력 데이터
        y: 타겟 라벨 (있는 경우)
        perplexity: t-SNE perplexity 파라미터
        n_iter: 반복 횟수
        figsize: 그래프 크기
    """
    # 데이터 크기가 큰 경우 샘플링
    if X.shape[0] > 1000:
        indices = np.random.choice(X.shape[0], 1000, replace=False)
        X_sample = X[indices]
        y_sample = y[indices] if y is not None else None
    else:
        X_sample = X
        y_sample = y
    
    # t-SNE 수행
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    X_tsne = tsne.fit_transform(StandardScaler().fit_transform(X_sample))
    
    plt.figure(figsize=figsize)
    
    if y_sample is not None:
        unique_labels = np.unique(y_sample)
        for i, label in enumerate(unique_labels):
            mask = y_sample == label
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=COLORS[i % len(COLORS)], alpha=0.7, s=50,
                       label=f'클래스 {label}')
        plt.legend()
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                   c=COLORS[0], alpha=0.7, s=50)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title('t-SNE 시각화')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_learning_progress(train_scores: List[float], 
                         val_scores: List[float] = None,
                         title: str = "학습 진행 과정",
                         figsize: Tuple[int, int] = (10, 6)):
    """
    학습 진행 과정을 시각화합니다.
    
    Args:
        train_scores: 훈련 점수 리스트
        val_scores: 검증 점수 리스트
        title: 그래프 제목
        figsize: 그래프 크기
    """
    plt.figure(figsize=figsize)
    
    epochs = range(1, len(train_scores) + 1)
    
    plt.plot(epochs, train_scores, 'o-', label='훈련 점수', 
            color=COLORS[0], linewidth=2)
    
    if val_scores is not None:
        plt.plot(epochs, val_scores, 'o-', label='검증 점수', 
                color=COLORS[1], linewidth=2)
    
    plt.xlabel('에포크')
    plt.ylabel('점수')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_distance_matrix(X: np.ndarray, labels: np.ndarray = None,
                        distance_metric: str = 'euclidean',
                        figsize: Tuple[int, int] = (10, 8)):
    """
    거리 행렬을 시각화합니다.
    
    Args:
        X: 입력 데이터
        labels: 데이터 라벨 (있는 경우)
        distance_metric: 거리 측정 방법
        figsize: 그래프 크기
    """
    from sklearn.metrics.pairwise import pairwise_distances
    
    # 샘플 수 제한 (시각화 성능을 위해)
    if X.shape[0] > 100:
        indices = np.random.choice(X.shape[0], 100, replace=False)
        X_sample = X[indices]
        labels_sample = labels[indices] if labels is not None else None
    else:
        X_sample = X
        labels_sample = labels
    
    # 거리 행렬 계산
    distance_matrix = pairwise_distances(X_sample, metric=distance_metric)
    
    plt.figure(figsize=figsize)
    
    # 히트맵 그리기
    im = plt.imshow(distance_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='거리')
    
    # 라벨이 있는 경우 색상 구분선 추가
    if labels_sample is not None:
        unique_labels = np.unique(labels_sample)
        boundaries = []
        current_pos = 0
        
        for label in unique_labels:
            count = np.sum(labels_sample == label)
            boundaries.append(current_pos + count)
            current_pos += count
        
        for boundary in boundaries[:-1]:
            plt.axhline(y=boundary-0.5, color='red', linewidth=2)
            plt.axvline(x=boundary-0.5, color='red', linewidth=2)
    
    plt.xlabel('샘플 인덱스')
    plt.ylabel('샘플 인덱스')
    plt.title(f'거리 행렬 ({distance_metric})')
    plt.tight_layout()
    plt.show()


def create_subplot_grid(n_plots: int, max_cols: int = 3) -> Tuple[int, int]:
    """
    서브플롯 그리드 크기를 계산합니다.
    
    Args:
        n_plots: 플롯 개수
        max_cols: 최대 열 수
    
    Returns:
        (행 수, 열 수) 튜플
    """
    n_cols = min(max_cols, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    return n_rows, n_cols