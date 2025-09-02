"""
그래프 기반 클러스터링 실습

이 실습에서는 네트워크 데이터에서 커뮤니티를 탐지하는 다양한 알고리즘을 학습하고,
소셜 네트워크 분석의 기본 개념을 익힙니다.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_sample_networks():
    """다양한 형태의 샘플 네트워크 생성"""
    
    networks = {}
    
    # 1. 카라테 클럽 네트워크 (유명한 실제 데이터)
    G_karate = nx.karate_club_graph()
    # 실제 분할 정보 추가
    true_communities_karate = {}
    for node in G_karate.nodes():
        if G_karate.nodes[node]['club'] == 'Mr. Hi':
            true_communities_karate[node] = 0
        else:
            true_communities_karate[node] = 1
    
    networks['karate'] = (G_karate, true_communities_karate, "Karate Club Network")
    
    # 2. 스톡스틱 블록 모델 (명확한 커뮤니티 구조)
    sizes = [20, 15, 25]  # 각 커뮤니티 크기
    probs = [[0.8, 0.1, 0.05],   # 커뮤니티 내/간 연결 확률
             [0.1, 0.7, 0.1],
             [0.05, 0.1, 0.6]]
    
    G_sbm = nx.stochastic_block_model(sizes, probs, seed=42)
    true_communities_sbm = {}
    start = 0
    for i, size in enumerate(sizes):
        for j in range(start, start + size):
            true_communities_sbm[j] = i
        start += size
    
    networks['sbm'] = (G_sbm, true_communities_sbm, "Stochastic Block Model")
    
    # 3. 바라바시-알버트 네트워크 (스케일-프리 네트워크)
    G_ba = nx.barabasi_albert_graph(100, 3, seed=42)
    # 차수 기반으로 임의의 커뮤니티 생성 (시연용)
    degrees = dict(G_ba.degree())
    degree_threshold = np.median(list(degrees.values()))
    true_communities_ba = {node: 0 if degrees[node] <= degree_threshold else 1 
                          for node in G_ba.nodes()}
    
    networks['barabasi'] = (G_ba, true_communities_ba, "Barabasi-Albert Network")
    
    # 4. 작은 세계 네트워크 (와츠-스트로가츠 모델)
    G_ws = nx.watts_strogatz_graph(60, 6, 0.3, seed=42)
    # 원형 배치 기반으로 커뮤니티 생성
    true_communities_ws = {node: node // 20 for node in G_ws.nodes()}
    
    networks['watts_strogatz'] = (G_ws, true_communities_ws, "Watts-Strogatz Network")
    
    return networks

def visualize_networks(networks):
    """네트워크 시각화"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink']
    
    for i, (key, (G, true_communities, title)) in enumerate(networks.items()):
        ax = axes[i]
        
        # 레이아웃 선택
        if key == 'karate':
            pos = nx.spring_layout(G, seed=42)
        elif key == 'sbm':
            pos = nx.spring_layout(G, seed=42)
        elif key == 'barabasi':
            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        else:  # watts_strogatz
            pos = nx.circular_layout(G)
        
        # 커뮤니티별로 다른 색상으로 노드 그리기
        for community_id in set(true_communities.values()):
            nodes_in_community = [node for node, comm in true_communities.items() 
                                if comm == community_id]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_community,
                                 node_color=colors[community_id % len(colors)],
                                 node_size=100, alpha=0.8, ax=ax)
        
        # 엣지 그리기
        nx.draw_networkx_edges(G, pos, alpha=0.5, width=0.5, ax=ax)
        
        ax.set_title(f'{title}\\nNodes: {G.number_of_nodes()}, '
                    f'Edges: {G.number_of_edges()}', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_modularity(G, communities):
    """모듈성 계산"""
    
    if isinstance(communities, dict):
        # 딕셔너리를 리스트로 변환
        partition = defaultdict(list)
        for node, comm in communities.items():
            partition[comm].append(node)
        communities = list(partition.values())
    
    m = G.number_of_edges()
    if m == 0:
        return 0
    
    Q = 0
    for community in communities:
        # 커뮤니티 내부 엣지 수
        subgraph = G.subgraph(community)
        internal_edges = subgraph.number_of_edges()
        
        # 커뮤니티 노드들의 총 차수
        total_degree = sum(G.degree(node) for node in community)
        
        # 모듈성 기여도 계산
        Q += internal_edges / m - (total_degree / (2 * m)) ** 2
    
    return Q

def girvan_newman_step(G):
    """Girvan-Newman 알고리즘의 한 단계 실행"""
    
    # 모든 엣지의 중심성 계산
    edge_betweenness = nx.edge_betweenness_centrality(G)
    
    # 가장 높은 중심성을 가진 엣지 찾기
    max_centrality = max(edge_betweenness.values())
    edges_to_remove = [edge for edge, centrality in edge_betweenness.items() 
                      if centrality == max_centrality]
    
    # 엣지 제거
    G_copy = G.copy()
    G_copy.remove_edges_from(edges_to_remove)
    
    return G_copy, edges_to_remove

def girvan_newman_clustering(G, max_communities=10):
    """Girvan-Newman 알고리즘으로 커뮤니티 탐지"""
    
    G_work = G.copy()
    results = []
    
    # 초기 상태 (전체가 하나의 커뮤니티)
    communities = [list(G.nodes())]
    modularity = calculate_modularity(G, communities)
    results.append((1, communities.copy(), modularity))
    
    for step in range(1, max_communities):
        if G_work.number_of_edges() == 0:
            break
        
        # 한 단계 실행
        G_work, removed_edges = girvan_newman_step(G_work)
        
        # 연결 성분 찾기
        communities = [list(component) for component in nx.connected_components(G_work)]
        
        # 모듈성 계산
        modularity = calculate_modularity(G, communities)
        results.append((len(communities), communities.copy(), modularity))
        
        print(f"Step {step}: Removed edges {removed_edges}, "
              f"Communities: {len(communities)}, Modularity: {modularity:.3f}")
    
    return results

def louvain_clustering(G):
    """Louvain 알고리즘으로 커뮤니티 탐지 (NetworkX 구현 사용)"""
    
    try:
        # NetworkX 2.5+ 에서 사용 가능
        communities = nx.community.louvain_communities(G, seed=42)
        community_dict = {}
        for i, community in enumerate(communities):
            for node in community:
                community_dict[node] = i
        
        modularity = calculate_modularity(G, communities)
        return community_dict, modularity
    
    except AttributeError:
        # 구버전 NetworkX를 위한 대안
        print("Louvain algorithm not available in this NetworkX version")
        return None, 0

def label_propagation_clustering(G):
    """Label Propagation 알고리즘으로 커뮤니티 탐지"""
    
    communities = nx.community.label_propagation_communities(G)
    community_dict = {}
    for i, community in enumerate(communities):
        for node in community:
            community_dict[node] = i
    
    modularity = calculate_modularity(G, communities)
    return community_dict, modularity

def spectral_clustering_graph(G, n_clusters=2):
    """스펙트럴 클러스터링으로 커뮤니티 탐지"""
    
    # 인접 행렬 생성
    adj_matrix = nx.adjacency_matrix(G).toarray()
    
    # 스펙트럴 클러스터링 적용
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', 
                                 random_state=42)
    labels = spectral.fit_predict(adj_matrix)
    
    # 결과를 딕셔너리로 변환
    community_dict = {node: labels[i] for i, node in enumerate(G.nodes())}
    
    # 모듈성 계산
    communities = defaultdict(list)
    for node, comm in community_dict.items():
        communities[comm].append(node)
    modularity = calculate_modularity(G, list(communities.values()))
    
    return community_dict, modularity

def compare_clustering_algorithms(G, true_communities, title):
    """다양한 클러스터링 알고리즘 비교"""
    
    print(f"\\n{title} 분석")
    print("="*50)
    
    algorithms = {}
    
    # 1. Girvan-Newman (최적 모듈성 선택)
    gn_results = girvan_newman_clustering(G, max_communities=8)
    best_gn = max(gn_results, key=lambda x: x[2])  # 최고 모듈성
    gn_communities = {}
    for i, community in enumerate(best_gn[1]):
        for node in community:
            gn_communities[node] = i
    algorithms['Girvan-Newman'] = (gn_communities, best_gn[2])
    
    # 2. Louvain
    louvain_communities, louvain_modularity = louvain_clustering(G)
    if louvain_communities:
        algorithms['Louvain'] = (louvain_communities, louvain_modularity)
    
    # 3. Label Propagation
    lp_communities, lp_modularity = label_propagation_clustering(G)
    algorithms['Label Propagation'] = (lp_communities, lp_modularity)
    
    # 4. Spectral Clustering
    n_true_communities = len(set(true_communities.values()))
    spec_communities, spec_modularity = spectral_clustering_graph(G, n_true_communities)
    algorithms['Spectral'] = (spec_communities, spec_modularity)
    
    # 결과 비교
    results_df = []
    
    for alg_name, (communities, modularity) in algorithms.items():
        # 성능 지표 계산
        true_labels = [true_communities[node] for node in G.nodes()]
        pred_labels = [communities[node] for node in G.nodes()]
        
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        n_communities = len(set(communities.values()))
        
        results_df.append({
            'Algorithm': alg_name,
            'Communities': n_communities,
            'Modularity': modularity,
            'ARI': ari,
            'NMI': nmi
        })
        
        print(f"{alg_name}:")
        print(f"  Communities: {n_communities}")
        print(f"  Modularity: {modularity:.3f}")
        print(f"  ARI: {ari:.3f}")
        print(f"  NMI: {nmi:.3f}")
    
    # 시각화
    visualize_clustering_results(G, algorithms, true_communities, title)
    
    return pd.DataFrame(results_df)

def visualize_clustering_results(G, algorithms, true_communities, title):
    """클러스터링 결과 시각화"""
    
    n_algorithms = len(algorithms) + 1  # +1 for true communities
    cols = min(3, n_algorithms)
    rows = (n_algorithms + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    if n_algorithms == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
    
    # 레이아웃 설정
    pos = nx.spring_layout(G, seed=42)
    
    # 실제 커뮤니티 시각화
    ax = axes[0, 0] if rows > 1 else axes[0]
    for community_id in set(true_communities.values()):
        nodes_in_community = [node for node, comm in true_communities.items() 
                            if comm == community_id]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_community,
                             node_color=colors[community_id % len(colors)],
                             node_size=100, alpha=0.8, ax=ax)
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)
    ax.set_title('True Communities', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # 알고리즘 결과 시각화
    for i, (alg_name, (communities, modularity)) in enumerate(algorithms.items(), 1):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        for community_id in set(communities.values()):
            nodes_in_community = [node for node, comm in communities.items() 
                                if comm == community_id]
            nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_community,
                                 node_color=colors[community_id % len(colors)],
                                 node_size=100, alpha=0.8, ax=ax)
        
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, ax=ax)
        ax.set_title(f'{alg_name}\\nModularity: {modularity:.3f}', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # 빈 subplot 숨기기
    for i in range(n_algorithms, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.suptitle(f'{title} - Community Detection Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def analyze_network_properties(G, title):
    """네트워크 기본 속성 분석"""
    
    print(f"\\n{title} 네트워크 분석")
    print("="*50)
    
    # 기본 통계
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    
    print(f"노드 수: {n_nodes}")
    print(f"엣지 수: {n_edges}")
    print(f"밀도: {density:.4f}")
    
    # 연결성
    is_connected = nx.is_connected(G)
    n_components = nx.number_connected_components(G)
    
    print(f"연결됨: {is_connected}")
    print(f"연결 성분 수: {n_components}")
    
    if is_connected:
        diameter = nx.diameter(G)
        avg_path_length = nx.average_shortest_path_length(G)
        print(f"지름: {diameter}")
        print(f"평균 최단 경로 길이: {avg_path_length:.3f}")
    
    # 차수 분포
    degrees = [G.degree(node) for node in G.nodes()]
    avg_degree = np.mean(degrees)
    max_degree = max(degrees)
    
    print(f"평균 차수: {avg_degree:.2f}")
    print(f"최대 차수: {max_degree}")
    
    # 클러스터링 계수
    avg_clustering = nx.average_clustering(G)
    print(f"평균 클러스터링 계수: {avg_clustering:.3f}")
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 차수 분포
    axes[0].hist(degrees, bins=20, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Degree')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Degree Distribution')
    axes[0].grid(True, alpha=0.3)
    
    # 클러스터링 계수 분포
    clustering_coeffs = [nx.clustering(G, node) for node in G.nodes()]
    axes[1].hist(clustering_coeffs, bins=20, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Clustering Coefficient')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Clustering Coefficient Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} - Network Properties', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def create_social_network_example():
    """실제적인 소셜 네트워크 예시 생성"""
    
    print("소셜 네트워크 시뮬레이션")
    print("="*50)
    
    # 가상의 소셜 네트워크 생성
    np.random.seed(42)
    
    # 사용자 그룹 정의
    groups = {
        'Students': 25,
        'Professors': 10,
        'Staff': 15,
        'Alumni': 20
    }
    
    G = nx.Graph()
    node_id = 0
    group_assignment = {}
    
    # 각 그룹 내 노드 생성 및 연결
    group_nodes = {}
    for group_name, size in groups.items():
        group_nodes[group_name] = list(range(node_id, node_id + size))
        
        # 그룹 내 연결 (높은 확률)
        for i in range(node_id, node_id + size):
            group_assignment[i] = group_name
            for j in range(i + 1, node_id + size):
                if np.random.random() < 0.3:  # 그룹 내 연결 확률
                    G.add_edge(i, j)
        
        node_id += size
    
    # 그룹 간 연결 (낮은 확률)
    inter_group_probs = {
        ('Students', 'Professors'): 0.05,
        ('Students', 'Staff'): 0.02,
        ('Students', 'Alumni'): 0.03,
        ('Professors', 'Staff'): 0.08,
        ('Professors', 'Alumni'): 0.06,
        ('Staff', 'Alumni'): 0.04
    }
    
    for (group1, group2), prob in inter_group_probs.items():
        for node1 in group_nodes[group1]:
            for node2 in group_nodes[group2]:
                if np.random.random() < prob:
                    G.add_edge(node1, node2)
    
    # 실제 커뮤니티 레이블 생성
    true_communities = {}
    group_to_id = {name: i for i, name in enumerate(groups.keys())}
    for node, group in group_assignment.items():
        true_communities[node] = group_to_id[group]
    
    print(f"생성된 네트워크:")
    print(f"- 총 노드 수: {G.number_of_nodes()}")
    print(f"- 총 엣지 수: {G.number_of_edges()}")
    print(f"- 그룹 수: {len(groups)}")
    
    for group_name, size in groups.items():
        print(f"  {group_name}: {size}명")
    
    # 네트워크 분석
    analyze_network_properties(G, "Social Network")
    
    # 커뮤니티 탐지 알고리즘 비교
    results_df = compare_clustering_algorithms(G, true_communities, "Social Network")
    
    return G, true_communities, results_df

def modularity_analysis():
    """모듈성 변화 분석"""
    
    print("\\n모듈성 분석")
    print("="*50)
    
    # 카라테 클럽 네트워크로 Girvan-Newman 과정 시각화
    G = nx.karate_club_graph()
    
    # Girvan-Newman 실행
    gn_results = girvan_newman_clustering(G, max_communities=10)
    
    # 모듈성 변화 그래프
    n_communities = [result[0] for result in gn_results]
    modularities = [result[2] for result in gn_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_communities, modularities, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Communities')
    plt.ylabel('Modularity')
    plt.title('Modularity vs Number of Communities (Girvan-Newman)')
    plt.grid(True, alpha=0.3)
    
    # 최적 지점 표시
    max_idx = np.argmax(modularities)
    plt.axvline(x=n_communities[max_idx], color='red', linestyle='--', 
               label=f'Optimal: {n_communities[max_idx]} communities')
    plt.axhline(y=modularities[max_idx], color='red', linestyle='--', alpha=0.5)
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print(f"최적 커뮤니티 수: {n_communities[max_idx]}")
    print(f"최대 모듈성: {modularities[max_idx]:.3f}")

def main():
    """메인 실습 함수"""
    
    print("그래프 기반 클러스터링 실습")
    print("="*50)
    
    # 1. 샘플 네트워크 생성 및 시각화
    print("\\n1. 다양한 네트워크 생성")
    networks = create_sample_networks()
    visualize_networks(networks)
    
    # 2. 각 네트워크에 대해 분석 수행
    print("\\n2. 네트워크별 커뮤니티 탐지")
    
    all_results = []
    
    for name, (G, true_communities, title) in networks.items():
        print(f"\\n처리 중: {title}")
        
        # 네트워크 속성 분석
        analyze_network_properties(G, title)
        
        # 커뮤니티 탐지 알고리즘 비교
        results_df = compare_clustering_algorithms(G, true_communities, title)
        results_df['Network'] = title
        all_results.append(results_df)
    
    # 3. 모듈성 분석
    modularity_analysis()
    
    # 4. 소셜 네트워크 예시
    print("\\n4. 소셜 네트워크 시뮬레이션")
    social_G, social_communities, social_results = create_social_network_example()
    all_results.append(social_results)
    
    # 5. 전체 결과 요약
    print("\\n" + "="*50)
    print("전체 결과 요약")
    print("="*50)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # 알고리즘별 평균 성능
    avg_performance = combined_results.groupby('Algorithm')[['Modularity', 'ARI', 'NMI']].mean()
    print("\\n알고리즘별 평균 성능:")
    print(avg_performance.round(3))
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['Modularity', 'ARI', 'NMI']
    for i, metric in enumerate(metrics):
        combined_results.boxplot(column=metric, by='Algorithm', ax=axes[i])
        axes[i].set_title(f'{metric} by Algorithm')
        axes[i].set_xlabel('Algorithm')
        axes[i].set_ylabel(metric)
    
    plt.suptitle('Algorithm Performance Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # 6. 주요 학습 포인트 정리
    print("\\n" + "="*50)
    print("주요 학습 포인트")
    print("="*50)
    print("1. 그래프 클러스터링은 네트워크 구조를 활용한 커뮤니티 탐지입니다.")
    print("2. 모듈성은 커뮤니티 품질을 측정하는 중요한 지표입니다.")
    print("3. Girvan-Newman은 분할적, Louvain은 응집적 접근법입니다.")
    print("4. Label Propagation은 빠르지만 불안정할 수 있습니다.")
    print("5. Spectral Clustering은 라플라시안 행렬의 고유벡터를 활용합니다.")
    print("6. 네트워크 유형에 따라 적합한 알고리즘이 다릅니다.")
    print("7. 실제 적용 시 도메인 지식과 결합하는 것이 중요합니다.")
    
    print("\\n실습을 완료했습니다!")

if __name__ == "__main__":
    main()