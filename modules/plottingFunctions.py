import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as patches




def plot_dbscan_pipeline(x, dbscanInstance, startPointIndex, targetCluster, 
                        greedySet, graph, singleCfResult, allCounterfactuals=None, figsize=(14, 10),
                        show_all_indexes=False, show_cluster_indexes=None, 
                        show_special_indexes=False, index_fontsize=8):
    """
    Visualize the complete DBSCAN counterfactual pipeline with point indexes
    
    Parameters:
    -----------
    x : np.ndarray
        The dataset (2D points)
    dbscanInstance : PredictDBSCAN
        Fitted DBSCAN instance
    startPointIndex : int
        Index of the point to explain
    targetCluster : int
        Target cluster
    greedySet : set
        Set of core points from greedy permutation
    graph : nx.Graph
        Graph modeling the target cluster
    singleCfResult : dict
        Result from findSingleCounterfactual containing counterfactual info
    figsize : tuple
        Figure size
    show_all_indexes : bool
        Whether to show indexes for all points
    show_cluster_indexes : int or list
        Show indexes only for specific cluster(s). None means don't show cluster indexes
    show_special_indexes : bool
        Whether to show indexes for special points (start, counterfactual, etc.)
    index_fontsize : int
        Font size for index annotations
    """
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    labels = dbscanInstance.labels_
    unique_labels = set(labels)
    
    # Color scheme
    colors = plt.cm.Set3(np.linspace(0.25, 1, len(unique_labels)))
    edge_color = '#2C3E50'
    greedy_color = '#E74C3C'
    start_point_color = '#8E44AD'
    counterfactual_color = '#27AE60'
    arrow_color = '#34495E'
    eps_color = '#BDC3C7'
    
    # 1. Plot all clusters with different colors
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Noise points
            class_member_mask = (labels == k)
            xy = x[class_member_mask]
            point_indices = np.where(class_member_mask)[0]
            ax.scatter(xy[:, 0], xy[:, 1], c='black', marker='x', 
                      s=30, alpha=0.6, label='Noise')
            
            # Add indexes for noise points if requested
            if show_all_indexes:
                for i, (point, idx) in enumerate(zip(xy, point_indices)):
                    ax.annotate(str(idx), (point[0], point[1]), 
                              xytext=(3, 3), textcoords='offset points',
                              fontsize=index_fontsize, alpha=0.7)
        else:
            # Regular clusters
            class_member_mask = (labels == k)
            xy = x[class_member_mask]
            point_indices = np.where(class_member_mask)[0]
            
            if k == targetCluster:
                ax.scatter(xy[:, 0], xy[:, 1], color=col, s=60, alpha=0.7, 
                          edgecolors='black', linewidth=1, 
                          label=f'Target Cluster {k}', zorder=3)
            else:
                ax.scatter(xy[:, 0], xy[:, 1], color=col, s=40, alpha=0.6,
                          label=f'Cluster {k}', zorder=2)
            
            # Add indexes for specific clusters or all points
            show_these_indexes = False
            '''(show_all_indexes or 
                                (show_cluster_indexes is not None and 
                                 (k == show_cluster_indexes or 
                                  (isinstance(show_cluster_indexes, (list, tuple)) and 
                                   k in show_cluster_indexes))))'''
            
            if show_these_indexes:
                for i, (point, idx) in enumerate(zip(xy, point_indices)):
                    ax.annotate(str(idx), (point[0], point[1]), 
                              xytext=(3, 3), textcoords='offset points',
                              fontsize=index_fontsize, alpha=0.8,
                              bbox=dict(boxstyle="round,pad=0.1", 
                                       facecolor='white', alpha=0.7, edgecolor='none'))
    
    # 2. Plot graph edges for the target cluster
    pos = {node: data['pos'] for node, data in graph.nodes(data=True)}
    
    # Draw edges
    
    for edge in graph.edges():
        x1, y1 = pos[edge[0]]
        x2, y2 = pos[edge[1]]
        ax.plot([x1, x2], [y1, y2], color=edge_color, alpha=0.3, 
                linewidth=1, zorder=1)
    
    # 3. Highlight core points selected in greedy permutation
    
    greedy_points = []
    for node in greedySet:
        if node in pos:  # Make sure node is in the graph
            point = pos[node]
            greedy_points.append(point)
            ax.scatter(point[0], point[1], c=greedy_color, s=120, 
                    marker='*', linewidth=2,
                    alpha=0.9, zorder=6)
            
            # Add index for greedy points if requested
            if show_special_indexes:
                ax.annotate(str(node), (point[0], point[1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=index_fontsize+1, fontweight='bold',
                        color='white',
                        bbox=dict(boxstyle="round,pad=0.2", 
                                facecolor=greedy_color, alpha=0.8))
                    
    if allCounterfactuals is not None:

        for point in allCounterfactuals:

            ax.plot(point[0], point[1], '*', color='forestgreen', markersize=10, zorder=5)
    
    # 4. Draw dotted epsilon neighborhoods around greedy points
    for point in greedy_points:
        eps_circle = Circle(point, dbscanInstance.eps, 
                           fill=False, color=eps_color, 
                           linewidth=1.5, linestyle=':', alpha=0.7, zorder=4)
        ax.add_patch(eps_circle)

        
    
    # 5. Plot start point
    start_point = x[startPointIndex]
    ax.scatter(start_point[0], start_point[1], 
              c=start_point_color, s=150, marker='o', 
               linewidth=3, alpha=0.9, edgecolors='white',
              label=f'Start Point (Cluster {labels[startPointIndex]})', zorder=8)
    
    # Add index for start point
    if show_special_indexes:
        ax.annotate(f'Start: {startPointIndex}', 
                   (start_point[0], start_point[1]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=index_fontsize+1, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=start_point_color, alpha=0.9))
    
    # 6. Plot counterfactual
    counterfactual = singleCfResult['counterfactual']
    ax.scatter(counterfactual[0], counterfactual[1], 
              c=counterfactual_color, s=200, marker='*', 
               linewidth=1, alpha=0.9, edgecolors='white',
              label='Counterfactual', zorder=8)
    
    # Add label for counterfactual
    if show_special_indexes:
        ax.annotate('CF', (counterfactual[0], counterfactual[1]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=index_fontsize+2, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=counterfactual_color, alpha=0.9))
    
    # 7. Draw arrow from start point to counterfactual
    arrow = FancyArrowPatch(start_point, counterfactual,
                           arrowstyle='->', mutation_scale=25, 
                           color=arrow_color, linewidth=3, 
                           alpha=0.8, zorder=7)
    ax.add_patch(arrow)
    
    # 8. Highlight closest core point
    closest_core = singleCfResult['closest_core_point']
    
    # Find the index of the closest core point
    closest_core_index = None
    for i, point in enumerate(x):
        if np.allclose(point, closest_core, rtol=1e-10):
            closest_core_index = i
            break
    
    ax.scatter(closest_core[0], closest_core[1], 
              c='orange', s=120, marker='D', 
              edgecolors='black', linewidth=2, alpha=0.9,
              label='Closest Core Point', zorder=7)
    
    # Add index for closest core point
    if show_special_indexes and closest_core_index is not None:
        ax.annotate(f'Core: {closest_core_index}', 
                   (closest_core[0], closest_core[1]), 
                   xytext=(10, -15), textcoords='offset points',
                   fontsize=index_fontsize+1, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor='orange', alpha=0.9))
    
    # Styling
    ax.set_title('DBSCAN Counterfactual Pipeline Visualization', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    
    # Custom legend entries for greedy points and eps circles
    ax.scatter([], [], c=greedy_color, s=120, marker='*', 
              edgecolors='white', linewidth=0.5, 
              label=f'Greedy Core Points (n={len(greedySet)})')
    
    # Add legend entry for eps neighborhoods
    ax.plot([], [], color=eps_color, linestyle=':', linewidth=1.5,
            label=f'ε-neighborhoods (ε={dbscanInstance.eps:.2f})')
    
    # Add legend entry for graph edges
    ax.plot([], [], color=edge_color, alpha=0.3, linewidth=1,
            label='Density connectivity graph')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig, ax



def plot_dbscan_single(x, dbscanInstance, startPointIndex, targetCluster, 
                        singleCfResult, figsize=(14, 10),
                        show_all_indexes=False, show_cluster_indexes=None, 
                        show_special_indexes=False, index_fontsize=8):
    """
    Visualize the complete DBSCAN counterfactual pipeline with point indexes
    
    Parameters:
    -----------
    x : np.ndarray
        The dataset (2D points)
    dbscanInstance : PredictDBSCAN
        Fitted DBSCAN instance
    startPointIndex : int
        Index of the point to explain
    targetCluster : int
        Target cluster
    greedySet : set
        Set of core points from greedy permutation
    graph : nx.Graph
        Graph modeling the target cluster
    singleCfResult : dict
        Result from findSingleCounterfactual containing counterfactual info
    figsize : tuple
        Figure size
    show_all_indexes : bool
        Whether to show indexes for all points
    show_cluster_indexes : int or list
        Show indexes only for specific cluster(s). None means don't show cluster indexes
    show_special_indexes : bool
        Whether to show indexes for special points (start, counterfactual, etc.)
    index_fontsize : int
        Font size for index annotations
    """
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    labels = dbscanInstance.labels_
    unique_labels = set(labels)
    
    # Color scheme
    colors = plt.cm.Set3(np.linspace(0.25, 1, len(unique_labels)))
    edge_color = '#2C3E50'
    start_point_color = '#8E44AD'
    counterfactual_color = '#27AE60'
    arrow_color = '#34495E'
    eps_color = '#BDC3C7'
    
    # 1. Plot all clusters with different colors
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Noise points
            class_member_mask = (labels == k)
            xy = x[class_member_mask]
            point_indices = np.where(class_member_mask)[0]
            ax.scatter(xy[:, 0], xy[:, 1], c='black', marker='x', 
                      s=30, alpha=0.6, label='Noise')
            
            # Add indexes for noise points if requested
            if show_all_indexes:
                for i, (point, idx) in enumerate(zip(xy, point_indices)):
                    ax.annotate(str(idx), (point[0], point[1]), 
                              xytext=(3, 3), textcoords='offset points',
                              fontsize=index_fontsize, alpha=0.7)
        else:
            # Regular clusters
            class_member_mask = (labels == k)
            xy = x[class_member_mask]
            point_indices = np.where(class_member_mask)[0]
            
            if k == targetCluster:
                ax.scatter(xy[:, 0], xy[:, 1], color=col, s=60, alpha=0.7, 
                          edgecolors='black', linewidth=1, 
                          label=f'Target Cluster {k}', zorder=3)
            else:
                ax.scatter(xy[:, 0], xy[:, 1], color=col, s=40, alpha=0.6,
                          label=f'Cluster {k}', zorder=2)
            
            # Add indexes for specific clusters or all points
            show_these_indexes = (show_all_indexes or 
                                (show_cluster_indexes is not None and 
                                 (k == show_cluster_indexes or 
                                  (isinstance(show_cluster_indexes, (list, tuple)) and 
                                   k in show_cluster_indexes))))
            
            if show_these_indexes:
                for i, (point, idx) in enumerate(zip(xy, point_indices)):
                    ax.annotate(str(idx), (point[0], point[1]), 
                              xytext=(3, 3), textcoords='offset points',
                              fontsize=index_fontsize, alpha=0.8,
                              bbox=dict(boxstyle="round,pad=0.1", 
                                       facecolor='white', alpha=0.7, edgecolor='none'))
    
    
    
    
    
    
    # 4. Draw dotted epsilon neighborhoods around greedy points
    
    eps_circle = Circle(singleCfResult['closest_core_point'], dbscanInstance.eps, 
                        fill=False, color=eps_color, 
                        linewidth=1.5, linestyle=':', alpha=0.7, zorder=4)
    ax.add_patch(eps_circle)
    
    # 5. Plot start point
    start_point = x[startPointIndex]
    ax.scatter(start_point[0], start_point[1], 
              c=start_point_color, s=150, marker='o', 
               linewidth=3, alpha=0.9, edgecolors='white',
              label=f'Start Point (Cluster {labels[startPointIndex]})', zorder=8)
    
    # Add index for start point
    if show_special_indexes:
        ax.annotate(f'Start: {startPointIndex}', 
                   (start_point[0], start_point[1]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=index_fontsize+1, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=start_point_color, alpha=0.9))
    
    # 6. Plot counterfactual
    counterfactual = singleCfResult['counterfactual']
    ax.scatter(counterfactual[0], counterfactual[1], 
              c=counterfactual_color, s=200, marker='*', 
               linewidth=1, alpha=0.9, edgecolors='white',
              label='Counterfactual', zorder=8)
    
    # Add label for counterfactual
    if show_special_indexes:
        ax.annotate('CF', (counterfactual[0], counterfactual[1]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=index_fontsize+2, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor=counterfactual_color, alpha=0.9))
    
    # 7. Draw arrow from start point to counterfactual
    arrow = FancyArrowPatch(start_point, counterfactual,
                           arrowstyle='->', mutation_scale=25, 
                           color=arrow_color, linewidth=3, 
                           alpha=0.8, zorder=7)
    ax.add_patch(arrow)
    
    # 8. Highlight closest core point
    closest_core = singleCfResult['closest_core_point']
    
    # Find the index of the closest core point
    closest_core_index = None
    for i, point in enumerate(x):
        if np.allclose(point, closest_core, rtol=1e-10):
            closest_core_index = i
            break
    
    ax.scatter(closest_core[0], closest_core[1], 
              c='orange', s=120, marker='D', 
              edgecolors='black', linewidth=2, alpha=0.9,
              label='Closest Core Point', zorder=7)
    
    # Add index for closest core point
    if show_special_indexes and closest_core_index is not None:
        ax.annotate(f'Core: {closest_core_index}', 
                   (closest_core[0], closest_core[1]), 
                   xytext=(10, -15), textcoords='offset points',
                   fontsize=index_fontsize+1, fontweight='bold',
                   color='white',
                   bbox=dict(boxstyle="round,pad=0.3", 
                            facecolor='orange', alpha=0.9))
    
    # Styling
    ax.set_title('DBSCAN Counterfactual Pipeline Visualization', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
 
    
    # Add legend entry for eps neighborhoods
    ax.plot([], [], color=eps_color, linestyle=':', linewidth=1.5,
            label=f'ε-neighborhoods (ε={dbscanInstance.eps:.2f})')
    
    # Add legend entry for graph edges
    ax.plot([], [], color=edge_color, alpha=0.3, linewidth=1,
            label='Density connectivity graph')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
             frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return fig, ax