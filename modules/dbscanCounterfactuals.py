import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, Circle
from sklearn.datasets import make_blobs
from typing import Dict, Iterable
import networkx as nx
from modules.utilityFunctions import physicalOptimizationActionable, createGraphActionable, moveToEpsilonDistanceVectorised, createGraph, greedyPermutation, physicalOptimization, moveToEpsilonDistance
from typing import List, Optional



class DbscanCounterfactuals:

    def __init__(self, data:np.ndarray, dbscanInstance:DBSCAN, neighbourhoods, neighbourhoodsDist, featureNames:Optional[Iterable[str]] = None):

        self.dbscanInstace:DBSCAN = dbscanInstance

        self.neighbourhoods = neighbourhoods

        self.neighbourhoodsDist = neighbourhoodsDist

        self.data = data

        self.indecesMapping = self.__mappingCorePointsData()

        #print('INDECE MAPPING', self.indecesMapping)

        #self.coreSampleIndices =  self.dbscanInstace.core_sample_indices_


    def findSingleCounterfactual(self, startPointIndex:int, targetCluster:int):

        assert self.__isPointNotAlreadyInTarget(
            startPointIndex=startPointIndex, targetCluster=targetCluster
            ), 'The point to explain is already in the target cluster'
        
        pointToExplain = self.data[startPointIndex]

        #targetClusterMask = self.dbscanInstace.labels_ == targetCluster

        #dataTargetCluster:np.ndarray = self.data[targetClusterMask]
        
        corePointsIndecesTarget:np.ndarray = self.indecesMapping[targetCluster]

        corePointsTarget:np.ndarray = self.data[corePointsIndecesTarget]

        distances = np.linalg.norm(corePointsTarget - pointToExplain, axis = 1)

        closestIndexInCorePoints = np.argmin(distances)

        closestCorePoint = corePointsTarget[closestIndexInCorePoints]

        closestCorePointIndex = corePointsIndecesTarget[closestIndexInCorePoints]


        counterfactual = self.__counterfactualGivenTargetPoint(
            pointToExplain=pointToExplain,
            targetCorePoint=closestCorePoint
        )

        return {
            'counterfactual': counterfactual,
            'closest_core_point': closestCorePoint,
            'closest_core_point_index': closestCorePointIndex,
            'original_point': pointToExplain,
            'core_points_target': corePointsTarget
        }
    

    def findSingleCounterfactualActionable(self, startPointIndex:int, targetCluster:int, nonActionableFeatures:List[int]):

        assert self.__isPointNotAlreadyInTarget(
            startPointIndex=startPointIndex, targetCluster=targetCluster
            ), 'The point to explain is already in the target cluster'
        
        actionableFeatures = np.arange(self.data.shape[1])[~np.isin(np.arange(self.data.shape[1]), nonActionableFeatures)]

        pointToExplain = self.data[startPointIndex]

        pointToExplainFiltered = np.array([pointToExplain[nonActionableFeatures]])
        pointToExplainActionable = np.array([pointToExplain[actionableFeatures]])

        filteredData:np.ndarray = self.data[:,nonActionableFeatures]
        
        corePointsIndecesTarget:np.ndarray = self.indecesMapping[targetCluster]
        corePointsTarget:np.ndarray = self.data[corePointsIndecesTarget]

        
        corePointsTargetFiltered:np.ndarray = corePointsTarget[:, nonActionableFeatures]
        corePointsTargetActionable:np.ndarray = corePointsTarget[:, actionableFeatures]

        # Get epsilon parameter consistently
        eps = self.dbscanInstace.get_params()['eps']

        distancesFiltered = np.linalg.norm(corePointsTargetFiltered - pointToExplainFiltered, axis = 1)
        distancesFilteredData = np.linalg.norm(filteredData - pointToExplainFiltered, axis = 1)

        indecesToSurvive = distancesFiltered <= eps
        indecesToSurviveData = distancesFilteredData <= eps

        # Filter the actionable points using the surviving indices
        survivingActionablePoints = corePointsTargetActionable[indecesToSurvive]
        if len(survivingActionablePoints) == 0:

            return  indecesToSurviveData, {
            'counterfactual': None,
            'closest_core_point': None,
            'closest_core_point_index': None,
            'original_point': pointToExplain,
            'core_points_target': corePointsTarget
        }

        distancesActionable = np.linalg.norm(survivingActionablePoints - pointToExplainActionable, axis = 1)

        closestIndexInSurviving = np.argmin(distancesActionable)
        
        survivingIndices = np.where(indecesToSurvive)[0]
        closestIndexInCorePoints = survivingIndices[closestIndexInSurviving]
        
        closestCorePoint = corePointsTarget[closestIndexInCorePoints]
        closestCorePointIndex = corePointsIndecesTarget[closestIndexInCorePoints]

        closestCorePointDisplacement = closestCorePoint.copy()

        displacement = pointToExplain - closestCorePointDisplacement

        displacement[nonActionableFeatures] = 0

        displacement_norm = np.linalg.norm(displacement)
        
        if displacement_norm > 0:
            unitVectorDisplacement = displacement / displacement_norm
        else:
            unitVectorDisplacement = np.zeros_like(displacement)

        

        counterfactual = moveToEpsilonDistance(
            start_point=pointToExplain,
            direction=unitVectorDisplacement, target_point=closestCorePoint, epsilon=eps
        )


        return indecesToSurviveData, {
            'counterfactual': counterfactual,
            'closest_core_point': closestCorePoint,
            'closest_core_point_index': closestCorePointIndex,
            'original_point': pointToExplain,
            'core_points_target': corePointsTarget
        }
    

    def findSingleCounterfactualActionableAlternative(self, startPointIndex:int, targetCluster:int, nonActionableFeatures:List[int]):

        assert self.__isPointNotAlreadyInTarget(
            startPointIndex=startPointIndex, targetCluster=targetCluster
            ), 'The point to explain is already in the target cluster'
        
        actionableFeatures = np.arange(self.data.shape[1])[~np.isin(np.arange(self.data.shape[1]), nonActionableFeatures)]

        pointToExplain = self.data[startPointIndex]

        pointToExplainFiltered = np.array([pointToExplain[nonActionableFeatures]])
        pointToExplainActionable = np.array([pointToExplain[actionableFeatures]])

        filteredData:np.ndarray = self.data[:,nonActionableFeatures]
        
        corePointsIndecesTarget:np.ndarray = self.indecesMapping[targetCluster] # Cahnge this if we do not wanna limit to core points
        corePointsTarget:np.ndarray = self.data[corePointsIndecesTarget]

        
        corePointsTargetFiltered:np.ndarray = corePointsTarget[:, nonActionableFeatures]
        corePointsTargetActionable:np.ndarray = corePointsTarget[:, actionableFeatures]
    
        # Get epsilon parameter consistently
        eps = self.dbscanInstace.get_params()['eps']

        

        distancesFiltered = np.linalg.norm(corePointsTargetFiltered - pointToExplainFiltered, axis = 1)
        distancesFilteredData = np.linalg.norm(filteredData - pointToExplainFiltered, axis = 1)

        indecesToSurvive = distancesFiltered <= eps
        indecesToSurviveData = distancesFilteredData <= eps

        # Filter the actionable points using the surviving indices
        survivingActionablePoints = corePointsTargetActionable[indecesToSurvive]

        if len(survivingActionablePoints) == 0:

            return  indecesToSurviveData, {
            'counterfactual': None,
            'closest_core_point': None,
            'closest_core_point_index': None,
            'original_point': pointToExplain,
            'core_points_target': corePointsTarget,
            'unreachable':True
            }

        survivingCorePointsTarget = corePointsTarget[indecesToSurvive]

        displacement =  survivingCorePointsTarget - pointToExplain

        displacement[:,nonActionableFeatures] = 0

        movedCorePoints = moveToEpsilonDistanceVectorised(
            start_point=pointToExplain,
            displacement=displacement,
            epsilon=eps
        )

        distancesMoved = np.linalg.norm(movedCorePoints - pointToExplain, axis = 1)

        closestIndexInSurviving = np.argmin(distancesMoved)

        counterfactual = movedCorePoints[closestIndexInSurviving]

        survivingIndices = np.where(indecesToSurvive)[0]

        closestIndexInCorePoints = survivingIndices[closestIndexInSurviving]

        closestCorePoint = corePointsTarget[closestIndexInCorePoints]
        closestCorePointIndex = corePointsIndecesTarget[closestIndexInCorePoints]

        #closestIndexInSurviving = np.argmin(distancesActionable)
        
        #survivingIndices = np.where(indecesToSurvive)[0]
        #closestIndexInCorePoints = survivingIndices[closestIndexInSurviving]
        
        #closestCorePoint = corePointsTarget[closestIndexInCorePoints]
        #closestCorePointIndex = corePointsIndecesTarget[closestIndexInCorePoints]

        #closestCorePointDisplacement = closestCorePoint.copy()

        

        


        return indecesToSurviveData, {
            'counterfactual': counterfactual,
            'closest_core_point': closestCorePoint,
            'closest_core_point_index': closestCorePointIndex,
            'original_point': pointToExplain,
            'core_points_target': corePointsTarget,
            'unreachable':False
        }

        
    

    def findMultipleCounterfactuals(self, startPointIndex:int, 
                                    targetCluster:int|np.int64|np.int32, nCounterfactuals:int, normalization:bool=False):

        allCounterfactuals = []

        singleCfMapping = self.findSingleCounterfactual(
            startPointIndex=startPointIndex, targetCluster=targetCluster
        )

        #allCounterfactuals.append(singleCfMapping['counterfactual'])

        startVertexIndex:int = singleCfMapping['closest_core_point_index']

        graph, characteristicDistance = createGraph(
        neighborhoods=self.neighbourhoods, neighborhoodsDist=self.neighbourhoodsDist,
        clusterLabels=self.dbscanInstace.labels_, targetCluster=targetCluster, X=self.data
        )

        graph:nx.Graph   
        
        '''greedySet = set(greedyPermutation(
            graph, startVertex=startVertexIndex, nPoints=nCounterfactuals, alpha=0.5, normalize=True
            ))'''
        greedySet = set(physicalOptimization(
            graph, pointToExplain=self.data[startPointIndex,:], startVertex=startVertexIndex, nPoints=nCounterfactuals, 
            alpha=1, carachteristicDistance=characteristicDistance, singleCounterfactual=singleCfMapping['counterfactual'], 
            normalization=normalization
            ))
        
        for node in greedySet:

            counterfactual = self.__counterfactualGivenTargetPoint(
                pointToExplain=self.data[startPointIndex],       
                targetCorePoint=graph.nodes[node]['pos']
            )
        
            allCounterfactuals.append(counterfactual)

        return allCounterfactuals, greedySet, graph, singleCfMapping
    

    def findMultipleCounterfactualsActionable(self, startPointIndex: int, 
                                        targetCluster: int|np.int64|np.int32, 
                                        nCounterfactuals: int, 
                                        nonActionableFeatures: List[int],
                                        normalization: bool = False):
    
        allCounterfactuals = []
        
        # Use the actionable version for single counterfactual
        indecesToSurviveData, singleCfMapping = self.findSingleCounterfactualActionable(
            startPointIndex=startPointIndex, 
            targetCluster=targetCluster,
            nonActionableFeatures=nonActionableFeatures
        )
        
        # If no valid counterfactual found, return empty results
        if singleCfMapping['counterfactual'] is None:
            return [], set(), None, singleCfMapping
        
        startVertexIndex: int = singleCfMapping['closest_core_point_index']
        
        # Create actionable-aware graph
        graph, characteristicDistance = createGraphActionable(
            neighborhoods=self.neighbourhoods, 
            neighborhoodsDist=self.neighbourhoodsDist,
            clusterLabels=self.dbscanInstace.labels_, 
            targetCluster=targetCluster, 
            X=self.data,
            nonActionableFeatures=nonActionableFeatures,
            pointToExplain=self.data[startPointIndex],
            eps=self.dbscanInstace.get_params()['eps']
        )
        
        graph: nx.Graph   
        
        # Use physical optimization with actionable constraints
        greedySet = set(physicalOptimizationActionable(
            graph, 
            pointToExplain=self.data[startPointIndex,:], 
            startVertex=startVertexIndex, 
            nPoints=nCounterfactuals, 
            alpha=1, 
            carachteristicDistance=characteristicDistance, 
            singleCounterfactual=singleCfMapping['counterfactual'], 
            normalization=normalization,
            nonActionableFeatures=nonActionableFeatures
        ))
        
        # Generate counterfactuals for each selected node
        for node in greedySet:
            counterfactual = self.__counterfactualGivenTargetPointActionable(
                pointToExplain=self.data[startPointIndex],       
                targetCorePoint=graph.nodes[node]['pos'],
                nonActionableFeatures=nonActionableFeatures
            )
            
            allCounterfactuals.append(counterfactual)
        
        return allCounterfactuals, greedySet, graph, singleCfMapping
    

    def findMultipleCounterfactualsAlternative(self, startPointIndex:int, 
                                    targetCluster:int|np.int64|np.int32, nCounterfactuals:int):

        allCounterfactuals = []

        singleCfMapping = self.findSingleCounterfactual(
            startPointIndex=startPointIndex, targetCluster=targetCluster
        )

        #allCounterfactuals.append(singleCfMapping['counterfactual'])

        startVertexIndex:int = singleCfMapping['closest_core_point_index']

        graph, characteristicDistance = createGraph(
        neighborhoods=self.neighbourhoods, neighborhoodsDist=self.neighbourhoodsDist,
        clusterLabels=self.dbscanInstace.labels_, targetCluster=targetCluster, X=self.data
        )

        graph:nx.Graph   
        
        '''greedySet = set(greedyPermutation(
            graph, startVertex=startVertexIndex, nPoints=nCounterfactuals, alpha=0.5, normalize=True
            ))'''
        greedySet = set(physicalOptimization(
            graph, pointToExplain=self.data[startPointIndex,:], startVertex=startVertexIndex, nPoints=nCounterfactuals, 
            alpha=0.5, carachteristicDistance=characteristicDistance, singleCounterfactual=singleCfMapping['counterfactual']
            ))
        
        for node in greedySet:

            counterfactual = self.__counterfactualGivenTargetPoint(
                pointToExplain=self.data[startPointIndex],       
                targetCorePoint=graph.nodes[node]['pos']
            )
        
            allCounterfactuals.append(counterfactual)

        return allCounterfactuals, greedySet, graph, singleCfMapping
    

    def computeDiversityGraph(graph: nx.Graph, nodes):
        """
        Compute the sum of pairwise shortest path distances between nodes.
        
        Args:
            graph: NetworkX graph
            nodes: Collection of nodes to compute distances between
        
        Returns:
            Sum of all pairwise distances, or float('inf') if any pair is disconnected
        """
        #total_distance = 0
        
        nodes_list = list(nodes)
        finalMatrix = np.empty((len(nodes_list), len(nodes_list)))
        # Iterate over all unique pairs of nodes
        for i in range(len(nodes_list)):
            finalMatrix[i, i] = 1
            for j in range(i + 1, len(nodes_list)):
                try:
                    distance = nx.shortest_path_length(graph, nodes_list[i], nodes_list[j], weight='weight')

                    finalMatrix[i, j] = 1/(1 + distance)

                    finalMatrix[j, i] = 1/(1 + distance)
                    
                except nx.NetworkXNoPath:
                    
                    # Nodes are disconnected
                    return float('inf')
        
    
        return np.linalg.det(finalMatrix)

    

    def __mappingCorePointsData(self):

        indecesMapping = {}

        for cluster in np.unique(self.dbscanInstace.labels_):

            if cluster != -1:

                labelsCorePoints = self.dbscanInstace.labels_[self.dbscanInstace.core_sample_indices_]

                labelsTargetClusterMask = labelsCorePoints == cluster

                corePointsIndecesTarget:np.ndarray = self.dbscanInstace.core_sample_indices_[labelsTargetClusterMask]

                indecesMapping[cluster] = corePointsIndecesTarget

        return indecesMapping
    

    def __counterfactualGivenTargetPointActionable(self, pointToExplain, targetCorePoint, nonActionableFeatures):
        """
        Generate counterfactual considering non-actionable features
        """
        # Calculate displacement
        displacement = pointToExplain - targetCorePoint
        
        # Set displacement to 0 for non-actionable features
        displacement[nonActionableFeatures] = 0
        
        # Calculate unit vector for displacement
        displacement_norm = np.linalg.norm(displacement)
        
        if displacement_norm > 0:
            unitVectorDisplacement = displacement / displacement_norm
        else:
            unitVectorDisplacement = np.zeros_like(displacement)
        
        # Generate counterfactual using moveToEpsilonDistance
        counterfactual = moveToEpsilonDistance(
            start_point=pointToExplain,
            direction=unitVectorDisplacement, 
            target_point=targetCorePoint, 
            epsilon=self.dbscanInstace.eps
        )
        
        return counterfactual

                
    

    def selectPointsToTest(self) -> Dict[int, Iterable[int]]:

        np.random.seed(0)

        allPoints:Dict[int, Iterable[int]] = {} # id of the point, target clusters

        allClusters = np.unique(self.dbscanInstace.labels_)

        allIndeces = np.arange(self.data.shape[0])

        for cluster in allClusters:

            targetClusters:np.ndarray = allClusters[allClusters != cluster]

            indecesCluster:np.ndarray = allIndeces[self.dbscanInstace.labels_ == cluster]

            pointsChosen = np.random.choice(indecesCluster, size=np.min([10, len(indecesCluster)]), replace=False)

            for pointId in pointsChosen:

                allPoints[pointId] = targetClusters

        return allPoints


    def __counterfactualGivenTargetPoint(self, pointToExplain, targetCorePoint):

        unitVectorDisplacement = (pointToExplain - targetCorePoint) / np.linalg.norm((targetCorePoint - pointToExplain))

        movement = unitVectorDisplacement * self.dbscanInstace.eps

        counterfactual = targetCorePoint + movement

        return counterfactual



    def __isPointNotAlreadyInTarget(self, startPointIndex:int, targetCluster:int):

        if self.dbscanInstace.labels_[startPointIndex] == targetCluster:

            return False
        
        else:

            return True
            



    def plot_dbscan_counterfactual(self,
                                   cfResults, 
                              start_point_index: int, 
                              target_cluster: int,
                              title: str = "DBSCAN Counterfactual Explanation",
                              figsize: tuple = (12, 10)):
        """
        Create a beautiful visualization of DBSCAN counterfactual explanation
        
        Parameters:
        -----------
        dbscan_cf : DbscanCounterfactuals
            Your DBSCAN counterfactuals object
        start_point_index : int
            Index of the point to explain
        target_cluster : int
            Target cluster to move to
        title : str
            Plot title
        style : str
            'cute', 'professional', or 'minimal'
        figsize : tuple
            Figure size
        """
        
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
        # Style configurations
        
        
        colors = {
            'original': '#E74C3C',        # Red
            'counterfactual': '#27AE60',  # Green
            'closest_core': '#F39C12',    # Orange
            'arrow': '#34495E',           # Dark gray
            'eps_circle': '#BDC3C7',      # Light gray
            'background': '#FFFFFF',      # White
            'text': '#2C3E50'
        }
        point_size = 100
        core_size = 60
        alpha = 0.9
        
        
        ax.set_facecolor(colors['background'])
        
        # Plot all DBSCAN clusters
        labels = self.dbscanInstace.labels_
        unique_labels = set(labels)
        cluster_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, cluster_colors):
            if k == -1:
                # Noise points
                class_member_mask = (labels == k)
                xy = self.data[class_member_mask]
                ax.scatter(xy[:, 0], xy[:, 1], c='lightgray', marker='x', 
                        s=30, alpha=0.6, label='Noise' if k == -1 else None)
            else:
                # Regular clusters
                class_member_mask = (labels == k)
                xy = self.data[class_member_mask]
                
                # Highlight target cluster
                if k == target_cluster:
                    ax.scatter(xy[:, 0], xy[:, 1], color=col, s=60, alpha=0.7, 
                            edgecolors='black', linewidth=1, 
                            label=f'Target Cluster {k}')
                else:
                    ax.scatter(xy[:, 0], xy[:, 1], color=col, s=40, alpha=0.5,
                            label=f'Cluster {k}' if k != -1 else None)
        
        # Plot core points for target cluster
        if len(cfResults['core_points_target']) > 0:
            ax.scatter(cfResults['core_points_target'][:, 0], 
                    cfResults['core_points_target'][:, 1], 
                    c='white', s=core_size, marker='o', 
                    edgecolors='black', linewidth=2, alpha=0.9,
                    label='Core Points (Target)', zorder=8)
        
        # Highlight the closest core point
        ax.scatter(cfResults['closest_core_point'][0], 
                cfResults['closest_core_point'][1], 
                c=colors['closest_core'], s=point_size*1.2, marker='*', 
                edgecolors='black', linewidth=3, alpha=alpha,
                label='Closest Core Point', zorder=9)
        
        # Plot eps circle around closest core point
        eps_circle = Circle(cfResults['closest_core_point'], 
                        self.dbscanInstace.eps,
                        fill=False, color=colors['eps_circle'], 
                        linewidth=2, linestyle='--', alpha=0.7,
                        label=f'ε-neighborhood (ε={self.dbscanInstace.eps:.2f})')
        ax.add_patch(eps_circle)
        
        # Plot the journey: original -> closest core -> counterfactual
        # Arrow from original to closest core
        arrow1 = FancyArrowPatch(
            cfResults['original_point'], cfResults['closest_core_point'],
            arrowstyle='->', mutation_scale=20, color=colors['arrow'],
            linewidth=2, alpha=0.7, linestyle=':', zorder=6
        )
        ax.add_patch(arrow1)
        
        # Arrow from closest core to counterfactual
        arrow2 = FancyArrowPatch(
            cfResults['closest_core_point'], cfResults['counterfactual'],
            arrowstyle='->', mutation_scale=25, color=colors['arrow'],
            linewidth=3, alpha=alpha, zorder=7
        )
        ax.add_patch(arrow2)
        
        # Plot original point
        original_cluster = labels[start_point_index]
        ax.scatter(cfResults['original_point'][0], cfResults['original_point'][1], 
                c=colors['original'], s=point_size*1.5, marker='o', 
                edgecolors='white', linewidth=3, alpha=alpha,
                label=f'Original Point (Cluster {original_cluster})', zorder=10)
        
        # Plot counterfactual point
        ax.scatter(cfResults['counterfactual'][0], cfResults['counterfactual'][1], 
                c=colors['counterfactual'], s=point_size*1.5, marker='*', 
                edgecolors='white', linewidth=3, alpha=alpha,
                label='Counterfactual', zorder=10)
        
        # Customize the plot
        ax.set_title(title, fontsize=16, fontweight='bold', color=colors['text'], pad=20)
        ax.set_xlabel('Feature 1', fontsize=12, color=colors['text'])
        ax.set_ylabel('Feature 2', fontsize=12, color=colors['text'])
        
        # Legend with custom styling
        legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                        frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        return fig, ax, cfResults



    """#' Function to calculate self developed "density cluster separability index" (DCSI)
    #' 
    #' eps = median distance to minPts*2-th neighbor for each class
    #' @param dist a distance matrix
    #' @param labels a vector with labels
    #' @param minPts minPts argument for core point definition
    calc_DCSI <- function(dist, labels, minPts = 5){

    dist <- as.matrix(dist)

    # compute core points: 
    # calculate eps for each class = median distance to minPts*2-th nearest neighbor
    # calculate distance to minPts-th nearest neighbor among points of same class
    ind_corePoints <- c()
    for(i in unique(labels)){

        ind_i <- which(labels == i) 

        dist_i <- dist[ind_i, ind_i]

        knn_graph_eps_i <- cccd::nng(dx = dist_i, k = minPts*2)
        knn_matrix_eps_i <- as.matrix(igraph::as_adjacency_matrix(knn_graph_eps_i))
        knn_weights_eps_i <- matrixcalc::hadamard.prod(knn_matrix_eps_i, dist_i) # add distances to knn-graph

        dist_kth_neighbor_eps_i <- apply(knn_weights_eps_i, 1, max) # maximum value of every row = distance to minPts*2-th neighbor
        eps_i <- median(dist_kth_neighbor_eps_i)


        # calculate core points
        knn_graph_i <- cccd::nng(dx = dist_i, k = minPts)
        knn_matrix_i <- as.matrix(igraph::as_adjacency_matrix(knn_graph_i))
        knn_weights_i <- matrixcalc::hadamard.prod(knn_matrix_i, dist_i) # add distances to knn-graph

        dist_kth_neighbor_i <- apply(knn_weights_i, 1, max)

        ind_corePoints_i <- which(dist_kth_neighbor_i <= eps_i)

        ind_corePoints <- c(ind_corePoints, ind_i[ind_corePoints_i])

    }

    # from now on, consider only core points
    dist_core <- dist[ind_corePoints, ind_corePoints]
    labels_core <- labels[ind_corePoints]

    # for each cluster i: 
    # separation = minimum distance between a core point of i and a core point that is not in i
    # connectedness = maximum distance in a MST built of the core points of cluster i
    Sep_list <- list()
    Conn_list <- list()



    """



