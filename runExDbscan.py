
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from typing import Dict, Tuple, List, Iterable
from modules.utilityFunctions import dbscan_opt, executeDice, addPointsToGraphExdbscan, addPointsToGraph, computeDiversityGraphDpp, loadResults, loadDbscanParameters, executeBaycon, computeDiversityGraph
from modules.dbscanCounterfactuals import DbscanCounterfactuals
from modules.customDbscan import PredictDBSCAN
from modules.constants import datasets
import numpy as np
from openml.datasets import OpenMLDataset
from modules.data import Dataset, loadDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm

np.random.seed(0)

toyDatasetPath = 'data/iclrDbscan/toyDataset.csv'

iclrDbscanPath:str = 'data/iclrDbscan/datasetsParameters.json'

pathDf = 'data/iclrDbscan/resultsDiversityNonActionableDPP.csv'


resultsDf = pd.read_csv(pathDf)

uniqueDatasets = np.unique(resultsDf['datasetName'])

for datasetName in tqdm(datasets):

    if datasetName in uniqueDatasets:

        continue

    dataset:OpenMLDataset = loadDataset(name=datasetName)

    target_name = dataset.default_target_attribute

    df, _, _, _ = dataset.get_data()

    df = df.drop_duplicates().dropna()

    featureNames = list(df.columns)

    X = df.drop(columns=[target_name]).to_numpy()

    featureNames = np.array(featureNames)[np.array(featureNames) != target_name]

    nFeatures = X.shape[1]

    nNonActionableFeatures = np.random.choice(np.arange(nFeatures//2), size=1)[0]

    nonActionableFeatures = np.random.choice(np.arange(nFeatures), size=nNonActionableFeatures)

    stringActionable = np.array(featureNames)[np.arange(nFeatures)[~np.isin(np.arange(nFeatures), nonActionableFeatures)]]

    if nFeatures > 20:

        

        continue

    df:pd.DataFrame = df.rename({target_name:'target'}, axis=1)

    

    
    try:
        
        df = df.astype(float)

    except:

        df['target'] = LabelEncoder().fit_transform(df['target'])

    X = StandardScaler().fit_transform(X=X)

    allParams = loadDbscanParameters(parametersPath=iclrDbscanPath,
                                    datasetName=datasetName,
                                    x=X)

    

    dbscanInstance:PredictDBSCAN = PredictDBSCAN(eps=allParams[datasetName][0], 
                                min_samples=allParams[datasetName][1])

    predictInstance, neighborhoods, neighborhoods_dist = dbscanInstance.fitCustom(X=X)

    y = dbscanInstance.predict(X=X)

    df['target'] = y

    allLabels = np.unique(dbscanInstance.labels_)

    cfInstance:DbscanCounterfactuals = DbscanCounterfactuals(
        data=X,dbscanInstance=dbscanInstance, neighbourhoods=neighborhoods,
        neighbourhoodsDist=neighborhoods_dist
    )

    pointsToTest:Dict[int, Iterable[int]] = cfInstance.selectPointsToTest()

    resultsDatasets = []

    for i, (indexPoint, targetClusters) in enumerate(tqdm(pointsToTest.items())):

        for targetCluster in targetClusters:

            if targetCluster != -1:

                try:

                    output = executeBaycon(
                        xData=X, labels=dbscanInstance.labels_,
                        model=dbscanInstance, indexStartPoint=indexPoint,
                        targetClass=int(targetCluster), plainCounterfactuals=True,
                        featureNames=featureNames, actionableFeatures=stringActionable
                    )

                except AssertionError:

                    continue

                predictions = np.array(output['predictions'])

                counterfactualsBaycon = np.array(output['counterfactuals'])

                nCounterfactuals = 10

                allCounterfactuals, greedySetExDbscan, graph, singleCfMapping = cfInstance.findMultipleCounterfactualsActionable(
                startPointIndex=indexPoint, targetCluster=targetCluster, nCounterfactuals=nCounterfactuals,
                normalization=True, nonActionableFeatures=nonActionableFeatures
                )            

                if len(counterfactualsBaycon) == 0:

                    proximityBaycon = np.nan

                    diversityBaycon = np.nan

                else:

                    validCounterfactualsBaycon:np.ndarray = counterfactualsBaycon[predictions == targetCluster][:10]

                    allProximitiesBaycon = np.linalg.norm(
                    validCounterfactualsBaycon-X[indexPoint], axis=1
                    )

                    newGraphBaycon, newNodeIndicesBaycon, _ = addPointsToGraph(
                    G=graph, newPoints=validCounterfactualsBaycon,
                    targetCluster=targetCluster, epsilon=dbscanInstance.get_params()['eps']
                    )

                    

                    diversityBaycon = computeDiversityGraphDpp(graph=newGraphBaycon, nodes=newNodeIndicesBaycon)#cfInstance.computeDiversityGraph(graph=graph, nodes=greedySet)

                    proximityBaycon = np.mean(allProximitiesBaycon)

                
                if len(allCounterfactuals)==0:
                    
                    allProximitiesDbscan = np.nan

                    diversityDbscan = np.nan

                else:
                    allProximitiesDbscan = np.linalg.norm(
                    allCounterfactuals-X[indexPoint], axis=1
                )

                    proximityDbscan = np.mean(allProximitiesDbscan)

                    

                    newGraphExDbscan, newNodeIndicesExDbscan, _ = addPointsToGraph(
                    G=graph, newPoints=allCounterfactuals, targetCluster=targetCluster, epsilon=dbscanInstance.get_params()['eps']
                )
                    
                    diversityExDbscan = computeDiversityGraphDpp(graph=newGraphExDbscan, nodes=newNodeIndicesExDbscan)
                            
                                                                        
                
                resultsDatasets.append(
                    [datasetName, indexPoint, targetCluster, proximityDbscan, proximityBaycon, diversityExDbscan, diversityBaycon, nFeatures]
                )

    resultsDatasetsDf = pd.DataFrame(resultsDatasets, columns=['datasetName','indexPoint','targetCluster','proximityDb','proximityBaycon', 'diversityDb', 'diversityBaycon', 'nFeatures'])

    resultsDf = pd.concat([resultsDf, resultsDatasetsDf], ignore_index=True)

    resultsDf.to_csv('data/iclrDbscan/resultsDiversityNonActionableDPP.csv', index=False)
            
            

            

            
        



