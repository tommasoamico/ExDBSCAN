from typeguard import typechecked
from ucimlrepo import fetch_ucirepo 
from sklearn.utils import Bunch
import numpy as np
from typing import Dict, Any, Optional, List, Iterable
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from modules.constants import featuresIris, featuresWine
from sklearn.datasets import fetch_openml
from sklearn.utils import Bunch
import openml


@typechecked
def loadDataset(name:str, version:int=1):

    data = openml.datasets.get_dataset(name, version=version)

    return data


class Dataset:

    @typechecked
    def __init__(self, datasetname:str):
        
        assert datasetname in ['diabetes', 'covertype', 'dryBean', 'penDigits',
                               'iris', 'wine'], f"Variable datasetname must be one of {['diabetes', 'covertype', 'dryBean', 'penDigits']}"
        
        self.datasetname:str = datasetname

        self.featureNames:Optional[List[str]] = None

    
    @typechecked
    def getData(self):

        if self.datasetname == 'diabetes':

            diabetesDataset:Dict[str, Any] = fetch_ucirepo(id=296)

            X:pd.DataFrame = diabetesDataset.data.features

            y:pd.DataFrame = diabetesDataset.data.targets

            return X.to_numpy(), y.to_numpy()
    
        elif self.datasetname == 'covertype':

            diabetesDataset:Dict[str, Any] = fetch_ucirepo(id=31)

            X:pd.DataFrame = diabetesDataset.data.features

            y:pd.DataFrame = diabetesDataset.data.targets

            X:np.ndarray = MinMaxScaler().fit_transform(X)

            return X, y.to_numpy()
        
        elif self.datasetname == 'dryBean':

            diabetesDataset:Dict[str, Any] = fetch_ucirepo(id=602)

            X:pd.DataFrame = diabetesDataset.data.features

            y:pd.DataFrame = diabetesDataset.data.targets

            X:np.ndarray = MinMaxScaler().fit_transform(X)

            return X, y.to_numpy()
        
        elif self.datasetname == 'penDigits':

            self.dataset:Dict[str, Any] = fetch_ucirepo(id=81) 
  
            # data (as pandas dataframes) 
            X:pd.DataFrame = self.dataset.data.features 
            y:pd.DataFrame = self.dataset.data.targets 

            X:np.ndarray = StandardScaler().fit_transform(X)

            self.featureNames:Iterable[str] = [str(i+1) for i in range(16)]

            return X, y.to_numpy().flatten()
        
        elif self.datasetname == 'iris':

            self.dataset:Dict[str, Any] = fetch_ucirepo(id=53) 
  
            # data (as pandas dataframes) 
            X:pd.DataFrame = self.dataset.data.features 
            y:pd.DataFrame = self.dataset.data.targets 

            X:np.ndarray = StandardScaler().fit_transform(X)

            self.featureNames:List[str] = featuresIris

            return X, y.to_numpy().flatten()
        

        elif self.datasetname == 'wine':

            self.dataset:Dict[str, Any] = fetch_ucirepo(id=109) 
  
            # data (as pandas dataframes) 
            X:pd.DataFrame = self.dataset.data.features 
            y:pd.DataFrame = self.dataset.data.targets 

            X:np.ndarray = StandardScaler().fit_transform(X)

            self.featureNames:List[str] = featuresWine

            return X, y.to_numpy().flatten()
        

    
        

    