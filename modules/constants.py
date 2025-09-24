from pathlib import Path
from typing import List


dataPath:Path = Path.cwd() / 'data'

modelsPath:Path = dataPath / 'models'

featuresIris:List[str] = ['sepal_length', 'sepal_width', 'petal_length', 'sepal_width']

featuresWine:List[str] = ['Alcohol', 'malicAcid', 'ash', 'ashAlcalinity', 'magnesium',
                       'totalPhenols', 'flavanoids', 'nonflavanoidPhenols',
                       'proanthocyanins', 'colorIntensity', 'hure', 'dilutedWines', 'proline']


datasets = ['hayes-roth', 'breast-w', 'wine', 'iris', 'ar3',
 'MindCave2',
 'ar5',
 'autoPrice',
 'baskball',
 'blood-transfusion-service-center',
 'bodyfat',
 'chscase_census2',
 'chscase_census6',
 'chscase_vine1',
 'diabetes',
 'diabetes_numeric',
 'disclosure_x_noise',
 'ecoli',
 'glass',
 'heart-statlog',
 'kc1-top5',
 'kc3',
 'libras_move',
 'liver-disorders',
 'longley',
 'machine_cpu',
 'mfeat-zernike',
 'mu284',
 'no2',
 'pyrim',
 'rabe_131',
 'sleep',
 'sonar',
 'strikes',
 'tecator',
 'triazines',
 'vehicle',
 'wdbc',
 'wisconsin',
 'SPECTF',
 'ar6', 'confidence', 'diggle_table_a1', 'prnn_fglass'
 ,'ionosphere', 'kc2', 'pm10', 'steel-plates-fault', 'MeanWhile1', 'diamonds', 'wine-quality-red',
 'wine-quality-white']