import pandas as pd
import numpy as np

import torch
from torch import nn

from skorch import NeuralNetBinaryClassifier
from sklearn.model_selection import GridSearchCV

class Classifier_torch(nn.Module):
    def __init__(self, activation, neurons, initializer):
        super().__init__()

        # Defition of layers
        # 8 -> 4 -> 4 -> 1

        self.dense_00 = nn.Linear(8, neurons, bias=True)
        initializer(self.dense_00.weight)
        self.activation_00 = activation

        self.dense_01 = nn.Linear(neurons, neurons, bias=True)
        initializer(self.dense_01.weight)
        self.activation_01 = activation

        self.dense_02 = nn.Linear(neurons, 1, bias=True)
        initializer(self.dense_02.weight)
        self.dropout = nn.Dropout(0.2)

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.dense_00(x)
        x = self.activation_00(x)
        x = self.dropout(x)

        x = self.dense_01(x)
        x = self.activation_01(x)
        x = self.dropout(x)

        x = self.dense_02(x)
        x = self.output(x)

        return x


def is_maligant_or_benign(valor):
    '''
    Turn M "Malignant" into int 1
    and B "Benign" into int 0

    retunt int
    '''
    if valor == 'M':
        return 1
    else:
        return 0


np.random.seed(123)
torch.manual_seed(123)

#######################################
## Data manipulation and transformation

df = pd.read_csv('xx_projetos/prostata_cancer/Prostate_Cancer.csv')
df = df.drop('id', axis=1)
df['diagnosis_result'] = df['diagnosis_result'].apply(is_maligant_or_benign)

pd_forecasters = df.drop('diagnosis_result', axis=1)
np_array_forecasters = np.array(pd_forecasters, dtype='float32')

pd_classes = pd.DataFrame() 
pd_classes['diagnosis_result'] = df['diagnosis_result'].copy()
np_array_classes = np.array(pd_classes, dtype='float32').squeeze(1)

# Classifier/Neural Network with Skorch to use in Sklearn
classificador_sklearn = NeuralNetBinaryClassifier(module=Classifier_torch,
                                                  lr=0.001,
                                                  optimizer__weight_decay=0.0001,
                                                  train_split=False)

params = {'batch_size': [5], # 5,10,20
          'max_epochs': [100], #50, 100, 200, 500, 1000
          'optimizer': [torch.optim.Adam], #torch.optim.Adam, torch.optim.SGD
          'criterion': [torch.nn.BCELoss], #torch.nn.HingeEmbeddingLoss, torch.nn.BCEWithLogitsLoss
          'module__activation': [nn.ReLU()], #nn.functional.tanh, nn.functional.relu,
          'module__neurons': [4], # 4, 8, 16, 
          'module__initializer': [torch.nn.init.uniform_]} #torch.nn.init.normal_ , torch.nn.init.uniform_

grid_search = GridSearchCV(estimator=classificador_sklearn, param_grid=params,scoring='accuracy', cv=10)
grid_search = grid_search.fit(np_array_forecasters, np_array_classes)

best_precise = grid_search.best_score_
best_parameters = grid_search.best_params_

print("\n#######################################")
print(f"Best Precise: {best_precise}" )
print("Best Parameters:\n\n", best_parameters, '\n')

