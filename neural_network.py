import pandas as pd
import numpy as np
import torch
import torch.nn as nn

np.random.seed(123)
torch.manual_seed(123)

class Classifier_torch(nn.Module):
    def __init__(self,):
        super().__init__()

        # Defition of layers
        # 8 -> 4 -> 4 -> 1

        self.dense_00 = nn.Linear(8, 4, bias=True)
        nn.init.uniform_(self.dense_00.weight)
        self.activation_00 = nn.ReLU()

        self.dense_01 = nn.Linear(4, 4, bias=True)
        nn.init.uniform_(self.dense_01.weight)
        self.activation_01 = nn.ReLU()

        self.dense_02 = nn.Linear(4, 1, bias=True)
        nn.init.uniform_(self.dense_02.weight)

        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.dense_00(x)
        x = self.activation_00(x)

        x = self.dense_01(x)
        x = self.activation_01(x)

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


#######################################
## Data manipulation and transformation

df = pd.read_csv('Prostate_Cancer.csv')
df = df.drop('id', axis=1)
df['diagnosis_result'] = df['diagnosis_result'].apply(is_maligant_or_benign)

pd_forecasters = df.drop('diagnosis_result', axis=1)
tensor_forecasters = torch.tensor(np.array(pd_forecasters), dtype=torch.float)

pd_classes = pd.DataFrame() 
pd_classes['diagnosis_result'] = df['diagnosis_result'].copy()
tensor_classes = torch.tensor(np.array(pd_classes), dtype=torch.float)

dataset = torch.utils.data.TensorDataset(tensor_forecasters, tensor_classes)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

neural_network = Classifier_torch()
erro_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(neural_network.parameters(), lr=0.001, weight_decay=0.0001)

## Traning...
for epoch in range(500):
    running_erro = 0.0
    for data in train_loader:
        forecasters, classes = data
        optimizer.zero_grad()

        output = neural_network(forecasters)
        values_loss = erro_function(output, classes)
        values_loss.backward()
        optimizer.step()

        running_erro += values_loss.item()

    print('Epoch %3d: loss %.5f' % (epoch+1, running_erro/len(train_loader)))

# Salvando o modelo treinado
torch.save(neural_network.state_dict(), 'checkpoint.pth')

