import torch
from torch import nn

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

neural_network_loaded = Classifier_torch()

state_dict = torch.load('checkpoint.pth')
neural_network_loaded.load_state_dict(state_dict)
neural_network_loaded.eval()

malignant_00 = {
    'radius': 23,
    'texture': 12,
    'perimeter': 151,
    'area': 954,
    'smoothness': 0.143,
    'compactness': 0.278,
    'symmetry': 0.242,
    'fractal_dimension': 0.079
}

malignant_01 = {
    'radius': 10,
    'texture': 19,
    'perimeter': 126,
    'area': 1152,
    'smoothness': 0.105,
    'compactness': 0.127,
    'symmetry': 0.192,
    'fractal_dimension': 0.06
}

benign_00 = {
    'radius': 19,
    'texture': 22,
    'perimeter': 87,
    'area': 572,
    'smoothness': 0.077,
    'compactness': 0.061,
    'symmetry': 0.135,
    'fractal_dimension': 0.06
}

benign_01 = {
    'radius': 19,
    'texture': 25,
    'perimeter': 75,
    'area': 428,
    'smoothness': 0.086,
    'compactness': 0.05,
    'symmetry': 0.15,
    'fractal_dimension': 0.059
}

benign_02 = {
    'radius': 23,
    'texture': 26,
    'perimeter': 54,
    'area': 255,
    'smoothness': 0.098,
    'compactness': 0.053,
    'symmetry': 0.168,
    'fractal_dimension': 0.072
}


list_forecasters = [malignant_00, malignant_01, benign_00, benign_01, benign_02]
waited_results = ['Malignant', 'Malignant', 'Inconclusive','Inconclusive', 'Benign']

print("\n####################################")
print("Between 70 and 100: Malignant\nBetween 30 and 70: Inconclusive\nBelow 30: Benign\n")

cont = 0
for forecasters in list_forecasters:    
    forecasters = list(forecasters.values())
    tensor_forecasters = torch.tensor(forecasters, dtype=torch.float)
    neural_network_loaded.eval()
    forecast = neural_network_loaded(tensor_forecasters)

    forecast = forecast.detach().numpy()
    forecast = float(forecast.item()) 
    if forecast >= 0.70:
        print(f"For the {cont+1}° sample the result is Malignant: {forecast * 100:.2f}% accuracy. Expected Result: {waited_results[cont]}")
    elif forecast > 0.30 and forecast < 0.70:
        print(f"For the {cont+1}° sample the result is Inconclusive: {forecast * 100:.2f}% - Expected Result: {waited_results[cont]}")
    else:
        print(f"For the {cont+1}° sample the result is Benign: {100-(forecast * 100):.2f}% accuracy - Expected Result: {waited_results[cont]}")

    cont+=1
print('\n')


