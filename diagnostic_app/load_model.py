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

def using_model(radius, texture, perimeter, area, smoothness, compactness, symmetry, fractal_dimension):
    
    neural_network_loaded = Classifier_torch()

    state_dict = torch.load('checkpoint.pth', weights_only=True)
    neural_network_loaded.load_state_dict(state_dict)
    neural_network_loaded.eval()

    sample = {
        'radius': radius,
        'texture': texture,
        'perimeter': perimeter,
        'area': area,
        'smoothness': smoothness,
        'compactness': compactness,
        'symmetry': symmetry,
        'fractal_dimension': fractal_dimension
    }

    forecasters = list(sample.values())
    tensor_forecasters = torch.tensor(forecasters, dtype=torch.float)
    neural_network_loaded.eval()
    forecast = neural_network_loaded(tensor_forecasters)

    forecast = forecast.detach().numpy()
    forecast = float(forecast.item()) 
    if forecast >= 0.70:
        return f"For the sample the result is Malignant: {forecast * 100:.2f}% accuracy."
    elif forecast > 0.30 and forecast < 0.70:
        return f"For the sample the result is Inconclusive: {forecast * 100:.2f}% accuracy."
    else:
        return f"For the sample the result is Benign: {100-(forecast * 100):.2f}% accuracy."

