import torch
import numpy as np


torch.manual_seed(4342)
num_residencias = 1000

# Gerando os dados simulados
temperatura = 15 + (torch.rand(num_residencias, 1) * (35 - 15))  # Temperature mean
luz_solar = 4 + (torch.rand(num_residencias, 1) * (12 - 4))  # Hours of daylight
aparelhos = torch.randint(5, 20, (num_residencias, 1))  # Numbers os household appliances
area_residencial = torch.randint(50, 300, (num_residencias, 1))  # Area of house (m²)

consumo_energia = (temperatura * 50) + (luz_solar * 30) + (aparelhos * 5) + (area_residencial * 0.3) + torch.normal(0, 200, (num_residencias, 1)) # Y


# X and Y
X_tensor = torch.cat([temperatura, luz_solar, aparelhos, area_residencial], dim=1)
Y_tensor = consumo_energia

# Scaler
X_mean = X_tensor.mean(dim=0)
X_std = X_tensor.std(dim=0)

X_tensor = (X_tensor - X_mean) / X_std 
