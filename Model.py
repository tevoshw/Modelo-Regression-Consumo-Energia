from Data import X_tensor, Y_tensor
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split

# Train and test split
X_train, X_test, y_train, y_test = train_test_split(X_tensor.numpy(), Y_tensor.numpy(), test_size=0.3, random_state=4)

# Convertendo para tensores novamente após a divisão
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Model
class CalculateEnergyModel(nn.Module):
    def __init__(self):
        super(CalculateEnergyModel, self).__init__()
        self.model = nn.Linear(4, 1)
        
    def forward(self, x):
        return self.model(x)
    
torch.manual_seed(4342)
model = CalculateEnergyModel()

# Loss and optimizer functions
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training
epochs = 10000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    y_preds = model(X_train)  
    loss = loss_function(y_preds, y_train)  

    loss.backward() 
    optimizer.step() 

    if epoch % 1000 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item():.5f}")
