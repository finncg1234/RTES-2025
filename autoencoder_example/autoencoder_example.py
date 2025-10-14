import kagglehub
from kagglehub import KaggleDatasetAdapter

import torch
from torch import nn, optim
import matplotlib.pyplot as plt

import pandas as pd

print(torch.__version__)
print(torch.cuda.is_available()) # True if CUDA is available, False otherwise



# Download latest version
file_path = kagglehub.dataset_download("mohankrishnathalla/diabetes-health-indicators-dataset")
file_path = file_path + "/diabetes_dataset.csv"
print("Path to dataset files:", file_path)

df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "mohankrishnathalla/diabetes-health-indicators-dataset",
  "diabetes_dataset.csv",
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records:", df.head())

df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(0)
df = (df - df.min()) / (df.max() - df.min())

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 15),
            nn.ReLU(),
            nn.Linear(15, 30),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
model = AE()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

epochs = 5
outputs = []
losses = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
data = torch.tensor(df.iloc[:, 0:30].values, dtype=torch.float32).to(device)

for epoch in range(epochs):
    for x in data:
        x = x.unsqueeze(0)

        reconstructed = model(x)
        loss = loss_function(reconstructed, x)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    outputs.append((epoch, x, reconstructed))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

plt.style.use('fivethirtyeight')
plt.figure(figsize=(8, 5))
plt.plot(losses, label='Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()