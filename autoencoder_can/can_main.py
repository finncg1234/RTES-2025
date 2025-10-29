import torch
from torch import nn, optim

from dataset import create_dataset, load_data
from autoencoder import AE

# model/training parameters
N = 10                              # number of can messages to pack into one input. (input layer size = 2N)
B = 32                              # batch size
epochs = 10                         # number of times to cycle through training set
lr = 1e-3                           # learning rate
wd = 1e-8                           # weight decay

# Which can trace to use
vehicle = "2016-chevrolet-silverado"
type = "extra-attack-free"

# grab the dataset
raw_data = load_data(vehicle, type)
can_dataset = create_dataset(raw_data, N)
can_dataset = can_dataset.to(torch.float32)
# confirm it looks as expected
# print(can_dataset[0:2])
print(can_dataset.shape)

# get the dataset loader
loader = torch.utils.data.DataLoader(
        dataset=can_dataset, batch_size=B, shuffle=True)

# instantiate our model
ae = AE(N * 2, False)
loss_function = nn.MSELoss()
optimizer = optim.Adam(ae.parameters(), lr=lr, weight_decay=wd)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ae.to(device)

outputs = []
losses = []

for epoch in range(epochs):
    for inputs in loader:
        inputs = inputs.to(device)

        reconstructed = ae(inputs)

        loss = loss_function(reconstructed, inputs)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    outputs.append((epoch, inputs, reconstructed))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
