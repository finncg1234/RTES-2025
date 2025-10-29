import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from autoencoder import AE

# Adjust as needed based on timing/hardware requirements
epochs = 10


def train_ae(use_bfw):
    save_path = "training_data\\loss_per_epoch_" + ("bfw.png" if (use_bfw) else "no_bfw.png")
    tensor_transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=True,
                            download=True, transform=tensor_transform)
    targets = dataset.targets
    mask = (targets == 2)
    indices = mask.nonzero(as_tuple=True)[0]

    # Just want to train on samples with label 2
    subset = Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        dataset=subset, batch_size=32, shuffle=True)

    model_bfw = AE(use_bfw)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model_bfw.parameters(), lr=1e-3, weight_decay=1e-8)

    
    outputs_bfw = []
    losses_bfw = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_bfw.to(device)

    for epoch in range(epochs):
        for images, labels in loader:

            images = images.view(-1, 28*28).to(device)

            reconstructed = model_bfw(images)
            loss = loss_function(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses_bfw.append(loss.item())

        outputs_bfw.append((epoch, images, reconstructed))
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    torch.save(model_bfw.state_dict(), "ae_mnist_digit2_" + ("bfw.pth" if use_bfw else "no_bfw.pth"))

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(9, 6))
    plt.plot(losses_bfw, label='Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch Count for NN ' + 'with BFW' if use_bfw else 'without BFW')
    plt.legend()
    # plt.show()
    plt.savefig(save_path, pad_inches=0.1)