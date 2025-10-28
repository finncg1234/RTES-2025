import torch
from torch import nn, optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from autoencoder import AE

def eval_ae(model_path):
    use_bfw = (model_path == "ae_mnist_digit2_bfw.pth") 
    tensor_transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=True,
                            download=True, transform=tensor_transform)
    loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AE(use_bfw)                      # re-initialize the same architecture
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()                      # switch to evaluation mode
    print("Model loaded!")

    dataiter = iter(loader)
    images, _ = next(dataiter)

    images = images.view(-1, 28*28).to(device)
    reconstructed = model(images)

    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(10,3))
    for i in range(10):
        axes[0, i].imshow(images[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].cpu().detach().numpy().reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
    plt.savefig("evaluation_data/" + ("reconstructed_images_bfw.png" if use_bfw else "reconstructed_images_no_bfw.png"))

    model.eval()

    # Collect reconstruction errors and true labels for all samples
    all_errors = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.view(-1, 28*28).to(device)
            labels = labels.to(device)

            reconstructed = model(images)
            errors = ((reconstructed - images) ** 2).mean(dim=1)

            all_errors.extend(errors.cpu().tolist())
            all_labels.extend((labels == 2).cpu().int().tolist())

    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)

    # Sweep a range of thresholds between min and max error
    thresholds = np.linspace(all_errors.min(), all_errors.max(), 50)
    precisions = []
    recalls = []

    fprs = []
    tprs = []

    for t in thresholds:
        preds = (all_errors < t).astype(int)

        TP = np.sum((preds == 1) & (all_labels == 1))
        FP = np.sum((preds == 1) & (all_labels == 0))
        FN = np.sum((preds == 0) & (all_labels == 1))
        TN = np.sum((preds == 0) & (all_labels == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        fprs.append(FP / (FP + TN))
        tprs.append(TP / (TP + FN))

    # Plot precision and recall vs threshold
    # plt.figure(figsize=(8,5))
    # plt.plot(thresholds, precisions, label="Precision", linewidth=2)
    # plt.plot(thresholds, recalls, label="Recall", linewidth=2)
    # plt.xlabel("Error Threshold")
    # plt.ylabel("Score")
    # plt.title("Precision and Recall vs Error Threshold")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # Plot ROC
    # Calculate AUC using trapezoidal rule
    roc_auc = np.trapezoid(tprs, fprs)

    plt.figure(figsize=(8,5))
    plt.plot(fprs, tprs, label=f"ROC (AUC = {roc_auc:.3f})", linewidth=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve" + (" using BFW" if use_bfw else " without using BFW"))
    plt.legend()
    plt.grid(True)
    plt.savefig("evaluation_data/" + ("ROC_bfw.png" if use_bfw else "ROC_no_bfw.png"), pad_inches=0.1)

    # Optional: precision–recall curve
    # plt.figure(figsize=(6,5))
    # plt.plot(recalls, precisions, marker='.')
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.title("Precision–Recall Curve")
    # plt.grid(True)
    # plt.show()