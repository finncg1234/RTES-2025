import time
import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset


from dataset import fetch_dataset
from autoencoder import AE
from config import Config, feature_extraction

torch.cuda.empty_cache()
# folder to save everything
save_path = "out\\"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(time.time())
# model/training parameters
ae_config = Config()

def dataset_loader():
    # Create TensorDataset that pairs inputs with labels
    
    dataset, labels = fetch_dataset(ae_config)
    # print(len(dataset))
    # print(len(labels))
    if ae_config.labeled:
        dataset = TensorDataset(dataset, labels)
    # get the dataset loader
    loader = torch.utils.data.DataLoader(
            dataset=dataset, batch_size=ae_config.B, shuffle=True)
    
    return loader


def train_ae():
    print("Training the model...")
    # get the features
    loader = dataset_loader()
    # instantiate our model
    ae = AE(ae_config.input_size, True)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(ae.parameters(), lr=ae_config.lr, weight_decay=ae_config.wd)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    torch.nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
    ae.to(device)

    losses = []

    for epoch in range(ae_config.epochs):

        for inputs in loader:
            inputs = inputs.to(device)

            reconstructed = ae(inputs)

            loss = loss_function(reconstructed, inputs)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1}/{ae_config.epochs}, Loss: {loss.item():.6f}")
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title("Loss over Time")
    plt.grid(True)
    plt.savefig(ae_config.model_path + "_training_loss.png")

    torch.save(ae.state_dict(), ae_config.model_path + ".pth")

def run_ae():
    print("Evaluating the model...")

    ae = AE(ae_config.input_size, True)
    ae.load_state_dict(torch.load(ae_config.model_path + ".pth"))
    ae.to(device)
    
    loader = dataset_loader()
    
    ae.eval()

    # Collect reconstruction errors and true labels for all samples
    all_errors = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            reconstructed = ae(inputs)
            errors = ((reconstructed - inputs) ** 2).mean(dim=1)

            all_errors.extend(errors.cpu().tolist())
            all_labels.extend((labels).cpu().int().tolist())

    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)
    print("There are " + str(sum(all_labels == 1)) + " attack frames out of " + str(len(all_labels)))
    # Plot histogram of errors, separated by label
    normal_errors = all_errors[all_labels == 0]  # Label 0 = normal
    attack_errors = all_errors[all_labels == 1]  # Label 1 = attack

    plt.hist(normal_errors, bins=50, alpha=0.6, label='Normal Traffic', color='blue', edgecolor='black')
    plt.hist(attack_errors, bins=50, alpha=0.6, label='Attack Traffic', color='red', edgecolor='black')

    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    plt.savefig(ae_config.file_path + "_error_distribution.png")
    plt.close()
    # # Sweep a range of thresholds between min and max error
    thresholds = np.linspace(all_errors.min(), all_errors.max(), 50)
    precisions = []
    recalls = []

    fprs = []
    tprs = []

    for t in thresholds:
        preds = (all_errors > t).astype(int)

        TP = np.sum((preds == 1) & (all_labels == 1))
        FP = np.sum((preds == 1) & (all_labels == 0))
        FN = np.sum((preds == 0) & (all_labels == 1))
        TN = np.sum((preds == 0) & (all_labels == 0))
        # print("TP: " + str(TP))
        # print("FP: " + str(FP))
        # print("FN: " + str(FN))
        # print("TN: " + str(TN))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        fprs.append(FP / (FP + TN))
        tprs.append(TP / (TP + FN))

    roc_auc = np.trapezoid(tprs, fprs)

    plt.figure(figsize=(8,5))
    plt.plot(fprs, tprs, label=f"ROC (AUC = {roc_auc:.3f})", linewidth=2)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(ae_config.file_path + "_eval.png")



def main():
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Whether to run or train the model")
    parser.add_argument(
        "-train",
        action="store_true",
        help="train the model"
    )
    parser.add_argument(
        "-run",
        action="store_true",
        help="run the model"
    )
    parser.add_argument(
        "-f",
        help="feature extraction strategy to use [1 for naive with no CAN DATA, 1 for including CAN DATA]",
        required=True
    )
    parser.add_argument(
        "-b",
        help="batch size to use for training",
        required=True
    )
    parser.add_argument(
        "-r",
        help="learning rate for training",
        required=True
    )
    parser.add_argument(
        "-e",
        help="number of epochs to train for",
        required=True
    )
    parser.add_argument(
        "-n",
        help="number of samples per input vector",
        required=True
    )
    parser.add_argument(
        "-v",
        help="vehicle type to use the data from, e.g. \'2016-chevrolet-silverado\'",
        required=True
    )
    parser.add_argument(
        "-d",
        help="attack type of dataset to analyze, e.g. \'extra-attack-free\'",
        required=True
    )
    parser.add_argument(
        "-l",
        help="whether the dataset is labeled or not",
        required=True
    )
    args = parser.parse_args()
    # decide whether to train or evaluate the model or both
    ae_config.N = int(args.n)
    ae_config.B = int(args.b)
    ae_config.epochs = int(args.e)
    ae_config.lr = float(args.r)*(10**-4)
    ae_config.labeled = (int(args.l) == 1)

    ae_config.fe = feature_extraction(int(args.f))

    ae_config.dtype = str(args.d)
    ae_config.vehicle = str(args.v)

    if ae_config.fe == feature_extraction.NAIVE:
        ae_config.input_size = ae_config.N * 6
    elif ae_config.fe == feature_extraction.NO_CAN_DATA:
        ae_config.input_size = ae_config.N * 2
    else:
        assert(0), "this feature extraction not currently supported."
    
    ae_config.model_path = (
        ".\\out\\" + 
        ae_config.vehicle + 
        "\\" + 
        "fe-" + str(ae_config.fe.value) +  # Use .value to get the integer from enum
        "_b-" + str(ae_config.B) + 
        "_input-size-" + str(ae_config.input_size) + 
        "_epochs-" + str(ae_config.epochs) + 
        "_lr-" + str(ae_config.lr).replace('.', '_')  # Replace '.' with '_'
    )
    ae_config.file_path = (
        ".\\out\\" + 
        ae_config.vehicle + 
        "\\" + 
        ae_config.dtype +
        "\\"
        "fe-" + str(ae_config.fe.value) +  # Use .value to get the integer from enum
        "_b-" + str(ae_config.B) + 
        "_input-size-" + str(ae_config.input_size) + 
        "_epochs-" + str(ae_config.epochs) + 
        "_lr-" + str(ae_config.lr).replace('.', '_')  # Replace '.' with '_'
    )

    print("Proceeding with config: " + str(ae_config))
    if (args.train):
        train_ae()
    if(args.run):
        run_ae()
    
if __name__ == "__main__":
    main()