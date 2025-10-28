import subprocess
import argparse
from autoencoder_train import train_ae
from autoencoder_eval import eval_ae

# For documentation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

model_no_bfw = "ae_mnist_digit2_no_bfw.pth"
model_bfw = "ae_mnist_digit2_bfw.pth"
def train():
    print("training model without BFW...")
    train_ae(False)
    print("training model with BFW...")
    train_ae(True)
    

def evaluate():
    print("running evaluation...")
    eval_ae(model_no_bfw)
    eval_ae(model_bfw)

def report():
    c = canvas.Canvas("autoencoder_example.pdf", pagesize=letter)
    width, height = letter
    cursor = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, cursor, "Results from Autoencoder Proof of Concept with MNIST Dataset")
    cursor -= 25
    c.setFont("Helvetica", 12)
    c.drawString(50, cursor, "Comparison of Loss with Number of Iterations/Epochs with and without BFW")
    cursor -= 10
    image_height = 150
    cursor -= image_height
    c.drawImage("training_data\\loss_per_epoch_bfw.png", width/2 - 225, cursor, width=200, height=image_height)
    c.drawImage("training_data\\loss_per_epoch_no_bfw.png", width/2 + 25, cursor, width=200, height=image_height)
    cursor -= 25
    c.drawString(50, cursor, "Example Reconstruction of MNIST Digits without BFW")
    image_height = 100
    cursor -= image_height + 5
    c.drawImage("evaluation_data\\reconstructed_images_no_bfw.png", 50, cursor, width=400, height=image_height)
    cursor -= 10
    c.setFont("Helvetica", 8)
    c.drawString(75, cursor, "Even though model only trained on digit 2, we see decent reconstruction for all digits")
    cursor -= 15
    c.setFont("Helvetica", 12)
    c.drawString(50, cursor, "Example Reconstruction of MNIST Digits with BFW")
    cursor -= image_height + 5
    c.drawImage("evaluation_data\\reconstructed_images_bfw.png", 50, cursor, width=400, height=image_height)
    cursor -= 10
    c.setFont("Helvetica", 8)
    c.drawString(75, cursor, "Using BFW, we can see there is poor reconstruction for digits other than 2.")
    cursor -= 10
    c.drawString(75, cursor, "This is desired as it will allow us to distinguish between a 2 and any other digit based on reconstruction error.")
    cursor -= 15
    c.setFont("Helvetica", 12)
    c.drawString(50, cursor, "Comparison of ROC (Receiver Operating Characteristic) Curve with and without BFW")
    cursor -= 10
    image_height = 150
    cursor -= image_height
    c.drawImage("evaluation_data\\ROC_no_bfw.png", width/2 - 225, cursor, width=200, height=image_height)
    c.drawImage("evaluation_data\\ROC_bfw.png", width/2 + 25, cursor, width=200, height=image_height)
    cursor -= 10
    c.setFont("Helvetica", 8)
    c.drawString(75, cursor, "Larger Area Under Curve is desirable since it allows us to select an error threshhold with both low false positive rate and high true positive rate.")
    cursor -= 10
    c.drawString(75, cursor, "We see that the AUC improves when we use the BFW which is consistent with our observations of the reconstructed images")
    c.showPage()   # finish this page
    c.save()       # save file

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run training or evaluation")

    parser.add_argument(
        "-t", "--train",
        action="store_true",
        help="Train the model first"
    )
    parser.add_argument(
        "-e", "--eval",
        action="store_true",
        help="Run evaluation of trained model"
    )

    args = parser.parse_args()

    if args.train:
        train()
    if args.eval:
        evaluate()
    report()



# subprocess.run(["python", "basic_example_eval.py"])

