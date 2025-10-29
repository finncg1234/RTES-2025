## This Repo

There are currently two main subfolders to check out:
- autoencoder_example: proof of concept for autoencoder architecture using MNIST dataset
- autoencoder_can:     workspace for getting an autoencoder one-class classifier to detect anomalies on CAN bus

There are also several files to look at in the root of this project:

- The annotated bibliography contains descriptions and citations for the literatures I have used in developing this project
- The Gantt Chart lays out the expectations for what will be accomplished by each project checkpoint.
- the CAN-architecture image shows the hardware layout that will be used later in the project
- ProjectReport.md is a living document that has my latest writing on the scope and impact of this project. The aim is that this will become the final report for this project.


### autoencoder_example

There are 4 python scripts in this folder: for defining the neural net, training, evaluation and orchestrating the entire example project.

To run this example, you will need python with a few machine learning libraries installed, the easiest way to get these may just be to run the program and see what python prompts you for install. You can run this example with the following command.
```bash
python.exe .\autoencoder_example.py -h
usage: autoencoder_example.py [-h] [-t] [-e]

Run training or evaluation

options:
  -h, --help   show this help message and exit
  -t, --train  Train the model first
  -e, --eval   Run evaluation of trained model
```

That is, if you have just cloned the repo, you will need to run both the training and evaluation. This will create some charts and print it all nicely into a pdf.

```bash
# from within \autoencoder_example folder
python.exe .\autoencoder_example.py -t -e
```
**NOTE**: The epoch count is defined in the autoencoder_train.py script. This defines how many times the training will iterate over the entire training set, so adjust as necessary if your hardware is limited. The intended outcome and conclusions drawn for the effectiveness of the BFW may only be visible for large number of epochs, but these results are preserved in the autoencoder_example_1000_epochs.pdf.
### autoencoder_can

This is still a work in progress, but to run the current setup, simply execute the can_main.py script.

``` bash
# from \autoencoder_can folder
python.exe .\can_main.py
```

There are no command line parameters to pass in although that will be coming in the next checkpoint.