# Intelligent Systems: Machine Learning and Optimisation (IMLO) project

University second year project involving the creation of a neural network to classify flowers based off the Flowers-102 dataset

## What's included

A python file (IMLO-code.py) containing all the code needed to either load an existing model or train an entirely new model. 

A pdf (IMLO-report.pdf) containing a report on the project which includes:
- An introduction giving details about what a Neural Network is and a brief discussion on its history
- An explaination about the Neural Network I have created, including details on the loss function and the network layers
- Results and Evaluation about my network discussing the developement and final accuracy reached.

## How to run the python file
The below information is also provided in the requirements.txt file, but also displayed here for convenience.

#### Prerequisites
These python packages must be installed prior to running the code.
- code
- torch
- os
- numpy
- matplotlib
- torchvision

This code should be run on a GPU
### To load in the preexisting, 64.5% model
#### From an unchanged version of the IMLO-code.py file:
- Run the file

#### If the file has been used to train the model previously:
- go the end of the file in the `if __name__ == "__main__":` section.
- ensure the line `net = train_new_model()` is commented out.
- ensure the line `net = load_exisiting_model("best_model.pt")` is NOT commented out.
- Run the file.

#### Expected results:
- If the Flowers102 dataset is not present in the folder, this will be downloaded which may take some time.

- Then 'Model successfully loaded' should be seen in the terminal, indicating the model has been found and loaded.

- Finally 'Accuracy on test data:  64.46576679134819 %' should be printed in the terminal, with the number showing the accuracy.


### To train the model from scratch
- go the end of the file in the `if __name__ == "__main__":` section.
- uncomment the line `net = train_new_model()`.
- comment out the line `net = load_exisiting_model("best_model.pt")`.
- run the file.

#### Expected results:
- If the Flowers102 dataset is not present in the folder, this will be downloaded which may take some time.

- Then the training will begin. This is set to run for 200 epochs, which should take aproximately 2 hours and 15 minutes when running on a PC with a RTX 3070 GPU with 16GB RAM.

- Throughout the training the following line will be printed for every epoch:
	Epoch []  -   Lr: []  -   Training Loss: []  -   Validation Loss: []  -   Val Accuracy: []

- Once training is finished, 'Finished Training' will be printed.

- Finally 'Accuracy on test data:  [] %' should be printed in the terminal, with the [] showing the accuracy.
