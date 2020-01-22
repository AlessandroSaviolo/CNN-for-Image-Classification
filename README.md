# Convolutional Neural Network for Image Classification

This project is part of a series of projects for the course _Selected Topics in Visual Recognition using Deep Learning_ that I attended during my exchange program at National Chiao Tung University (Taiwan). See `task.pdf` for the details of the assignment. See `report.pdf` for the report containing the representation and the analysis of the produced results.

The purpose of this project is to implement a Convolutional Neural Network for Image Classification. The methodology implemented consists of 4 steps:

1. Hyperparameters tuning

2. Data preprocessing

3. Building the model architecture and fitting the model with the training data

4. Computing predictions on the test data

At first, the dataset is augmented and balanced, in order to prevent overfitting and provide some bias towards the minority classes
while training the model. The data is splitted in training and validation, in order to obtain an estimate of the final accuracy of the
model. Then, the model architecture is fixed and it is fitted with the augmentated training data, by monitoring the validation accuracy.
Once the model is built and fitted, the best weight computed during the training phase are restored and the final model is used for
predicting the test data.

The [submission.csv](https://drive.google.com/open?id=18reV_So6zCEGOlY3gWqmHQKTz4_-OZQ4) file containing the predictions on the test set is provided.

The [weights.best.hdf5](https://drive.google.com/open?id=1VcB9HDtPdH0WwP29ygxK22wLrukyUrr-) file containing the weights of the final model is provided.

## 1. Dataset

- [Dataset](https://drive.google.com/open?id=1gwLcH2zjSW55aou6mYZ7BbeEE21rGAxD)

## 2. Project Structure

- `main.py` : main function, use it to change task ('r' or 'c') and hyperparameters (i.e., learning rate, number of epochs)

- `model.py` : contains the regression and classification neural network models

- `regression.py` : run regression using the relative model from model.py, use it to change the hyperparameters of the model (i.e., number of neurons)

- `classification.py` : run classification using the relative model from model.py, use it to change the hyperparameters of the model (i.e., number of neurons)

- `utilities.py` : contains plot functions and common functions among the different files (i.e., load dataset which is used both for regression and classification)

- `deep_classification.py` : deep classifier used to plot the distribution of latent features at different training stages. It contains also the deep model
