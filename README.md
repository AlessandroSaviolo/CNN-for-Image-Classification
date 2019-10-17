The purpose of this document is to present and analyze the methodology used to solve the Kaggle challenge for the first Homework.
The methodology consists of 4 steps:
1. Hyperparameters tuning
2. Data preprocessing
3. Building the model architecture and fitting the model with the training data
4. Computing predictions on the test data.

At first, the dataset is augmented and balanced, in order to prevent overfitting and provide some bias towards the minority classes
while training the model. The data is splitted in training and validation, in order to obtain an estimate of the final accuracy of the
model. Then, the model architecture is fixed and it is fitted with the augmentated training data, by monitoring the validation accuracy.
Once the model is built and fitted, the best weight computed during the training phase are restored and the final model is used for
predicting the test data.
