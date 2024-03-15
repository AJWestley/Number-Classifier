# Number Classifier

A drawing pad app that tries to guess what digit you wrote onto the canvas.

## DigitClassifier

The classifying is done by the `DigitClassifier` class. 
This class makes use of a logistic regression model which has been pretrained with the MNIST dataset.

Overall, it works like a regular logistic regression model, except that it is made to take in a numpy array containing greyscale png data. 
It then flattens the data and scales it to suit the format of the mnist data.

## The GUI

The gui is written with tkinter, and when the `predict` button is pushed: a snapshot of the canvas is taken, converted to a png, then that png is processed by the classifier.
