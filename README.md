# Number Classifier

A drawing pad app that tries to guess what digit you wrote onto the canvas.

## DigitClassifier

The classifying is done by the `DigitClassifier` class. 
This class makes use of a Convolutional Neural Network which has been pretrained with the MNIST dataset.

## The GUI

The gui is written with tkinter, and when the `predict` button is pushed: a snapshot of the canvas is taken, converted to a png, then that png is processed by the classifier.
