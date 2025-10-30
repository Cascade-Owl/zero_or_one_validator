import cv2
import nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import preprocess_dataset, preprocess_image
import tkinter as tk
from tkinter import messagebox


def validate(validation_set):

    np.random.seed(39)

    X, y = preprocess_dataset(validation_set)

    dense1 = nn.Dense_layer(784, 128, weights_regularizer=1.2e-10)
    dense2 = nn.Dense_layer(128, 64, weights_regularizer=0)
    dense3 = nn.Dense_layer(64, 16, weights_regularizer=0)
    dense4 = nn.Dense_layer(16, 1, weights_regularizer=0)

    dropout1 = nn.Dropout(0.05)
    dropout2 = nn.Dropout(0.0)
    dropout3 = nn.Dropout(0.0)
    dropout4 = nn.Dropout(0.0)

    activation1 = nn.ReLU_activation()
    activation2 = nn.ReLU_activation()
    activation3 = nn.ReLU_activation()
    activation4 = nn.Sigmoid_activation()

    loss = nn.BinaryCrossEntropy_loss()

    params = np.load("zero_or_one_validator_model.npz")

    dense1.weights, dense1.biases = params["w1"], params["b1"]
    dense2.weights, dense2.biases = params["w2"], params["b2"]
    dense3.weights, dense3.biases = params["w3"], params["b3"]
    dense4.weights, dense4.biases = params["w4"], params["b4"]

    dense1.forward(X)
    dropout1.forward(dense1.output)
    activation1.forward(dropout1.output)

    dense2.forward(activation1.output)
    dropout2.forward(dense2.output)
    activation2.forward(dropout2.output)  

    dense3.forward(activation2.output)
    dropout3.forward(dense3.output)
    activation3.forward(dropout3.output)

    dense4.forward(activation3.output)
    dropout4.forward(dense4.output)
    activation4.forward(dropout4.output)

    normal_loss = loss.calculate(activation4.output, y)
    reg_loss = loss.regularization_loss(dense1) + loss.regularization_loss(dense2) + loss.regularization_loss(dense3)

    total_loss = normal_loss + reg_loss
    predictions = (activation4.output > 0.5).astype(int)
    acc = np.mean(y == predictions)
  
    print(f"Validation acc: {acc:.3f}, Validation loss: {total_loss:.3f}")


def predict(img_path):
    x = preprocess_image(img_path)

    dense1 = nn.Dense_layer(784, 128, weights_regularizer=1.2e-10)
    dense2 = nn.Dense_layer(128, 64, weights_regularizer=0)
    dense3 = nn.Dense_layer(64, 16, weights_regularizer=0)
    dense4 = nn.Dense_layer(16, 1, weights_regularizer=0)

    dropout1 = nn.Dropout(0.05)
    dropout2 = nn.Dropout(0.0)
    dropout3 = nn.Dropout(0.0)
    dropout4 = nn.Dropout(0.0)

    activation1 = nn.ReLU_activation()
    activation2 = nn.ReLU_activation()
    activation3 = nn.ReLU_activation()
    activation4 = nn.Sigmoid_activation()

    params = np.load("zero_or_one_validator_model.npz")

    dense1.weights, dense1.biases = params["w1"], params["b1"]
    dense2.weights, dense2.biases = params["w2"], params["b2"]
    dense3.weights, dense3.biases = params["w3"], params["b3"]
    dense4.weights, dense4.biases = params["w4"], params["b4"]

    dense1.forward(X)
    dropout1.forward(dense1.output)
    activation1.forward(dropout1.output)

    dense2.forward(activation1.output)
    dropout2.forward(dense2.output)
    activation2.forward(dropout2.output)  

    dense3.forward(activation2.output)
    dropout3.forward(dense3.output)
    activation3.forward(dropout3.output)

    dense4.forward(activation3.output)
    dropout4.forward(dense4.output)
    activation4.forward(dropout4.output)

    img_color = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis("off")
    plt.show()

    root = tk.Tk()
    root.withdraw()

    if activation4.output > 0.5:
        messagebox.showinfo("Number 0 is detected in the image!")

    else:
        messagebox.showwarning("Number 1 is detected in the image!")
