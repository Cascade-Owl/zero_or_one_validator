from prediction import validate, predict
from training import train
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from scipy import ndimage
import nn
import matplotlib.pyplot as plt

training_set_path = "./zero_or_one_training_set"
train(training_set_path)

validation_set_path = "./zero_or_one_validation_set"
validate(validation_set_path)

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import nn
import matplotlib.pyplot as plt


class DrawingApp:
    def __init__(self, canvas_size=400, brush_size=12):
        self.canvas_size = canvas_size
        self.brush_size = brush_size
        self.root = tk.Tk()
        self.root.title("Draw 0 or 1")

        self.canvas = tk.Canvas(
            self.root, width=canvas_size, height=canvas_size, bg="black"
        )
        self.canvas.pack()

        self.button_predict = tk.Button(self.root, text="Predict", command=self.predict)
        self.button_predict.pack(side=tk.LEFT, padx=10, pady=10)

        self.button_clear = tk.Button(self.root, text="Clear", command=self.clear)
        self.button_clear.pack(side=tk.RIGHT, padx=10, pady=10)

        self.button_show = tk.Button(
            self.root, text="Show Preprocessed", command=self.show_preprocessed
        )
        self.button_show.pack(pady=5)

        self.image = Image.new("L", (canvas_size, canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.params = np.load("zero_or_one_validator_model.npz")

        self.root.mainloop()

    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill=0)

    def preprocess(self):

        x = np.array(self.image, dtype=np.float32)  # convert to float

        coords = np.argwhere(x > 0)
        if coords.size == 0:
            return np.zeros((1, 28 * 28))  

        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        x_crop = x[y0 : y1 + 1, x0 : x1 + 1]

        img_pil = Image.fromarray(x_crop)
        img_pil.thumbnail((20, 20), Image.Resampling.LANCZOS)
        x_resized = np.array(img_pil)

        final_img = np.zeros((28, 28), dtype=np.float32)
        y_offset = (28 - x_resized.shape[0]) // 2
        x_offset = (28 - x_resized.shape[1]) // 2
        final_img[
            y_offset : y_offset + x_resized.shape[0],
            x_offset : x_offset + x_resized.shape[1],
        ] = x_resized

        final_img /= 255.0
        return final_img.reshape(1, 28 * 28)

    def predict(self):
        x = self.preprocess()

        dense1 = nn.Dense_layer(784, 128, weights_regularizer=1.2e-10)
        dense2 = nn.Dense_layer(128, 64, weights_regularizer=0)
        dense3 = nn.Dense_layer(64, 16, weights_regularizer=0)
        dense4 = nn.Dense_layer(16, 1, weights_regularizer=0)

        activation1 = nn.ReLU_activation()
        activation2 = nn.ReLU_activation()
        activation3 = nn.ReLU_activation()
        activation4 = nn.Sigmoid_activation()

        dense1.weights, dense1.biases = self.params["w1"], self.params["b1"]
        dense2.weights, dense2.biases = self.params["w2"], self.params["b2"]
        dense3.weights, dense3.biases = self.params["w3"], self.params["b3"]
        dense4.weights, dense4.biases = self.params["w4"], self.params["b4"]

        dense1.forward(x)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        activation3.forward(dense3.output)
        dense4.forward(activation3.output)
        activation4.forward(dense4.output)

        prob = activation4.output[0][0]
        if prob > 0.5:
            messagebox.showinfo("Prediction", f"Number 1 detected!")
        else:
            messagebox.showwarning("Prediction", f"Number 0 detected!")

    def show_preprocessed(self):
        
        x = self.preprocess()
        preprocessed_img = x.reshape(28, 28)
        original_img = np.array(self.image)

        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(original_img, cmap="gray")
        plt.title("Original Canvas")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(preprocessed_img, cmap="gray")
        plt.title("Preprocessed (28x28) Image")
        plt.axis("off")

        plt.show()


while True:
    choice = input("Choose input method: (D)raw or (F)ile path, (Q)uit: ").upper()

    if choice == "D":
        DrawingApp()
    elif choice == "F":
        img_path = input("Enter an image path to predict: ")
        predict(img_path)
    elif choice == "Q":
        break
    else:
        print("Invalid choice. Please select D, F, or Q.")
