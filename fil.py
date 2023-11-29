import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.io import loadmat
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Import necessary libraries

# Load Fashion MNIST dataset using scipy
fashion_mnist_data = loadmat('fashion_mnist_data.mat')

# Extract training and testing data along with labels
train_images = fashion_mnist_data['train_images']
train_labels = fashion_mnist_data['train_labels'].flatten()
test_images = fashion_mnist_data['test_images']
test_labels = fashion_mnist_data['test_labels'].flatten()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten images
train_images_flat = train_images.reshape((train_images.shape[0], -1))
test_images_flat = test_images.reshape((test_images.shape[0], -1))

# Define class names for each label
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Create a Sequential model using Keras
model = Sequential([
    Dense(units=128, activation='relu', input_dim=784),
    Dense(units=128, activation='relu'),
    Dense(units=10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# Train the model
model.fit(train_images_flat, train_labels, epochs=5)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images_flat, test_labels)
print("\nTest accuracy: " + str(test_acc))

# Build a Tkinter window for visualization
window = tk.Tk()
window.title("Fashion MNIST Classification with Neural Network")
window.geometry("800x600")

# Function to choose a file using file dialog
def choose_file():
    file_path = filedialog.askopenfilename()
    print(f'Selected file: {file_path}')

# Create a button to choose a file
file_button = tk.Button(window, text="Choose File", command=choose_file)
file_button.pack()

# Start the Tkinter main loop
window.mainloop()
