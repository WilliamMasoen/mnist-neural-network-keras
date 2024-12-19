import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

# Load the MNIST dataset for digits classification (training and testing sets)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Visualize the first 10 training images (digits) from the dataset
plt.figure(figsize=(5, 2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.imshow(x_train[i], cmap='gray')  # Display image in grayscale
    plt.axis('off')  # Hide axis labels
plt.show()

# Normalize the pixel values to range from 0 to 1 by dividing by 255
x_train = x_train / 255
x_test = x_test / 255

# Create a Sequential model to stack layers in order
model = keras.models.Sequential()

# Flatten the 28x28 images into 1D arrays (784 elements)
model.add(keras.layers.Flatten(input_shape=[28, 28]))

# First hidden layer with 300 neurons and ReLU activation function
model.add(keras.layers.Dense(300, activation="relu"))

# Second hidden layer with 100 neurons and ReLU activation function
model.add(keras.layers.Dense(100, activation="relu"))

# Output layer with 10 neurons (one per class) and softmax activation for classification
model.add(keras.layers.Dense(10, activation="softmax"))

# Print model summary to view the layers and number of parameters
model.summary()

# Compile the model with sparse categorical crossentropy loss and stochastic gradient descent optimizer
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd")

# Train the model for 20 epochs using the training data
model.fit(x_train, y_train, epochs=20)

# Predict the labels of the first 10 test instances
y_pred = model.predict(x_test)

# Visualize the predicted labels and display the corresponding images
plt.figure(figsize=(5, 2))
for i in range(10):
    plt.subplot(5, 2, i + 1)
    plt.title('Predicted: ' + str(np.argmax(y_pred[i])))  # Display predicted label
    plt.imshow(x_test[i], cmap='gray')  # Display image in grayscale
    plt.axis('off')  # Hide axis labels
plt.show()
