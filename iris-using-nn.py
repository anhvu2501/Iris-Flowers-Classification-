# Iris flower classification using Neural Networks 

from email.charset import add_charset
from unittest import result
import numpy as np
import pandas as pd
from simplejson import load
import tensorflow as tf
from keras import backend
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load data
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

# This data is already converted to the set of numbers (float for X and integer for y) 
# print(iris_data)
# print(X)
# print(y)

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train = keras.utils.to_categorical(y_train, 3)
y_test = keras.utils.to_categorical(y_test, 3)

# Create the Model
backend.clear_session() # Clear the Keras Backend

# Set random seed values for Numpy and Tensorflow
np.random.seed(42)  
tf.random.set_seed(42)

# Use Sequential API to create the model
# 7 layers total
model = models.Sequential()
# dense layer is just a regular densele-connected Neural Networks layer
model.add(layers.Dense(300, input_shape=(4,), activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(100, activation='relu', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(500, activation='relu', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(3, activation='softmax'))

# Compile the Model
model.compile(optimizer=optimizers.Adam(learning_rate= 0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# print(X_train.shape)
# print(y_train.shape)

# Train the model
history = model.fit(X_train, y_train, batch_size=5, epochs=100)

# Once the training is complete, plot the loss and accuracy metrics of the model
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()

# Evaluate the Model
results = model.evaluate(X_test, y_test) # return the loss value and metrics value for the model
print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))

class_names = iris_data.target_names
# Get the features for the first 5 test samples
X_new = X_test[:5]
# Predict the classes for those 5 samples
y_pred = model.predict_classes(X_new)
print(np.array(class_names)[y_pred])
# Get the actual class names of the first 5 samples
y_new = y_test[:5]
print(np.array(class_names)[y_new])