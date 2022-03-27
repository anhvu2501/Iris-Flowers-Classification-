# Iris flower classification using Neural Networks 

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend
from matplotlib import pyplot as plt
from pandas import read_csv
from simplejson import load
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers

### Load data from sklearn (The data is already in the form the set of appropriate numbers)
iris_data = load_iris()
X = iris_data.data
y = iris_data.target


### Load raw data from kaggle. The data is not processed yet
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
# names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# dataset = read_csv(url, names=names)
# array = dataset.values
# X = array[:, 0:4] # take all values in each row from column 0 to 4 => input features 
# y = array[:, 4]

# # Process to encode the raw y into number representation of y
# # And also need to convert X from type = 'Object' into type 'float'
# X = X.astype(float)
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y)


# This data is already converted to the set of numbers (float for X and integer for y) 
# print(iris_data)
# print(X)
# print(y)

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Convert y_train and y_test into one-hot representation form (from integer number to one-hot)
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

### NOTE
# If we dont convert y_train and y_test into one-hot form. We can use sparese_categorical_crossentropy instead #
# model.compile(optimizer=optimizers.Adam(learning_rate= 0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


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
predict_x = model.predict(X_test) 
classes_x = np.argmax(predict_x,axis=1)
print(np.array(class_names)[classes_x])
# Get the actual class names of the first 5 samples
y_new = y_test[:5]
print(np.array(class_names)[y_new])
