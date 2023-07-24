import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import feature_column
from tensorflow.python.keras import layers
from sklearn.model_selection import train_test_split

tf.enable_eager_execution()

dataframe = pd.read_csv('digikala.csv')
dataframe = dataframe.values

x = dataframe[:, 0:4]
y = dataframe[:, 4]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

model = tf.keras.Sequential([
  layers.Dense(128, activation='relu', input_shape=(4,)),
  layers.Dense(128, activation='relu'),
  layers.Dense(1),
])

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'],
              run_eagerly=True)

model.fit(x_train, y_train, epochs=150)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy", accuracy)

