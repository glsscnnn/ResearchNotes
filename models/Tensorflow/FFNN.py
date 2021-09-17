import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np

MLP = keras.Sequential([
    layers.Dense(128, activation="sigmoid", name="input_layer"),
    layers.Dense(32, activation="sigmoid", name="hl1"),
    layers.Dense(64, activation="sigmoid", name="hl2"),
    layers.Dense(64, activation="sigmoid", name="hl3"),
    layers.Dense(10, activation="sigmoid", name="output_layer"),
])

x = np.random.randn(128, 1) # fake data
print(MLP(x)) # predictions
