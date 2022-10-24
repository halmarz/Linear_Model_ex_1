# Linear_Model_ex_1


import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

x = np.random.random((2, 3))
y = np.random.randint(0, 2, (2, 2))

model = tf.keras.experimental.LinearModel()
model.compile(optimizer='sgd', loss='mse')
model.fit(x, y, epochs=5)
model.summary()
x = np.random.random((2, 3))
y = np.random.randint(0, 2, (2, 2))

model = tf.keras.experimental.LinearModel()
model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
model.fit(x, y, epochs=5)
model.summary()


def dana_fn(**kwargs):
    print("Liczba przekazanych parametrów:",len(kwargs))
    
    for key, item in kwargs.items():
        print ("Klucz:", key, "Wartość:", item)
dana_fn(a=1,b=2,c=3,d=18, e=12)
