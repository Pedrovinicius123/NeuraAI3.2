from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from alive_progress import alive_bar
from neuratron import Model

from brain import generate_brain_, feed_forward, backpropagation

import matplotlib.pyplot as plt
import numpy as np

X, y = make_regression(n_samples=300, random_state=98, n_features=3, noise=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)
brain, inp, out = generate_brain_(10, 4)

outputs = feed_forward(brain, inp, X_train)
backpropagation(brain, out)

print(outputs)
