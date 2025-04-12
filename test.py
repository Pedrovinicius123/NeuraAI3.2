from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from alive_progress import alive_bar
from neuratron import Model

import matplotlib.pyplot as plt
import numpy as np

X, y = make_regression(n_samples=300, random_state=98, n_features=3, n_targets=1, noise=20)
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size=0.2)
model = Model(shape=(20, 20), lr=0.001)
epochs = 40000

with alive_bar(epochs) as bar:
    anterior_loss = None
    for i in range(epochs):
        output = model.feed_forward(X_train, 1)
        loss = mean_squared_error(output, y_train)
        score = r2_score(output, y_train)
        model.backward_pass(y_train)

        if loss < 1000 and anterior_loss and anterior_loss < loss:
            break

        anterior_loss = loss
        print(loss, score)
        bar()