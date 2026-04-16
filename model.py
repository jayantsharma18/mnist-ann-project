from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
import numpy as np

class HistoryObject:
    def __init__(self, loss, val_loss):
        self.history = {
            'loss': loss,
            'val_loss': val_loss
        }

def train_model():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    x_train, x_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    model = MLPClassifier(
        hidden_layer_sizes=(64, 64, 32),
        activation='relu',
        learning_rate_init=1e-4,
        max_iter=15,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )

    losses = []
    val_losses = []
    for i in range(1, 16):
        model.max_iter = i
        model.warm_start = True
        model.fit(x_train, y_train)
        losses.append(model.loss_)
        val_losses.append(model.best_validation_score_)

    history = HistoryObject(losses, val_losses)
    x_test_reshaped = x_test.reshape(-1, 28, 28)
    return model, history, x_test_reshaped, y_test
