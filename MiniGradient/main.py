import numpy as np
import matplotlib.pyplot as plt

# Veri Seti Oluşturuldu
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)

# Model Parametreleri
w = np.random.randn(2, 1)

# Mini Gradyan İnişi Fonksiyonu
def mini_batch_gradient_descent(X, Y, w, learning_rate = 0.01,epochs = 100, batch_size = 10):
    m = len(Y)
    loss_values = []

    for epoch in range (epochs):
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X[shuffled_indices]
        Y_shuffled = Y[shuffled_indices]

        for i in range (0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]

            gradient = 2/batch_size * X_batch.T.dot(X_batch.dot(w) - Y_batch)
            w = w - learning_rate * gradient

            loss_value = np.mean((X.dot(w) - Y) ** 2)
            loss_values.append(loss_value)

    return w, loss_values

x_b = np.c_[np.ones((len(X), 1)), X]

w, loss_values = mini_batch_gradient_descent(x_b, Y, w, learning_rate = 0.01, epochs = 100, batch_size = 10)

plt.plot(loss_values)
plt.title("Mini Batch Gradient Descent")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()
