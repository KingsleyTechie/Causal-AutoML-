import numpy as np

class CausalDiscovery:
    def __init__(self, max_iter=1000, lr=0.01):
        self.max_iter = max_iter
        self.lr = lr

    def discover(self, data):
        n = data.shape[1]
        W = np.random.uniform(0, 0.1, (n, n))
        for _ in range(self.max_iter):
            pred = data @ W
            loss = np.mean((pred - data) ** 2)
            grad = 2 * data.T @ (pred - data) / data.shape[0]
            W -= self.lr * grad
        np.fill_diagonal(W, 0)
        return W
