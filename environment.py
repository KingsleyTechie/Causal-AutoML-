import numpy as np

class CausalEnv:
    def __init__(self, n_variables=5):
        self.n = n_variables
        self.graph = self._generate_random_graph()
        
    def _generate_random_graph(self):
        W = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j and np.random.rand() < 0.3:
                    W[i, j] = np.random.uniform(0.5, 1.0)
        return W

    def reset(self):
        self.state = np.random.uniform(-1, 1, self.n)
        return self.state

    def step(self, action):
        noise = np.random.normal(0, 0.1, self.n)
        self.state = self.state @ self.graph + action + noise
        reward = -np.sum(np.abs(self.state))
        return self.state, reward
