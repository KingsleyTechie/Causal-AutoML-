import numpy as np

class ZeroShotPolicySynthesizer:
    def __init__(self, causal_matrix):
        self.W = causal_matrix

    def compute_policy(self, state):
        target = np.argmax(np.sum(np.abs(self.W), axis=0))
        direction = -np.sign(state[target])
        magnitude = min(1.0, max(0.1, np.abs(state[target])))
        return direction * magnitude
