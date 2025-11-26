import numpy as np
from causal_discovery import CausalDiscovery
from policy_synthesizer import ZeroShotPolicySynthesizer

class AutoMLController:
    def __init__(self, max_budget=3):
        self.max_budget = max_budget

    def select_policy(self, dataset):
        candidates = [100, 300, 500]  # Number of iterations for causal discovery
        best_policy = None
        best_score = -np.inf
        used_budget = 0
        for iter_count in candidates:
            if used_budget + 1 > self.max_budget:
                break
            cd = CausalDiscovery(max_iter=iter_count)
            W = cd.discover(dataset)
            policy = ZeroShotPolicySynthesizer(W)
            score = -np.sum(np.abs(W))
            if score > best_score:
                best_score = score
                best_policy = policy
            used_budget += 1
        return best_policy
