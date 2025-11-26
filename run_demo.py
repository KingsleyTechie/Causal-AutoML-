import numpy as np
from environment import CausalEnv
from automl_controller import AutoMLController

def collect_data(env, steps=500):
    data = []
    state = env.reset()
    for _ in range(steps):
        action = np.random.uniform(-0.5, 0.5, env.n)
        state, _ = env.step(action)
        data.append(state)
    return np.array(data)

def evaluate_policy(env, policy, episodes=20):
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        action = policy.compute_policy(state)
        _, reward = env.step(action)
        rewards.append(reward)
    return np.mean(rewards)

def main():
    env = CausalEnv(n_variables=5)
    dataset = collect_data(env)
    controller = AutoMLController(max_budget=2)
    policy = controller.select_policy(dataset)
    performance = evaluate_policy(env, policy)
    print(f"Zero-shot policy average reward: {performance}")

if __name__ == "__main__":
    main()
