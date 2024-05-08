import torch
import numpy as np
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.evaluation import evaluate_policy

class RandomAgent:
    """Random Agent that selects actions randomly from the action space."""
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self):
        """Select a random action from the action space."""
        mutation_type = self.action_space[0].sample()
        location = self.action_space[1].sample()
        # Generate a one-hot encoded vector for bases
        num_bases = np.random.randint(1, self.action_space[2].nvec.size + 1)
        bases_vector = np.zeros((num_bases, 5))
        for i in range(num_bases):
            base_index = np.random.randint(0, 5)
            bases_vector[i, base_index] = 1
        return (mutation_type, location, bases_vector)

class BaseAgent:
    """Base class for all agents, providing save, load, and evaluate functionalities."""
    def __init__(self, model):
        self.model = model

    def train(self, total_timesteps=10000, callback=None):
        """Train the agent."""
        self.model.learn(total_timesteps=total_timesteps, callback=callback)

    def save(self, path):
        """Save the trained model."""
        self.model.save(path)

    def load(self, path, env):
        """Load a trained model."""
        self.model = self.model.__class__.load(path, env=env)

    def evaluate(self, env, n_episodes=10):
        """Evaluate the trained model."""
        mean_reward, std_reward = evaluate_policy(self.model, env, n_eval_episodes=n_episodes)
        print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

class DQNAgent(BaseAgent):
    """DQN Agent with specified network architecture."""
    def __init__(self, env, net_arch):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = DQN("MlpPolicy", env, policy_kwargs={"net_arch": net_arch}, verbose=1, tensorboard_log="./dqn_gene_tensorboard/", device=device)
        super().__init__(model)

class A2CAgent(BaseAgent):
    """A2C Agent with specified network architecture."""
    def __init__(self, env, net_arch):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = A2C("MlpPolicy", env, policy_kwargs={"net_arch": net_arch}, verbose=1, tensorboard_log="./a2c_gene_tensorboard/", device=device)
        super().__init__(model)

class PPOAgent(BaseAgent):
    """PPO Agent with specified network architecture."""
    def __init__(self, env, net_arch):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": net_arch}, verbose=1, tensorboard_log="./ppo_gene_tensorboard/", device=device)
        super().__init__(model)