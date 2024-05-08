import argparse
import json
from Environment import GeneSequenceEnv
from Agents import DQNAgent, A2CAgent, PPOAgent

def main():
    parser = argparse.ArgumentParser(description="Run reinforcement learning models on a gene sequence environment.")
    
    # Load network architecture configuration
    with open('architectures.json', 'r', encoding="UTF-8") as f:
        architectures = json.load(f)
    
    default_arch = next(iter(architectures))  # Get the first key in the dictionary

    parser.add_argument("--agent", type=str, choices=['dqn', 'a2c', 'ppo', 'random'], default='dqn',
                        help="Select the agent to use.")
    parser.add_argument("--arch", type=str, default=default_arch,
                        help="Neural network architecture to use (defined in architectures.json).")
    parser.add_argument("--timesteps", type=int, default=10000,
                        help="Total training timesteps.")
    args = parser.parse_args()

    env = GeneSequenceEnv(max_length=10)
    net_arch = architectures[args.arch]

    if args.agent == 'dqn':
        agent = DQNAgent(env, net_arch)
    elif args.agent == 'a2c':
        agent = A2CAgent(env, net_arch)
    elif args.agent == 'ppo':
        agent = PPOAgent(env, net_arch)

    agent.train(total_timesteps=args.timesteps)

if __name__ == "__main__":
    main()
