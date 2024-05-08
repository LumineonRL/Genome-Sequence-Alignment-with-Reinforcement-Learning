# Gene Sequence Alignment with Reinforcement Learning

## Installation

- Python `3.10.6`. Some packages used in this project may not be available or may cause compatibility issues on different Python installations.
- CUDA `12.1`. `torch` dependencies in `dependencies.txt` may need to be modified for non-GPU installations if you can not fulfill this requirement. Note that training is lengthy in
this project even on a power CUDA-enabled GPU.
- Package dependencies are included in this repository's `dependencies.txt` folder. Please run  `pip install -r dependencies.txt` to install on your venv.

## Running

The application can be ran by using the `python main.py` command from your terminal of choice.

`python main.py --help` can be ran for a detailed list of all available arguments.

These arguments are as follows:

- `--agent [DQN|A2C|PPO|random]`: Specifies the reinforcement learning agent to use. Defaults to DQN. Random will make entirely random selections.
- `--net-arch [config1|config2|...]`: Specifies MLP network architectures as defined in `configurations.json.`. Defaults to the first architecture defined in that file.
- `--train-steps [number]`: Specifies the number of training steps. Defaults to 10000.
- `--evaluate`: Flag to run evaluation after training.

## Structure

The following is a list of the modules contained within this repository. Many of the modules build on top of each other, so this will begin at the lowest level and work its way up.

### Gene.py

Defines a `Gene` object that is used throughout all steps of training. Controls the max length of a sequence, generating sequences, types of mutations that can occur in a sequence, as well as implementations for each of those mutations.

This module also contains a helper class, `SequenceProcessor`, that takes in a `Gene` object and processes (via padding and one-hot encoding) so that the environment (discussed below) can handle it.

### Environment.py

Contains the `GeneSequenceEnv` class, which is tasked with creating an environment for an `Agent` to observe two formatted `Gene` objects. Some core components of the environment as follows:

- Observation space: The current observable state of the sequence alignment task.
- Action space: The actions that an agent traversing the environment can take.
- Reset: Sets the environment back to an initial state by randomly generating a gene, mutating it, and processing it for the environment. Also resets any reward received.
- Step: Take an action as defined by the action space to progress the environment to a different state.
- Reward: Structure to provide feedback to an agent to let them know if they're performing a "good job" or not when traversing.

### Agents.py

Contains implementations of DQN, A2C, and PPO agents, as well as a random agent that performs entirely random actions without attempting to learn and optimal traversal strategy.

### main.py

The entry point for running the environment and its agents.

### configurations.json

Contains different user-defined MLP network architectures. The size of the list indicates the number of hidden layers and each element represents the number of nodes in that layer. Users are encouraged to experiment with various network configurations. The `--net-arch` argument can be used in `main.py` to specify which configuration to use during training.

### dependencies.txt

Contains all package dependencies used in this project.