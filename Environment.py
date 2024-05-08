import random
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Tuple, Box
from Gene import Gene, SequenceProcessor


class GeneSequenceEnv(gym.Env):
    def __init__(self, max_length=100, max_steps=10):
        super(GeneSequenceEnv, self).__init__()
        self.max_length = max_length
        self.max_steps = max_steps
        self.sequence_processor = SequenceProcessor(self.max_length)
        self.action_space = Tuple((
            Discrete(3),  # Mutation Type: 0 - Substitution, 1 - Insertion, 2 - Deletion
            Discrete(self.max_length),  # Location
            MultiDiscrete([5] * self.max_length)  # Bases vector, one-hot encoding
        ))
        self.observation_space = Box(low=0, high=1, shape=(self.max_length, 5, 2), dtype=np.float32)
        self.original_sequence = None
        self.mutated_sequence = None
        self.processed_sequences = None
        self.one_hot_original = None
        self.one_hot_mutated = None
        self.current_reward = 0
        self.action_history = []
        self.current_step = 0 

    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode with valid mutations resulting in a distinct sequence."""
        np.random.seed(seed) if seed is not None else None
        attempts = 0
        max_attempts = 10  # Prevent infinite loops by limiting mutation attempts

        while True:
            gene = Gene(self.max_length)
            gene.generate()  # Generate the initial sequence
            _, mutation_events, mutation_details = gene.introduce_mutations()  # Apply mutations

            if list(gene.sequence) != list(gene.original_sequence):
                break  # Exit loop if mutated sequence is different from the original

            attempts += 1
            if attempts >= max_attempts:
                print("Failed to produce a distinct mutated sequence after several attempts.")
                break  # Exit loop to prevent infinite retrying

        # Process sequences with consistent padding and encoding
        padding_method = random.choice(self.sequence_processor.padding_methods)
        padded_original = self.sequence_processor.pad_sequence(list(gene.original_sequence), padding_method)
        padded_mutated = self.sequence_processor.pad_sequence(list(gene.sequence), padding_method)

        self.one_hot_original = self.sequence_processor.one_hot_encode(padded_original)
        self.one_hot_mutated = self.sequence_processor.one_hot_encode(padded_mutated)

        self.original_sequence = gene.original_sequence
        self.mutated_sequence = gene.sequence  # Ensure these are updated for use in step method
        self.current_step = 0
        self.action_history = []

        return (self.one_hot_original, self.one_hot_mutated)

    def step(self, action):
        self.current_step += 1
        if self.current_step > self.max_steps:
            return self.processed_sequences, -1000, True, {}

        mutation_type, location, bases_vector = action
        self.apply_mutation(mutation_type, location, bases_vector)
        done = self.check_if_done()
        reward = self.compute_reward() if done else -1
        self.current_reward += reward
        return self.processed_sequences, reward, done, {}

    def apply_mutation(self, mutation_type, location, bases_vector):
        bases = self._decode_bases(bases_vector)
        if mutation_type == 0:
            self.mutated_sequence = self.apply_substitution(
                self.mutated_sequence, location, bases
            )
        elif mutation_type == 1:
            self.mutated_sequence = self.apply_insertion(
                self.mutated_sequence, location, bases
            )
        elif mutation_type == 2:
            self.mutated_sequence = self.apply_deletion(
                self.mutated_sequence, location, len(bases)
            )
        self.processed_sequences = self.sequence_processor.process_sequences(
            self.original_sequence, self.mutated_sequence
        )

    def apply_substitution(self, sequence, location, bases):
        return sequence[:location] + "".join(bases) + sequence[location + len(bases) :]

    def apply_insertion(self, sequence, location, bases):
        return sequence[:location] + "".join(bases) + sequence[location:]

    def apply_deletion(self, sequence, location, length):
        return sequence[:location] + sequence[location + length :]

    def check_if_done(self):
        return np.array_equal(self.processed_sequences[0], self.processed_sequences[1])

    def compute_reward(self):
        if not self.check_if_done():
            return -1  # Penalize each step that does not lead to a solution
        return 100  # Base reward for solving

    def _decode_bases(self, bases_vector):
        base_to_index = {0: "A", 1: "C", 2: "G", 3: "T", 4: "N"}
        return "".join(
            base_to_index.get(np.argmax(b), "?") for b in bases_vector if np.any(b)
        )

    def render(self, mode="human"):
        if mode == "human":
            original = "".join(self._decode_bases(self.one_hot_original))
            mutated = "".join(self._decode_bases(self.one_hot_mutated))
            print("Original Sequence: ", original)
            print("Mutated Sequence: ", mutated)
            print(f"Current Reward: {self.current_reward}")
