import copy
import math

import numpy as np
from scipy.stats import beta


class Gene:
    def __init__(self, max_length):
        """
        Initialize the Gene object with the maximum sequence length.
        
        Parameters:
            max_length (int): Maximum length of the DNA sequence.
        """
        self.max_length = max_length
        self.original_sequence = None
        self.sequence = None
        self.mutation_types = ["Substitution", "Insertion", "Deletion"]
        # ,'Duplication', 'Inversion', 'Translocation']

    def generate(self):
        """
        Generate a random DNA sequence based on a beta distribution for its length.
        """
        length = self.determine_length()
        self.original_sequence = "".join(
            np.random.choice(["A", "C", "G", "T"], size=length)
        )
        self.sequence = self.original_sequence.ljust(self.max_length, "N")

    def determine_length(self):
        """
        Determine the sequence length using a scaled beta distribution.
        """
        beta_distribution = beta(5, 2)  # Beta distribution with alpha=5, beta=2
        length = np.floor(self.max_length * beta_distribution.rvs()).astype(int)
        return length

    def introduce_mutations(self):
        """
        Introduce a series of random mutations into the sequence.
        Reset the sequence to the original and repeat mutation attempts until a change is effected or a limit is reached.
        """
        attempts = 0
        mutation_details = []
        while True:
            attempts += 1
            mutated_sequence = copy.deepcopy(
                self.original_sequence
            )  # Start with a fresh copy of the original sequence
            m = len(mutated_sequence)  # Store the length of the mutated sequence
            num_mutations, mutation_events = self.select_mutations()

            # Apply each selected mutation and collect details
            for mutation in mutation_events:
                mutated_sequence, mutation_type, start, length = self.mutate(
                    mutated_sequence, mutation, m
                )
                mutation_details.append((mutation_type, start, length))
                m = len(mutated_sequence)  # Update m to the new length after mutation

            if mutated_sequence != self.original_sequence:
                self.sequence = mutated_sequence
                break
            elif attempts > 5:  # Limit attempts to prevent infinite loops
                break
        return num_mutations, mutation_events, mutation_details

    def select_mutations(self):
        """
        Randomly select the number and types of mutations to introduce.
        """
        num_mutations = np.random.randint(1, int(math.log10(self.max_length)) + 1)
        mutation_events = np.random.choice(self.mutation_types, size=num_mutations)
        return num_mutations, mutation_events

    def mutate(self, sequence, mutation_type, m):
        """
        Apply a mutation to the sequence based on the specified type, using current length m of the sequence.
        """
        if mutation_type == "Substitution":
            return self._substitution(sequence, m)
        elif mutation_type == "Insertion":
            return self._insertion(sequence, m)
        elif mutation_type == "Deletion":
            return self._deletion(sequence, m)
        elif mutation_type == "Duplication":
            pass  # To be implemented later
        elif mutation_type == "Inversion":
            pass  # To be implemented later
        elif mutation_type == "Translocation":
            pass  # To be implemented later
        return sequence

    def _substitution(self, sequence, m):
        """
        Perform a substitution mutation on the sequence.
        """
        # 1) Sample the length of the substitution, s
        s = np.random.geometric(p=0.9)  # s ~ Geom(0.9)
        while s > m:  # 1a) Ensure s <= m
            s = np.random.geometric(p=0.9)

        # 2) Determine the start location for the substitution
        start = np.random.randint(0, m - s)

        # 3) Determine new bases
        new_bases = np.random.choice(["A", "C", "G", "T"], size=s, replace=True)
        while any(
            new_bases[i] == sequence[start + i] for i in range(s)
        ):  # 3a) Ensure new bases are different
            new_bases = np.random.choice(["A", "C", "G", "T"], size=s, replace=True)

        # 4) Replace bases
        sequence = sequence[:start] + "".join(new_bases) + sequence[start + s :]

        # 5) Return the mutated sequence and mutation details
        return sequence, "Substitution", start, s

    def _insertion(self, sequence, m):
        """
        Perform an insertion mutation on the sequence.
        """
        # 1) Sample the length of the insertion, s
        s = np.random.geometric(p=0.9)  # s ~ Geom(0.9)
        while s > m:  # 1a) Ensure s <= m
            s = np.random.geometric(p=0.9)

        # 2) Determine the start location for the insertion
        start = np.random.randint(
            0, m + 1
        )  # Including m because insertion can be at the end

        # 3) Determine new bases
        new_bases = np.random.choice(["A", "C", "G", "T"], size=s, replace=True)

        # 4) Insert new base(s) at index
        sequence = sequence[:start] + "".join(new_bases) + sequence[start:]

        # 5) Return the mutated sequence and mutation details
        return sequence, "Insertion", start, s

    def _deletion(self, sequence, m):
        """
        Perform a deletion mutation on the sequence.
        """
        # 1) Sample the length of the deletion, s
        s = np.random.geometric(p=0.9)  # s ~ Geom(0.9)
        while (
            m - s < 1
        ):  # 1a) Ensure that after deletion, there is at least 1 base left
            s = np.random.geometric(p=0.9)

        # 2) Determine the start location for the deletion
        start = np.random.randint(0, m - s)  # Ensure deletion stays within bounds

        # 3) Remove bases from start location up to start+s
        sequence = sequence[:start] + sequence[start + s :]

        # 4) Return the mutated sequence and mutation details
        return sequence, "Deletion", start, s


class SequenceProcessor:
    def __init__(self, max_length):
        """
        Initialize the processor with a maximum sequence length.
        :param max_length: int, the maximum length N of sequences after padding
        """
        self.max_length = max_length
        self.padding_methods = ["left", "right", "center"]

    def pad_sequence(self, sequence, method):
        """
        Pad the sequence to the maximum length based on the specified method.
        :param sequence: list, the DNA sequence to pad
        :param method: str, the padding method ('left', 'right', 'center')
        :return: list, the padded sequence
        """
        padding_length = self.max_length - len(sequence)
        pad_symbol = "N"

        if method == "left":
            padded_sequence = [pad_symbol] * padding_length + sequence
        elif method == "right":
            padded_sequence = sequence + [pad_symbol] * padding_length
        elif method == "center":
            left_pad = padding_length // 2
            right_pad = padding_length - left_pad
            padded_sequence = (
                [pad_symbol] * left_pad + sequence + [pad_symbol] * right_pad
            )

        return padded_sequence

    def one_hot_encode(self, sequence):
        """
        Convert the sequence to one-hot encoded format.
        :param sequence: list, the sequence of DNA bases
        :return: np.array, the one-hot encoded sequence
        """
        base_to_index = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        one_hot = np.zeros((len(sequence), 5))
        for i, base in enumerate(sequence):
            if base in base_to_index:
                one_hot[i, base_to_index[base]] = 1
        return one_hot

    def process_sequences(self, original, mutated):
        """
        Process both original and mutated sequences with padding and one-hot encoding.
        
        Parameters:
            original (list): The original DNA sequence.
            mutated (list): The mutated DNA sequence.
        
        Returns:
            tuple: One-hot encoded original and mutated sequences.
        """
        if original is None or mutated is None:
            raise ValueError("Cannot process None sequences.")

        # Convert strings to lists if necessary
        original_list = list(original) if isinstance(original, str) else original
        mutated_list = list(mutated) if isinstance(mutated, str) else mutated

        # Pad sequences
        padded_original = self.pad_sequence(original_list, "left")
        padded_mutated = self.pad_sequence(mutated_list, "left")

        # One-hot encode the padded sequences
        one_hot_original = self.one_hot_encode(padded_original)
        one_hot_mutated = self.one_hot_encode(padded_mutated)

        return (one_hot_original, one_hot_mutated)
