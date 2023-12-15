import numpy as np
from neural_network.layer import Activation


class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__(
            activation=lambda input: np.tanh(input),
            activation_prime=lambda input: 1 - np.power(np.tanh(input), 2),
        )


class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__(
            activation=lambda input: 1 / (1 + np.exp(-input)),
            activation_prime=lambda input: np.exp(-input)
            / np.power(1 + np.exp(-input), 2),
        )
