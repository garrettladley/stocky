from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np


@dataclass
class Layer(ABC):
    input: np.ndarray
    output: np.ndarray

    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        ...


class Dense(Layer):
    def __init__(self, input_size, output_size) -> None:
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(
            2 / (input_size + output_size)
        )
        self.bias = np.zeros((output_size, 1))

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input_size = input
        return np.dot(self.weights, self.input_size) + self.bias

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        weights_gradient = np.dot(output_gradient, self.input_size.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


class Activation(Layer):
    def __init__(
        self,
        activation: Callable[[float], float] = lambda x: x,
        activation_prime: Callable[[float], float] = lambda _: 1,
    ) -> None:
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input: np.ndarray) -> float:
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient: float, _) -> float:
        return np.multiply(self.activation_prime(self.input), output_gradient)
