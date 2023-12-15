from abc import ABC, abstractmethod
import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def loss(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        ...

    @abstractmethod
    def loss_prime(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        ...


class MSE(LossFunction):
    def loss(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return np.mean(np.power(actual - predicted, 2))

    def loss_prime(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        return 2 * (predicted - actual) / np.size(predicted)


class CrossEntropy(LossFunction):
    def loss(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
        return -np.mean(
            actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)
        )

    def loss_prime(self, actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
        predicted = np.clip(predicted, 1e-15, 1 - 1e-15)
        return -(actual / predicted - (1 - actual) / (1 - predicted)) / np.size(
            predicted
        )
