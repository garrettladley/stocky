from dataclasses import dataclass
from typing import Optional
import numpy as np
from neural_network.dataset import Dataset
from neural_network.layer import Layer
from neural_network.loss_functions import LossFunction
from neural_network.evaluate import Accuracies, Losses, TrainingStats


@dataclass
class Model:
    network: list[Layer]
    loss_function: LossFunction
    learning_rate: float = 1e-3

    def predict(self, input: np.ndarray) -> np.ndarray:
        output = input.copy()
        for layer in self.network:
            output = layer.forward(output)
        return output

    def train(
        self,
        train_set: Dataset,
        test_set: Optional[Dataset] = None,
        epochs: int = 1_000,
        eval_training_every: int = 10,
    ) -> TrainingStats:
        training_losses = np.zeros(epochs // eval_training_every)
        training_accuracies = np.zeros(epochs // eval_training_every)

        if test_set is not None:
            testing_losses = np.zeros(epochs // eval_training_every)
            testing_accuracies = np.zeros(epochs // eval_training_every)

        for e in range(1, epochs + 1):
            for x, y in zip(train_set.xs, train_set.ys):
                output = self.predict(x)

                grad = self.loss_function.loss_prime(actual=y, predicted=output)

                for layer in reversed(self.network):
                    grad = layer.backward(grad, self.learning_rate)

            if e % eval_training_every == 0:
                print(
                    f"Epoch {e}/{epochs} | Average Error {self.calc_avg_loss(train_set)} | Average Accuracy {self.calc_avg_accuracy(train_set)}"
                )

            if e % eval_training_every == 0:
                eval_index = (e // eval_training_every) - 1
                training_losses[eval_index] = self.calc_avg_loss(train_set)
                training_accuracies[eval_index] = self.calc_avg_accuracy(train_set)

                if test_set is not None:
                    testing_losses[eval_index] = self.calc_avg_loss(test_set)
                    testing_accuracies[eval_index] = self.calc_avg_accuracy(test_set)

        return TrainingStats(
            losses=Losses(
                training_losses=training_losses,
                testing_losses=None if test_set is None else testing_losses,
                eval_training_every=eval_training_every,
            ),
            accuracies=Accuracies(
                training_accuracies=training_accuracies,
                testing_accuracies=None if test_set is None else testing_accuracies,
                eval_training_every=eval_training_every,
            ),
        )

    def calc_avg_accuracy(self, dataset: Dataset) -> float:
        return np.average(
            [
                1 if np.argmax(self.predict(x)) == np.argmax(y) else 0
                for x, y in zip(dataset.xs, dataset.ys)
            ]
        )

    def calc_avg_loss(self, dataset: Dataset) -> float:
        return np.average(
            [
                self.loss_function.loss(actual=y, predicted=self.predict(x))
                for x, y in zip(dataset.xs, dataset.ys)
            ]
        )
