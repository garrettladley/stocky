from typing import Optional
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class Losses:
    training_losses: np.ndarray
    eval_training_every: int
    testing_losses: Optional[np.ndarray] = None

    def plot_training_loss_curve(self, save_to: Optional[Path] = None) -> None:
        iterations = np.arange(
            0,
            len(self.training_losses) * self.eval_training_every,
            self.eval_training_every,
        )
        plt.plot(iterations, self.training_losses, label="Training Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.show()

        if save_to is not None:
            plt.savefig(save_to)

    def plot_testing_loss_curve(self, save_to: Optional[Path] = None) -> None:
        if self.testing_losses is None:
            raise ValueError("No test losses were provided.")
        iterations = np.arange(
            0,
            len(self.testing_losses) * self.eval_training_every,
            self.eval_training_every,
        )
        plt.plot(iterations, self.testing_losses, label="Test Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Test Loss Curve")
        plt.legend()
        plt.show()

        if save_to is not None:
            plt.savefig(save_to)

    def plot_vs_loss_curves(self, save_to: Optional[Path] = None) -> None:
        if self.testing_losses is None:
            raise ValueError("No test losses were provided.")
        iterations = np.arange(
            0,
            len(self.training_losses) * self.eval_training_every,
            self.eval_training_every,
        )
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(iterations, self.training_losses, label="Training Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(iterations, self.testing_losses, label="Testing Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Testing Loss")
        plt.legend()

        plt.tight_layout()
        plt.show()

        if save_to is not None:
            plt.savefig(save_to)


@dataclass
class Accuracies:
    training_accuracies: np.ndarray
    eval_training_every: int
    testing_accuracies: Optional[np.ndarray] = None

    def plot_training_accuracy_curve(self, save_to: Optional[Path] = None) -> None:
        iterations = np.arange(
            0,
            len(self.training_accuracies) * self.eval_training_every,
            self.eval_training_every,
        )
        plt.plot(iterations, self.training_accuracies, label="Training Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy Curve")
        plt.legend()
        plt.show()

        if save_to is not None:
            plt.savefig(save_to)

    def plot_testing_accuracy_curve(self, save_to: Optional[Path] = None) -> None:
        if self.testing_accuracies is None:
            raise ValueError("No test accuracies were provided.")
        iterations = np.arange(
            0,
            len(self.testing_accuracies) * self.eval_training_every,
            self.eval_training_every,
        )
        plt.plot(iterations, self.testing_accuracies, label="Test Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.title("Test Accuracy Curve")
        plt.legend()
        plt.show()

        if save_to is not None:
            plt.savefig(save_to)

    def plot_vs_accuracies(self, save_to: Optional[Path] = None) -> None:
        if self.testing_accuracies is None:
            raise ValueError("No test accuracies were provided.")
        iterations = np.arange(
            0,
            len(self.training_accuracies) * self.eval_training_every,
            self.eval_training_every,
        )
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(iterations, self.training_accuracies, label="Training Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(iterations, self.testing_accuracies, label="Testing Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.title("Testing Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

        if save_to is not None:
            plt.savefig(save_to)


@dataclass
class TrainingStats:
    losses: Optional[Losses] = None
    accuracies: Optional[Accuracies] = None
