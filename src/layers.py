import numpy as np


class Linear:
    """
    Поўнасувязны (fully connected) слой.

    Выконвае:
        Y = X @ W + b
    """

    def __init__(self, input_size, output_size):
        """
        input_size  — колькасць уваходных значэнняў.
        output_size — колькасць нейронаў у слоі.
        """

        self.input_size = input_size
        self.output_size = output_size

        # Ініцыялізацыя ваг
        self.weights = (
            np.random.randn(
                input_size,
                output_size
            ).astype(np.float32)
            * np.sqrt(2.0 / input_size)
        )

        # Зрушэнне
        self.bias = np.zeros(
            output_size,
            dtype=np.float32
        )

        # Значэнні, неабходныя для backward
        self.input = None

        # Градыенты
        self.grad_weights = None
        self.grad_bias = None

    def forward(self, x):
        """
        Прамы праход.

        X → XW + b
        """

        self.input = x

        return x @ self.weights + self.bias

    def backward(self, grad_output):
        """
        Зваротны праход.

        grad_output — градыент памылкі
                      адносна выхаду слоя.

        Вяртае градыент адносна ўваходу.
        """

        # Градыент адносна ваг
        self.grad_weights = self.input.T @ grad_output

        # Градыент адносна bias
        self.grad_bias = np.sum(
            grad_output,
            axis=0
        )

        # Градыент адносна ўваходу
        grad_input = grad_output @ self.weights.T

        return grad_input

    def update(self, learning_rate):
        """
        Абнаўленне ваг з дапамогай SGD.
        """

        self.weights -= (
            learning_rate * self.grad_weights
        )

        self.bias -= (
            learning_rate * self.grad_bias
        )