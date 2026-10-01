import numpy as np


class ReLU:
    """
    Функцыя актывацыі ReLU.

    ReLU(x) = max(0, x)
    """

    def __init__(self):
        self.input = None

    def forward(self, x):
        """
        Прамы праход.
        """

        self.input = x

        return np.maximum(
            0,
            x
        )

    def backward(self, grad_output):
        """
        Зваротны праход.
        """

        grad_input = grad_output.copy()

        grad_input[self.input <= 0] = 0

        return grad_input


class Softmax:
    """
    Функцыя Softmax для выхаднога слоя.

    Ператварае значэнні ў верагоднасці
    належнасці да 10 класаў.
    """

    def __init__(self):
        self.output = None

    def forward(self, x):
        """
        Прамы праход.

        Для стабільнасці вылічэнняў
        ад кожнага радка аднімаецца максімальнае значэнне.
        """

        shifted = x - np.max(
            x,
            axis=1,
            keepdims=True
        )

        exp_values = np.exp(shifted)

        self.output = (
            exp_values /
            np.sum(
                exp_values,
                axis=1,
                keepdims=True
            )
        )

        return self.output

    def backward(self, grad_output):
        """
        Зваротны праход Softmax.

        У далейшым пры выкарыстанні
        Softmax + CrossEntropy можна будзе
        выкарыстоўваць спрошчаны градыент.
        """

        return grad_output