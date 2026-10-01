import numpy as np


class CrossEntropyLoss:
    """
    Cross-Entropy Loss для шматкласавой класіфікацыі.
    """

    def __init__(self):
        self.probabilities = None
        self.labels = None

    def forward(self, probabilities, labels):
        """
        Разлічвае значэнне функцыі страты.

        probabilities — вынік Softmax, форма (batch_size, 10)
        labels        — правільныя нумары класаў
        """

        self.probabilities = probabilities
        self.labels = labels

        batch_size = probabilities.shape[0]

        # Абмяжоўваем верагоднасці,
        # каб пазбегнуць log(0)
        probabilities = np.clip(
            probabilities,
            1e-7,
            1.0
        )

        # Верагоднасць правільнага класа
        correct_probabilities = probabilities[
            np.arange(batch_size),
            labels
        ]

        # Cross-Entropy
        loss = -np.mean(
            np.log(correct_probabilities)
        )

        return loss

    def backward(self):
        """
        Вяртае градыент адносна ўваходу Softmax.

        Для звязкі Softmax + Cross-Entropy
        градыент мае просты выгляд.
        """

        batch_size = self.probabilities.shape[0]

        grad = self.probabilities.copy()

        grad[
            np.arange(batch_size),
            self.labels
        ] -= 1.0

        grad /= batch_size

        return grad