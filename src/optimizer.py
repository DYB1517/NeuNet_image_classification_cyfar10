class SGD:
    """
    Stochastic Gradient Descent.

    Абнаўляе параметры слоя:
        parameter = parameter - learning_rate * gradient
    """

    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def step(self, layers):
        """
        Аднаўленне параметраў усіх Linear-слаёў.
        """

        for layer in layers:
            if hasattr(layer, "weights"):
                layer.weights -= (
                    self.learning_rate *
                    layer.grad_weights
                )

                layer.bias -= (
                    self.learning_rate *
                    layer.grad_bias
                )