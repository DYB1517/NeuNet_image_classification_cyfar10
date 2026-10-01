import numpy as np

from src.model import MLP
from src.loss import CrossEntropyLoss


def main():
    print("=== MODEL TEST ===")

    # Ствараем мадэль
    model = MLP(
        input_size=3072,
        hidden1_size=128,
        hidden2_size=64,
        output_size=10
    )

    print("Architecture:")
    print("3072 -> 128 -> 64 -> 10")

    # Колькасць параметраў
    parameters = model.count_parameters()

    print("Trainable parameters:", parameters)

    if parameters > 1_000_000:
        raise ValueError(
            "Number of parameters exceeds 1,000,000!"
        )

    print("Parameter limit: OK")

    # Ствараем маленькі тэставы batch
    batch_size = 8

    X = np.random.rand(
        batch_size,
        3072
    ).astype(np.float32)

    y = np.random.randint(
        0,
        10,
        size=batch_size
    )

    print()
    print("Input shape:", X.shape)
    print("Labels shape:", y.shape)

    # Forward
    probabilities = model.forward(X)

    print()
    print("=== FORWARD TEST ===")
    print("Output shape:", probabilities.shape)

    print(
        "Probability sums:",
        np.sum(probabilities, axis=1)
    )

    # Loss
    loss_function = CrossEntropyLoss()

    loss = loss_function.forward(
        probabilities,
        y
    )

    print("Loss:", loss)

    # Backward
    grad = loss_function.backward()

    model.backward(grad)

    print()
    print("=== BACKWARD TEST ===")
    print("Backward pass: OK")

    print()
    print("=== PREDICTION TEST ===")

    predictions = model.predict(X)

    print("Predictions:", predictions)
    print("True labels:", y)

    print()
    print("Model test completed successfully.")


if __name__ == "__main__":
    main()