import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from src.data import load_cifar10
from src.preprocessing import preprocess_images
from src.augmentation import get_scaling_flipping
from src.model import MLP


BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.001

HIDDEN1_SIZE = 256
HIDDEN2_SIZE = 128
HIDDEN3_SIZE = 64


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def apply_augmentation(images, augmentation):
    augmented_images = []

    for image in images:
        image = image.reshape(3, 32, 32)
        image = np.transpose(image, (1, 2, 0))

        result = augmentation(image=image)
        augmented_image = result["image"]

        augmented_image = np.transpose(
            augmented_image,
            (2, 0, 1)
        )

        augmented_images.append(augmented_image)

    return np.asarray(
        augmented_images,
        dtype=np.uint8
    )


def calculate_accuracy(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == y_batch
            ).sum().item()

            total += y_batch.size(0)

    return correct / total


def calculate_loss(model, loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)

            loss = loss_function(
                logits,
                y_batch
            )

            batch_size = y_batch.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

    return total_loss / total_samples


def train():
    print("Loading CIFAR-10...")

    X_train, y_train, X_test, y_test = load_cifar10()

    device = get_device()

    print()
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()

    print("=== TRAINING CONFIGURATION ===")

    print(
        f"Architecture: 3072 -> "
        f"{HIDDEN1_SIZE} -> "
        f"{HIDDEN2_SIZE} -> "
        f"{HIDDEN3_SIZE} -> 10"
    )

    print("Batch size:", BATCH_SIZE)
    print("Epochs:", EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Augmentation: Scaling + Flipping")

    model = MLP(
        input_size=3072,
        hidden1_size=HIDDEN1_SIZE,
        hidden2_size=HIDDEN2_SIZE,
        hidden3_size=HIDDEN3_SIZE,
        output_size=10
    )

    model = model.to(device)

    parameters = model.count_parameters()

    print("Trainable parameters:", parameters)

    if parameters > 1_000_000:
        raise ValueError(
            "Number of parameters exceeds 1,000,000!"
        )

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE
    )

    augmentation = get_scaling_flipping()

    print()
    print("Preparing test data...")

    X_test = preprocess_images(X_test)

    X_test_tensor = torch.from_numpy(
        X_test
    )

    y_test_tensor = torch.from_numpy(
        y_test.astype(np.int64)
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        "Test shape:",
        X_test.shape
    )

    print()
    print("Preparing training evaluation data...")

    X_train_eval = preprocess_images(
        X_train[:10000]
    )

    y_train_eval = y_train[:10000]

    X_train_eval_tensor = torch.from_numpy(
        X_train_eval
    )

    y_train_eval_tensor = torch.from_numpy(
        y_train_eval.astype(np.int64)
    )

    train_eval_dataset = TensorDataset(
        X_train_eval_tensor,
        y_train_eval_tensor
    )

    train_eval_loader = DataLoader(
        train_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(
        "Training evaluation shape:",
        X_train_eval.shape
    )

    print()
    print("=== TRAINING STARTED ===")

    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.time()

        model.train()

        indices = np.random.permutation(
            len(X_train)
        )

        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]

        epoch_loss = 0.0
        samples_processed = 0

        progress_bar = tqdm(
            range(
                0,
                len(X_train),
                BATCH_SIZE
            ),
            desc=f"Epoch {epoch}/{EPOCHS}"
        )

        for start in progress_bar:

            end = min(
                start + BATCH_SIZE,
                len(X_train)
            )

            X_batch = X_train_shuffled[
                start:end
            ]

            y_batch = y_train_shuffled[
                start:end
            ]

            X_batch = apply_augmentation(
                X_batch,
                augmentation
            )

            X_batch = preprocess_images(
                X_batch
            )

            X_batch = torch.from_numpy(
                X_batch
            )

            y_batch = torch.from_numpy(
                y_batch.astype(np.int64)
            )

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(X_batch)

            loss = loss_function(
                logits,
                y_batch
            )

            loss.backward()

            optimizer.step()

            batch_count = y_batch.size(0)

            epoch_loss += (
                loss.item() * batch_count
            )

            samples_processed += batch_count

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        epoch_loss /= samples_processed

        train_accuracy = calculate_accuracy(
            model,
            train_eval_loader,
            device
        )

        test_loss = calculate_loss(
            model,
            test_loader,
            loss_function,
            device
        )

        test_accuracy = calculate_accuracy(
            model,
            test_loader,
            device
        )

        epoch_time = (
            time.time() - epoch_start
        )

        print()

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss: {epoch_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Test Loss: {test_loss:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{test_accuracy * 100:.2f}%"
        )

        print(
            f"Time: {epoch_time:.2f} s"
        )

        print()

    print("=== TRAINING COMPLETED ===")


if __name__ == "__main__":
    train()