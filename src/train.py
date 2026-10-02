
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import mlflow
from torch.utils.data import TensorDataset, DataLoader, Subset
from tqdm import tqdm

from src.data import load_cifar10
from src.model import MLPMixer
from src.augmentation import (
    get_rotation_translation,
    get_scaling_flipping,
    get_brightness_contrast,
    get_scaling_rotation,
    get_cropping_flipping,
    get_translation_brightness,
    get_rotation_noise,
    get_scaling_cropping,
    get_contrast_noise,
    get_noise_blurring,
    get_rotation_translation_pair,
    get_rotation_scaling_flipping,
    get_geometric_four,
)


# =========================
# CONFIGURATION
# =========================

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

VAL_SIZE = 5000
NUM_WORKERS = 0

MODEL_PATH = "best_mixer.pth"
SEED = 42

# Выбраная аугментацыя для эксперыменту
AUGMENTATION = "brightness_contrast"

# =========================
# REPRODUCIBILITY
# =========================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# DEVICE
# =========================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def build_augmentation(name):
    augmentations = {
        "rotation_translation": get_rotation_translation,
        "scaling_flipping": get_scaling_flipping,
        "brightness_contrast": get_brightness_contrast,

        "scaling_rotation": get_scaling_rotation,
        "cropping_flipping": get_cropping_flipping,
        "translation_brightness": get_translation_brightness,
        "rotation_noise": get_rotation_noise,
        "scaling_cropping": get_scaling_cropping,
        "contrast_noise": get_contrast_noise,
        "noise_blurring": get_noise_blurring,
        "rotation_translation_pair": get_rotation_translation_pair,

        "rotation_scaling_flipping": get_rotation_scaling_flipping,
        "geometric_four": get_geometric_four,

        "none": lambda: None,
    }

    if name not in augmentations:
        raise ValueError(f"Unknown augmentation: {name}")

    return augmentations[name]()

    if name not in augmentations:
        raise ValueError(f"Unknown augmentation: {name}")

    return augmentations[name]()

# =========================
# DATA PREPARATION
# =========================

def prepare_data(augmentation_name):
    print("Loading CIFAR-10...")

    X_train, y_train, X_test, y_test = load_cifar10()

    # Validation split
    indices = np.random.permutation(len(X_train))

    val_indices = indices[:VAL_SIZE]
    train_indices = indices[VAL_SIZE:]

    X_train_part = X_train[train_indices]
    y_train_part = y_train[train_indices]

    X_val = X_train[val_indices]
    y_val = y_train[val_indices]

    # Convert to image format [N, 3, 32, 32]
    X_train_part = X_train_part.reshape(-1, 3, 32, 32)
    X_val = X_val.reshape(-1, 3, 32, 32)
    X_test = X_test.reshape(-1, 3, 32, 32)

    # Normalize to [0, 1]
    X_train_part = X_train_part.astype(np.float32) / 255.0
    X_val = X_val.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # Data augmentation
    train_transform = build_augmentation(augmentation_name)

    # Convert arrays to tensors
    X_train_tensor = torch.from_numpy(X_train_part)
    y_train_tensor = torch.from_numpy(
        y_train_part.astype(np.int64)
    )

    X_val_tensor = torch.from_numpy(X_val)
    y_val_tensor = torch.from_numpy(y_val.astype(np.int64))

    X_test_tensor = torch.from_numpy(X_test)
    y_test_tensor = torch.from_numpy(y_test.astype(np.int64))

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )

    val_dataset = TensorDataset(
        X_val_tensor,
        y_val_tensor
    )

    test_dataset = TensorDataset(
        X_test_tensor,
        y_test_tensor
    )

    # Apply augmentation only to training data
    class AugmentedDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, transform):
            self.dataset = dataset
            self.transform = transform

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            image, label = self.dataset[index]

            if self.transform is not None:
                # PyTorch CHW -> NumPy HWC для Albumentations
                image = image.permute(1, 2, 0).numpy()

                # Albumentations вяртае dict з ключом "image"
                image = self.transform(image=image)["image"]

                # NumPy HWC -> PyTorch CHW
                image = torch.from_numpy(
                    np.ascontiguousarray(image)
                ).permute(2, 0, 1).float()

            return image, label

    train_dataset = AugmentedDataset(
        train_dataset,
        train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    return train_loader, val_loader, test_loader


# =========================
# METRICS
# =========================

def evaluate(model, loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(
                device,
                non_blocking=True
            )
            labels = labels.to(
                device,
                non_blocking=True
            )

            logits = model(images)

            loss = loss_function(logits, labels)

            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size

            predictions = torch.argmax(
                logits,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

    return total_loss / total, correct / total


# =========================
# TRAINING
# =========================

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_function,
    device,
    scaler
):
    model.train()

    total_loss = 0.0
    total = 0

    progress = tqdm(
        loader,
        desc="Training",
        leave=False
    )

    for images, labels in progress:
        images = images.to(
            device,
            non_blocking=True
        )
        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        # Mixed precision on CUDA
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=(device.type == "cuda")
        ):
            logits = model(images)
            loss = loss_function(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return total_loss / total

def main():
    set_seed(SEED)

    device = get_device()
    mlflow.set_experiment("CIFAR10_MLP_Mixer")

    run_name = AUGMENTATION

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "model": "MLP-Mixer",
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "SGD",
            "scheduler": "CosineAnnealingLR",
            "seed": SEED,
            "val_size": VAL_SIZE,
            "augmentation": AUGMENTATION,
        })

        print("Device:", device)

        if device.type == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))
            torch.backends.cudnn.benchmark = True

        train_loader, val_loader, test_loader = prepare_data(AUGMENTATION)

        model = MLPMixer(
            image_size=32,
            patch_size=4,
            channels=256,
            token_hidden=128,
            channel_hidden=512,
            num_blocks=3,
            num_classes=10
        ).to(device)

        parameters = model.count_parameters()
        print("Model parameters:", parameters)

        mlflow.log_param("model_parameters", parameters)

        if parameters > 1_000_000:
            raise ValueError("Parameter limit exceeded!")

        loss_function = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            nesterov=True
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=EPOCHS,
            eta_min=0.0001
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(device.type == "cuda")
        )

        best_accuracy = 0.0
        best_epoch = 0
        final_train_loss = 0.0

        print("\n=== TRAINING STARTED ===")

        for epoch in range(1, EPOCHS + 1):
            start_time = time.time()

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_function,
                device,
                scaler
            )

            val_loss, val_accuracy = evaluate(
                model,
                val_loader,
                loss_function,
                device
            )

            scheduler.step()
            final_train_loss = train_loss

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_epoch = epoch

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_accuracy": best_accuracy,
                        "val_loss": val_loss,
                    },
                    MODEL_PATH
                )

            elapsed = time.time() - start_time

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)

            print(f"\nEpoch {epoch}/{EPOCHS}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
            print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print(f"Time: {elapsed:.2f}s")

        # Load best checkpoint
        checkpoint = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(checkpoint["model_state_dict"])

        # Final evaluation
        test_loss, test_accuracy = evaluate(
            model,
            test_loader,
            loss_function,
            device
        )

        mlflow.log_metrics({
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "best_val_accuracy": best_accuracy,
            "best_epoch": best_epoch,
        })

        # Save checkpoint to MLflow
        mlflow.log_artifact(MODEL_PATH)

        print("\n=== FINAL RESULTS ===")
        print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
        print(f"Best Epoch: {best_epoch}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Train Loss: {final_train_loss:.4f}")
        print(f"Test Loss: {test_loss:.4f}")

        print("\nTraining completed.")


if __name__ == "__main__":
    main()