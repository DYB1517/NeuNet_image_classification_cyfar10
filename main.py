from src.data import load_cifar10, CLASS_NAMES, show_images
from src.preprocessing import preprocess_images
from src.augmentation import (
    get_rotation_translation,
    get_scaling_flipping,
    get_brightness_contrast
)

import matplotlib.pyplot as plt


def show_augmentation_examples(images, labels):
    """
    Паказвае арыгінальную выяву
    і вынікі трох варыянтаў аугментацыі.
    """

    rotation_translation = get_rotation_translation()
    scaling_flipping = get_scaling_flipping()
    brightness_contrast = get_brightness_contrast()

    plt.figure(figsize=(12, 8))

    for i in range(5):
        image = images[i].reshape(3, 32, 32)
        image = image.transpose(1, 2, 0)

        # Пераўтвараем 0..255 у 0..1 толькі для адлюстравання
        image_display = image / 255.0

        # 1. Арыгінал
        plt.subplot(4, 5, i + 1)
        plt.imshow(image_display)
        plt.title("Original")
        plt.axis("off")

        # 2. Rotation + Translation
        augmented = rotation_translation(image=image_display)
        aug_image = augmented["image"]

        plt.subplot(4, 5, i + 6)
        plt.imshow(aug_image)
        plt.title("Rotation + Translation")
        plt.axis("off")

        # 3. Scaling + Flipping
        augmented = scaling_flipping(image=image_display)
        aug_image = augmented["image"]

        plt.subplot(4, 5, i + 11)
        plt.imshow(aug_image)
        plt.title("Scaling + Flipping")
        plt.axis("off")

        # 4. Brightness + Contrast
        augmented = brightness_contrast(image=image_display)
        aug_image = augmented["image"]

        plt.subplot(4, 5, i + 16)
        plt.imshow(aug_image)
        plt.title("Brightness + Contrast")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    print("Loading CIFAR-10...")

    X_train, y_train, X_test, y_test = load_cifar10()

    print()
    print("=== CIFAR-10 INFORMATION ===")

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("Data type:", X_train.dtype)

    # Паказваем прыклады аугментацыі
    print()
    print("Showing augmentation examples...")

    show_augmentation_examples(
        X_train,
        y_train
    )

    # Базавая preprocessing
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    print()
    print("=== PREPROCESSING INFORMATION ===")

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("Data type:", X_train.dtype)

    print(
        "Pixel value range:",
        X_train.min(),
        "-",
        X_train.max()
    )


if __name__ == "__main__":
    main()