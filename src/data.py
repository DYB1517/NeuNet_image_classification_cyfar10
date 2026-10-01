import os
import pickle
import tarfile

import numpy as np


# Шлях да архіва CIFAR-10
# Вызначаем каранёвую папку праекта
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

ARCHIVE_PATH = os.path.join(
    DATA_DIR,
    "cifar-10-python.tar.gz"
)

EXTRACT_PATH = DATA_DIR

CIFAR10_FOLDER = os.path.join(
    EXTRACT_PATH,
    "cifar-10-batches-py"
)

# Папка, у якую будзе распакаваны CIFAR-10
EXTRACT_PATH = "data"

# Назва папкі пасля распакоўкі
CIFAR10_FOLDER = os.path.join(
    EXTRACT_PATH,
    "cifar-10-batches-py"
)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

def extract_cifar10():
    """
    Распакоўвае CIFAR-10 з архіва,
    калі ён яшчэ не быў распакаваны.
    """

    if os.path.exists(CIFAR10_FOLDER):
        print("CIFAR-10 already extracted.")
        return

    if not os.path.exists(ARCHIVE_PATH):
        raise FileNotFoundError(
            f"CIFAR-10 archive not found: {ARCHIVE_PATH}"
        )

    print("Extracting CIFAR-10...")

    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(EXTRACT_PATH)

    print("Extraction completed.")


def load_batch(file_path):
    """
    Загружае адзін batch CIFAR-10.
    """

    with open(file_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    images = np.array(
        batch[b"data"],
        dtype=np.float32
    )

    labels = np.array(
        batch[b"labels"],
        dtype=np.int64
    )

    return images, labels


def load_cifar10():
    """
    Загружае ўвесь CIFAR-10.

    Вяртае:
        X_train — навучальныя выявы
        y_train — меткі навучальных выяў
        X_test  — тэставыя выявы
        y_test  — меткі тэставых выяў
    """

    extract_cifar10()

    train_images = []
    train_labels = []

    # CIFAR-10 мае 5 навучальных batch-файлаў
    for i in range(1, 6):
        file_path = os.path.join(
            CIFAR10_FOLDER,
            f"data_batch_{i}"
        )

        images, labels = load_batch(file_path)

        train_images.append(images)
        train_labels.append(labels)

    # Аб'ядноўваем усе навучальныя batch
    X_train = np.concatenate(
        train_images,
        axis=0
    )

    y_train = np.concatenate(
        train_labels,
        axis=0
    )

    # Загружаем тэставы batch
    test_path = os.path.join(
        CIFAR10_FOLDER,
        "test_batch"
    )

    X_test, y_test = load_batch(test_path)

    return X_train, y_train, X_test, y_test

def show_images(images, labels, class_names, count=10):
    """
    Паказвае некалькі малюнкаў CIFAR-10.
    """

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))

    for i in range(count):
        image = images[i].reshape(3, 32, 32)
        image = np.transpose(image, (1, 2, 0))
        image = image.astype(np.uint8)

        plt.subplot(2, 5, i + 1)
        plt.imshow(image)
        plt.title(class_names[labels[i]])
        plt.axis("off")

    plt.tight_layout()
    plt.show()