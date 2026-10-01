import numpy as np


def normalize_images(images):
    """
    Нармалізацыя значэнняў пікселяў з дыяпазону 0..255
    у дыяпазон 0..1.
    """

    return images.astype(np.float32) / 255.0


def preprocess_images(images):
    """
    Базавая падрыхтоўка CIFAR-10:

    1. Нармалізацыя пікселяў.
    2. Пераўтварэнне выявы ў вектар.
    """

    images = normalize_images(images)

    # Для CIFAR-10 выявы ўжо маюць форму (N, 3072),
    # таму дадатковы reshape не патрэбны.
    images = images.reshape(images.shape[0], -1)

    return images