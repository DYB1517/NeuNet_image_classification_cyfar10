import albumentations as A


def get_rotation_translation():
    return A.Compose([
        A.Rotate(
            limit=(-15, 15),
            p=0.5
        ),
        A.Affine(
            translate_percent={
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1)
            },
            p=0.5
        )
    ])


def get_scaling_flipping():
    return A.Compose([
        A.Affine(
            scale=(0.9, 1.1),
            p=0.5
        ),
        A.HorizontalFlip(
            p=0.5
        )
    ])


def get_brightness_contrast():
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        )
    ])