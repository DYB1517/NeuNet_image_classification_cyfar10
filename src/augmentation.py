import albumentations as A


# 1. Паварот + зрух
def get_rotation_translation():
    return A.Compose([
        A.Rotate(limit=(-15, 15), p=0.5),
        A.Affine(
            translate_percent={
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1)
            },
            p=0.5
        )
    ])


# 2. Маштабаванне + адлюстраванне
def get_scaling_flipping():
    return A.Compose([
        A.Affine(scale=(0.9, 1.1), p=0.5),
        A.HorizontalFlip(p=0.5)
    ])


# 3. Яркасць + кантраснасць
def get_brightness_contrast():
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        )
    ])


# 4. Маштабаванне + паварот
def get_scaling_rotation():
    return A.Compose([
        A.Affine(scale=(0.9, 1.1), p=0.5),
        A.Rotate(limit=15, p=0.5)
    ])


# 5. Абразанне + адлюстраванне
def get_cropping_flipping():
    return A.Compose([
        A.PadIfNeeded(
            min_height=36,
            min_width=36,
            border_mode=0
        ),
        A.RandomCrop(height=32, width=32, p=1.0),
        A.HorizontalFlip(p=0.5)
    ])


# 6. Перанос + яркасць
def get_translation_brightness():
    return A.Compose([
        A.Affine(
            translate_percent={
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1)
            },
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0,
            p=0.5
        )
    ])


# 7. Паварот + шум
def get_rotation_noise():
    return A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.GaussNoise(p=0.5)
    ])


# 8. Маштабаванне + абразанне
def get_scaling_cropping():
    return A.Compose([
        A.Affine(scale=(0.9, 1.1), p=0.5),
        A.PadIfNeeded(
            min_height=36,
            min_width=36,
            border_mode=0
        ),
        A.RandomCrop(height=32, width=32, p=1.0)
    ])


# 9. Кантраснасць + шум
def get_contrast_noise():
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(p=0.5)
    ])


# 10. Шум + размыццё
def get_noise_blurring():
    return A.Compose([
        A.GaussNoise(p=0.5),
        A.GaussianBlur(blur_limit=(3, 3), p=0.5)
    ])


# 11. Паварот + перанос
def get_rotation_translation_pair():
    return A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.Affine(
            translate_percent={
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1)
            },
            p=0.5
        )
    ])


# 12. Паварот + маштабаванне + flipping
def get_rotation_scaling_flipping():
    return A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.Affine(scale=(0.9, 1.1), p=0.5),
        A.HorizontalFlip(p=0.5)
    ])


# 13. Паварот + зрух + маштабаванне + flipping
def get_geometric_four():
    return A.Compose([
        A.Rotate(limit=15, p=0.5),
        A.Affine(
            translate_percent={
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1)
            },
            p=0.5
        ),
        A.Affine(scale=(0.9, 1.1), p=0.5),
        A.HorizontalFlip(p=0.5)
    ])