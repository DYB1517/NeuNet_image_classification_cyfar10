import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    """
    Адзін блок MLP-Mixer.

    Складаецца толькі з Linear-слаёў
    і функцыі актывацыі.

    Token mixing:
        змешвае інфармацыю паміж patches.

    Channel mixing:
        змешвае інфармацыю паміж feature channels.
    """

    def __init__(
        self,
        num_tokens,
        channels,
        token_hidden,
        channel_hidden
    ):
        super().__init__()

        # Token mixing
        self.token_mixing = nn.Sequential(
            nn.Linear(
                num_tokens,
                token_hidden
            ),
            nn.GELU(),
            nn.Linear(
                token_hidden,
                num_tokens
            )
        )

        # Channel mixing
        self.channel_mixing = nn.Sequential(
            nn.Linear(
                channels,
                channel_hidden
            ),
            nn.GELU(),
            nn.Linear(
                channel_hidden,
                channels
            )
        )

    def forward(self, x):
        # x:
        # [batch, tokens, channels]

        # -------------------------
        # Token mixing
        # -------------------------

        residual = x

        x = x.transpose(1, 2)

        x = self.token_mixing(x)

        x = x.transpose(1, 2)

        x = x + residual

        # -------------------------
        # Channel mixing
        # -------------------------

        residual = x

        x = self.channel_mixing(x)

        x = x + residual

        return x


class MLPMixer(nn.Module):
    """
    MLP-Mixer для CIFAR-10.

    Не выкарыстоўвае:
        Conv
        Conv2d
        Pooling layers
        Attention
        BatchNorm
        LayerNorm

    Навучальныя слаі:
        толькі Linear.
    """

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        channels=288,
        token_hidden=64,
        channel_hidden=576,
        num_blocks=3,
        num_classes=10
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size

        # Колькасць patches па адной восі
        patches_per_side = (
            image_size // patch_size
        )

        # Агульная колькасць patches
        num_tokens = (
            patches_per_side *
            patches_per_side
        )

        # Колькасць значэнняў у адным patch
        patch_dim = (
            patch_size *
            patch_size *
            3
        )

        self.num_tokens = num_tokens
        self.patch_dim = patch_dim
        self.channels = channels

        # Patch embedding
        self.patch_embedding = nn.Linear(
            patch_dim,
            channels
        )

        # MLP-Mixer blocks
        self.blocks = nn.ModuleList([
            MixerBlock(
                num_tokens=num_tokens,
                channels=channels,
                token_hidden=token_hidden,
                channel_hidden=channel_hidden
            )
            for _ in range(num_blocks)
        ])

        # Classification head
        self.classifier = nn.Linear(
            channels,
            num_classes
        )

    def create_patches(self, x):
        """
        Пераўтварае выяву:

        [B, 3, 32, 32]

        у:

        [B, 64, 48]
        """

        batch_size = x.shape[0]

        p = self.patch_size

        # [B, 3, 8, 4, 8, 4]
        x = x.reshape(
            batch_size,
            3,
            self.image_size // p,
            p,
            self.image_size // p,
            p
        )

        # [B, 8, 8, 3, 4, 4]
        x = x.permute(
            0,
            2,
            4,
            1,
            3,
            5
        )

        # [B, 64, 48]
        x = x.reshape(
            batch_size,
            self.num_tokens,
            self.patch_dim
        )

        return x

    def forward(self, x):
        """
        Forward pass.

        Уваход:
            [B, 3072]
            або
            [B, 3, 32, 32]

        Выхад:
            [B, 10]
        """

        # Калі ўваход — плоскі вектар
        if x.dim() == 2:
            x = x.reshape(
                -1,
                3,
                self.image_size,
                self.image_size
            )

        # Ствараем patches
        x = self.create_patches(x)

        # Patch embedding
        x = self.patch_embedding(x)

        # Mixer blocks
        for block in self.blocks:
            x = block(x)

        # Average па patches
        x = x.mean(dim=1)

        # Classification
        x = self.classifier(x)

        return x

    def predict(self, x):
        """
        Вяртае нумар прадказанага класа.
        """

        with torch.no_grad():
            logits = self.forward(x)

            return torch.argmax(
                logits,
                dim=1
            )

    def count_parameters(self):
        """
        Колькасць навучальных параметраў.
        """

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )