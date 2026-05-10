import torch.nn as nn
from torch import Tensor


class EyeCNN(nn.Module):
    """
    Свёрточная сеть для классификации глазных снимков.

    Архитектура: 4 блока [Conv2d → BatchNorm2d → ReLU → MaxPool2d],
    затем AdaptiveAvgPool2d(1) → Dropout → Linear.

    AdaptiveAvgPool2d(1) сжимает пространственные размеры до (1×1) независимо
    от входного разрешения — это позволяет использовать одну архитектуру
    для 32×32 и 128×128 без изменения кода.
    """

    def __init__(self, img_size: int, num_classes: int) -> None:
        super().__init__()
        self.img_size    = img_size
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # Блок 1: 3 → 32 каналов
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Блок 2: 32 → 64 каналов
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Блок 3: 64 → 128 каналов
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Блок 4: 128 → 256 каналов
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # (B, 256, H, W) → (B, 256, 1, 1)
        )

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)  # (B, 256, 1, 1)
        x = x.flatten(1)      # (B, 256)
        return self.head(x)   # (B, num_classes)
