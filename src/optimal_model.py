import torch
from torch import Tensor


class NearestCentroidClassifier:
    """
    Оптимальный байесовский классификатор для модели AWGN с детерминированными сигналами.

    При наблюдении y = x_c + n, где n ~ N(0, σ²·I) и x_c — известный шаблон класса c,
    логарифм функции правдоподобия равен:

        log p(y | c) = -||y - x_c||² / (2σ²) + const

    Поскольку σ одинаково для всех классов, оптимальное решение:

        ĉ = argmax_c log p(y | c) = argmin_c ||y - x_c||²

    Таким образом, оптимальный классификатор — минимальное евклидово расстояние
    до центроида класса. Центроиды x_c вычисляются как средние изображения класса.
    """

    def __init__(self) -> None:
        self.centroids: Tensor | None = None  # (n_classes, C, H, W)

    def fit(self, centroids: Tensor) -> "NearestCentroidClassifier":
        """Сохраняет эталоны классов. centroids: (n_classes, C, H, W)."""
        self.centroids = centroids
        return self

    def predict(self, images: Tensor) -> Tensor:
        """
        Классифицирует батч изображений по минимуму евклидова расстояния.

        Args:
            images: (B, C, H, W)
        Returns:
            preds:  (B,) — предсказанные индексы классов
        """
        if self.centroids is None:
            raise RuntimeError("Вызовите fit() перед predict().")

        device    = images.device
        centroids = self.centroids.to(device)

        b    = images.shape[0]
        flat = images.view(b, -1)                      # (B, D)
        ctrs = centroids.view(centroids.shape[0], -1)  # (n_classes, D)

        # ||a - b||² = ||a||² + ||b||² - 2·aᵀb  — вычисляем за один матмул
        dists = (
            flat.pow(2).sum(1, keepdim=True)   # (B, 1)
            + ctrs.pow(2).sum(1).unsqueeze(0)  # (1, n_classes)
            - 2 * flat @ ctrs.T                # (B, n_classes)
        )
        return dists.argmin(dim=1)  # (B,)
