import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import ExperimentConfig


class EyeDataset:
    """Загружает датасет с Kaggle, возвращает DataLoader'ы и центроиды классов."""

    # ImageNet нормализация — стандарт для предобученных моделей и честного сравнения
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(self, config: ExperimentConfig) -> None:
        self.config   = config
        self.data_dir = Path(config.raw_data_dir)
        # Имена классов в порядке, который задаёт ImageFolder (алфавит папок).
        # Заполняется после первого вызова get_loaders().
        self.class_names: list[str] = []

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def download(self) -> None:
        """Скачивает датасет с Kaggle если данные ещё не существуют."""
        if self.data_dir.exists() and any(self.data_dir.iterdir()):
            print("Dataset already present, skipping download.")
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        os.system(
            f"kaggle datasets download -d {self.config.kaggle_dataset} "
            f"-p {self.data_dir} --unzip"
        )

    def get_loaders(
        self,
        img_size: int,
        train_fraction: float = 1.0,
    ) -> tuple[DataLoader, DataLoader]:
        """Возвращает (train_loader, test_loader) для заданного разрешения изображений."""
        root = self._find_dataset_root()

        train_ds = ImageFolder(root / "train", transform=self._train_transform(img_size))
        test_ds  = ImageFolder(root / "test",  transform=self._test_transform(img_size))

        # Сохраняем имена классов в алфавитном порядке (так работает ImageFolder)
        self.class_names = train_ds.classes

        if train_fraction < 1.0:
            n        = int(len(train_ds) * train_fraction)
            indices  = torch.randperm(len(train_ds))[:n].tolist()
            train_ds = Subset(train_ds, indices)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=(self.config.device == "cuda"),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=(self.config.device == "cuda"),
        )
        return train_loader, test_loader

    def compute_centroids(self, loader: DataLoader) -> torch.Tensor:
        """
        Вычисляет среднее изображение каждого класса — эталон для оптимального классификатора.

        Математически: x̄_c = (1/N_c) Σ x_i  для всех i с меткой c.
        Эти центроиды используются как детерминированные шаблоны в оптимальном классификаторе.

        Returns:
            centroids: Tensor (n_classes, C, H, W)
        """
        device = torch.device(self.config.device)

        sums:   torch.Tensor | None = None
        counts: torch.Tensor | None = None

        for images, labels in loader:
            images   = images.to(device)
            n_cls    = int(labels.max().item()) + 1

            if sums is None:
                sums   = torch.zeros(n_cls, *images.shape[1:], device=device)
                counts = torch.zeros(n_cls, device=device)

            for c in range(n_cls):
                mask = labels == c
                if mask.any():
                    sums[c]   += images[mask].sum(0)
                    counts[c] += mask.sum()

        centroids = sums / counts.view(-1, 1, 1, 1)
        return centroids  # (n_classes, C, H, W)

    # ------------------------------------------------------------------
    # Приватные методы
    # ------------------------------------------------------------------

    def _train_transform(self, img_size: int) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(self._MEAN, self._STD),
        ])

    def _test_transform(self, img_size: int) -> transforms.Compose:
        # Тестовые изображения — без аугментации, только resize и нормализация
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self._MEAN, self._STD),
        ])

    def _find_dataset_root(self) -> Path:
        """
        Находит корневую папку датасета после распаковки Kaggle.
        Kaggle иногда создаёт вложенную директорию при --unzip.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Директория {self.data_dir} не найдена. Запустите download() сначала."
            )

        # Проверяем сначала сам data_dir, затем его прямые поддиректории
        candidates = [self.data_dir, *(p for p in self.data_dir.iterdir() if p.is_dir())]
        for candidate in candidates:
            if (candidate / "train").exists():
                return candidate

        raise FileNotFoundError(
            f"Не удалось найти папку 'train' в {self.data_dir}. "
            "Проверьте структуру скачанного датасета."
        )
