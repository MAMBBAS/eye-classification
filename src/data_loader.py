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
        """Скачивает датасет с Kaggle если данные ещё не существуют.

        В Kaggle Notebook датасет монтируется через UI и уже доступен —
        вызов kaggle CLI не нужен и заблокирован сетью.
        """
        if self.data_dir.exists() and any(self.data_dir.iterdir()):
            print(f"Датасет уже доступен: {self.data_dir}")
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        ret = os.system(
            f"kaggle datasets download -d {self.config.kaggle_dataset} "
            f"-p {self.data_dir} --unzip"
        )
        if ret != 0:
            raise RuntimeError(
                "Не удалось скачать датасет. В Kaggle Notebook добавьте датасет "
                "через кнопку '+ Add Input' в правой панели."
            )

    def get_loaders(
        self,
        img_size: int,
        train_fraction: float = 1.0,
    ) -> tuple[DataLoader, DataLoader]:
        """
        Возвращает (train_loader, test_loader).

        Датасет не имеет готового train/test разбиения — делаем его вручную
        стратифицированно по классам (80/20 по умолчанию из config.train_ratio).
        """
        root    = self._find_dataset_root()
        # Загружаем весь датасет с тестовым transform (без аугментаций)
        full_ds = ImageFolder(root, transform=self._test_transform(img_size))
        self.class_names = full_ds.classes

        train_idx, test_idx = self._stratified_split(full_ds, self.config.train_ratio)

        # Применяем train_fraction поверх train-сплита
        if train_fraction < 1.0:
            n          = int(len(train_idx) * train_fraction)
            train_idx  = train_idx[:n]

        # Тренировочному сплиту нужен отдельный датасет с аугментациями
        train_ds_aug = ImageFolder(root, transform=self._train_transform(img_size))

        train_ds = Subset(train_ds_aug, train_idx)
        test_ds  = Subset(full_ds,      test_idx)

        kw = dict(batch_size=self.config.batch_size, num_workers=2,
                  pin_memory=(self.config.device == "cuda"))
        return (DataLoader(train_ds, shuffle=True,  **kw),
                DataLoader(test_ds,  shuffle=False, **kw))

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

    def _stratified_split(
        self, dataset: ImageFolder, train_ratio: float
    ) -> tuple[list[int], list[int]]:
        """
        Стратифицированный сплит: сохраняет пропорции классов в train и test.
        Внутри каждого класса перемешиваем случайно для разных повторений Монте-Карло.
        """
        from collections import defaultdict
        class_indices: dict[int, list[int]] = defaultdict(list)
        for idx, (_, label) in enumerate(dataset.samples):
            class_indices[label].append(idx)

        train_idx, test_idx = [], []
        for indices in class_indices.values():
            perm  = torch.randperm(len(indices)).tolist()
            split = int(len(indices) * train_ratio)
            train_idx.extend([indices[i] for i in perm[:split]])
            test_idx.extend( [indices[i] for i in perm[split:]])

        return train_idx, test_idx

    def _find_dataset_root(self) -> Path:
        """
        Находит папку с подпапками классов после распаковки Kaggle.
        Поддерживает структуру: data/raw/dataset/{class}/ или data/raw/{class}/.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Директория {self.data_dir} не найдена. Запустите download() сначала."
            )

        # Папка с классами — та, в которой лежат только директории (не файлы изображений)
        candidates = [self.data_dir, *(p for p in self.data_dir.iterdir() if p.is_dir())]
        for candidate in candidates:
            subdirs = [p for p in candidate.iterdir() if p.is_dir()]
            # Признак папки с классами: все поддиректории содержат изображения
            if subdirs and all(
                any(f.suffix.lower() in (".jpg", ".jpeg", ".png") for f in d.iterdir())
                for d in subdirs
            ):
                return candidate

        raise FileNotFoundError(
            f"Не удалось найти папки классов в {self.data_dir}. "
            "Проверьте структуру датасета."
        )
