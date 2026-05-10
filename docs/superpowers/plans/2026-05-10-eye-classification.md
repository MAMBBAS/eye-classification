# Eye Disease Classification — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Построить полный ML-пайплайн классификации глазных заболеваний с гауссовским шумом и Монте-Карло экспериментами для Google Colab.

**Architecture:** Классовая архитектура с единым `config.py` как источником всех параметров. Каждый модуль — один класс с чётким интерфейсом. `main.ipynb` содержит только оркестрацию.

**Tech Stack:** Python 3.10, PyTorch, torchvision, scikit-learn, matplotlib, seaborn, kaggle API.

---

## Файловая карта

| Файл | Класс | Ответственность |
|------|-------|-----------------|
| `config.py` | `ExperimentConfig` | Все гиперпараметры как dataclass |
| `src/data_loader.py` | `EyeDataset` | Загрузка, препроцессинг, DataLoader'ы, центроиды |
| `src/noise.py` | `GaussianNoise` | SNR↔σ конвертация, применение шума |
| `src/optimal_model.py` | `NearestCentroidClassifier` | Оптимальный классификатор (минимум евклидова расстояния) |
| `src/cnn_model.py` | `EyeCNN` | Свёрточная сеть (4 блока Conv→BN→ReLU→Pool) |
| `src/trainer.py` | `Trainer` | Обучение CNN, fit центроидов, оценка точности |
| `src/experiment.py` | `MonteCarloRunner` | Внешний цикл по SNR и повторениям |
| `src/evaluate.py` | `Evaluator` | Все графики и confusion matrix'ы |
| `main.ipynb` | — | Colab-точка входа, только оркестрация |
| `requirements.txt` | — | Зависимости |

---

### Task 1: Структура проекта и requirements.txt

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`

- [ ] **Step 1: Создать `requirements.txt`**

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
kaggle>=1.5.16
Pillow>=9.5.0
tqdm>=4.65.0
```

- [ ] **Step 2: Создать `src/__init__.py`**

```python
```
(пустой файл, делает src Python-пакетом)

- [ ] **Step 3: Проверить в Colab**

```python
# Validation cell — запустить в Colab
!pip install -r requirements.txt -q
import torch, torchvision, sklearn, matplotlib, seaborn
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
```

Ожидаемый вывод: версии библиотек без ошибок, `CUDA available: True` в Colab с GPU runtime.

---

### Task 2: config.py

**Files:**
- Create: `config.py`

- [ ] **Step 1: Написать `config.py`**

```python
from dataclasses import dataclass, field
import torch


@dataclass
class ExperimentConfig:
    # --- Данные ---
    kaggle_dataset: str              = "gunavenkatdoddi/eye-diseases-classification"
    raw_data_dir: str                = "data/raw"
    img_sizes: tuple[int, ...]       = (32, 128)
    train_ratio: float               = 0.8
    batch_size: int                  = 32

    # --- Шум ---
    # Уровни SNR в dB: от хорошего сигнала (20) до сильного шума (-5)
    snr_levels_db: tuple[float, ...] = (20.0, 15.0, 10.0, 5.0, 0.0, -5.0)

    # --- Монте-Карло ---
    n_repetitions: int                 = 15
    # train_fractions: для серии "точность vs число выборок"
    train_fractions: tuple[float, ...] = (0.25, 0.50, 1.0)
    # SNR фиксируется при эксперименте с числом выборок
    fixed_snr_for_samples_exp: float   = 10.0

    # --- CNN ---
    num_epochs: int      = 20
    learning_rate: float = 1e-3

    # --- Пути ---
    results_dir: str = "results"
    drive_dir: str   = "/content/drive/MyDrive/eye_classification"

    # --- Устройство (определяется автоматически) ---
    device: str = field(default="", init=True)

    def __post_init__(self) -> None:
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
```

- [ ] **Step 2: Проверить в Colab**

```python
# Validation cell
import sys; sys.path.insert(0, ".")
from config import ExperimentConfig

cfg = ExperimentConfig()
assert cfg.device in ("cuda", "cpu")
assert len(cfg.snr_levels_db) == 6
assert len(cfg.img_sizes) == 2
print("Config OK:", cfg.device, cfg.img_sizes, cfg.snr_levels_db)
```

Ожидаемый вывод: `Config OK: cuda (32, 128) (20.0, 15.0, 10.0, 5.0, 0.0, -5.0)`

---

### Task 3: src/data_loader.py

**Files:**
- Create: `src/data_loader.py`

- [ ] **Step 1: Написать `src/data_loader.py`**

```python
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from config import ExperimentConfig


class EyeDataset:
    """Загружает датасет с Kaggle, возвращает DataLoader'ы и центроиды классов."""

    # ImageNet нормализация — стандарт для transfer-ready моделей
    _MEAN = (0.485, 0.456, 0.406)
    _STD  = (0.229, 0.224, 0.225)

    def __init__(self, config: ExperimentConfig) -> None:
        self.config   = config
        self.data_dir = Path(config.raw_data_dir)
        # Имена классов в порядке, который задаёт ImageFolder (алфавит папок)
        self.class_names: list[str] = []

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def download(self) -> None:
        """Скачивает датасет с Kaggle если папка ещё не существует."""
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
        """Возвращает (train_loader, test_loader) для заданного разрешения."""
        root = self._find_dataset_root()

        train_ds = ImageFolder(root / "train", transform=self._train_transform(img_size))
        test_ds  = ImageFolder(root / "test",  transform=self._test_transform(img_size))

        # Запоминаем имена классов в порядке ImageFolder (алфавит)
        self.class_names = train_ds.classes

        if train_fraction < 1.0:
            n       = int(len(train_ds) * train_fraction)
            indices = torch.randperm(len(train_ds))[:n].tolist()
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
        """Вычисляет среднее изображение каждого класса — эталон для оптимального классификатора."""
        device    = torch.device(self.config.device)
        n_classes = len(self.config.img_sizes)  # placeholder — реальное число из первого батча

        sums: torch.Tensor | None = None
        counts = torch.zeros(100)  # временный размер, уточняется после первого батча

        for images, labels in loader:
            images = images.to(device)
            n_cls  = int(labels.max().item()) + 1

            if sums is None:
                sums   = torch.zeros(n_cls, *images.shape[1:], device=device)
                counts = torch.zeros(n_cls, device=device)

            for c in range(n_cls):
                mask = labels == c
                if mask.any():
                    sums[c]   += images[mask].sum(0)
                    counts[c] += mask.sum()

        # centroids: (n_classes, C, H, W)
        centroids = sums / counts.view(-1, 1, 1, 1)
        return centroids

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
        # Тестовые изображения — без аугментации, только нормализация
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(self._MEAN, self._STD),
        ])

    def _find_dataset_root(self) -> Path:
        """Находит корень датасета после распаковки Kaggle (иногда создаёт вложенную папку)."""
        candidates = [self.data_dir, *self.data_dir.iterdir()] if self.data_dir.exists() else []
        for candidate in candidates:
            if candidate.is_dir() and (candidate / "train").exists():
                return candidate
        raise FileNotFoundError(
            f"Не удалось найти папку 'train' в {self.data_dir}. "
            "Запустите dataset.download() или проверьте структуру."
        )
```

- [ ] **Step 2: Проверить в Colab (после скачивания данных)**

```python
# Validation cell
from config import ExperimentConfig
from src.data_loader import EyeDataset

cfg = ExperimentConfig()
ds  = EyeDataset(cfg)
ds.download()  # первый запуск скачивает, повторные пропускают

train_loader, test_loader = ds.get_loaders(img_size=32)

images, labels = next(iter(train_loader))
print("Batch shape:", images.shape)       # (32, 3, 32, 32)
print("Labels:", labels[:8].tolist())
print("Classes:", ds.class_names)         # ['cataract', 'diabetic_retinopathy', ...]
print("Train batches:", len(train_loader))
print("Test batches:", len(test_loader))

centroids = ds.compute_centroids(train_loader)
print("Centroids shape:", centroids.shape)  # (4, 3, 32, 32)
```

Ожидаемый вывод: `Batch shape: torch.Size([32, 3, 32, 32])`, 4 класса, centroids `(4, 3, 32, 32)`.

---

### Task 4: src/noise.py

**Files:**
- Create: `src/noise.py`

- [ ] **Step 1: Написать `src/noise.py`**

```python
import math
import torch
from torch import Tensor


class GaussianNoise:
    """
    Аддитивный белый гауссовский шум (AWGN).

    Математическая модель:
        y = x + n,  n ~ N(0, σ²·I)

    SNR в децибелах:
        SNR_dB = 10·log10(P_signal / σ²)
        P_signal = E[x²]  — вычисляется по текущему батчу

    Из этого следует:
        σ² = P_signal / 10^(SNR_dB / 10)
        σ  = sqrt(P_signal) / 10^(SNR_dB / 20)
    """

    @staticmethod
    def snr_to_sigma(images: Tensor, snr_db: float) -> float:
        """Вычисляет σ шума по мощности сигнала и заданному SNR."""
        # Мощность сигнала — среднее квадратичное значение пикселей батча
        p_signal = images.pow(2).mean().item()
        if p_signal == 0:
            return 0.0
        sigma_sq = p_signal / (10 ** (snr_db / 10))
        return math.sqrt(sigma_sq)

    @staticmethod
    def add(images: Tensor, snr_db: float) -> Tensor:
        """Возвращает зашумлённую копию батча при заданном SNR."""
        sigma = GaussianNoise.snr_to_sigma(images, snr_db)
        noise = torch.randn_like(images) * sigma
        # Не изменяем исходный тензор
        return images + noise
```

- [ ] **Step 2: Проверить в Colab**

```python
# Validation cell
import torch
from src.noise import GaussianNoise

# Синтетический батч: единичные значения пикселей
x = torch.ones(8, 3, 32, 32)

# При SNR=20 dB шум должен быть слабым (σ≈0.1 при P=1)
noisy_high_snr = GaussianNoise.add(x, snr_db=20.0)
noisy_low_snr  = GaussianNoise.add(x, snr_db=0.0)

sigma_high = GaussianNoise.snr_to_sigma(x, 20.0)
sigma_low  = GaussianNoise.snr_to_sigma(x, 0.0)

print(f"SNR=20 dB → σ={sigma_high:.4f}")
print(f"SNR= 0 dB → σ={sigma_low:.4f}")
print(f"Отношение σ: {sigma_low/sigma_high:.1f}x (ожидаем ~10x)")
print(f"Оригинал mean: {x.mean():.3f}, Зашумлённый (20dB): {noisy_high_snr.mean():.3f}")

# Проверка: шум не меняет форму тензора
assert noisy_high_snr.shape == x.shape
assert sigma_low > sigma_high, "При меньшем SNR σ должна быть больше"
print("noise.py OK")
```

Ожидаемый вывод: `SNR=20 dB → σ≈0.1`, `SNR=0 dB → σ≈1.0`, отношение ~10x.

---

### Task 5: src/optimal_model.py

**Files:**
- Create: `src/optimal_model.py`

- [ ] **Step 1: Написать `src/optimal_model.py`**

```python
import torch
from torch import Tensor


class NearestCentroidClassifier:
    """
    Оптимальный классификатор для модели AWGN с детерминированными сигналами.

    При наблюдении y = x_i + n (n ~ N(0, σ²·I)) оптимальное решение —
    минимум евклидова расстояния до эталонного изображения класса:

        ĉ = argmin_i ||y - x_i||²

    Эталоны x_i задаются методом fit() как средние изображения классов.
    """

    def __init__(self) -> None:
        self.centroids: Tensor | None = None  # (n_classes, C, H, W)

    def fit(self, centroids: Tensor) -> "NearestCentroidClassifier":
        """Сохраняет эталоны классов. centroids: (n_classes, C, H, W)."""
        self.centroids = centroids
        return self

    def predict(self, images: Tensor) -> Tensor:
        """
        Классифицирует батч изображений.

        images: (B, C, H, W)
        returns: (B,) — предсказанные индексы классов
        """
        if self.centroids is None:
            raise RuntimeError("Вызовите fit() перед predict().")

        device     = images.device
        centroids  = self.centroids.to(device)

        # Разворачиваем в (B, D) и (n_classes, D) для вычисления расстояний
        b    = images.shape[0]
        flat = images.view(b, -1)                     # (B, D)
        ctrs = centroids.view(centroids.shape[0], -1) # (n_classes, D)

        # Квадратичные расстояния: (B, n_classes)
        # ||a - b||² = ||a||² + ||b||² - 2·aᵀb
        dists = (
            flat.pow(2).sum(1, keepdim=True)
            + ctrs.pow(2).sum(1).unsqueeze(0)
            - 2 * flat @ ctrs.T
        )
        return dists.argmin(dim=1)  # (B,)
```

- [ ] **Step 2: Проверить в Colab**

```python
# Validation cell
import torch
from src.optimal_model import NearestCentroidClassifier

# 4 класса, изображения 3×32×32
centroids = torch.zeros(4, 3, 32, 32)
for i in range(4):
    centroids[i] = i  # класс i имеет пиксели равные i

clf = NearestCentroidClassifier()
clf.fit(centroids)

# Батч: 4 изображения, каждое близко к своему центроиду
test_images = torch.zeros(4, 3, 32, 32)
for i in range(4):
    test_images[i] = i + 0.01  # чуть зашумлено

preds = clf.predict(test_images)
print("Predictions:", preds.tolist())  # ожидаем [0, 1, 2, 3]
assert preds.tolist() == [0, 1, 2, 3], "Классификатор вернул неверные метки"
print("optimal_model.py OK")
```

Ожидаемый вывод: `Predictions: [0, 1, 2, 3]`.

---

### Task 6: src/cnn_model.py

**Files:**
- Create: `src/cnn_model.py`

- [ ] **Step 1: Написать `src/cnn_model.py`**

```python
import torch.nn as nn
from torch import Tensor


class EyeCNN(nn.Module):
    """
    Свёрточная сеть для классификации глазных снимков.

    Архитектура: 4 блока [Conv2d → BatchNorm → ReLU → MaxPool],
    затем AdaptiveAvgPool → Dropout → Linear.

    AdaptiveAvgPool позволяет использовать одну архитектуру
    для разных размеров входного изображения (32×32 и 128×128).
    """

    def __init__(self, img_size: int, num_classes: int) -> None:
        super().__init__()
        # img_size сохраняем для документации, архитектура его не требует
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
            # AdaptiveAvgPool сжимает пространственные размеры до 1×1
            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)   # (B, 256, 1, 1)
        x = x.flatten(1)       # (B, 256)
        return self.head(x)    # (B, num_classes)
```

- [ ] **Step 2: Проверить в Colab**

```python
# Validation cell
import torch
from src.cnn_model import EyeCNN

for img_size in [32, 128]:
    model  = EyeCNN(img_size=img_size, num_classes=4)
    dummy  = torch.randn(8, 3, img_size, img_size)
    output = model(dummy)
    print(f"img_size={img_size}: output shape = {output.shape}")
    assert output.shape == (8, 4), f"Ожидали (8, 4), получили {output.shape}"

n_params = sum(p.numel() for p in EyeCNN(128, 4).parameters())
print(f"Параметры модели: {n_params:,}")  # ~0.5M
print("cnn_model.py OK")
```

Ожидаемый вывод: `output shape = torch.Size([8, 4])` для обоих размеров, ~500k параметров.

---

### Task 7: src/trainer.py

**Files:**
- Create: `src/trainer.py`

- [ ] **Step 1: Написать `src/trainer.py`**

```python
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from config import ExperimentConfig
from src.cnn_model import EyeCNN
from src.noise import GaussianNoise
from src.optimal_model import NearestCentroidClassifier


class Trainer:
    """Обучает CNN и оптимальный классификатор, оценивает точность на зашумлённых данных."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)

    def train_cnn(self, model: EyeCNN, loader: DataLoader) -> EyeCNN:
        """Обучает CNN на чистых данных. Возвращает обученную модель."""
        model     = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        # Снижаем lr вдвое каждые 7 эпох
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

        model.train()
        for epoch in range(self.config.num_epochs):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                criterion(model(images), labels).backward()
                optimizer.step()
            scheduler.step()

        return model

    def fit_optimal(self, centroids: Tensor) -> NearestCentroidClassifier:
        """Создаёт оптимальный классификатор из центроидов классов."""
        clf = NearestCentroidClassifier()
        clf.fit(centroids)
        return clf

    @torch.no_grad()
    def evaluate(
        self,
        model: EyeCNN | NearestCentroidClassifier,
        loader: DataLoader,
        snr_db: float,
    ) -> float:
        """Оценивает точность модели на тест-данных с заданным уровнем шума."""
        is_cnn = isinstance(model, EyeCNN)
        if is_cnn:
            model.eval()

        correct = total = 0
        for images, labels in loader:
            images = images.to(self.device)
            noisy  = GaussianNoise.add(images, snr_db)

            if is_cnn:
                preds = model(noisy).argmax(dim=1)
            else:
                preds = model.predict(noisy)

            correct += (preds.cpu() == labels).sum().item()
            total   += len(labels)

        return correct / total
```

- [ ] **Step 2: Проверить в Colab (требует данных)**

```python
# Validation cell
from config import ExperimentConfig
from src.data_loader import EyeDataset
from src.cnn_model import EyeCNN
from src.trainer import Trainer

cfg     = ExperimentConfig()
cfg.num_epochs = 1  # быстрая проверка
ds      = EyeDataset(cfg)
trainer = Trainer(cfg)

train_loader, test_loader = ds.get_loaders(img_size=32)
centroids = ds.compute_centroids(train_loader)

# Тест CNN
cnn = EyeCNN(img_size=32, num_classes=4)
cnn = trainer.train_cnn(cnn, train_loader)
acc_cnn = trainer.evaluate(cnn, test_loader, snr_db=20.0)
print(f"CNN accuracy (1 epoch, SNR=20dB): {acc_cnn:.3f}")
assert 0.0 < acc_cnn <= 1.0

# Тест оптимального
optimal = trainer.fit_optimal(centroids)
acc_opt = trainer.evaluate(optimal, test_loader, snr_db=20.0)
print(f"Optimal accuracy (SNR=20dB): {acc_opt:.3f}")
assert 0.0 < acc_opt <= 1.0

print("trainer.py OK")
```

Ожидаемый вывод: обе точности в диапазоне (0, 1), без исключений.

---

### Task 8: src/experiment.py

**Files:**
- Create: `src/experiment.py`

- [ ] **Step 1: Написать `src/experiment.py`**

```python
import numpy as np
import torch

from config import ExperimentConfig
from src.cnn_model import EyeCNN
from src.data_loader import EyeDataset
from src.trainer import Trainer


class MonteCarloRunner:
    """
    Реализует метод Монте-Карло для оценки устойчивости классификаторов к шуму.

    Для каждого повторения:
      1. Случайный train/test сплит
      2. Обучение обоих классификаторов
      3. Оценка точности при каждом уровне SNR

    Результат: матрица точностей (n_snr_levels × n_repetitions) для каждого классификатора.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self.config  = config
        self.dataset = EyeDataset(config)
        self.trainer = Trainer(config)

    def run(
        self,
        img_size: int,
        train_fraction: float = 1.0,
    ) -> dict[str, np.ndarray]:
        """
        Запускает Монте-Карло эксперимент.

        Returns:
            {
                "cnn":     ndarray (n_snr, n_repetitions),
                "optimal": ndarray (n_snr, n_repetitions),
            }
        """
        cfg   = self.config
        n_snr = len(cfg.snr_levels_db)
        n_rep = cfg.n_repetitions

        results: dict[str, np.ndarray] = {
            "cnn":     np.zeros((n_snr, n_rep)),
            "optimal": np.zeros((n_snr, n_rep)),
        }

        for rep in range(n_rep):
            print(f"  [img={img_size} frac={train_fraction:.0%}] "
                  f"Повторение {rep+1}/{n_rep}", end="\r")

            # Новый случайный сплит для каждого повторения
            train_loader, test_loader = self.dataset.get_loaders(img_size, train_fraction)
            centroids = self.dataset.compute_centroids(train_loader)

            # Обучаем оба классификатора на одних и тех же данных
            cnn = self.trainer.train_cnn(
                EyeCNN(img_size, num_classes=len(self.dataset.class_names or [""] * 4)),
                train_loader,
            )
            optimal = self.trainer.fit_optimal(centroids)

            # Оцениваем при каждом уровне шума
            for snr_idx, snr_db in enumerate(cfg.snr_levels_db):
                results["cnn"][snr_idx, rep]     = self.trainer.evaluate(cnn,     test_loader, snr_db)
                results["optimal"][snr_idx, rep] = self.trainer.evaluate(optimal, test_loader, snr_db)

        print()  # завершить строку \r
        return results

    def run_samples_experiment(self, img_size: int) -> dict[float, dict[str, np.ndarray]]:
        """
        Серия 2: точность vs число обучающих выборок при фиксированном SNR.

        Returns:
            { train_fraction: {"cnn": ndarray(n_rep), "optimal": ndarray(n_rep)} }
        """
        cfg     = self.config
        snr_db  = cfg.fixed_snr_for_samples_exp
        results = {}

        for frac in cfg.train_fractions:
            print(f"  Fraction={frac:.0%}")
            r = self.run(img_size=img_size, train_fraction=frac)
            # Берём среднее по SNR-уровням нет — берём только фиксированный SNR
            snr_idx = list(cfg.snr_levels_db).index(snr_db)
            results[frac] = {
                "cnn":     r["cnn"][snr_idx],
                "optimal": r["optimal"][snr_idx],
            }

        return results
```

- [ ] **Step 2: Проверить в Colab (быстрая версия)**

```python
# Validation cell — используем 2 повторения и 1 эпоху для быстрой проверки
from config import ExperimentConfig
from src.experiment import MonteCarloRunner

cfg = ExperimentConfig()
cfg.n_repetitions = 2
cfg.num_epochs    = 1

runner  = MonteCarloRunner(cfg)
results = runner.run(img_size=32, train_fraction=1.0)

print("Keys:", list(results.keys()))
print("CNN shape:", results["cnn"].shape)     # (6, 2)
print("Optimal shape:", results["optimal"].shape)  # (6, 2)
print("CNN mean per SNR:", results["cnn"].mean(axis=1).round(3))
print("Optimal mean per SNR:", results["optimal"].mean(axis=1).round(3))
assert results["cnn"].shape == (6, 2)
print("experiment.py OK")
```

Ожидаемый вывод: shape `(6, 2)`, числа в диапазоне [0, 1].

---

### Task 9: src/evaluate.py

**Files:**
- Create: `src/evaluate.py`

- [ ] **Step 1: Написать `src/evaluate.py`**

```python
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from config import ExperimentConfig
from src.cnn_model import EyeCNN
from src.noise import GaussianNoise
from src.optimal_model import NearestCentroidClassifier


class Evaluator:
    """Строит все графики и confusion matrix'ы, сохраняет на диск и в Google Drive."""

    def __init__(self, config: ExperimentConfig, class_names: list[str]) -> None:
        self.config       = config
        self.class_names  = class_names
        self.results_dir  = Path(config.results_dir)
        self.plots_dir    = self.results_dir / "plots"
        self.cm_dir       = self.results_dir / "confusion_matrices"

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.cm_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Серия 1: Точность vs SNR
    # ------------------------------------------------------------------

    def accuracy_vs_snr(
        self,
        results_by_size: dict[int, dict[str, np.ndarray]],
    ) -> None:
        """
        Основной график: точность vs SNR для обоих классификаторов и обоих размеров.

        results_by_size: { img_size: {"cnn": (n_snr, n_rep), "optimal": (n_snr, n_rep)} }
        """
        n_sizes = len(self.config.img_sizes)
        fig, axes = plt.subplots(1, n_sizes, figsize=(7 * n_sizes, 5))
        if n_sizes == 1:
            axes = [axes]

        colors = {"cnn": "#2196F3", "optimal": "#FF5722"}
        labels = {"cnn": "CNN (обучение с учителем)", "optimal": "Оптимальный (ближайший центроид)"}
        snr    = list(self.config.snr_levels_db)

        for ax, img_size in zip(axes, self.config.img_sizes):
            results = results_by_size[img_size]
            for clf_name, data in results.items():
                mean = data.mean(axis=1)
                std  = data.std(axis=1)
                ax.plot(snr, mean, marker="o", label=labels[clf_name],
                        color=colors[clf_name], linewidth=2)
                ax.fill_between(snr, mean - std, mean + std,
                                alpha=0.2, color=colors[clf_name])

            ax.set_title(f"Разрешение {img_size}×{img_size}", fontsize=13)
            ax.set_xlabel("SNR (дБ)", fontsize=11)
            ax.set_ylabel("Точность", fontsize=11)
            ax.set_ylim(0, 1.05)
            ax.invert_xaxis()  # высокий SNR слева = чистый сигнал слева
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)

        fig.suptitle("Точность классификации vs уровень шума (SNR)", fontsize=14, y=1.02)
        plt.tight_layout()
        self._save_fig(fig, self.plots_dir / "accuracy_vs_snr.png")

    # ------------------------------------------------------------------
    # Серия 2: Точность vs число обучающих выборок
    # ------------------------------------------------------------------

    def accuracy_vs_samples(
        self,
        results_by_fraction: dict[float, dict[str, np.ndarray]],
        img_size: int,
    ) -> None:
        """
        График: точность vs доля обучающих данных при фиксированном SNR.

        results_by_fraction: { fraction: {"cnn": (n_rep,), "optimal": (n_rep,)} }
        """
        fractions   = list(results_by_fraction.keys())
        clf_names   = list(next(iter(results_by_fraction.values())).keys())
        x           = np.arange(len(fractions))
        width       = 0.35
        colors      = {"cnn": "#2196F3", "optimal": "#FF5722"}
        labels      = {"cnn": "CNN", "optimal": "Оптимальный"}

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, clf_name in enumerate(clf_names):
            means = [results_by_fraction[f][clf_name].mean() for f in fractions]
            stds  = [results_by_fraction[f][clf_name].std()  for f in fractions]
            bars  = ax.bar(x + i * width, means, width,
                           label=labels[clf_name], color=colors[clf_name],
                           yerr=stds, capsize=5, alpha=0.85)

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([f"{int(f * 100)}%" for f in fractions], fontsize=11)
        ax.set_xlabel("Объём обучающей выборки", fontsize=11)
        ax.set_ylabel("Точность", fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"Точность vs число обучающих выборок\n"
            f"(SNR = {self.config.fixed_snr_for_samples_exp} дБ, {img_size}×{img_size})",
            fontsize=13,
        )
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        self._save_fig(fig, self.plots_dir / f"accuracy_vs_samples_{img_size}.png")

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        model: EyeCNN | NearestCentroidClassifier,
        loader: DataLoader,
        snr_db: float,
        title: str,
    ) -> None:
        """Строит и сохраняет confusion matrix при заданном SNR."""
        device    = torch.device(self.config.device)
        is_cnn    = isinstance(model, EyeCNN)
        all_preds: list[int] = []
        all_true:  list[int] = []

        if is_cnn:
            model.eval()

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                noisy  = GaussianNoise.add(images, snr_db)

                preds = model(noisy).argmax(1) if is_cnn else model.predict(noisy)
                all_preds.extend(preds.cpu().tolist())
                all_true.extend(labels.tolist())

        cm  = confusion_matrix(all_true, all_preds)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax,
        )
        ax.set_xlabel("Предсказание", fontsize=11)
        ax.set_ylabel("Истина",       fontsize=11)
        ax.set_title(title,           fontsize=12)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        safe_name = title.replace(" ", "_").replace("=", "").replace(",", "").replace(".", "")
        self._save_fig(fig, self.cm_dir / f"{safe_name}.png")

    # ------------------------------------------------------------------
    # Сохранение на Google Drive
    # ------------------------------------------------------------------

    def save_to_drive(self) -> None:
        """Копирует папку results/ в Google Drive для сохранения между сессиями."""
        drive = Path(self.config.drive_dir)
        if not drive.exists():
            print("Google Drive не смонтирован, пропускаем сохранение.")
            return
        dest = drive / "results"
        shutil.copytree(str(self.results_dir), str(dest), dirs_exist_ok=True)
        print(f"Результаты сохранены в {dest}")

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _save_fig(fig: plt.Figure, path: Path) -> None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Сохранено: {path}")
```

- [ ] **Step 2: Проверить в Colab**

```python
# Validation cell
import numpy as np
from config import ExperimentConfig
from src.evaluate import Evaluator

cfg = ExperimentConfig()
cfg.n_repetitions = 2

# Синтетические результаты для проверки графиков
fake_results = {
    32:  {"cnn": np.random.uniform(0.5, 0.9, (6, 2)), "optimal": np.random.uniform(0.4, 0.8, (6, 2))},
    128: {"cnn": np.random.uniform(0.6, 0.95, (6, 2)), "optimal": np.random.uniform(0.5, 0.85, (6, 2))},
}
evaluator = Evaluator(cfg, class_names=["cataract", "diabetic_retinopathy", "glaucoma", "normal"])
evaluator.accuracy_vs_snr(fake_results)

fake_samples = {
    0.25: {"cnn": np.array([0.55, 0.57]), "optimal": np.array([0.45, 0.48])},
    0.50: {"cnn": np.array([0.70, 0.72]), "optimal": np.array([0.55, 0.58])},
    1.00: {"cnn": np.array([0.85, 0.87]), "optimal": np.array([0.65, 0.68])},
}
evaluator.accuracy_vs_samples(fake_samples, img_size=128)
print("evaluate.py OK — графики отображены")
```

Ожидаемый вывод: два графика отображаются в Colab без ошибок.

---

### Task 10: main.ipynb

**Files:**
- Create: `main.ipynb`

- [ ] **Step 1: Написать `main.ipynb`**

Файл создаётся как JSON. Каждая ячейка — отдельный блок. Ниже — содержимое ячеек:

**Ячейка 1 — Установка и клонирование:**
```python
# @title Ячейка 1: Установка зависимостей
!pip install kaggle torch torchvision scikit-learn matplotlib seaborn tqdm -q

# Настройка Kaggle API (загрузи kaggle.json в Colab перед этим)
import os
os.makedirs("/root/.kaggle", exist_ok=True)
# Если файл не загружен, создай вручную:
# with open("/root/.kaggle/kaggle.json", "w") as f:
#     f.write('{"username":"ВАШ_USERNAME","key":"ВАШ_API_KEY"}')
!chmod 600 /root/.kaggle/kaggle.json 2>/dev/null || true
```

**Ячейка 2 — Монтирование Google Drive:**
```python
# @title Ячейка 2: Google Drive
from google.colab import drive
drive.mount("/content/drive")
```

**Ячейка 3 — Импорты:**
```python
# @title Ячейка 3: Импорты
import sys
sys.path.insert(0, ".")

from config import ExperimentConfig
from src.data_loader import EyeDataset
from src.experiment import MonteCarloRunner
from src.evaluate import Evaluator

cfg = ExperimentConfig()
print(f"Device: {cfg.device}")
print(f"Image sizes: {cfg.img_sizes}")
print(f"SNR levels: {cfg.snr_levels_db}")
print(f"Repetitions: {cfg.n_repetitions}")
```

**Ячейка 4 — Загрузка данных:**
```python
# @title Ячейка 4: Загрузка и проверка данных
dataset = EyeDataset(cfg)
dataset.download()

train_loader, test_loader = dataset.get_loaders(img_size=128)
print(f"Классы: {dataset.class_names}")
print(f"Обучающих батчей: {len(train_loader)}")
print(f"Тестовых батчей:  {len(test_loader)}")
```

**Ячейка 5 — Серия 1: Точность vs SNR:**
```python
# @title Ячейка 5: Монте-Карло — точность vs SNR (~1.5 часа)
runner  = MonteCarloRunner(cfg)
results_by_size: dict[int, dict] = {}

for img_size in cfg.img_sizes:
    print(f"\n=== Размер {img_size}×{img_size} ===")
    results_by_size[img_size] = runner.run(img_size=img_size, train_fraction=1.0)

print("\nГотово!")
```

**Ячейка 6 — Серия 2: Точность vs выборки:**
```python
# @title Ячейка 6: Монте-Карло — точность vs число выборок
results_by_fraction = runner.run_samples_experiment(img_size=128)
print("Готово!")
```

**Ячейка 7 — Графики:**
```python
# @title Ячейка 7: Построение графиков
evaluator = Evaluator(cfg, class_names=dataset.class_names)

# График 1: точность vs SNR
evaluator.accuracy_vs_snr(results_by_size)

# График 2: точность vs число выборок
evaluator.accuracy_vs_samples(results_by_fraction, img_size=128)
```

**Ячейка 8 — Confusion matrices:**
```python
# @title Ячейка 8: Матрицы ошибок
from src.cnn_model import EyeCNN
from src.trainer import Trainer

trainer = Trainer(cfg)

# Обучаем финальную модель на 100% данных для confusion matrix
train_loader_128, test_loader_128 = dataset.get_loaders(img_size=128)
centroids = dataset.compute_centroids(train_loader_128)

final_cnn     = trainer.train_cnn(EyeCNN(128, num_classes=4), train_loader_128)
final_optimal = trainer.fit_optimal(centroids)

# Матрицы при низком и высоком шуме
for snr_db in [20.0, 0.0]:
    label = "мало шума" if snr_db == 20.0 else "много шума"
    evaluator.plot_confusion_matrix(
        final_cnn, test_loader_128, snr_db,
        title=f"CNN — SNR={snr_db} дБ ({label})",
    )
    evaluator.plot_confusion_matrix(
        final_optimal, test_loader_128, snr_db,
        title=f"Оптимальный — SNR={snr_db} дБ ({label})",
    )
```

**Ячейка 9 — Сохранение:**
```python
# @title Ячейка 9: Сохранение на Google Drive
evaluator.save_to_drive()
print("Все результаты сохранены!")
```

- [ ] **Step 2: Проверить структуру**

```python
# Validation cell
from pathlib import Path
import json

nb = json.loads(Path("main.ipynb").read_text())
n_cells = len(nb["cells"])
print(f"Ячеек в ноутбуке: {n_cells}")  # ожидаем 9
assert n_cells == 9
print("main.ipynb OK")
```

---

## Финальная проверка

После реализации всех задач запустить в Colab по порядку ячейки 1–9 и убедиться:

- [ ] Данные скачиваются без ошибок
- [ ] `dataset.class_names` содержит 4 класса
- [ ] Монте-Карло завершается за ~1.5 часа на GPU Colab
- [ ] Два графика точности сохранены в `results/plots/`
- [ ] Четыре confusion matrix сохранены в `results/confusion_matrices/`
- [ ] Папка `results/` скопирована на Google Drive
