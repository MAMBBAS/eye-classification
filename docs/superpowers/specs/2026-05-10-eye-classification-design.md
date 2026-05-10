# Eye Disease Classification — Design Spec
**Date:** 2026-05-10  
**Environment:** Google Colab + Google Drive  
**Dataset:** `gunavenkatdoddi/eye-diseases-classification` (Kaggle)

---

## 1. Цель

Учебный проект по курсу «Основы теории принятия решений». Исследование устойчивости алгоритмов классификации глазных заболеваний к аддитивному гауссовскому шуму методом Монте-Карло.

**4 класса:** Normal, Cataract, Glaucoma, Diabetic Retinopathy  
**2 классификатора:** NearestCentroid (оптимальный, априори известные образы) + CNN (обучение с учителем)  
**2 размера изображений:** 32×32 и 128×128  

---

## 2. Структура файлов

```
eye_disease_classifier/
├── config.py              # ExperimentConfig dataclass — все параметры в одном месте
├── src/
│   ├── data_loader.py     # EyeDataset — загрузка, препроцессинг, сплиты, центроиды
│   ├── noise.py           # GaussianNoise — SNR↔σ, применение шума
│   ├── cnn_model.py       # EyeCNN — nn.Module, 4 свёрточных слоя
│   ├── optimal_model.py   # NearestCentroidClassifier — минимум евклидова расстояния
│   ├── trainer.py         # Trainer — цикл обучения CNN, fit центроидов
│   ├── experiment.py      # MonteCarloRunner — внешний цикл по SNR и повторениям
│   └── evaluate.py        # Evaluator — метрики, confusion matrix, графики
├── main.ipynb             # Точка входа в Colab — только оркестрация
└── results/               # Создаётся автоматически, зеркалится на Google Drive
    ├── plots/
    └── confusion_matrices/
```

---

## 3. Config

```python
@dataclass
class ExperimentConfig:
    img_sizes: tuple[int, ...]         = (32, 128)
    train_ratio: float                 = 0.8
    batch_size: int                    = 32
    class_names: tuple[str, ...]       = ("Normal", "Cataract", "Glaucoma", "Diabetic Retinopathy")
    snr_levels_db: tuple[float, ...]   = (20.0, 15.0, 10.0, 5.0, 0.0, -5.0)
    n_repetitions: int                 = 15
    train_fractions: tuple[float, ...] = (0.25, 0.50, 1.0)
    num_epochs: int                    = 20
    learning_rate: float               = 1e-3
    drive_dir: str                     = "/content/drive/MyDrive/eye_classification"
    results_dir: str                   = "results"
```

---

## 4. Поток данных

```
config.py
    ↓
EyeDataset       →  DataLoader(train) + DataLoader(test) + centroids tensor
    ↓
GaussianNoise    →  зашумлённые батчи по заданному SNR (σ вычисляется из мощности сигнала)
    ↓
Trainer          →  обученная EyeCNN / подогнанный NearestCentroidClassifier
    ↓
MonteCarloRunner →  dict{"cnn": ndarray(n_snr, n_rep), "optimal": ndarray(n_snr, n_rep)}
    ↓
Evaluator        →  графики + confusion matrices → results/ + Google Drive
```

---

## 5. Интерфейсы классов

### data_loader.py
```python
class EyeDataset:
    def download(self) -> None
    def get_loaders(self, img_size: int, train_fraction: float = 1.0) -> tuple[DataLoader, DataLoader]
    def compute_centroids(self, loader: DataLoader) -> Tensor  # (n_classes, C, H, W)
```

### noise.py
```python
class GaussianNoise:
    @staticmethod
    def snr_to_sigma(images: Tensor, snr_db: float) -> float
    @staticmethod
    def add(images: Tensor, snr_db: float) -> Tensor
```

### cnn_model.py
```python
class EyeCNN(nn.Module):
    def __init__(self, img_size: int, num_classes: int): ...
    def forward(self, x: Tensor) -> Tensor
```
Архитектура: Conv2d(3→32) → Conv2d(32→64) → Conv2d(64→128) → Conv2d(128→256) → AdaptiveAvgPool → Linear  
BatchNorm + ReLU + MaxPool после каждого свёрточного блока. Dropout(0.5) перед головой.

### optimal_model.py
```python
class NearestCentroidClassifier:
    def fit(self, centroids: Tensor) -> None       # сохраняет эталоны классов
    def predict(self, images: Tensor) -> Tensor    # argmin евклидова расстояния до центроида
```

### trainer.py
```python
class Trainer:
    def train_cnn(self, model: EyeCNN, loader: DataLoader) -> EyeCNN
    def fit_optimal(self, centroids: Tensor) -> NearestCentroidClassifier
    def evaluate(self, model, loader: DataLoader, snr_db: float) -> float  # accuracy
```

### experiment.py
```python
class MonteCarloRunner:
    def run(self, img_size: int, train_fraction: float = 1.0) -> dict[str, np.ndarray]
    # Возвращает {"cnn": (n_snr, n_rep), "optimal": (n_snr, n_rep)}
```

### evaluate.py
```python
class Evaluator:
    def accuracy_vs_snr(self, results: dict, img_size: int) -> None
    def accuracy_vs_samples(self, results_by_fraction: dict) -> None
    def plot_confusion_matrix(self, model, loader, snr_db: float, title: str) -> None
    def save_all(self) -> None  # results/ + Google Drive
```

---

## 6. Эксперименты

### Серия 1 — Точность vs SNR
- Для каждого `img_size` ∈ {32, 128}
- Для каждого `snr_db` ∈ {20, 15, 10, 5, 0, -5}
- 15 повторений Монте-Карло
- Результат: mean ± std accuracy на тест-сете

### Серия 2 — Точность vs число обучающих выборок
- Фиксированный SNR = 10 dB
- Для каждого `train_fraction` ∈ {0.25, 0.5, 1.0}
- 15 повторений
- Сравниваем оба классификатора

### Confusion matrix
- При SNR = 20 dB (чистые данные)
- При SNR = 0 dB (высокий шум)
- Для CNN и NearestCentroid отдельно

---

## 7. Математическая модель шума

Сигнал: **x** — изображение, нормализованное в [0, 1]  
Шум: **n** ~ N(0, σ²·I)  
Наблюдение: **y = x + n**  

SNR в dB:  
```
SNR_dB = 10 · log10(P_signal / σ²)
P_signal = mean(x²)          # мощность сигнала по батчу
σ² = P_signal / 10^(SNR_dB/10)
```

---

## 8. Обязательные графики для отчёта

1. Точность vs SNR — линейный график, обе модели, оба размера изображений
2. Точность vs число обучающих выборок — bar chart
3. Confusion matrix при низком шуме (SNR=20 dB)
4. Confusion matrix при высоком шуме (SNR=0 dB)
