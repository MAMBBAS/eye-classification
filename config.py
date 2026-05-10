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
    num_classes: int                 = 4

    # --- Шум ---
    # Уровни SNR в dB: от хорошего сигнала (20) до сильного шума (-5)
    snr_levels_db: tuple[float, ...] = (20.0, 15.0, 10.0, 5.0, 0.0, -5.0)

    # --- Монте-Карло ---
    n_repetitions: int                 = 15
    # Доли обучающей выборки для серии "точность vs число выборок"
    train_fractions: tuple[float, ...] = (0.25, 0.50, 1.0)
    # SNR фиксируется в эксперименте с числом выборок
    fixed_snr_for_samples_exp: float   = 10.0

    # --- CNN ---
    num_epochs: int      = 20
    learning_rate: float = 1e-3

    # --- Пути ---
    results_dir: str = "results"
    drive_dir: str   = "/content/drive/MyDrive/eye_classification"

    # --- Устройство (определяется автоматически через __post_init__) ---
    device: str = field(default="")

    def __post_init__(self) -> None:
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
