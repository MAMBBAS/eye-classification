import numpy as np

from config import ExperimentConfig
from src.cnn_model import EyeCNN
from src.data_loader import EyeDataset
from src.trainer import Trainer


class MonteCarloRunner:
    """
    Реализует метод Монте-Карло для статистической оценки устойчивости классификаторов к шуму.

    Для каждого повторения:
      1. Новый случайный train/test сплит
      2. Обучение CNN и подгонка оптимального классификатора на одних данных
      3. Оценка точности обоих при каждом уровне SNR

    Результат — матрица [n_snr_levels × n_repetitions] для каждого классификатора,
    из которой вычисляются mean ± std для построения графиков.
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
        Запускает Монте-Карло эксперимент для одного размера изображения.

        Args:
            img_size:       Разрешение входных изображений (32 или 128).
            train_fraction: Доля обучающей выборки (1.0 = всё).

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
            print(
                f"  [img={img_size}px  frac={train_fraction:.0%}]  "
                f"Повторение {rep + 1}/{n_rep}",
                end="\r",
            )

            # Каждое повторение — новый случайный сплит (это суть метода Монте-Карло)
            train_loader, test_loader = self.dataset.get_loaders(img_size, train_fraction)
            centroids = self.dataset.compute_centroids(train_loader)

            # Оба классификатора обучаются на одном сплите для честного сравнения
            cnn = self.trainer.train_cnn(
                EyeCNN(img_size, num_classes=cfg.num_classes),
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
        Серия 2: точность vs объём обучающей выборки при фиксированном SNR.

        Возвращает результаты для каждой доли выборки из config.train_fractions.

        Returns:
            { train_fraction: {"cnn": ndarray(n_rep), "optimal": ndarray(n_rep)} }
        """
        cfg     = self.config
        snr_db  = cfg.fixed_snr_for_samples_exp
        snr_idx = list(cfg.snr_levels_db).index(snr_db)
        results: dict[float, dict[str, np.ndarray]] = {}

        for frac in cfg.train_fractions:
            print(f"\n  Fraction = {frac:.0%}")
            full_results = self.run(img_size=img_size, train_fraction=frac)
            # Извлекаем только строку для фиксированного SNR
            results[frac] = {
                "cnn":     full_results["cnn"][snr_idx],
                "optimal": full_results["optimal"][snr_idx],
            }

        return results
