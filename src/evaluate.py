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

    _COLORS = {"cnn": "#2196F3", "optimal": "#FF5722"}
    _LABELS = {"cnn": "CNN (обучение с учителем)", "optimal": "Оптимальный (ближайший центроид)"}

    def __init__(self, config: ExperimentConfig, class_names: list[str]) -> None:
        self.config      = config
        self.class_names = class_names

        self.plots_dir = Path(config.results_dir) / "plots"
        self.cm_dir    = Path(config.results_dir) / "confusion_matrices"
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

        Полоса вокруг линии — ±1 стандартное отклонение по повторениям Монте-Карло.

        Args:
            results_by_size: { img_size: {"cnn": (n_snr, n_rep), "optimal": (n_snr, n_rep)} }
        """
        n_sizes = len(self.config.img_sizes)
        fig, axes = plt.subplots(1, n_sizes, figsize=(7 * n_sizes, 5), sharey=True)
        if n_sizes == 1:
            axes = [axes]

        snr = list(self.config.snr_levels_db)

        for ax, img_size in zip(axes, self.config.img_sizes):
            results = results_by_size[img_size]
            for clf_name, data in results.items():
                mean = data.mean(axis=1)
                std  = data.std(axis=1)
                ax.plot(
                    snr, mean,
                    marker="o", linewidth=2,
                    label=self._LABELS[clf_name],
                    color=self._COLORS[clf_name],
                )
                ax.fill_between(snr, mean - std, mean + std, alpha=0.2, color=self._COLORS[clf_name])

            ax.set_title(f"Разрешение {img_size}×{img_size}", fontsize=13)
            ax.set_xlabel("SNR (дБ)", fontsize=11)
            ax.set_ylabel("Точность", fontsize=11)
            ax.set_ylim(0, 1.05)
            ax.invert_xaxis()  # слева — чистый сигнал (высокий SNR), справа — шум
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.4)

        fig.suptitle("Точность классификации vs уровень шума", fontsize=14)
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

        Args:
            results_by_fraction: { fraction: {"cnn": (n_rep,), "optimal": (n_rep,)} }
            img_size:            Размер изображений (для заголовка).
        """
        fractions = list(results_by_fraction.keys())
        clf_names = list(next(iter(results_by_fraction.values())).keys())
        x         = np.arange(len(fractions))
        width     = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, clf_name in enumerate(clf_names):
            means = [results_by_fraction[f][clf_name].mean() for f in fractions]
            stds  = [results_by_fraction[f][clf_name].std()  for f in fractions]
            ax.bar(
                x + i * width, means, width,
                label=self._LABELS[clf_name],
                color=self._COLORS[clf_name],
                yerr=stds, capsize=5, alpha=0.85,
            )

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
        """Строит confusion matrix при заданном уровне шума и сохраняет в PNG."""
        device = torch.device(self.config.device)
        is_cnn = isinstance(model, EyeCNN)
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

        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in title)
        self._save_fig(fig, self.cm_dir / f"{safe_name}.png")

    # ------------------------------------------------------------------
    # Сохранение на Google Drive
    # ------------------------------------------------------------------

    def save_to_drive(self) -> None:
        """Копирует папку results/ в Google Drive для сохранения между Colab-сессиями."""
        drive = Path(self.config.drive_dir)
        if not drive.exists():
            print("Google Drive не смонтирован — пропускаем сохранение на Drive.")
            return
        dest = drive / "results"
        shutil.copytree(str(Path(self.config.results_dir)), str(dest), dirs_exist_ok=True)
        print(f"Результаты сохранены в {dest}")

    # ------------------------------------------------------------------
    # Вспомогательное
    # ------------------------------------------------------------------

    @staticmethod
    def _save_fig(fig: plt.Figure, path: Path) -> None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"  Сохранено: {path}")
