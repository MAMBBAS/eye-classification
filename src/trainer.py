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
        """
        Обучает CNN на чистых (незашумлённых) данных.

        Шум добавляется только при оценке (evaluate), не при обучении — это
        классический сценарий «обучение на чистых, тест на зашумлённых».
        """
        model     = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        # Снижаем lr вдвое каждые 7 эпох для стабилизации обучения
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

        model.train()
        for _ in range(self.config.num_epochs):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                criterion(model(images), labels).backward()
                optimizer.step()
            scheduler.step()

        return model

    def fit_optimal(self, centroids: Tensor) -> NearestCentroidClassifier:
        """Создаёт оптимальный классификатор из предвычисленных центроидов классов."""
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
        """
        Оценивает долю правильных классификаций на тест-данных с заданным уровнем шума.

        Шум добавляется батч-по-батчу чтобы σ вычислялась из реальной мощности сигнала.
        """
        is_cnn = isinstance(model, EyeCNN)
        if is_cnn:
            model.eval()

        correct = total = 0
        for images, labels in loader:
            images = images.to(self.device)
            noisy  = GaussianNoise.add(images, snr_db)

            preds = model(noisy).argmax(dim=1) if is_cnn else model.predict(noisy)

            correct += (preds.cpu() == labels).sum().item()
            total   += len(labels)

        return correct / total
