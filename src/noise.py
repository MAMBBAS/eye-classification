import math

import torch
from torch import Tensor


class GaussianNoise:
    """
    Аддитивный белый гауссовский шум (AWGN).

    Математическая модель сигнала:
        y = x + n,   n ~ N(0, σ²·I)

    где x — нормализованное изображение, n — шум.

    Отношение сигнал/шум в децибелах:
        SNR_dB = 10·log₁₀(P_signal / σ²)
        P_signal = E[x²]  — вычисляется как среднее квадратов пикселей батча

    Из определения SNR выражаем σ:
        σ² = P_signal / 10^(SNR_dB / 10)
        σ  = √P_signal / 10^(SNR_dB / 20)
    """

    @staticmethod
    def snr_to_sigma(images: Tensor, snr_db: float) -> float:
        """Вычисляет стандартное отклонение шума σ по мощности сигнала и SNR."""
        p_signal = images.pow(2).mean().item()
        if p_signal == 0:
            return 0.0
        sigma_sq = p_signal / (10 ** (snr_db / 10))
        return math.sqrt(sigma_sq)

    @staticmethod
    def add(images: Tensor, snr_db: float) -> Tensor:
        """Возвращает зашумлённую копию батча при заданном SNR. Оригинал не изменяется."""
        sigma = GaussianNoise.snr_to_sigma(images, snr_db)
        noise = torch.randn_like(images) * sigma
        return images + noise
