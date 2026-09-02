"""
IDR MVP -- AI Package
======================
Lightweight AI models for vehicle motion and forward velocity estimation.
"""

from .dataset import IMUScaler, IMUSequenceDataset
from .model import build_1d_cnn_velocity_model, NumpyCNNInference
from .trainer import VelocityModelTrainer
from .evaluator import VelocityEvaluator, VelocityMetrics

__all__ = [
    "IMUScaler",
    "IMUSequenceDataset",
    "build_1d_cnn_velocity_model",
    "NumpyCNNInference",
    "VelocityModelTrainer",
    "VelocityEvaluator",
    "VelocityMetrics",
]
