"""
IDR MVP -- AI Velocity Model Trainer
=====================================
Orchestrates training, validation, checkpointing, and TFLite model export for the 1D CNN
forward velocity estimator.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np

from .model import build_1d_cnn_velocity_model, NumpyCNNInference


class VelocityModelTrainer:
    """
    Manages model compilation, training loops, callbacks, evaluation, and multi-format exports.
    """

    def __init__(
        self,
        input_shape: Tuple[int, int] = (50, 6),
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 35,
    ):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.history = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        verbose: int = 1,
    ) -> Dict[str, Any]:
        """
        Trains the 1D CNN model with EarlyStopping and ReduceLROnPlateau callbacks.
        """
        import keras

        self.model = build_1d_cnn_velocity_model(
            input_shape=self.input_shape,
            learning_rate=self.learning_rate,
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=verbose,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-5,
                verbose=verbose,
            ),
        ]

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            verbose=verbose,
        )

        return self.history.history

    def export_models(
        self,
        export_dir: Union[str, Path],
        model_name: str = "velocity_cnn",
    ) -> Dict[str, Path]:
        """
        Exports the trained model into:
        1. Keras native format (.keras)
        2. TFLite model (.tflite) for Android deployment per TECH_STACK.md §5
        3. Compressed NumPy weights (.npz) for pure-NumPy runtime
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        export_dir = Path(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        paths = {}

        # 1. Save Keras format
        keras_path = export_dir / f"{model_name}.keras"
        self.model.save(keras_path)
        paths["keras"] = keras_path

        # 2. Save NumPy weights
        npz_path = export_dir / f"{model_name}_weights.npz"
        np_engine = NumpyCNNInference.from_keras_model(self.model)
        np_engine.save_weights_npz(npz_path)
        paths["npz"] = npz_path

        # 3. Export TFLite
        try:
            import tensorflow as tf
            tflite_path = export_dir / f"{model_name}.tflite"
            converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            paths["tflite"] = tflite_path
        except Exception as e:
            print(f"  [Warning] TFLite conversion encountered: {e}")

        return paths
